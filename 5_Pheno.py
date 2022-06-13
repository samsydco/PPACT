#!/usr/bin/env python3

# Create DF containing: subject, age, ECA, Movie, FD

import os
import glob
import tqdm
import pandas as pd
import numpy as np
import deepdish as dd
from scipy.stats import ttest_ind
from settings import *

phenofigdir = figdir+'pheno/'

new_pheno_df = pd.read_csv(phenopath+'sam_all.csv')

subs = [sub.split('/')[-1][4:-5] for sub in glob.glob(fmripreppath + '*.html')]

new_pheno_df = new_pheno_df[new_pheno_df['IDENT_SUBID'].isin(subs)].reset_index(drop=True)

new_pheno_df = new_pheno_df.replace(' ', np.nan)
new_pheno_df = new_pheno_df.apply(pd.to_numeric, errors='coerce').fillna(new_pheno_df)

# MISSING DATA!
# Missing phenotypic data for some subjects:
missingsubs = []
for sub in subs:
	if len(new_pheno_df[new_pheno_df['IDENT_SUBID']==sub])==0:
		missingsubs.append(sub)

# How many timepoints are censored?
motion_outliers = {}
nTR = {}
for sub in tqdm.tqdm(new_pheno_df['IDENT_SUBID']):
	fdir = 'ses-V2W2' if os.path.isdir(fmripreppath + 'sub-' + sub + '/ses-V2W2/') else 'ses-V1W2'
	fname = os.path.join(fmripreppath + 'sub-' + sub + '/' + fdir + '/func/' + 'sub-' + sub + '_' + fdir + '_task-MOVIE_run-1_desc-confounds_timeseries.tsv')
	if not os.path.isfile(fname): fname = fname.replace('_run-1','')
	conf = np.genfromtxt(fname, names=True)
	motion_outliers[sub] = len([k for k in conf.dtype.names if 'motion_outlier' in k])
new_pheno_df['motion_outliers'] = new_pheno_df['IDENT_SUBID'].map(motion_outliers)


# Filling in some missing data with zeros for control subjects
for c in ['W1_CGH_AGE_ADOPT', 'W1_CGH_SUM_EARLYAGE', 'W1_CGH_SUM_LATEAGE']:
	new_pheno_df[c].loc[new_pheno_df['GROUP'] == 'Comparison'] = 0.0
	#new_pheno_df[c] = new_pheno_df[c].fillna(0)
new_pheno_df['Group'] = np.where((new_pheno_df.GROUP == 'Comparison'),'Control','ECA')
# Based on Nim's recommendation
new_pheno_df['Group'][new_pheno_df['IDENT_SUBID']=='PA087'] = 'Control'
new_pheno_df['Group'][new_pheno_df['IDENT_SUBID']=='PA208'] = 'Control'

# Making plots
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="W2_SS_NEW_TOTAL_MEAN.J", y="W1_CGH_AGE_ADOPT",
                hue='W1_CGH_SUM_SWITCH',#palette="ch:r=-.2,d=.3_r",
				linewidth=0,
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Child\'s felt attachment security')
ax.set_ylabel('Age of adoption (months)')
plt.savefig(phenofigdir+'Attachment_vs_adoption.png')

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x='motion_outliers', y="W2_SS_NEW_TOTAL_MEAN.J", 
                hue='W1_CGH_SUM_SWITCH',#palette="ch:r=-.2,d=.3_r",
				linewidth=0,
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Motion outliers')
ax.set_ylabel('Child\'s felt attachment security')
plt.savefig(phenofigdir+'Motion_vs_Attachment.png',bbox_inches = "tight")

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="W1_CGH_AGE_ADOPT", y='W1_CGH_SUM_SWITCH',
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Age of adoption (months)')
ax.set_ylabel('Number of caregiver switches')
plt.tight_layout()
plt.savefig(phenofigdir+'Adoption_age_vs_caregiver_switches.png',bbox_inches = "tight")

sns.set_theme(style="darkgrid", font_scale=2)
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="W1_CGH_AGE_ADOPT", y="motion_outliers",
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Age of adoption (months)')
ax.set_ylabel('Motion Outliers')
plt.tight_layout()
plt.savefig(phenofigdir+'Motion_vs_Adoption_age.png',bbox_inches = "tight")

tempdf = new_pheno_df.rename(columns={'W2_SS_NEW_TOTAL_MEAN.J': 'Felt attachment security', 'GROUP': 'ECA Group'})
sns.set_theme(style="darkgrid", font_scale=2)
sns.displot(tempdf, x='Felt attachment security', col="Group")
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_2.png')
sns.displot(tempdf, x='Felt attachment security', col="ECA Group", height=4, aspect=1.75)
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_1.png')
sns.displot(tempdf, x='motion_outliers',col='Group')
plt.savefig(phenofigdir+'Motion_vs_ECA_group_2.png')
sns.displot(tempdf, x='motion_outliers', col="ECA Group", height=4, aspect=1.75)
plt.savefig(phenofigdir+'Motion_vs_ECA_group_1.png')
thresh = 0.2*225 #45 TRs is 20% of all TRs
tempdf2 = tempdf[tempdf['motion_outliers']<thresh]
sns.displot(tempdf2, x='motion_outliers',col='Group')
plt.savefig(phenofigdir+'Motion_vs_ECA_group_thresh_2.png')
sns.displot(tempdf2, x='motion_outliers', col="ECA Group", height=4, aspect=1.75)
plt.savefig(phenofigdir+'Motion_vs_ECA_group_thresh_1.png')
ndrop = tempdf[tempdf['motion_outliers']>thresh]
ECAdrop = len(ndrop[ndrop['Group']=='ECA'])
ttest = ttest_ind(tempdf['motion_outliers'][tempdf['Group']=='Control'], tempdf['motion_outliers'][tempdf['Group']=='ECA'])

sns.set_theme(font_scale=1)
tempdf = new_pheno_df.rename(columns={'W2_SS_NEW_TOTAL_MEAN.J': 'Felt attachment security','W1_CGH_SUM_SWITCH':'Number of Switches','W1_CGH_AGE_ADOPT':'Adoption Age','W2_SS_NEW_AVAILABILITY_MEAN':'Parent Availability','W2_SS_NEW_RELYSTRESS_MEAN':'Stress reliance','W2_SS_NEW_COMMUNICATION_MEAN':'Communication subscale','W1_CGH_SUM_EARLYAGE':'Earliest disruption','W1_CGH_SUM_LATEAGE':'Latest disruption','W2_AGE':'Age'})
tempdf = tempdf.drop(['W2_MOVIE','IDENT_SUBID','GROUP'], 1)
tempdf['GENDER'] = tempdf['GENDER'].replace({'Female': 0, 'Male': 1})
tempdf['Group'] = tempdf['Group'].replace({'Control': 0, 'ECA': 1})
tempdf = tempdf.dropna()
tempdf = tempdf.astype(float)
f, ax = plt.subplots()
corr = tempdf.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05,annot_kws={"fontsize":8})
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=14)
plt.savefig(phenofigdir+'ECA_heatmap_all.png',bbox_inches = "tight")

tempdf = tempdf[tempdf['motion_outliers']<thresh]
f, ax = plt.subplots()
corr = tempdf.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05,annot_kws={"fontsize":8})
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=14)
plt.savefig(phenofigdir+'ECA_heatmap_all_thresh.png',bbox_inches = "tight")



import seaborn as sns
sns.set_theme(style="darkgrid")
sns.displot(
    new_pheno_df, x="age", col="GROUP", row="GENDER_CHILD",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True))



Phenodf = pd.read_csv(phenopath+'schema_inventory.csv').drop([0]).drop('Unnamed: 4',axis=1)
Phenodf = Phenodf.drop(Phenodf[Phenodf.age_truncated.isnull()].index).reset_index()
Phenodf['Group'] = np.where((Phenodf.GROUP == 'C'),'Control','ECA')
GROUPkey = {'c':'comparison','PI':'previously institutionalized','DA':'domestically adopted','DC':'disrupted care','IFC':'international foster care'}
GENDERkey = {1:'female',0:'male'}

subs=[sub.split('/')[-1].split('.')[0] for sub in glob.glob(prepath+'*.h5')]


Phenodf['FDmean'] = np.nan
Phenodf['FDstd']  = np.nan
Phenodf['FDmax']  = np.nan
for sub in tqdm.tqdm(subs):
	subtmp = sub.replace('sub-','')+'_V2'
	conf = np.genfromtxt(os.path.join(fmripreppath + sub + '/ses-V2W2/func/' + sub + '_ses-V2W2_task-MOVIE_run-1_desc-confounds_timeseries.tsv'), names=True)
	FD = np.nan_to_num(conf['framewise_displacement'])
	Phenodf['FDmean'].loc[Phenodf['IDENT_SUBID'] == subtmp] = np.mean(FD)
	Phenodf['FDstd'].loc[Phenodf['IDENT_SUBID'] == subtmp]  = np.std(FD)
	Phenodf['FDmax'].loc[Phenodf['IDENT_SUBID'] == subtmp]  = np.max(FD)

Phenodf.to_csv(phenopath+'Phenodf.csv', index=False)