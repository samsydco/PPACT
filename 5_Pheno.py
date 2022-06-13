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

#new_pheno_df = pd.read_csv(phenopath+'SHIRLEY_adversityVARS_for_Sam_analysis.csv')
new_pheno_df = pd.read_csv(phenopath+'sam_all.csv')

subs = [sub.split('/')[-1][4:-5] for sub in glob.glob(fmripreppath + '*.html')]

new_pheno_df = new_pheno_df[new_pheno_df['ID'].isin(subs)].reset_index(drop=True)

Phenodf = pd.read_csv(phenopath+'schema_inventory.csv').drop([0]).drop('Unnamed: 4',axis=1)
Phenodf['IDENT_SUBID'] = Phenodf['IDENT_SUBID'].str.replace('_V2','')
Phenodf = Phenodf.rename(columns={'IDENT_SUBID': 'ID'})
new_pheno_df = new_pheno_df.merge(Phenodf, how='left', on='ID')
new_pheno_df = new_pheno_df.replace(' ', np.nan)
new_pheno_df = new_pheno_df.apply(pd.to_numeric, errors='coerce').fillna(new_pheno_df)

# MISSING DATA!
# Create Missing data CSV for Nim and Lisa
# subjects whose adversity characterization differs between dataframes!
nanGROUP = list(new_pheno_df['ID'][new_pheno_df['GROUP_x'].isna()])
missingdf = new_pheno_df[new_pheno_df['GROUP_x'].isna()]
tempdf = new_pheno_df[~new_pheno_df['ID'].isin(nanGROUP)]
prob_subs = tempdf.loc[~(tempdf['GROUP_x'] == tempdf['GROUP_y'])]
prob_subs['GROUP_x'] = np.nan
missingdf = missingdf.append(prob_subs, ignore_index=True)
# ages do not match between dataframes!
age_diff = np.array(new_pheno_df['age_truncated']) - np.array(new_pheno_df['age_rounded_to_years'])
#subjects missing schema score:
noschema = list(tempdf['ID'][tempdf['SS_NEW_TOTAL_MEAN.J'].isna()])
missingdf = missingdf.append(tempdf[tempdf['SS_NEW_TOTAL_MEAN.J'].isna()], ignore_index=True)
tempdf = tempdf[~tempdf['ID'].isin(noschema)]
# no adoption age
noadoptdf = tempdf[tempdf['CGH_AGE_LIVE'].isna()]
noadoptdf = noadoptdf[~noadoptdf['GROUP_x'].isin(['C'])]
missingdf = missingdf.append(noadoptdf, ignore_index=True)
noadoptlist = np.array(noadoptdf['ID'])
# Missing phenotypic data for some subjects:
missingsubs = []
for sub in subs:
	if len(new_pheno_df[new_pheno_df['ID']==sub])==0:
		missingsubs.append(sub)
new_index = pd.RangeIndex(len(missingsubs))
new_df = pd.DataFrame(np.nan, index=new_index, columns=missingdf.columns)
new_df['ID'] = np.resize(missingsubs,len(new_df))
missingdf = missingdf.append(new_df,ignore_index=True)

missingdf.to_csv(phenopath+'missingdata.csv', index=False)

# How many timepoints are censored?
motion_outliers = {}
nTR = {}
for sub in tqdm.tqdm(new_pheno_df['ID']):
	fdir = 'ses-V2W2' if os.path.isdir(fmripreppath + 'sub-' + sub + '/ses-V2W2/') else 'ses-V1W2'
	fname = os.path.join(fmripreppath + 'sub-' + sub + '/' + fdir + '/func/' + 'sub-' + sub + '_' + fdir + '_task-MOVIE_run-1_desc-confounds_timeseries.tsv')
	if not os.path.isfile(fname): fname = fname.replace('_run-1','')
	conf = np.genfromtxt(fname, names=True)
	motion_outliers[sub] = len([k for k in conf.dtype.names if 'motion_outlier' in k])
new_pheno_df['motion_outliers'] = new_pheno_df['ID'].map(motion_outliers)


# Filling in some missing data with zeros for control subjects
for c in ['CGH_AGE_LIVE', 'CGH_AGE_ADOPT', 'CGH_SUM_EARLYAGE', 'CGH_SUM_LATEAGE']:
	new_pheno_df[c].loc[new_pheno_df['GROUP_x'] == 'C'] = 0.0
	#new_pheno_df[c] = new_pheno_df[c].fillna(0)
new_pheno_df['Group'] = np.where((new_pheno_df.GROUP_x == 'C'),'Control','ECA')
# Based on Nim's recommendation
new_pheno_df['Group'][new_pheno_df['ID']=='PA087'] = 'Control'
new_pheno_df['Group'][new_pheno_df['ID']=='PA208'] = 'Control'

# Making plots
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="SS_NEW_TOTAL_MEAN.J", y="CGH_AGE_ADOPT",
                hue="cgh_switch_groups",#palette="ch:r=-.2,d=.3_r",
				linewidth=0,
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Child\'s felt attachment security')
ax.set_ylabel('Age of adoption (months)')
plt.savefig(phenofigdir+'Attachment_vs_adoption.png')

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x='motion_outliers', y="SS_NEW_TOTAL_MEAN.J", 
                hue="cgh_switch_groups",#palette="ch:r=-.2,d=.3_r",
				linewidth=0,
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Motion outliers')
ax.set_ylabel('Child\'s felt attachment security')
plt.savefig(phenofigdir+'Motion_vs_Attachment.png',bbox_inches = "tight")

f, ax = plt.subplots()
sns.stripplot(x="cgh_switch_groups", y='CGH_SUM_SWITCH', data=new_pheno_df, jitter=True,ax=ax)
ax.set_xlabel('Binned number of caregiver switches')
ax.set_ylabel('Actual number of caregiver switches')
plt.savefig(phenofigdir+'Caregiver_switches.png')

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="CGH_AGE_ADOPT", y="cgh_switch_groups",
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Age of adoption (months)')
ax.set_ylabel('Binned number of caregiver switches')
plt.tight_layout()
plt.savefig(phenofigdir+'Adoption_age_vs_caregiver_switches.png',bbox_inches = "tight")

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="CGH_AGE_ADOPT", y="motion_outliers",
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Age of adoption (months)')
ax.set_ylabel('Motion Outliers')
plt.tight_layout()
plt.savefig(phenofigdir+'Motion_vs_Adoption_age.png',bbox_inches = "tight")

tempdf = new_pheno_df.rename(columns={'SS_NEW_TOTAL_MEAN.J': 'Felt attachment security', 'GROUP_x': 'ECA Group'})
sns.set_theme(style="darkgrid", font_scale=2)
sns.displot(tempdf, x='Felt attachment security', col="Group")
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_2.png')
sns.displot(tempdf, x='Felt attachment security', col="ECA Group")
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_1.png')
sns.displot(tempdf, x='motion_outliers',col='Group')
plt.savefig(phenofigdir+'Motion_vs_ECA_group_2.png')
sns.displot(tempdf, x='motion_outliers', col="ECA Group")
plt.savefig(phenofigdir+'Motion_vs_ECA_group_1.png')
thresh = 0.2*225 #45 TRs is 20% of all TRs
tempdf2 = tempdf[tempdf['motion_outliers']<thresh]
sns.displot(tempdf2, x='motion_outliers',col='Group')
plt.savefig(phenofigdir+'Motion_vs_ECA_group_thresh_2.png')
sns.displot(tempdf2, x='motion_outliers', col="ECA Group")
plt.savefig(phenofigdir+'Motion_vs_ECA_group_thresh_1.png')
ndrop = tempdf[tempdf['motion_outliers']>thresh]
ECAdrop = len(ndrop[ndrop['Group']=='ECA'])
ttest = ttest_ind(tempdf['motion_outliers'][tempdf['Group']=='Control'], tempdf['motion_outliers'][tempdf['Group']=='ECA'])

sns.set_theme(font_scale=1)
tempdf = new_pheno_df.rename(columns={'SS_NEW_TOTAL_MEAN.J': 'Felt attachment security', 'GROUP_x': 'ECA Group','GENDER_CHILD':'Gender','CGH_SUM_SWITCH':'Number of Switches','CGH_AGE_ADOPT':'Adoption Age','SS_NEW_AVAILABILITY_MEAN':'Parent Availability','SS_NEW_RELYSTRESS_MEAN':'Stress reliance','SS_NEW_COMMUNICATION_MEAN':'Communication subscale','cgh_switch_groups':'Binned switches','CGH_AGE_LIVE':'Age living with parents'})
tempdf = tempdf.drop(['shirley1_no0','ID','comps0_creas1','GROUP_y','ECA Group','MOVIE','DEM_3_GENDER_CHILD','age_rounded_to_years','CGH_SUM_EARLYAGE','CGH_SUM_LATEAGE','age_truncated','Group'], 1)
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


#new_pheno_df.to_csv('temp.csv', index=False)


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