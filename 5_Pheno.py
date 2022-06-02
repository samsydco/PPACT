#!/usr/bin/env python3

# Create DF containing: subject, age, ECA, Movie, FD

import os
import glob
import tqdm
import pandas as pd
import numpy as np
from settings import *

phenofigdir = figdir+'pheno/'

new_pheno_df = pd.read_csv(phenopath+'SHIRLEY_adversityVARS_for_Sam_analysis.csv')

subs = [sub.split('/')[-1][4:-5] for sub in glob.glob(fmripreppath + '*.html')]

new_pheno_df = new_pheno_df[new_pheno_df['ID'].isin(subs)].reset_index(drop=True)

Phenodf = pd.read_csv(phenopath+'schema_inventory.csv').drop([0]).drop('Unnamed: 4',axis=1)
Phenodf['IDENT_SUBID'] = Phenodf['IDENT_SUBID'].str.replace('_V2','')
Phenodf = Phenodf.rename(columns={'IDENT_SUBID': 'ID'})
new_pheno_df = new_pheno_df.merge(Phenodf, how='left', on='ID')
new_pheno_df = new_pheno_df.replace(' ', np.nan)
new_pheno_df = new_pheno_df.apply(pd.to_numeric, errors='coerce').fillna(new_pheno_df)

# MISSING DATA!
# subjects whose adversity characterization differs between dataframes!
nanGROUP = list(new_pheno_df['ID'][new_pheno_df['GROUP_x'].isna()])
tempdf = new_pheno_df[~new_pheno_df['ID'].isin(nanGROUP)]
prob_subs = tempdf.loc[~(tempdf['GROUP_x'] == tempdf['GROUP_y'])]
# ages do not match between dataframes!
age_diff = np.array(df2['age_truncated']) - np.array(df2['age_rounded_to_years'])
#subjects missing schema score:
noschema = list(tempdf['ID'][tempdf['SS_NEW_TOTAL_MEAN.J'].isna()])
tempdf = tempdf[~tempdf['ID'].isin(noschema)]
# no adoption age
noadoptdf = tempdf[tempdf['CGH_AGE_LIVE'].isna()]
noadoptdf = noadoptdf[~noadoptdf['GROUP_x'].isin(['C'])]
noadoptlist = np.array(noadoptdf['ID'])

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

tempdf = new_pheno_df.rename(columns={'SS_NEW_TOTAL_MEAN.J': 'Felt attachment security', 'GROUP_x': 'ECA Group'})
sns.set_theme(style="darkgrid", font_scale=2)
sns.displot(tempdf, x='Felt attachment security', col="Group")
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_2.png')
sns.displot(tempdf, x='Felt attachment security', col="ECA Group")
plt.savefig(phenofigdir+'Attachment_vs_ECA_group_1.png')

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


new_pheno_df.to_csv('temp.csv', index=False)


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