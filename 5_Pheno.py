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

# subjects whose adversity characterization differs between dataframes!
prob_subs = new_pheno_df.loc[~(new_pheno_df['GROUP_x'] == new_pheno_df['GROUP_y'])]
# ages do not match between dataframes!
age_diff = np.array(df2['age_truncated']) - np.array(df2['age_rounded_to_years'])


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="SS_NEW_TOTAL_MEAN.J", y="CGH_AGE_ADOPT",
                hue="cgh_switch_groups",
                palette="ch:r=-.2,d=.3_r",linewidth=0,
                data=new_pheno_df, ax=ax)
ax.set_xlabel('Child\'s felt attachment security')
ax.set_ylabel('Age of adoption (months)')
plt.savefig(phenofigdir+'Attachment_vs_adoption')


sns.stripplot(x="cgh_switch_groups", y='CGH_SUM_SWITCH', data=new_pheno_df, jitter=True)

f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="CGH_AGE_ADOPT", y="cgh_switch_groups",
                data=new_pheno_df, ax=ax)

sns.set_theme(style="darkgrid", font_scale=2)
sns.displot(
    new_pheno_df, x="SS_NEW_TOTAL_MEAN.J", col="GROUP_x"
)


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