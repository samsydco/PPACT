#!/usr/bin/env python3

# Create DF containing: subject, age, ECA, Movie, FD
# Create histograms of FD
# 

import os
import glob
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from settings import *


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
	
	
fig,ax = plt.subplots(1,3,figsize=(9,3))
fig.patch.set_facecolor('white')
sns.histplot(data=Phenodf, x="FDmean", hue="Group", multiple="stack",ax=ax[0],legend=False)
g = sns.histplot(data=Phenodf, x="FDstd", hue="Group", multiple="stack",ax=ax[1],legend=False)
ax[1].set(ylabel=None)
sns.histplot(data=Phenodf, x="FDmax", hue="Group", multiple="stack",ax=ax[2])
ax[2].set(ylabel=None)
plt.tight_layout()
fig.savefig(figdir+'QA/FD.png')

fig,ax = plt.subplots(1,2,figsize=(10,5))
fig.patch.set_facecolor('white')
for i in range(len(Phenodf)):
	row = Phenodf[['FDmean','FDstd','FDmax']].iloc[i]
	if Phenodf['Group'].iloc[i] == 'Control':
		ax[0].plot(row,linestyle='-', marker='o')
	else:
		ax[1].plot(row,linestyle='-', marker='o')
ax[0].set_title('Control')
ax[1].set_title('ECA')	
ax[0].set_ylabel('Value')
fig.savefig(figdir+'QA/FDdots.png')
	
col = 'FDmax'
scipy.stats.ttest_ind(
Phenodf[col][Phenodf[col].notna()].loc[Phenodf['Group'] == 'Control'], 
Phenodf[col][Phenodf[col].notna()].loc[Phenodf['Group'] != 'Control'])

