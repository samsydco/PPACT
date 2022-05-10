#!/usr/bin/env python3

# Create histograms of FD

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from settings import *

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
	
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

