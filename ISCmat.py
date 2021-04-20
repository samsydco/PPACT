#!/usr/bin/env python3

# organize ISC values in a matrix acording to ECA status,
# movie, age

import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from settings import *

movies = ['Homeward Bound', 'Shirley']
ROIs = glob.glob(ISCdir+'*')
sns.set(font_scale=3) 

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
for movie in movies:
	Phenocopy = Phenodf[Phenodf['MOVIE']==movie].sort_values(by=['GROUP','age_truncated','DEM_3_GENDER_CHILD'])
	subvec = list(Phenocopy['IDENT_SUBID'])
	groupcnt = Phenocopy.groupby('GROUP').count()['Group']
	ticks = [0]
	for i,v in enumerate(groupcnt[:-1]):
		ticks.append(v+np.sum(groupcnt[:i])) if i>0 else ticks.append(v)
	[v for v in Phenocopy.groupby('GROUP').count()['Group']]
	ticklabels = ['C', 'DA', 'DC', 'IFC', 'PI']
	for roi in ROIs:
		roi_short = roi.split('/')[-1][:-3]
		roidict = dd.io.load(roi)
		fig,ax = plt.subplots(1,3,figsize=(30,10))
		for i,comp in enumerate(roidict.keys()):
			compmat = np.zeros((len(subvec),len(subvec)))
			for pair in roidict[comp].keys():
				if pair[4:9]+'_V2' in subvec:
					idx1 = subvec.index(pair[4:9]+'_V2')
					idx2 = subvec.index(pair[17:22]+'_V2')
					compmat[idx1,idx2] = roidict[comp][pair]
			sns.heatmap(compmat, center=0, ax=ax[i])
			ax[i].set_yticks(ticks)
			ax[i].set_yticklabels(ticklabels)
			ax[i].set_xticks(ticks)
			ax[i].set_xticklabels(ticklabels)
			ax[i].set_title(comp)
		plt.suptitle(roi_short,y=0.9)
		plt.tight_layout()
		fig.savefig(figdir+'ISC/'+movie[:4]+'_'+roi_short+'.png')
		plt.close()
			

					
			


