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
ticklabels = ['C', 'DA', 'DC', 'IFC', 'PI']
badsubjs = [['PA231_V2', 'PA092_V2', 'PA164_V2', 'PA077_V2', 'PA155_V2', 'PA074_V2'],['PA042_V2', 'PA017_V2', 'PA085_V2']]

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
for mi,movie in enumerate(movies):
	Phenocopy = Phenodf[Phenodf['MOVIE']==movie].sort_values(by=['GROUP','age_truncated','DEM_3_GENDER_CHILD'])
	Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
	subvec = list(Phenocopy['IDENT_SUBID'])
	nsub = len(subvec)
	groupcnt = Phenocopy.groupby('GROUP').count()['Group']
	# make a matrix that's one within a group and zero in off diagonal
	groupmat = np.zeros((nsub,nsub))
	groupmat2 = groupmat.copy()
	groupmat3 = groupmat.copy()
	groupmat2[0:groupcnt[0],0:groupcnt[0]] = np.ones((groupcnt[0],groupcnt[0]))
	groupmat3[0:groupcnt[0],0:groupcnt[0]] = np.ones((groupcnt[0],groupcnt[0]))
	groupmat3[groupcnt[0]:nsub,groupcnt[0]:nsub] = np.ones((nsub-groupcnt[0],nsub-groupcnt[0]))
	ticks = [0]
	for i,v in enumerate(groupcnt[:-1]):
		ticks.append(v+np.sum(groupcnt[:i])) if i>0 else ticks.append(v)
		groupmat[ticks[i]:ticks[i+1],ticks[i]:ticks[i+1]] = np.ones((v,v))
	groupmat[ticks[i+1]:nsub,ticks[i+1]:nsub] = np.ones((groupcnt[-1],groupcnt[-1]))
	np.fill_diagonal(groupmat, 0)
	np.fill_diagonal(groupmat2, 0)
	np.fill_diagonal(groupmat3, 0)
	for roi in ROIs:
		roi_short = roi.split('/')[-1][:-3]
		#if len(glob.glob(figdir+'ISC/'+movie[:4]+'_'+roi_short+'.png'))!=1:
		roidict = dd.io.load(roi)
		fig,ax = plt.subplots(1,3,figsize=(30,10))
		fig.set_facecolor('white')
		for i,comp in enumerate(roidict.keys()):
			compmat = np.zeros((nsub,nsub))
			for pair in roidict[comp].keys():
				if pair[4:9]+'_V2' in subvec and pair[17:22]+'_V2' in subvec:
					idx1 = subvec.index(pair[4:9]+'_V2')
					idx2 = subvec.index(pair[17:22]+'_V2')
					compmat[idx1,idx2] = roidict[comp][pair]
			sns.heatmap(compmat, center=0, ax=ax[i])
			ax[i].set_yticks(ticks)
			ax[i].set_yticklabels(ticklabels)
			ax[i].set_xticks(ticks)
			ax[i].set_xticklabels(ticklabels)
			ax[i].set_title(comp)
		plt.suptitle(roi_short,fontsize=30)
		plt.tight_layout()
		
		fig.savefig(figdir+'ISC_nobad/'+movie[:4]+'_'+roi_short+'.png')
		plt.close()
			

					
			


