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
	for roi in tqdm.tqdm(ROIs):
		roidict = dd.io.load(roi)
		fig,ax = plt.subplots(1,3,figsize=(25,8))
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
		plt.tight_layout()
		fig.savefig(figdir+'ISC/'+movie[:4]+'_'+roi.split('/')[-1][:-3]+'.png')
			
			
			
			h=ax.imshow(compmat)
			fig.colorbar(h,ax=ax,fraction=0.046, pad=0.04)
			plt.tight_layout()
					
					



from itertools import permutations
subpairs = list(permutations(Phenodf['IDENT_SUBID'],2))
Pairdf = pd.DataFrame(columns=['pair','MOVIE','ECA','Agemean','Agediff'])
for pair in tqdm.tqdm(subpairs):
	movie = Phenodf[Phenodf['IDENT_SUBID'] == pair[0]]['MOVIE'].iloc[0]
	if movie == Phenodf[Phenodf['IDENT_SUBID'] == pair[1]]['MOVIE'].iloc[0]:
		Pairdf = Pairdf.append({'pair': ' '.join(pair),
					  'MOVIE': movie,
					  'ECA': Phenodf[Phenodf['IDENT_SUBID'] == pair[0]]['GROUP'].iloc[0] + ' ' + Phenodf[Phenodf['IDENT_SUBID'] == pair[1]]['GROUP'].iloc[0],
					  'Agemean': Phenodf.loc[Phenodf['IDENT_SUBID'].isin(pair)]['age_truncated'].mean(),
					  'Agediff': abs(Phenodf.loc[Phenodf['IDENT_SUBID'].isin(pair)]['age_truncated'].diff().iloc[1])},ignore_index = True)
Pairdf = Pairdf.replace({'PI C':'C PI','DC PI':'PI DC','DA C':'C DA','DA PI':'PI DA','IFC DC':'DC IFC','IFC C':'C IFC','DC C':'C DC','IFC PI':'PI IFC','IFC DA':'DA IFC','DC DA':'DA DC'})


ROIs = glob.glob(ISCdir+'*')

for roi in ROIs:
	roidict = dd.io.load(roi)
	Pairdfcopy = Pairdf.copy()
	for comp in tqdm.tqdm(roidict.keys()):
		Pairdfcopy[comp] = np.nan
		for pair in roidict[comp].keys():
			pairtmp = pair.replace(".h5","_V2").replace('sub-','')
			Pairdfcopy.loc[Pairdfcopy['pair']==pairtmp,comp] = roidict[comp][pair]
		for m in movies:
			pivot = Pairdfcopy[Pairdfcopy['MOVIE']==m].pivot_table(values=comp,index=['ECA','Agemean'])
			


