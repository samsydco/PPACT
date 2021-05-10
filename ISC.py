#!/usr/bin/env python3


# Run pairwise ISC
# Use DF containing: subject, age, ECA, Movie
# Run ISC in sub-groups

import os
import glob
import tqdm
import pandas as pd
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
from itertools import permutations
from settings import *


Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf[Phenodf['FDmean'].notna()].reset_index().drop(['level_0','index'],axis=1)

subs = glob.glob(parpath+'*.h5')
subpairs = list(permutations(subs,2))
ROIs = list(dd.io.load(subs[0]).keys())[1:]

for roi in tqdm.tqdm(ROIs):
	if not any([roi in r for r in glob.glob(ISCdir+'ISCall/'+'*')]):
		roidict = {'allvISC':{},'meanISC':{},'patternISC':{}}
		for pair in subpairs:
			pairs1 = pair[0].split('/')[-1]
			pairs2 = pair[1].split('/')[-1]
			if Phenodf[Phenodf['IDENT_SUBID'] == pairs2[4:9] + '_V2']['MOVIE'].iloc[0] == Phenodf[Phenodf['IDENT_SUBID'] == pairs1[4:9] + '_V2']['MOVIE'].iloc[0]:
				pairstr = pairs1+' '+pairs2
				try:
					v1 = dd.io.load(pair[0],'/'+roi)
					v2 = dd.io.load(pair[1],'/'+roi)
				except:
					continue
				nvox = nvoxreal = v1.shape[0]
				goodvox=np.arange(nvox)
				badvox = []
				i = 0
				for v in range(nvox):
					try:
						i+=pearsonr(v1[v,:],v2[v,:])[0]
					except:
						nvoxreal-=1
						badvox.append(v)
				goodvox = np.delete(goodvox, badvox)
				roidict['allvISC'][pairstr] = i/nvoxreal
				roidict['meanISC'][pairstr]    = pearsonr(np.nanmean(v1,0),np.nanmean(v2,0))[0]
				roidict['patternISC'][pairstr] = pearsonr(np.mean(v1[goodvox,:],1),np.mean(v2[goodvox,:],1))[0]
		dd.io.save(ISCdir+'ISCall/'+roi+'.h5',roidict)
		
	








