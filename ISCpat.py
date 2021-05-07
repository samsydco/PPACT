#!/usr/bin/env python3


# Run pairwise pattern ISC seperately in each movie event:
# Attachment, seperation, reunion

import os
import glob
import tqdm
import pandas as pd
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
from itertools import permutations
from settings import *

TR = 0.8 # seconds
Events = [[0,25, 138,180],[0,56,125,180]]

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf[Phenodf['FDmean'].notna()].reset_index().drop(['level_0','index'],axis=1)

subs = glob.glob(parpath+'*.h5')
subpairs = list(permutations(subs,2))
ROIs = list(dd.io.load(subs[0]).keys())[1:]

for roi in tqdm.tqdm([r for r in ROIs if len(r)!=5]):#tqdm.tqdm(ROIs):
	roidict = {k:{k:{} for k in range(3)} for k in ['allvISC','meanISC','patternISC']}
	for pair in subpairs:
		pairs1 = pair[0].split('/')[-1]
		pairs2 = pair[1].split('/')[-1]
		mov1 = movies.index(Phenodf[Phenodf['IDENT_SUBID'] == pairs2[4:9] + '_V2']['MOVIE'].iloc[0])
		mov2 = movies.index(Phenodf[Phenodf['IDENT_SUBID'] == pairs1[4:9] + '_V2']['MOVIE'].iloc[0])
		pairstr = pairs1+' '+pairs2
		v1 = dd.io.load(pair[0],'/'+roi)
		v2 = dd.io.load(pair[1],'/'+roi)
		goodvox = np.arange(len(v1))
		goodvox = np.delete(goodvox, np.unique( np.concatenate( (np.where( np.isnan( np.mean(v1,1)))[0], np.where(np.isnan(np.mean(v2,1)))[0]))))
		for event in range(3):
			e1 = v1[goodvox,np.round(Events[mov1][event]/TR):\
							np.round(Events[mov1][event+1]/TR)]
			e2 = v2[goodvox,np.round(Events[mov2][event]/TR):\
							np.round(Events[mov2][event+1]/TR)]
			roidict['patternISC'][event][pairstr] = pearsonr(np.mean(e1,1),np.mean(e2,1))[0]
			if mov1 == mov2:
				i=0
				for v in goodvox:
					i+=pearsonr(v1[v,:],v2[v,:])[0]
				roidict['allvISC'][event][pairstr] = i/len(goodvox)
				roidict['meanISC'][event][pairstr] = pearsonr(np.nanmean(e1,0),np.nanmean(e2,0))[0]
	dd.io.save(ISCdir+'ISCpat/'+roi+'.h5',roidict)
				
			
		
		








