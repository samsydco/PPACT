#!/usr/bin/env python3

# Run pairwise ISC in verts

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
subs = glob.glob(prepath+'*.h5')
subpairs = list(permutations(subs,2))
hemis = ['L','R']

ISCdict = {k:{} for k in hemis}
for sub1 in tqdm.tqdm(subs):
	pair1 = sub1.split('/')[-1]
	for sub2 in subs:
		pair2 = sub2.split('/')[-1]
		pairstr = pair1+' '+pair2
		if sub1!=sub2 and Phenodf[Phenodf['IDENT_SUBID'] == pair1[4:9] + '_V2']['MOVIE'].iloc[0] == Phenodf[Phenodf['IDENT_SUBID'] == pair2[4:9] + '_V2']['MOVIE'].iloc[0] and pairstr not in ISCdict['L'].keys():
			for hemi in hemis:
				v1 = dd.io.load(sub1,'/'+hemi)
				v2 = dd.io.load(sub2,'/'+hemi)
				nvox = v1.shape[0]
				ISCdict[hemi][pairstr] = np.zeros(nvox)
				for v in range(nvox):
					try:
						ISCdict[hemi][pairstr][v]=pearsonr(v1[v,:],v2[v,:])[0]
					except:
						continue
dd.io.save(ISCdir+'ISC_vert.h5',ISCdict)
					


