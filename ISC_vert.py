#!/usr/bin/env python3

# Run pairwise ISC in verts

import glob
import tqdm
import pandas as pd
import numpy as np
import deepdish as dd
from scipy.stats import pearsonr
from settings import *

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf[Phenodf['FDmean'].notna()].reset_index().drop(['level_0','index'],axis=1)
subs = glob.glob(prepath+'*.h5')
hemis = ['L','R']
nvox = 40962

ISCdict = {k:{'pairs':[],'ISCs':np.zeros(nvox)} for k in hemis}
for sub1 in tqdm.tqdm(subs):
	pair1 = sub1.split('/')[-1]
	for sub2 in subs:
		pair2 = sub2.split('/')[-1]
		pairstr1 = pair1+' '+pair2
		pairstr2 = pair2+' '+pair1
		if sub1!=sub2 and Phenodf[Phenodf['IDENT_SUBID'] == pair1[4:9] + '_V2']['MOVIE'].iloc[0] == Phenodf[Phenodf['IDENT_SUBID'] == pair2[4:9] + '_V2']['MOVIE'].iloc[0] and pairstr1 not in ISCdict['L']['pairs'] and pairstr2 not in ISCdict['L']['pairs']:
			for hemi in hemis:
				v1 = dd.io.load(sub1,'/'+hemi)
				v2 = dd.io.load(sub2,'/'+hemi)
				ISCdict[hemi]['pairs'].append(pairstr1)
				ISCdict[hemi]['ISCs'] = np.append(ISCdict[hemi]['ISCs'], np.zeros(nvox))
				for v in range(nvox):
					try:
						ISCdict[hemi]['ISCs'][v,-1] = pearsonr(v1[v,:],v2[v,:])[0]
					except:
						continue
dd.io.save(ISCdir+'ISC_vert_L.h5',ISCdict['L'])
dd.io.save(ISCdir+'ISC_vert_R.h5',ISCdict['R'])					


