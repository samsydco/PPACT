#!/usr/bin/env python3


# calculate between group ISC
# also calculate within group ISC


import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from itertools import combinations_with_replacement,combinations
from settings import *

ROIs = glob.glob(ISCdir+'ISCall/'+'*')
comps = ['C','DA','PI','ECA']
compb = list(combinations_with_replacement(comps,2))
compb = [p for p in compb if p not in (('PI', 'ECA'),('DA', 'ECA'))]

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
savedict = {k:{k:{k:{} for k in ['allvISC', 'meanISC', 'patternISC']} for k in [roi.split('/')[-1][:-3] for roi in ROIs]} for k in movies}
for mi,movie in enumerate(movies):
	Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
	Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
	subvec = list(Phenocopy['IDENT_SUBID'])
	for roi in tqdm.tqdm(ROIs):
		roi_short = roi.split('/')[-1][:-3]
		roidict = dd.io.load(roi)
		for comp in roidict.keys():
			ISCs = {k:[] for k in compb}
			ISCe = {k:[] for k in list(combinations([c for c in compb if c[0]==c[1]],2))}
			ISCb = {k:[] for k in [c for c in compb if c[0]!=c[1]]}
			for pair in roidict[comp].keys():
				if pair[4:9]+'_V2' in subvec and pair[17:22]+'_V2' in subvec:
					idx1 = subvec.index(pair[4:9]+'_V2')
					idx2 = subvec.index(pair[17:22]+'_V2')
					if Phenocopy['Group'].iloc[idx1] == 'ECA':
						if Phenocopy['Group'].iloc[idx2] == 'ECA':
							ISCs[('ECA', 'ECA')].append(roidict[comp][pair])
							if Phenocopy['GROUP'].iloc[idx1] in comps and \
							Phenocopy['GROUP'].iloc[idx2] in comps:
								if Phenocopy['GROUP'].iloc[idx1] != \
								Phenocopy['GROUP'].iloc[idx2]:
									ISCs[('DA', 'PI')].append(roidict[comp][pair])
								else:
									ECA1 = Phenocopy['GROUP'].iloc[idx1]
									ISCs[(ECA1, ECA1)].append(roidict[comp][pair])
						else:
							ISCs[('C', 'ECA')].append(roidict[comp][pair])
							if Phenocopy['GROUP'].iloc[idx1] in comps:
								ISCs[('C', Phenocopy['GROUP'].iloc[idx1])]. append( roidict[comp][pair])
					elif Phenocopy['Group'].iloc[idx2] == 'ECA':
						ISCs[('C', 'ECA')].append(roidict[comp][pair])
						if Phenocopy['GROUP'].iloc[idx2] in comps:
							ISCs[('C', Phenocopy['GROUP'].iloc[idx2])]. append( roidict[comp][pair])
					else:
						ISCs[('C', 'C')].append(roidict[comp][pair])
			for i,v in ISCs.items():
				ISCs[i] = np.mean(v)
			for i in ISCe.keys():
				ISCe[i] = ISCs[i[0]] - ISCs[i[1]]
			for i in ISCb.keys():
				ISCb[i] = ISCs[i]/(np.sqrt(ISCs[(i[0], i[0])])*np.sqrt(ISCs[(i[1], i[1])]))
			savedict[movie][roi_short][comp]['ISCs'] = ISCs
			savedict[movie][roi_short][comp]['ISCe'] = ISCe
			savedict[movie][roi_short][comp]['ISCb'] = ISCb
dd.io.save(ISCdir+'ISCw_ISCb.h5',savedict)
			
					
					
			
					
	