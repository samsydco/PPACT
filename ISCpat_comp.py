#!/usr/bin/env python3


# Compare pattern ISCs between groups during:
# Attachment, seperation, reunion


import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from itertools import combinations_with_replacement,combinations
from settings import *

movies = ['Homeward Bound', 'Shirley']
moviecomp = list(combinations_with_replacement(movies,2))
ROIs = glob.glob(ISCdir+'ISCpat/'+'*')
comps = ['C','DA','PI','ECA']
compb = list(combinations_with_replacement(comps,2))
compb = [p for p in compb if p not in (('PI', 'ECA'),('DA', 'ECA'))]

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
Phenodf = Phenodf[~Phenodf['IDENT_SUBID'].isin(badsubjs[0]+badsubjs[1])]
subvec = list(Phenodf['IDENT_SUBID'])
savedict = {k:{k:{k:{k:{} for k in [0,1,2]} for k in ['allvISC', 'meanISC', 'patternISC']} for k in [roi.split('/')[-1][:-3] for roi in ROIs]} for k in moviecomp}
for mi,movpair in enumerate(moviecomp):
	for roi in tqdm.tqdm(ROIs):
		roi_short = roi.split('/')[-1][:-3]
		roidict = dd.io.load(roi)
		isccomps = list(roidict.keys()) if movpair[0]==movpair[1] else ['patternISC']
		for comp in isccomps:
			for event in range(3):
				ISCs = {k:[] for k in compb}
				ISCe = {k:[] for k in list(combinations([c for c in compb if c[0]==c[1]],2))}
				ISCb = {k:[] for k in [c for c in compb if c[0]!=c[1]]}
				for pair in roidict[comp][event].keys():
					if pair[4:9] + '_V2' in subvec and pair[17:22] + '_V2' in subvec:
						if Phenodf[Phenodf['IDENT_SUBID'] == pair[4:9] + '_V2']['MOVIE'].iloc[0] == movpair[0] and Phenodf[Phenodf['IDENT_SUBID'] == pair[17:22] + '_V2']['MOVIE'].iloc[0] == movpair[1]:
							idx1 = subvec.index(pair[4:9]+'_V2')
							idx2 = subvec.index(pair[17:22]+'_V2')
							if Phenodf['Group'].iloc[idx1] == 'ECA':
								if Phenodf['Group'].iloc[idx2] == 'ECA':
									ISCs[('ECA', 'ECA')].append(roidict[comp][event][pair])
									if Phenodf['GROUP'].iloc[idx1] in comps and \
									Phenodf['GROUP'].iloc[idx2] in comps:
										if Phenodf['GROUP'].iloc[idx1] != \
										Phenodf['GROUP'].iloc[idx2]:
											ISCs[('DA', 'PI')].append(roidict[comp][event][pair])
										else:
											ECA1 = Phenodf['GROUP'].iloc[idx1]
											ISCs[(ECA1, ECA1)].append(roidict[comp][event][pair])
								else:
									ISCs[('C', 'ECA')].append(roidict[comp][event][pair])
									if Phenodf['GROUP'].iloc[idx1] in comps:
										ISCs[('C', Phenodf['GROUP'].iloc[idx1])]. append( roidict[comp][event][pair])
							elif Phenodf['Group'].iloc[idx2] == 'ECA':
								ISCs[('C', 'ECA')].append(roidict[comp][event][pair])
								if Phenodf['GROUP'].iloc[idx2] in comps:
									ISCs[('C', Phenodf['GROUP'].iloc[idx2])]. append( roidict[comp][event][pair])
							else:
								ISCs[('C', 'C')].append(roidict[comp][event][pair])
				for i,v in ISCs.items():
					ISCs[i] = np.mean(v)
				for i in ISCe.keys():
					ISCe[i] = ISCs[i[0]] - ISCs[i[1]]
				for i in ISCb.keys():
					ISCb[i] = ISCs[i]/(np.sqrt(ISCs[(i[0], i[0])])*np.sqrt(ISCs[(i[1], i[1])]))
				savedict[movpair][roi_short][comp][event]['ISCs'] = ISCs
				savedict[movpair][roi_short][comp][event]['ISCe'] = ISCe
				savedict[movpair][roi_short][comp][event]['ISCb'] = ISCb
dd.io.save(ISCdir+'ISCpatw_ISCpatb.h5',savedict)




