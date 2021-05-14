#!/usr/bin/env python3


# calculate between group ISC
# also calculate within group ISC

import os
import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from itertools import combinations_with_replacement,combinations
from settings import *

savedir = ISCdir+'ISCw_ISCb/'
ROIs = glob.glob(ISCdir+'ISCall/'+'*')
comps = ['C','DA','PI','ECA']
compb = list(combinations_with_replacement(comps,2))
compb = [p for p in compb if p not in (('PI', 'ECA'),('DA', 'ECA'))]

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
savedict = {k:{k:{k:{} for k in ['allvISC', 'meanISC', 'patternISC']} for k in [roi.split('/')[-1][:-3] for roi in ROIs]} for k in movies}
for mi,movie in enumerate(movies):
	for shuffle in tqdm.tqdm(range(nshuffle)):
		Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
		Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
		subvec = list(Phenocopy['IDENT_SUBID'])
		if shuffle !=0:
			idx = np.random.permutation(len(subvec))
			Phenocopy['GROUP'] = [Phenocopy['GROUP'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['GROUP'])]
			Phenocopy['Group'] = [Phenocopy['Group'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['Group'])]
	
		for roi in ROIs:
			roi_short = roi.split('/')[-1][:-3]
			roidict = dd.io.load(roi)
			savefile = savedir+movie+'_'+roi_short+'.h5'
			savedict = dd.io.load(savefile) if os.path.exists(savefile) else {k:{k:{} for k in ['ISCs','ISCe','ISCb']} for k in ['allvISC', 'meanISC', 'patternISC']}
			for comp in roidict.keys():
				if not os.path.exists(savefile):
					ISCs = {k:np.zeros(nshuffle) for k in compb}
					ISCe = {k:np.zeros(nshuffle) for k in list(combinations([c for c in compb if c[0]==c[1]],2))}
					ISCb = {k:np.zeros(nshuffle) for k in [c for c in compb if c[0]!=c[1]]}
				else:
					ISCs = savedict[comp]['ISCs']
					ISCe = savedict[comp]['ISCe']
					ISCb = savedict[comp]['ISCb']
				ISCs_tmp = {k:[] for k in compb}
				for pair in roidict[comp].keys():
					if pair[4:9]+'_V2' in subvec and pair[17:22]+'_V2' in subvec:
						idx1 = subvec.index(pair[4:9]+'_V2')
						idx2 = subvec.index(pair[17:22]+'_V2')
						if Phenocopy['Group'].iloc[idx1] == 'ECA':
							if Phenocopy['Group'].iloc[idx2] == 'ECA':
								ISCs_tmp[('ECA', 'ECA')].append(roidict[comp][pair])
								if Phenocopy['GROUP'].iloc[idx1] in comps and \
								Phenocopy['GROUP'].iloc[idx2] in comps:
									if Phenocopy['GROUP'].iloc[idx1] != \
									Phenocopy['GROUP'].iloc[idx2]:
										ISCs_tmp[('DA', 'PI')].append(roidict[comp][pair])
									else:
										ECA1 = Phenocopy['GROUP'].iloc[idx1]
										ISCs_tmp[(ECA1, ECA1)].append(roidict[comp][pair])
							else:
								ISCs_tmp[('C', 'ECA')].append(roidict[comp][pair])
								if Phenocopy['GROUP'].iloc[idx1] in comps:
									ISCs_tmp[('C', Phenocopy['GROUP'].iloc[idx1])]. append( roidict[comp][pair])
						elif Phenocopy['Group'].iloc[idx2] == 'ECA':
							ISCs_tmp[('C', 'ECA')].append(roidict[comp][pair])
							if Phenocopy['GROUP'].iloc[idx2] in comps:
								ISCs_tmp[('C', Phenocopy['GROUP'].iloc[idx2])]. append( roidict[comp][pair])
						else:
							ISCs_tmp[('C', 'C')].append(roidict[comp][pair])
				for i in ISCs.keys():
					ISCs[i][shuffle] = np.mean(ISCs_tmp[i])
				for i in ISCe.keys():
					ISCe[i][shuffle] = ISCs[i[0]][shuffle] - ISCs[i[1]][shuffle]
				for i in ISCb.keys():
					ISCb[i][shuffle] = ISCs[i][shuffle]/(np.sqrt(ISCs[(i[0], i[0])][shuffle])*np.sqrt(ISCs[(i[1], i[1])][shuffle]))
				savedict[comp]['ISCs'] = ISCs
				savedict[comp]['ISCe'] = ISCe
				savedict[comp]['ISCb'] = ISCb
			dd.io.save(savefile,savedict)
				
				

			
					
					
			
					
	