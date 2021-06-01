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

comps = ['C','DA','PI','ECA']
compb = list(combinations_with_replacement(comps,2))
compb = [p for p in compb if p not in (('PI', 'ECA'),('DA', 'ECA'))]
nvox = 40962

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
data = {'L':dd.io.load(ISCdir+'ISC_vert_L.h5'),'R':dd.io.load(ISCdir+'ISC_vert_R.h5')}
savedict = {k:{k:{} for k in ['ISCs','ISCe','ISCb']} for k in ['L','R']}
for mi,movie in enumerate(movies):
	for hemi,vals in data.items():
		print(movie,hemi)
		ISCs = {k:np.zeros((nvox,nshuffle+1)) for k in compb}
		ISCe = {k:np.zeros((nvox,nshuffle+1)) for k in list(combinations([c for c in compb if c[0]==c[1]],2))}
		ISCb = {k:np.zeros((nvox,nshuffle+1)) for k in [c for c in compb if c[0]!=c[1]]}
		for shuffle in tqdm.tqdm(range(nshuffle+1)):
			ISCsi = {k:0 for k in compb}
			Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
			Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
			subvec = list(Phenocopy['IDENT_SUBID'])
			if shuffle !=0:
				np.random.seed(shuffle)
				idx = np.random.permutation(len(subvec))
				Phenocopy['GROUP'] = [Phenocopy['GROUP'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['GROUP'])]
				Phenocopy['Group'] = [Phenocopy['Group'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['Group'])]		
			for pi,pair in enumerate(vals['pairs']):
				if pair[4:9]+'_V2' in subvec and pair[17:22]+'_V2' in subvec:
					idx1 = subvec.index(pair[4:9]+'_V2')
					idx2 = subvec.index(pair[17:22]+'_V2')
					if Phenocopy['Group'].iloc[idx1] == 'ECA':
						if Phenocopy['Group'].iloc[idx2] == 'ECA':
							ISCs[('ECA', 'ECA')][:,shuffle] += vals['ISCs'][:,pi+1]
							ISCsi[('ECA', 'ECA')] += 1
							if Phenocopy['GROUP'].iloc[idx1] in comps and \
							Phenocopy['GROUP'].iloc[idx2] in comps:
								if Phenocopy['GROUP'].iloc[idx1] != \
								Phenocopy['GROUP'].iloc[idx2]:
									ISCs[('DA', 'PI')][:,shuffle] += vals['ISCs'][:,pi+1]
									ISCsi[('DA', 'PI')] += 1
								else:
									ECA1 = Phenocopy['GROUP'].iloc[idx1]
									ISCs[(ECA1, ECA1)][:,shuffle] += vals['ISCs'][:,pi+1]
									ISCsi[(ECA1, ECA1)] += 1
						else:
							ISCs[('C', 'ECA')][:,shuffle] += vals['ISCs'][:,pi+1]
							ISCsi[('C', 'ECA')] += 1
							if Phenocopy['GROUP'].iloc[idx1] in comps:
								ISCs[('C', Phenocopy['GROUP'].iloc[idx1])][:,shuffle]  += vals['ISCs'][:,pi+1]
								ISCsi[('C', Phenocopy['GROUP'].iloc[idx1])] += 1
					elif Phenocopy['Group'].iloc[idx2] == 'ECA':
						ISCs[('C', 'ECA')][:,shuffle] += vals['ISCs'][:,pi+1]
						ISCsi[('C', 'ECA')] += 1
						if Phenocopy['GROUP'].iloc[idx2] in comps:
							ISCs[('C', Phenocopy['GROUP'].iloc[idx2])][:,shuffle] += vals['ISCs'][:,pi+1]
							ISCsi[('C', Phenocopy['GROUP'].iloc[idx2])] += 1
					else:
						ISCs[('C', 'C')][:,shuffle] += vals['ISCs'][:,pi+1]
						ISCsi[('C', 'C')] += 1
			for i,v in ISCs.items():
				ISCs[i][:,shuffle] = v[:,shuffle] / ISCsi[i]
			for i in ISCe.keys():
				ISCe[i][:,shuffle] = ISCs[i[0]][:,shuffle] - ISCs[i[1]][:,shuffle]
			for i in ISCb.keys():
				ISCb[i][:,shuffle] = ISCs[i][:,shuffle]/(np.sqrt(ISCs[(i[0], i[0])][:,shuffle])*np.sqrt(ISCs[(i[1], i[1])][:,shuffle]))
		savedict[hemi]['ISCs'] = ISCs
		savedict[hemi]['ISCe'] = ISCe
		savedict[hemi]['ISCb'] = ISCb

dd.io.save(ISCdir+'ISCw_ISCb_vert.h5',savedict)
				
				

			
					
					
			
					
	