#!/usr/bin/env python3

# shISC

import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations_with_replacement,combinations
from settings import *

ROIs = [r.split('/')[-1][:-3] for r in glob.glob(ISCdir+'ISCall/'+'*')]
comps = ['Control','ECA']
compb = list(combinations_with_replacement(comps,2))
Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()

savedict = {k:{k:{k:np.zeros(nshuffle+1) for k in ['ISCe','ISCb']} for k in ROIs} for k in movies}
for mi,movie in enumerate(movies):
	Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
	Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
	subvec = list(Phenocopy['IDENT_SUBID'])
	for shuffle in tqdm.tqdm(range(nshuffle+1)):
		if shuffle !=0:
			np.random.seed(shuffle)
			idx = np.random.permutation(len(subvec))
			Phenocopy['GROUP'] = [Phenocopy['GROUP'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['GROUP'])]
			Phenocopy['Group'] = [Phenocopy['Group'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['Group'])]
		compdict = {}
		for comp in comps:
			subs = list(Phenocopy['IDENT_SUBID'][Phenocopy['Group'] == comp])
			compdict[comp] = [subs[:len(subs)//2],subs[len(subs)//2:]]
		for roi in ROIs:
			ISCw = {}
			ISCb_ = []
			for comp in compb:
				if comp[0] == comp[1]:
					dall = [[],[]]
					for h in [0,1]:
						for sub in compdict[comp[0]][h]:
							dall[h].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi))
						dall[h] = np.mean(np.nanmean(dall[h],0),0)
					ISCw[comp] = pearsonr(dall[0],dall[1])
				else:
					dall = [[[],[]],[[],[]]]
					for h1 in [0,1]:
						for h2 in [0,1]:
							for sub in compdict[comp[h1]][h2]:
								dall[h1][h2].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi))
							dall[h1][h2] = np.mean(np.nanmean(dall[h1][h2],0),0)
					for h1 in [0,1]:
						for h2 in [0,1]:
							ISCb_.append(pearsonr(dall[0][h1],dall[1][h2]))
			savedict[movie][roi]['ISCe'][shuffle] = ISCw[('Control', 'Control')] - ISCw[('ECA', 'ECA')]
			savedict[movie][roi]['ISCb'][shuffle] = np.mean(ISCb_) / (np.sqrt(ISCw[('Control', 'Control')]) * np.sqrt(ISCw[('ECA', 'ECA')]))
			
dd.io.save(ISCdir+'ISCsh.py',savedict)
				
				

