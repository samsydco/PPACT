#!/usr/bin/env python3

# ISFC between amygdala and all other parcels

import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from scipy.stats import pearsonr
from settings import *


ROIs = [r.split('/')[-1][:-3] for r in glob.glob(ISCdir+'ISCall/'+'*')]
amgrois = ['R_AMG','L_AMG']
ROIs = [r for r in ROIs if r not in amgrois]
comps = ['Control','ECA','across']

Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()
savedict = {k:{k:{k:{k:np.zeros(nshuffle+1) for k in ['ISCe','ISCb']} for k in amgrois} for k in ROIs} for k in movies}
for mi,movie in enumerate(movies):
	for shuffle in tqdm.tqdm(range(nshuffle+1)):
		Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
		Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
		subvec = list(Phenocopy['IDENT_SUBID'])
		if shuffle !=0:
			np.random.seed(shuffle)
			idx = np.random.permutation(len(subvec))
			Phenocopy['GROUP'] = [Phenocopy['GROUP'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['GROUP'])]
			Phenocopy['Group'] = [Phenocopy['Group'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['Group'])]
		for roi in ROIs:
			ISCs = {k:{k:[] for k in comps} for k in amgrois}
			for sub1 in subvec:
				ID1 = Phenocopy['Group'][Phenocopy['IDENT_SUBID'] == sub1].values[0]
				for sub2 in subvec:
					ID2 = Phenocopy['Group'][Phenocopy['IDENT_SUBID'] == sub2].values[0]
					if sub1!=sub2:
						comp = ID1 if ID1 == ID2 else 'across'
						for amg in amgrois:
							try:
								roi1 = np.nanmean(dd.io.load(parpath + 'sub-' + sub1[:-3] + '.h5', '/'+roi),0)
								roi2 = np.nanmean(dd.io.load(parpath + 'sub-' + sub2[:-3] + '.h5', '/'+amg),0)
								ISCs[amg][comp].append(pearsonr(roi1,roi2)[0])
							except:
								continue
			for amg in amgrois:
				savedict[movie][roi][amg]['ISCe'] = np.mean(ISCs[amg]['Control']) - np.mean(ISCs[amg]['ECA'])
				savedict[movie][roi][amg]['ISCb'] = np.mean(ISCs[amg]['across']) / (np.sqrt(np.mean(ISCs[amg]['Control'])) * np.sqrt(np.mean(ISCs[amg]['ECA'])))
				
dd.io.save(ISCdir+'ISFC.h5',savedict)
							
						
	

							


