#!/usr/bin/env python3

# shISC

import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations_with_replacement
from settings import *

def create_phenocopy(Phenodf,movie):
	mi = movies.index(movie)
	Phenocopy = Phenodf[Phenodf['MOVIE']==movie]
	Phenocopy = Phenocopy[~Phenocopy['IDENT_SUBID'].isin(badsubjs[mi])]
	subvec = list(Phenocopy['IDENT_SUBID'])
	return Phenocopy,subvec

TR = 0.8 # seconds
Events = [[0,25, 138,180],[0,56,125,180]]
nsh = 5
ROIs = [r.split('/')[-1][:-3] for r in glob.glob(ISCdir+'ISCall/'+'*')]
comps = ['Control','ECA']
compb = list(combinations_with_replacement(comps,2))
moviecomp = list(combinations_with_replacement(movies,2))
Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()

savedict = {k:{k:{k:np.zeros((nsh,3,nshuffle+1)) for k in ['ISCe','ISCb']} for k in ROIs} for k in moviecomp}
for movpair in moviecomp:
	Phenocopy1,subvec1 = create_phenocopy(Phenodf,movpair[0])
	Phenocopy2,subvec2 = create_phenocopy(Phenodf,movpair[1])
	for shuffle in tqdm.tqdm(range(nshuffle+1)):
		if shuffle !=0:
			np.random.seed(shuffle)
			idx1 = np.random.permutation(len(subvec1))
			Phenocopy1['Group'] = [Phenocopy1['Group'].iloc[idx1[vi]] for vi,val in enumerate(Phenocopy1['Group'])]
			idx2 = np.random.permutation(len(subvec2))
			Phenocopy2['Group'] = [Phenocopy2['Group'].iloc[idx2[vi]] for vi,val in enumerate(Phenocopy2['Group'])]
		compl = np.min([len(Phenocopy1['IDENT_SUBID'][Phenocopy1['Group'] == comp]) for comp in comps]+[len(Phenocopy2['IDENT_SUBID'][Phenocopy2['Group'] == comp]) for comp in comps])//2
		for s in range(nsh):
			np.random.seed(s)
			compdict = {}
			for comp in comps:
				subs1 = list(Phenocopy1['IDENT_SUBID'][Phenocopy1['Group'] == comp])
				np.random.shuffle(subs1)
				subs2 = list(Phenocopy2['IDENT_SUBID'][Phenocopy2['Group'] == comp])
				np.random.shuffle(subs2)
				compdict[comp] = [subs1[:compl],subs2[compl:compl*2]]
			for roi in ROIs:
				for event in range(3):
					ISCw = {}
					ISCb_ = []
					for comp in compb:
						if comp[0] == comp[1]:
							dall = [[],[]]
							for h in [0,1]:
								for sub in compdict[comp[0]][h]:
									mi = movies.index(movpair[h])
									try:
										dall[h].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi,sel=dd.aslice[:,int(np.round(Events[mi][event]/TR)):int(np.round(Events[mi][event+1]/TR))]))
									except:
										continue
								dall[h] = np.mean(np.nanmean(dall[h],0),1)
							ISCw[comp[0]] = pearsonr(dall[0],dall[1])[0]
						else:
							dall = [[[],[]],[[],[]]]
							for h1 in [0,1]:
								for h2 in [0,1]:
									for sub in compdict[comp[h1]][h2]:
										mi = movies.index(movpair[h2])
										try:
											dall[h1][h2].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi,sel=dd.aslice[:,int(np.round(Events[mi][event]/TR)):int(np.round(Events[mi][event+1]/TR))]))
										except:
											continue
									dall[h1][h2] = np.mean(np.nanmean(dall[h1][h2],0),1)
							if movpair[0] == movpair[1]:
								for h1 in [0,1]:
									for h2 in [0,1]:
										ISCb_.append(pearsonr(dall[0][h1],dall[1][h2])[0])
							else:
								ISCb_ = [pearsonr(dall[0][0],dall[1][1])[0], pearsonr(dall[0][1],dall[1][0])[0]]
					savedict[movpair][roi]['ISCe'][s,event,shuffle] = ISCw['Control'] - ISCw['ECA']
					savedict[movpair][roi]['ISCb'][s,event,shuffle] = np.mean(ISCb_) / (np.sqrt(ISCw['Control']) * np.sqrt(ISCw['ECA']))
			
dd.io.save(ISCdir+'ISCsh_pat.h5',savedict)			
				

