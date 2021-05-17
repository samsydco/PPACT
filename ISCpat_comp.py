#!/usr/bin/env python3


# Compare pattern ISCs between groups during:
# Attachment, seperation, reunion

import os
import glob
import tqdm
import pandas as pd
import deepdish as dd
import numpy as np
from itertools import combinations_with_replacement,combinations
from settings import *

savedir = ISCdir+'ISCpatw_ISCpatb/'
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
for movpair in moviecomp:
	for shuffle in tqdm.tqdm(range(nshuffle)):
		Phenocopy = Phenodf
		if shuffle !=0:
			idx = np.random.permutation(len(subvec))
			Phenocopy['GROUP'] = [Phenocopy['GROUP'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['GROUP'])]
			Phenocopy['Group'] = [Phenocopy['Group'].iloc[idx[vi]] for vi,val in enumerate(Phenocopy['Group'])]
	
		for roi in ROIs:
			roi_short = roi.split('/')[-1][:-3]
			roidict = dd.io.load(roi)
			savefile = savedir+'_'.join([movpair[0],movpair[1],roi_short])+'.h5'
			savedict = dd.io.load(savefile) if os.path.exists(savefile) else {k:{k:{} for k in [0,1,2]} for k in ['allvISC', 'meanISC', 'patternISC']}
			isccomps = list(roidict.keys()) if movpair[0]==movpair[1] else ['patternISC']
			for comp in isccomps:
				for event in range(3):
					if not os.path.exists(savefile):
						ISCs = {k:np.zeros(nshuffle) for k in compb}
						ISCe = {k:np.zeros(nshuffle) for k in list(combinations([c for c in compb if c[0]==c[1]],2))}
						ISCb = {k:np.zeros(nshuffle) for k in [c for c in compb if c[0]!=c[1]]}
					else:
						ISCs = savedict[comp]['ISCs']
						ISCe = savedict[comp]['ISCe']
						ISCb = savedict[comp]['ISCb']
					ISCs_tmp = {k:[] for k in compb}
					for pair in roidict[comp][event].keys():
						if pair[4:9] + '_V2' in subvec and pair[17:22] + '_V2' in subvec:
							if Phenocopy[Phenocopy['IDENT_SUBID'] == pair[4:9] + '_V2']['MOVIE'].iloc[0] == movpair[0] and Phenocopy[Phenocopy['IDENT_SUBID'] == pair[17:22] + '_V2']['MOVIE'].iloc[0] == movpair[1]:
								idx1 = subvec.index(pair[4:9]+'_V2')
								idx2 = subvec.index(pair[17:22]+'_V2')
								if Phenocopy['Group'].iloc[idx1] == 'ECA':
									if Phenocopy['Group'].iloc[idx2] == 'ECA':
										ISCs_tmp[('ECA', 'ECA')].append(roidict[comp][event][pair])
										if Phenocopy['GROUP'].iloc[idx1] in comps and \
										Phenocopy['GROUP'].iloc[idx2] in comps:
											if Phenocopy['GROUP'].iloc[idx1] != \
											Phenocopy['GROUP'].iloc[idx2]:
												ISCs_tmp[('DA', 'PI')].append(roidict[comp][event][pair])
											else:
												ECA1 = Phenocopy['GROUP'].iloc[idx1]
												ISCs_tmp[(ECA1, ECA1)].append(roidict[comp][event][pair])
									else:
										ISCs_tmp[('C', 'ECA')].append(roidict[comp][event][pair])
										if Phenocopy['GROUP'].iloc[idx1] in comps:
											ISCs_tmp[('C', Phenocopy['GROUP'].iloc[idx1])]. append( roidict[comp][event][pair])
								elif Phenocopy['Group'].iloc[idx2] == 'ECA':
									ISCs_tmp[('C', 'ECA')].append(roidict[comp][event][pair])
									if Phenocopy['GROUP'].iloc[idx2] in comps:
										ISCs_tmp[('C', Phenocopy['GROUP'].iloc[idx2])]. append( roidict[comp][event][pair])
								else:
									ISCs_tmp[('C', 'C')].append(roidict[comp][event][pair])
					for i in ISCs.keys():
						ISCs[i][shuffle] = np.mean(ISCs_tmp[i])
					for i in ISCe.keys():
						ISCe[i][shuffle] = ISCs[i[0]][shuffle] - ISCs[i[1]][shuffle]
					for i in ISCb.keys():
						ISCb[i][shuffle] = ISCs[i][shuffle]/(np.sqrt(ISCs[(i[0], i[0])][shuffle])*np.sqrt(ISCs[(i[1], i[1])][shuffle]))
					savedict[comp][event]['ISCs'] = ISCs
					savedict[comp][event]['ISCe'] = ISCe
					savedict[comp][event]['ISCb'] = ISCb
			dd.io.save(savefile,savedict)




