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

nsh = 10
ROIs = [r.split('/')[-1][:-3] for r in glob.glob(ISCdir+'ISCall/'+'*')]
amgrois = ['R_AMG','L_AMG']
comps = ['Control','ECA']
compb = list(combinations_with_replacement(comps,2))
Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf.drop(Phenodf[Phenodf.FDmax.isnull()].index).reset_index()

ISCw = {k:{k:{k:np.zeros((nsh,nshuffle+1)) for k in comps} for k in ROIs} for k in movies}
ISCb_ = {k:{k:{k:{k:[] for k in range(nshuffle+1)} for k in range(nsh)} for k in ROIs} for k in movies}
ISFCw = {k:{k:{k:{k:np.zeros((nsh,nshuffle+1)) for k in comps} for k in amgrois} for k in ROIs} for k in movies}
ISFCb = {k:{k:{k:{k:{k:{k:[] for k in ['C_amg_vs_ECA_par','ECA_amg_vs_C_par']} for k in range(nshuffle+1)} for k in range(nsh)} for k in amgrois} for k in ROIs} for k in movies}
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
		compl = np.min([len(Phenocopy['IDENT_SUBID'][Phenocopy['Group'] == comp]) for comp in comps])//2
		for s in range(nsh):
			np.random.seed(s)
			compdict = {}
			for comp in comps:
				subs = list(Phenocopy['IDENT_SUBID'][Phenocopy['Group'] == comp])
				np.random.shuffle(subs)
				compdict[comp] = [subs[:compl],subs[compl:compl*2]]
			for roi in ROIs:
				for comp in compb:
					if comp[0] == comp[1]:
						dall = [[],[]]
						amgl = {k:[[],[]] for k in amgrois}
						for h in [0,1]:
							for sub in compdict[comp[0]][h]:
								try:
									dall[h].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi))
								except:
									continue
								for amg in amgrois:
									try:
										amgl[amg][h].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+amg))
									except:
										continue
							dall[h] = np.mean(np.nanmean(dall[h],0),0)
							for amg in amgrois:
								amgl[amg][h] = np.mean(np.nanmean(amgl[amg][h],0),0)
						ISCw[movie][roi][comp[0]][s,shuffle] = pearsonr(dall[0],dall[1])[0]
						for amg in amgrois:
							ISFCw[movie][roi][amg][comp[0]][s,shuffle] = np.mean([pearsonr(amgl[amg][0],dall[1])[0], pearsonr(amgl[amg][1],dall[0])[0]])
					else:
						dall = [[[],[]],[[],[]]]
						amgl = {k:[[[],[]],[[],[]]] for k in amgrois}
						for h1 in [0,1]:
							for h2 in [0,1]:
								for sub in compdict[comp[h1]][h2]:
									try:
										dall[h1][h2].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+roi))
									except:
										continue
									for amg in amgrois:
										try:
											amgl[amg][h1][h2].append(dd.io.load(parpath + 'sub-' + sub[:-3] + '.h5', '/'+amg))
										except:
											continue
								dall[h1][h2] = np.mean(np.nanmean(dall[h1][h2],0),0)
								for amg in amgrois:
									amgl[amg][h1][h2] = np.mean(np.nanmean(amgl[amg][h1][h2],0),0)
						for h1 in [0,1]:
							for h2 in [0,1]:
								ISCb_[movie][roi][s][shuffle].append(pearsonr(dall[0][h1],dall[1][h2])[0])
								for amg in amgrois:
									ISFCb[movie][roi][amg][s][shuffle]['C_amg_vs_ECA_par']. append(pearsonr(amgl[amg][0][h1], dall[1][h2])[0])
									ISFCb[movie][roi][amg][s][shuffle]['ECA_amg_vs_C_par']. append(pearsonr(amgl[amg][1][h1], dall[0][h2])[0])

savedict = {k:{k:{k:np.zeros(nshuffle+1) for k in ['ISCe','ISCb']} for k in ROIs} for k in movies}
isfc_ISCe = {k:{k:{k:np.zeros(nshuffle+1) for k in amgrois} for k in ROIs} for k in movies}
isfc_ISCb = {k:{k:{k:{k:np.zeros(nshuffle+1) for k in ['C_amg_vs_ECA_par','ECA_amg_vs_C_par']} for k in amgrois} for k in ROIs} for k in movies}
for movie in movies:
	for shuffle in tqdm.tqdm(range(nshuffle+1)):
		for roi in ROIs:
			Control = np.mean(ISCw[movie][roi]['Control'][:,shuffle],0)
			ECA = np.mean(ISCw[movie][roi]['ECA'][:,shuffle],0)
			savedict[movie][roi]['ISCe'][shuffle] = Control - ECA
			ISCb = np.mean([np.mean(ISCb_[movie][roi][s][shuffle]) for s in range(nsh)])
			savedict[movie][roi]['ISCb'][shuffle] = ISCb / (np.sqrt(Control) * np.sqrt(ECA))
			for amg in amgrois:
				Control = np.mean(ISFCw[movie][roi][amg]['Control'][:,shuffle],0)
				ECA = np.mean(ISFCw[movie][roi][amg]['ECA'][:,shuffle],0)
				C_amg_vs_ECA_par = np.mean([np.mean(ISFCb[movie][roi][amg][s][shuffle]['C_amg_vs_ECA_par']) for s in range(nsh)])
				ECA_amg_vs_C_par = np.mean([np.mean(ISFCb[movie][roi][amg][s][shuffle]['ECA_amg_vs_C_par']) for s in range(nsh)])
				isfc_ISCe[movie][roi][amg][shuffle] = Control - ECA
				isfc_ISCb[movie][roi][amg]['C_amg_vs_ECA_par'][shuffle] = C_amg_vs_ECA_par / (np.sqrt(Control) * np.sqrt(ECA))
				isfc_ISCb[movie][roi][amg]['ECA_amg_vs_C_par'][shuffle] = ECA_amg_vs_C_par / (np.sqrt(Control) * np.sqrt(ECA))
		
combodict = {k:{k:np.zeros(nshuffle+1) for k in ['ISCe','ISCb']} for k in ROIs}
combo_ISFCe = {k:{k:np.zeros(nshuffle+1) for k in amgrois} for k in ROIs}
combo_ISFCb = {k:{k:{k:np.zeros(nshuffle+1) for k in ['C_amg_vs_ECA_par','ECA_amg_vs_C_par']} for k in amgrois} for k in ROIs}
combo2_ISFCe = {k:np.zeros(nshuffle+1) for k in ROIs}
combo2_ISFCb = {k:{k:np.zeros(nshuffle+1) for k in ['C_amg_vs_ECA_par','ECA_amg_vs_C_par']} for k in ROIs}
for roi in tqdm.tqdm(ROIs):
	for shuffle in range(nshuffle+1):
		Control = np.mean(np.concatenate([ISCw['Homeward Bound'][roi]['Control'][:,shuffle], ISCw['Shirley'][roi]['Control'][:,shuffle]]),0)
		ECA     = np.mean(np.concatenate([ISCw['Homeward Bound'][roi]['ECA'][:,shuffle], ISCw['Shirley'][roi]['ECA'][:,shuffle]]),0)
		ISCb = [np.mean(ISCb_['Homeward Bound'][roi][s][shuffle]) for s in range(nsh)] + [np.mean(ISCb_['Shirley'][roi][s][shuffle]) for s in range(nsh)]
		combodict[roi]['ISCe'][shuffle] = Control - ECA
		combodict[roi]['ISCb'][shuffle] = np.mean(ISCb)/(np.sqrt(Control)*np.sqrt(ECA))
		Control_amg = []
		ECA_amg = []
		C_amg_vs_ECA_par = []
		ECA_amg_vs_C_par = []
		for a,amg in enumerate(amgrois):
			Control_amg.append(np.mean(np.concatenate([ISFCw['Homeward Bound'][roi][amg]['Control'][:,shuffle], ISFCw['Shirley'][roi][amg]['Control'][:,shuffle]]),0))
			ECA_amg.append(np.mean(np.concatenate([ISFCw['Homeward Bound'][roi][amg]['ECA'][:,shuffle], ISFCw['Shirley'][roi][amg]['ECA'][:,shuffle]]),0))
			C_amg_vs_ECA_par.append(np.mean([np.mean(ISFCb['Homeward Bound'][roi][amg][s][shuffle]['C_amg_vs_ECA_par']) for s in range(nsh)] + [np.mean(ISFCb['Shirley'][roi][amg][s][shuffle]['C_amg_vs_ECA_par']) for s in range(nsh)]))
			ECA_amg_vs_C_par.append(np.mean([np.mean(ISFCb['Homeward Bound'][roi][amg][s][shuffle]['ECA_amg_vs_C_par']) for s in range(nsh)] + [np.mean(ISFCb['Shirley'][roi][amg][s][shuffle]['ECA_amg_vs_C_par']) for s in range(nsh)]))
			combo_ISFCe[roi][amg][shuffle] = Control_amg[a] - ECA_amg[a]
			combo_ISFCb[roi][amg]['C_amg_vs_ECA_par'][shuffle] = C_amg_vs_ECA_par[a]/(np.sqrt(Control_amg[a]) * np.sqrt(ECA_amg[a]))
			combo_ISFCb[roi][amg]['ECA_amg_vs_C_par'][shuffle] = ECA_amg_vs_C_par[a]/(np.sqrt(Control_amg[a]) * np.sqrt(ECA_amg[a]))
		combo2_ISFCe[roi][shuffle] = np.mean(Control_amg) - np.mean(ECA_amg)
		combo2_ISFCb[roi]['C_amg_vs_ECA_par'][shuffle] = np.mean(C_amg_vs_ECA_par)/(np.sqrt(np.mean(Control_amg)) * np.sqrt(np.mean(ECA_amg)))
		combo2_ISFCb[roi]['ECA_amg_vs_C_par'][shuffle] = np.mean(ECA_amg_vs_C_par)/(np.sqrt(np.mean(Control_amg)) * np.sqrt(np.mean(ECA_amg)))
					
					
dd.io.save(ISCdir+'ISCsh_avg.h5',  {'moviedict':savedict, 'combodict':combodict})
dd.io.save(ISCdir+'ISFCsh_avg.h5', {'ISCe':isfc_ISCe, 'ISCb':isfc_ISCb, 'comboISCe':combo_ISFCe, 'comboISCb':combo_ISFCb, 'combo2ISCe':combo2_ISFCe, 'combo2ISCb':combo2_ISFCb})				
				

