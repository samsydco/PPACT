#!/usr/bin/env python3


# Plot differences in mean ISC between groups
# For subcortical ROIs
# Use: ISCdir+'ISCw_ISCb.h5'


import glob
import pandas as pd
import deepdish as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from settings import *
plt.rcParams.update({'font.size': 15})

savedir = ISCdir+'ISCw_ISCb/'
ROIfs = [f for f in glob.glob(savedir+'*') if any(l in f for l in ['HPC','AMG'])]
ROIfs2 = [f for f in glob.glob(savedir[:-1]+'_combo/*') if any(l in f for l in ['HPC','AMG'])]
ROIfs=ROIfs+ROIfs2
figdir_ = figdir+'subcort_ISC/'
		
for r in ROIfs:
	roif = dd.io.load(r,'/meanISC')
	rsplit = r.split('/')[-1].split('_')
	movie = rsplit[0] if len(rsplit)==3 else ''
	roi = '_'.join(rsplit[1:])[:-3] if len(rsplit)==3 else '_'.join(rsplit)[:-3]
	for isc in ['ISCb','ISCe']:
		xticks = []; ISCs = []; violins = [];
		for comp in roif[isc].keys():
			xticks.append(comp[0]+' vs '+comp[1] if isc == 'ISCb' else comp[0][0]+' vs '+comp[1][0])
			ISCs.append(roif[isc][comp][0])
			v = roif[isc][comp][1:]
			violins.append(v[~np.isnan(v)])
		if isc == 'ISCe': xticks=xticks[:4]; ISCs=ISCs[:4]; violins=violins[:4]
		plt.rcParams.update({'font.size': 15})
		fig,ax = plt.subplots()
		y = 0 if isc=='ISCe' else 1
		ax.axhline(y=y, color='gray', linestyle='--',linewidth=2)
		parts = ax.violinplot(violins, np.arange(len(ISCs)), showmeans=False, showmedians=False, showextrema=False)
		ax.scatter(xticks, ISCs,color='k',s=80)
		ax.set_ylabel(isc)
		ax.set_title(roi+' '+movie)
		ax.set_ylim([-0.04,0.04]) if isc == 'ISCe' else ax.set_ylim([0,2])
		for pc in parts['bodies']:
			pc.set_facecolor('k')
		fig.savefig(figdir_+'_'.join([movie,roi,isc])+'.png', bbox_inches="tight")


ISCpat = dd.io.load(ISCdir+'ISCpatw_ISCpatb.h5')
movpairs = list(ISCpat.keys())
for roi in ROIs:
	for isc in ['ISCe','ISCb']:
		fig, ax = plt.subplots(1,3,figsize=(16,5))
		for event in range(3):
			y = 0 if isc=='ISCe' else 1
			ax[event].axhline(y=y, color='gray', linestyle='--',linewidth=3)
			for movpair in movpairs:
				xticks = []; ISCs = [];
				for comp in ISCpat[movpair][roi]['patternISC'][event][isc].keys():
					xticks.append(comp[0]+' vs '+comp[1] if isc == 'ISCb' else comp[0][0]+' vs '+comp[1][0])
					ISCs.append(ISCpat[movpair][roi]['patternISC'][event][isc][comp])
					if isc == 'ISCe': xticks=xticks[:4]; ISCs=ISCs[:4]
				ax[event].scatter(xticks, ISCs,label=movpair[0]+' - '+movpair[1])
				if event==0: ax[event].set_ylabel(isc)
				ax[event].set_title('Event '+str(event))
		ax[event].legend(bbox_to_anchor=(1, 1.05))
		fig.suptitle(roi, y=1)
		fig.savefig(figdir_+roi+'_'+isc+'_pat.png', bbox_inches="tight")
	





