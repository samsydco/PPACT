#!/usr/bin/env python3


# Plot differences in mean ISC between groups
# For subcortical ROIs
# Use: ISCdir+'ISCw_ISCb.h5'


import glob
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt
from settings import *
plt.rcParams.update({'font.size': 15})

data=dd.io.load(ISCdir+'ISCsh.h5')
ROIs = [r for r in data[movies[0]].keys() if any(l in r for l in ['HPC','AMG'])]
figdir_ = figdir+'subcort_ISC/'

for movie in movies:
	for roi in ROIs:
		for isc in ['ISCb','ISCe']:
			ISC =    np.nanmean(data[movie][roi][isc][:,0])
			violin = np.nanmean(data[movie][roi][isc][:,1:],0)
			violin = violin[~np.isnan(violin)]
			fig,ax = plt.subplots()
			y = 0 if isc=='ISCe' else 1
			ax.axhline(y=y, color='gray', linestyle='--',linewidth=2)
			parts = ax.violinplot(violin, showmeans=False, showmedians=False, showextrema=False)
			ax.scatter(1,ISC,color='k',s=80)
			ax.set_xticks([1])
			ax.set_xticklabels(['C vs ECA'])
			ax.set_ylabel(isc)
			ax.set_title(roi+' '+movie)
			#ax.set_ylim([-0.04,0.04]) if isc == 'ISCe' else ax.set_ylim([0,2])
			for pc in parts['bodies']:
				pc.set_facecolor('k')
			fig.savefig(figdir_+'_'.join([movie,roi,isc])+'_ISCsh.png', bbox_inches="tight")

savedir = ISCdir+'ISCpatw_ISCpatb/'		
ROIfs = [r for r in glob.glob(savedir+'*') if any(l in r for l in ['HPC','AMG'])]

for r in ROIfs:
	roif = dd.io.load(r,'/patternISC')
	rsplit = r.split('/')[-1].split('_')
	movpair = '_'.join(rsplit[:-2])
	roi = '_'.join(rsplit[-2:])[:-3]
	for isc in ['ISCb','ISCe']:
		fig, ax = plt.subplots(1,3,figsize=(16,5))
		for event in range(3):
			xticks = []; ISCs = []; violins = [];
			for comp in roif[event][isc].keys():
				xticks.append(comp[0]+' vs '+comp[1] if isc == 'ISCb' else comp[0][0]+' vs '+comp[1][0])
				ISCs.append(roif[event][isc][comp][0])
				v = roif[event][isc][comp][1:]
				violins.append(v[~np.isnan(v)])
			if isc == 'ISCe': xticks=xticks[:4]; ISCs=ISCs[:4]; violins=violins[:4]
			y = 0 if isc=='ISCe' else 1
			ax[event].axhline(y=y, color='gray', linestyle='--',linewidth=2)
			parts = ax[event].violinplot(violins, np.arange(len(ISCs)), showmeans=False, showmedians=False, showextrema=False)
			ax[event].scatter(xticks, ISCs,color='k',s=80)
			if event==0: ax[event].set_ylabel(isc)
			ax[event].set_title('Event '+str(event))
			for pc in parts['bodies']:
				pc.set_facecolor('k')
			ax[event].set_ylim([-0.08,0.08]) if isc == 'ISCe' else ax[event].set_ylim([-1,2])
		fig.suptitle(roi, y=1)
		fig.savefig(figdir_+'_'.join([movpair,roi,isc])+'_pat.png', bbox_inches="tight")



