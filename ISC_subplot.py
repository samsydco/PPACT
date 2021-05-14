#!/usr/bin/env python3


# Plot differences in mean ISC between groups
# For subcortical ROIs
# Use: ISCdir+'ISCw_ISCb.h5'



import pandas as pd
import deepdish as dd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from settings import *
plt.rcParams.update({'font.size': 15})

ISCbg = dd.io.load(ISCdir+'ISCw_ISCb.h5')
ROIs = [r for r in ISCbg[movies[0]].keys() if len(r)==5]
figdir_ = figdir+'subcort_ISC/'

for roi in ROIs:
	for isc in ['ISCe','ISCb']:
		fig, ax = plt.subplots()
		y = 0 if isc=='ISCe' else 1
		ax.axhline(y=y, color='gray', linestyle='--',linewidth=2)
		for movie in movies:
			xticks = []; ISCs = [];
			for comp in ISCbg[movie][roi]['meanISC'][isc].keys():
				xticks.append(comp[0]+' vs '+comp[1] if isc == 'ISCb' else comp[0][0]+' vs '+comp[1][0])
				ISCs.append(ISCbg[movie][roi]['meanISC'][isc][comp])
				if isc == 'ISCe': xticks=xticks[:4]; ISCs=ISCs[:4]
			ax.scatter(xticks, ISCs,label=movie)
			ax.set_ylabel(isc)
			ax.set_title(roi)
		ax.legend()
		fig.savefig(figdir_+roi+'_'+isc+'.png')
				
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
	





