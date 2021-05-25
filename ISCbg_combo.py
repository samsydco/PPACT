#!/usr/bin/env python3

# Combine both movies when calculating ISCe and ISCb
# Average pre-subtraction, or geometric mean


import glob
import tqdm
import deepdish as dd
import numpy as np
from settings import *

savedir = ISCdir+'ISCw_ISCb/'
newsavedir = ISCdir+'ISCw_ISCb_combo/'
ROIs = [r.split('/')[-1] for r in glob.glob(ISCdir+'ISCall/'+'*')]

for roi in tqdm.tqdm(ROIs):
	HB = dd.io.load(savedir+movies[0]+'_'+roi)
	Sh = dd.io.load(savedir+movies[1]+'_'+roi)
	savedict = {k:{k:{} for k in ['ISCe','ISCb']} for k in ['meanISC']}
	for i in HB['meanISC']['ISCe'].keys():
		ISC1 = np.mean([HB['meanISC']['ISCs'][i[0]],Sh['meanISC']['ISCs'][i[0]]],0)
		ISC2 = np.mean([HB['meanISC']['ISCs'][i[1]],Sh['meanISC']['ISCs'][i[1]]],0)
		savedict['meanISC']['ISCe'][i] = ISC1 - ISC2
	for i in HB['meanISC']['ISCb'].keys():
		ISCtop = np.mean([HB['meanISC']['ISCs'][i],Sh['meanISC']['ISCs'][i]],0)
		ISCbot1 = np.mean([HB['meanISC']['ISCs'][(i[0],i[0])], \
						   Sh['meanISC']['ISCs'][(i[0],i[0])]],0)
		ISCbot2 = np.mean([HB['meanISC']['ISCs'][(i[1],i[1])], \
						   Sh['meanISC']['ISCs'][(i[1],i[1])]],0)
		savedict['meanISC']['ISCb'][i] = ISCtop/(np.sqrt(ISCbot1)*np.sqrt(ISCbot2))
	dd.io.save(newsavedir+roi,savedict)
	
			
			
					
	