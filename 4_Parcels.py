#!/usr/bin/env python3

# Create parcels

import tqdm
import glob
import pandas as pd
import numpy as np
import deepdish as dd
import nibabel.freesurfer.io as free
from settings import *

for sub in tqdm.tqdm(glob.glob(prepath+'*.h5')):
	sub_short = sub.split('/')[-1]
	roidict = {}
	for hemi in glob.glob(path+'annot/*'):
		lab = free.read_annot(hemi)
		for ri,roi_tmp in enumerate(lab[2]):
			roi=roi_tmp.decode("utf-8")
			roi_short=roi_tmp.decode("utf-8")[11:]
			vall = np.where(lab[0]==ri)[0]
			hemi = (hemi.split('/')[-1][0]).upper()
			roidict[roi_short] = dd.io.load(sub,'/'+hemi,sel=dd.aslice[vall,:])
	if os.path.exists(subprepath+sub_short):
		subrois = dd.io.load(subprepath+sub_short)
		for roi in subrois.keys():
			roidict[roi] = subrois[roi]
	dd.io.save(parpath+sub_short,roidict)
	

