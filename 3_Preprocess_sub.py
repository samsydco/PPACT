#!/usr/bin/env python3

import os
import tqdm
import glob
import nibabel as nib
import numpy as np
import deepdish as dd
from sklearn import linear_model
from scipy import stats
from settings import *

subs = glob.glob('%ssub*.html'%(fmripreppath))
subs = [s.replace('.html', '').replace(fmripreppath, '') for s in subs]
subs = [sub for sub in subs if not os.path.isfile(subprepath + sub + '.h5')]

mask_path = '_ses-V2W2_task-MOVIE_run-1_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz'
dpath = '_ses-V2W2_task-MOVIE_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
conf_path = '_ses-V2W2_task-MOVIE_run-1_desc-confounds_timeseries.tsv'
ROIs = {'L_HPC':17,'R_HPC':53,'L_AMG':18,'R_AMG':54} # From: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

for sub in tqdm.tqdm(subs):
	fpath = fmripreppath+sub+'/ses-V2W2/func/'+sub
	nii = nib.load(fpath+dpath).get_fdata()
	mask = nib.load(fpath+mask_path)
	mask = mask.get_fdata()
	roidict = {}
	for roi,idx in ROIs.items():
		mask_ = mask == idx
		roi_ = nii[mask_]

		# Use regressors for:
		# -CSF
		# -WhiteMatter
		# -FramewiseDisplacement
		# -All cosine bases for drift (0.008 Hz = 125s)
		# -X, Y, Z and derivatives
		# -RotX, RotY, RotZ and derivatives
		
		conf = np.genfromtxt(fpath+conf_path, names=True)
		motion = np.column_stack((conf['trans_x'],
								  conf['trans_y'],
								  conf['trans_z'],
								  conf['rot_x'],
								  conf['rot_y'],
								  conf['rot_z']))
		reg = np.column_stack((conf['csf'],
							   conf['white_matter'],
			  np.nan_to_num(conf['framewise_displacement']),
			  np.column_stack([conf[k] for k in conf.dtype.names if 'cosine' in k]),
							   motion,
							   np.vstack((np.zeros((1,motion.shape[1])), 
										  np.diff(motion, axis=0)))))

		regr = linear_model.LinearRegression()
		regr.fit(reg, roi_.T)
		roi_ = roi_ - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
		roidict[roi] = stats.zscore(roi_, axis=1)
	dd.io.save(subprepath + sub + '.h5',roidict)