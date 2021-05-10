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
ROIs = {'L_HPC':[17],'R_HPC':[53],'L_AMG':[18],'R_AMG':[54]} # From: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

# Find intersection of ROIs across participants
for roi,idx in ROIs.items():
	mask = np.ones((81,96,81),dtype=bool)
	for sub in subs:
		fpath = fmripreppath+sub+'/ses-V2W2/func/'+sub
		submask = nib.load(fpath+mask_path) if sub!='sub-PA003' else nib.load(fpath+mask_path[:20]+'_acq-2'+mask_path[20:])
		submask = submask.get_fdata() == idx[0]
		if submask.shape[2]==81:
			mask[np.where(np.logical_or(mask == False, submask == False))] = False
	ROIs[roi].append(mask)
		

for sub in tqdm.tqdm(subs):
	fpath = fmripreppath+sub+'/ses-V2W2/func/'+sub
	nii = nib.load(fpath+dpath).get_fdata() if sub!='sub-PA003' else nib.load(fpath+dpath[:20]+'_acq-2'+dpath[20:]).get_fdata()
	if nii.shape[2]==81:
		roidict = {}
		for roi,idx in ROIs.items():
			mask = idx[1]
			roi_ = nii[mask]

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