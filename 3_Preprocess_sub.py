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
subs_temp = [sub for sub in subs if not os.path.isfile(subprepath + sub + '.h5')]
if len(subs_temp)>0:
	print('New subjects need to be run. Will re-run preprocessing on all',\
		 ' subjects to find intersection of subcortial roi')

mask_path = '_task-MOVIE_run-1_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz'
dpath = '_task-MOVIE_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
conf_path = '_task-MOVIE_run-1_desc-confounds_timeseries.tsv'
ROIs = {'L_HPC':[17],'R_HPC':[53],'L_AMG':[18],'R_AMG':[54]} # From: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

# Find intersection of ROIs across participants
fdir = {k:[] for k in subs}
for roi,idx in ROIs.items():
	mask = np.ones((81,96,81),dtype=bool)
	for sub in subs:
		fdir[sub] = 'ses-V2W2' if os.path.isdir(fmripreppath + sub + '/ses-V2W2/') else 'ses-V1W2'
		fpath = fmripreppath+sub+'/'+fdir[sub]+'/func/'+sub
		fname = fpath + '_' + fdir[sub] + mask_path
		if sub == 'sub-PA003': fname = fpath+ '_' + fdir[sub] + mask_path[:11] + '_acq-2' + mask_path[11:]
		if not os.path.isfile(fname): fname = fname.replace('_run-1','')
		submask = nib.load(fname)
		submask = submask.get_fdata() == idx[0]
		if submask.shape[2]==81:
			mask[np.where(np.logical_or(mask == False, submask == False))] = False
	ROIs[roi].append(mask)
		

for sub in tqdm.tqdm(subs):
	fpath = fmripreppath+sub+'/'+fdir[sub]+'/func/'+sub
	fname = fpath + '_' + fdir[sub] + dpath
	if sub == 'sub-PA003': fname = fpath + '_' + fdir[sub] + dpath[:11] + '_acq-2' + dpath[11:]
	if not os.path.isfile(fname): fname = fname.replace('_run-1','')
	nii = nib.load(fname).get_fdata()
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
		
			conf = np.genfromtxt(fpath + '_' + fdir[sub] + conf_path, names=True)
			motion = np.column_stack((conf['trans_x'],
									  conf['trans_y'],
									  conf['trans_z'],
									  conf['rot_x'],
									  conf['rot_y'],
									  conf['rot_z']))
			reg = np.column_stack((conf['csf'],
								   conf['white_matter'],
				  np.nan_to_num(conf['framewise_displacement']),
				  np.column_stack([conf[k] for k in conf.dtype.names if ('cosine' in k) or ('motion_outlier' in k)]),
								   motion,
								   np.vstack((np.zeros((1,motion.shape[1])), 
											  np.diff(motion, axis=0)))))

			regr = linear_model.LinearRegression()
			regr.fit(reg, roi_.T)
			roi_ = roi_ - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
			roidict[roi] = stats.zscore(roi_, axis=1)
		dd.io.save(subprepath + sub + '.h5',roidict)