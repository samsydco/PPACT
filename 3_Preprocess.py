#!/usr/bin/env python3

import glob
import nibabel as nib
import pandas as pd
import numpy as np
import os
import h5py
from scipy import stats
from settings import *

subs = glob.glob('%ssub*.html'%(fmripreppath))
subs = [s.replace('.html', '') for s in subs]
subs = [s.replace(fmripreppath, '') for s in subs]
subs = [sub for sub in subs if not os.path.isfile(prepath + sub + '.h5') and sub not in bad_sub_dict]

Phenodf = pd.read_csv(phenopath+'schema_inventory.csv')

for sub in subs:
    print('Processing subject ', sub)
    Demo = {'Age': Phenodf['age_truncated'][Phenodf['IDENT_SUBID'] == sub[4:]+'_V2'].iloc[0],
           'Sex': Phenodf['DEM_3_GENDER_CHILD'][Phenodf['IDENT_SUBID'] == sub[4:]+'_V2'].iloc[0]}
    with h5py.File(os.path.join(prepath + sub + '.h5')) as hf:
        grp = hf.create_group('Pheno')
        for k,v in Demo.items():
            grp.create_dataset(k,data=v)
    D = dict()
    for hem in ['L', 'R']:
        fname = os.path.join(fmripreppath + sub + '/ses-V2W2/func/' + sub + '_ses-V2W2_task-MOVIE_run-1_space-fsaverage6_hemi-' + hem + '_bold.func.gii')
        print('      Loading ', fname)
        gi = nib.load(fname)
        D[hem] = np.column_stack([gi.darrays[t].data for t in range(len(gi.darrays))])
        
    # Use regressors for:
    # -CSF
    # -WhiteMatter
    # -FramewiseDisplacement
    # -All cosine bases for drift (0.008 Hz = 125s)
    # -X, Y, Z and derivatives
    # -RotX, RotY, RotZ and derivatives
    
    conf = np.genfromtxt(os.path.join(fmripreppath + sub + '/ses-V2W2/func/' + sub + '_ses-V2W2_task-MOVIE_run-1_desc-confounds_timeseries.tsv'), names=True)
    motion = np.column_stack((conf['trans_x'],\
							  conf['trans_y'],\
							  conf['trans_z'],\
							  conf['rot_x'],\
							  conf['rot_y'],\
							  conf['rot_z']))
    reg = np.column_stack((conf['csf'],conf['white_matter'],\
		  np.nan_to_num(conf['framewise_displacement']),\
		  np.column_stack([conf[k] for k in conf.dtype.names if 'cosine' in k]),\
		  motion,\
		  np.vstack((np.zeros((1, motion.shape[1])), np.diff(motion, axis=0)))))
    print('      Cleaning and zscoring')
    for hem in ['L', 'R']:
        regr = linear_model.LinearRegression()
        regr.fit(reg, D[hem].T)
        D[hem] = D[hem] - np.dot(regr.coef_, reg.T) - regr.intercept_[:, np.newaxis]
        # Note 8% of values on cortical surface are NaNs, and the following will therefore throw an error
        D[hem] = stats.zscore(D[hem], axis=1)
    with h5py.File(os.path.join(prepath + sub + '.h5')) as hf:
        hf.create_dataset('L', data=D['L'])
        hf.create_dataset('R', data=D['R'])
        hf.create_dataset('reg',data=reg)


    