#!/usr/bin/env python3

# Create DF containing: subject, age, ECA, Movie, FD

import os
import glob
import tqdm
import pandas as pd
import numpy as np
from settings import *

Phenodf = pd.read_csv(phenopath+'schema_inventory.csv').drop([0]).drop('Unnamed: 4',axis=1)
Phenodf = Phenodf.drop(Phenodf[Phenodf.age_truncated.isnull()].index).reset_index()
Phenodf['Group'] = np.where((Phenodf.GROUP == 'C'),'Control','ECA')
GROUPkey = {'c':'comparison','PI':'previously institutionalized','DA':'domestically adopted','DC':'disrupted care','IFC':'international foster care'}
GENDERkey = {1:'female',0:'male'}

subs=[sub.split('/')[-1].split('.')[0] for sub in glob.glob(prepath+'*.h5')]


Phenodf['FDmean'] = np.nan
Phenodf['FDstd']  = np.nan
Phenodf['FDmax']  = np.nan
for sub in tqdm.tqdm(subs):
	subtmp = sub.replace('sub-','')+'_V2'
	conf = np.genfromtxt(os.path.join(fmripreppath + sub + '/ses-V2W2/func/' + sub + '_ses-V2W2_task-MOVIE_run-1_desc-confounds_timeseries.tsv'), names=True)
	FD = np.nan_to_num(conf['framewise_displacement'])
	Phenodf['FDmean'].loc[Phenodf['IDENT_SUBID'] == subtmp] = np.mean(FD)
	Phenodf['FDstd'].loc[Phenodf['IDENT_SUBID'] == subtmp]  = np.std(FD)
	Phenodf['FDmax'].loc[Phenodf['IDENT_SUBID'] == subtmp]  = np.max(FD)

Phenodf.to_csv(phenopath+'Phenodf.csv', index=False)