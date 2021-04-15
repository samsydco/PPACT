#!/usr/bin/env python3


# Run pairwise ISC
# Use DF containing: subject, age, ECA, Movie
# Run ISC in sub-groups

import glob
import tqdm
import pandas as pd
import deepdish as dd
from itertools import permutations
from settings import *


Phenodf = pd.read_csv(phenopath+'Phenodf.csv')
Phenodf = Phenodf[Phenodf['FDmean'].notna()].reset_index().drop(['level_0','index'],axis=1)

subs = glob.glob(parpath+'*.h5')
subpairs = list(permutations(subs,2))
ROIs = list(dd.io.load(subs[0]).keys())[1:]

for roi in tqdm.tqdm(ROIs):
	for pair in subpairs:
		v1 = dd.io.load(pair[0],'/'+roi)
		v2 = dd.io.load(pair[1],'/'+roi)
		
	








