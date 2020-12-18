#!/usr/bin/env python3

# Check to see if there are any new subjects in flywheel with movies
# Count number of subjects on sophon with movies
# Current count is 174

import subprocess as sp
import glob
import tqdm
from settings import *
fwls = [path + 'fw', 'ls']
fwget = [path+'fw','export','bids',datapath,'--project','PACCT_study_w2','--session']

fwlist = sp.check_output(fwls+[fwpath]).splitlines()
fwlist = [x.decode("utf-8") for x in fwlist]
fwlist = [s[1:] for s in fwlist if 'PA' in s and len(s)==6]

# If anything prints, new subject with functional run has been added
for sub in tqdm.tqdm(fwlist):#list(set(fwlist).difference(subs)):
	out = sp.run(fwget+[sub+'_V2W2'])
	print(sub,out)


