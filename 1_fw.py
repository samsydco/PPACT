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
fwlist = [s[5:] for s in fwlist if 'PA' in s]

new_subs = []
#new_subs = [283,282,251,250,249,248,263,119,204,135,307,296,168]
#new_subs = [251, 250, 263, 204, 296]
newsubs = ['PA'+str(s) for s in new_subs]

for sub in tqdm.tqdm(newsubs):
	out = sp.run(fwget+[sub+'_V2W2'])
	print(sub,out)


