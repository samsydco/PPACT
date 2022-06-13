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
prob_subs = [311]
#new_subs = [105, 147, 217, 218, 219, 251, 261, 263, 278, 292, 293, 296, 297, 304, 307, 317, 320, 335]
newsubs = ['PA'+str(s) for s in new_subs]

for sub in tqdm.tqdm(newsubs):
	out = sp.run(fwget+[sub+'_V2W2'])
	print(sub,out)


