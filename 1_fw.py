#!/usr/bin/env python3

# Check to see if there are any new subjects in flywheel with movies
# Count number of subjects on sophon with movies
# Current count is 174

import subprocess as sp 
import glob
from settings import *
fwls = [path + 'fw', 'ls']

subs = [s.split(path)[-1][4:] for s in glob.glob(path+'sub*')]

fwlist = sp.check_output(fwls+[fwpath]).splitlines()
fwlist = [x.decode("utf-8") for x in fwlist]
fwlist = [s[3:] for s in fwlist if 'PA' in s and len(s)==8]

# If anything prints, new subject with functional run has been added
for sub in list(set(fwlist).difference(subs)):
	subpath = fwpath + '/' + sub
	try:
		for ses_tmp in sp.check_output(fwls + [subpath]).splitlines():
			ses = ses_tmp.decode("utf-8")[-10:]
			if len([i.decode("utf-8") for i in sp.check_output(fwls + [subpath + '/' + ses]).splitlines() if 'func' in i.decode("utf-8")])>0:
				print(sub,ses)
	except:
		continue
		
movsubs = [sub for sub in subs if len(glob.glob(path+'sub-'+sub+'/ses*/func/*'))>0]



# Notes if need to download more subjects in future
'''
sp.run(["aws","s3","cp","s3://fcp-indi/data/Archives/HBN/MRI/"+site+"/%s"%(sub),path,"--no-sign-request"])
fwpath export bids [Project in Flywheel] [Destination folder] [optional flags]
/data/PACCT/fw export bids --project PACCT_study_w2 .
'''

