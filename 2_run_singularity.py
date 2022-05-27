#!/usr/bin/env python3

# import variables from 3_fmriprep.py:
import subprocess as sp 
import glob
import pandas as pd
import numpy as np
import os
import tqdm 
import datetime
from settings import *
	
plist = []
prob_s = {}
done_sub = []
for sub  in glob.glob(datapath+'/sub*'):
	sub_ = sub.split('/')[-1]
	if (len(glob.glob(fmripreppath+sub_+'/*/func/sub*fs*'))==0 and
		   len(glob.glob(sub+'/*/anat/*T1*'))>0 and
		   len(glob.glob(sub+'/*/func/*MOVIE*'))>0):
		plist.append(sub_)
	elif (len(glob.glob(sub+'/*/anat/*T1*'))==0 or
		   len(glob.glob(sub+'/*/func/*MOVIE*'))==0):
		prob_s[sub]={'anat':glob.glob(sub+'/*/anat/*T1*'),
					 'movie':glob.glob(sub+'/*/func/*MOVIE*')}
	elif len(glob.glob(fmripreppath+sub_+'/*/func/sub*fs*'))>0:
		done_sub.append(sub_)
plist = plist[:75]
nchunk = 4 # number of items per chunk (maybe use 10?)
pchunk = [plist[x:x+nchunk] for x in range(0,len(plist),nchunk)]

for chunk in tqdm.tqdm(pchunk):
	pstr = ' '.join([c[4:] for c in chunk])
	date = str(datetime.datetime.now())[0:19].replace(' ','_')
	f = open("%sfmriprep_cmdoutput/%s_%s.txt"%(path,date,pstr.replace(" ","_")), "w")
	command = ('singularity run \
		-B '+datapath+':/data \
		-B '+outputdr+':/out \
		-B '+path+'scratch/:/work \
		--cleanenv \
		'+path+'PACCT_code/fmriprep-20.2.1.simg \
		/data /out \
		participant \
		--ignore slicetiming \
		--output-spaces MNI152NLin2009cAsym fsaverage:den-41k \
		--fs-license-file /data/fs_license.txt \
		-w /work/ \
		--participant-label '+pstr+' -t MOVIE').split()
	p = sp.Popen(command,\
				 stdin=sp.PIPE, stderr=sp.PIPE,\
				 universal_newlines=True,stdout=f)
	p.communicate('/n')[1]