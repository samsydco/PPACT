#!/usr/bin/env python3

# create list of all subjects for which we have movie data
# save list to txt file

import glob
from settings import *
	
sub_list = []
for sub  in glob.glob(datapath+'/sub*'):
	sub_ = sub.split('/')[-1]
	if len(glob.glob(fmripreppath+sub_+'/*/func/sub*fs*'))>0:
		sub_list.append(sub_)
	elif (len(glob.glob(fmripreppath+sub_+'/*/func/sub*fs*'))==0 and
		   len(glob.glob(sub+'/*/anat/*T1*'))>0 and
		   len(glob.glob(sub+'/*/func/*MOVIE*'))>0):
		sub_list.append(sub_)
	
with open('subject_list.txt', 'w') as f:
    for item in sub_list:
        f.write("%s\n" % item)
		
		