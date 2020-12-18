#!/usr/bin/env python3

import subprocess as sp 
import glob
import pandas as pd
import os
from settings import *

def yesnofun(sub):
    while True:
        try:
            yesno = input("%s: Type \"y\" for \"yes\", \"n\" for \"no\", and \
                                \"m\" for \"maybe\".\n(If you need a break, type \"break\".)\n"%(sub))
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue

        if yesno not in ['y','n','m','break']:
            print("Sorry, your response must be \"y\", \"n\", or \"m\".")
            continue
        elif yesno=="break":
            break
        else:
            #Answer is good.
            break
    return yesno

# Right now all RAs should have already entered their information:
while True:
	initials = input("Please type your initials - should be same two letters every time you come to lab.")
	T1file = T1path+initials+"_T1.csv"
	if not glob.glob(T1path+initials+'_*'):
		print("You used different initials last time!")
		continue
	else:
		break
# If T1file exists, import info, if not, create it
if os.path.exists(T1file):
	df = pd.read_csv(T1file)
else:
	df = pd.DataFrame(columns=['sub', 'yesno','anatmv'])

subs = glob.glob('%ssub*'%(datapath))
print('You have %s scans left.'%(len(subs) - len(df)))
for sub in subs:
	sub_temp = sub.replace(datapath,"")
	if sub_temp not in df['sub'].values:
		anatmv = 0
		fbase = sub+'/ses-V2W2/'
		BIDsf = fbase+'anat/'
		anat1 = BIDsf+sub_temp+'_ses-V2W2_acq-MPR_rec-vNav_T1w.nii.gz'
		if not os.path.exists(BIDsf):
			anatmv = 1
			sp.run(['cp','-r',sub+'/anat/',fbase])
		# Look at T1's in fsleyes pause, do fmriprep if ok
		if os.path.exists(anat1):
			sp.run(["fsleyes",anat1])
		elif os.path.exists(BIDsf+sub_temp+'_ses-V2W2_acq-MPR_rec-Norm_T1w.nii.gz'):
			sp.run(['fsleyes',BIDsf+sub_temp+'_ses-V2W2_acq-MPR_rec-Norm_T1w.nii.gz'])
		elif os.path.exists("%s/ses-V2W2/anat/%s_acq-3_T1w.nii.gz"%(sub,sub_temp)):
			sp.run(["fsleyes","%s/ses-V2W2/anat/%s_acq-3_T1w.nii.gz"%(sub,sub_temp)])
		else:
			anatmv = 2
			print('No anatomical for '+sub_temp+' ???')
		if anatmv ==2:
			yesno = 'n'
		else:
			yesno = yesnofun(sub)
		
		if yesno in ['y','n','m']:
			df = df.append({"sub":sub_temp, "yesno":yesno, "anatmv":anatmv},
						   ignore_index=True)
		else:
			break
df.to_csv(T1file, index=False)





