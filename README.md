# PPACT
Data analysis for PPACT dataset

Description of each .py script:
Scripts to run in numbered order:
1_fw.py - check for new subjects on flywheel and download those subjects             
2_run_singularity.py - runs fmriprep on newly downloaded subjects
3_Preprocess.py - regresses-out motion in new subjects (in cortical space)
3_Preprocess_sub.py - regresses-out motion in new subjects (in sub-cortical spaces: hippocampus and amygdala)
4_Parcels.py - saves each subject's data into 100 cortical parcels
5_Pheno.py - makes dataframe of pheno-type (ECA vs non-ECA data), gender, and framewise displacement (FD)
FD_Check.py - Makes plots of framewise displacement (FD)

ISC Code:
ISC_vert.py - run pairwise ISC in vertex-space 
- obtain a measure of overall pairwise ISC

ISC.py - run pairwise ISC in parcels (in all vertexes, mean of all vertexes, pattern ISC over all movie)
ISCmat.py - put pairwise ISC (in parcels) values from ISC.py into matrix according to ECA status and plot

ISCbg_vert.py - pairwise ISC both within and between groups in vertex-space
ISCsh.py - split half ISC both within- and between-groups in parcel-space (also in subcortical areas)
ISCbg_combo.py - combines both movies when calculating between-group ISC and ISC group differences. - in parcel-space     

ISCpat.py - pairwise pattern ISC in 3 movie events
ISCsh_pat.py - split-half pattern ISC in 3 movie events
Honestly not sure difference between these two:
ISCpat_comp.py   
ISCpat_comp_eff.py
   
ISFC.py

ISCbg.py  
ISC_subplot.py - plots ISCs for subcortical ROIs 
                     
settings.py
