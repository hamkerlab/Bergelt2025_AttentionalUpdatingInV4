# -*- coding: utf-8 -*-
"""
@author: juschu

extract subset of experimental setups
with defined, fixed attention position

Following experiment of Marino and Mazer (2018):
 - 1 fixation point (FP), 1 saccade target (ST)
 - 1 attention position (AP) - fixed
 - multiple random bars as stimuli at different positions above and below FP/ST

 - store generated setups
"""


##############################
#### imports and settings ####
##############################
# import standard libraries
import os
import sys
from shutil import copyfile

# import files from this folder
from saving import load_dict_from_hdf5

# import files from subfolders
# parameters
from parameters.params_general import params

if __name__ == "__main__":

    ## Definitions
    # fixed attention position
    if len(sys.argv) > 1:
        AP = sys.argv[1]
    else:
        AP = '[ 0. -7.]'

    # folder of all saved setups
    source_folder = f'Results/setup/'
    # folder for new, extracted sub-setups
    target_folder = f'Results/setup_{AP}'

    # files, that should be copied
    files = ['0_barpos.txt', 'output.txt']
    
    ## Extract setups
    # get trials according to attention position
    trialsPerAP = load_dict_from_hdf5(f"{source_folder}/trialsPerAP.hdf5")
    # print(trialsPerAP.keys())

    # copy for all trials with given attention position defined files
    for trial in trialsPerAP[AP]:
        print(f"Copy files for trial {trial}")
        saveDir = f"{target_folder}/{int(trial)}"
        os.makedirs(saveDir)
        for f in files:
            copyfile(f"{source_folder}/{int(trial)}/{f}", f"{saveDir}/{f}")