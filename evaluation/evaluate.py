"""
@author: juschu

Main script for evaluation
"""

##############################
#### imports and settings ####
##############################
# import standard libraries
import os
import sys
# import files from this folder
from processData import getTrialsPerAP, getBarposAndOnsetCombinations, extract_eyepos, extract_LIPdata, extract_onsetPos
from plotting import plot_setup, plot_revCorrelation, plot_actLIP, plot_actV4L4, plot_actAll, plot_ratesOverTime
# parameters
from parameters.params_general import params


######################
#### Main program ####
######################
if __name__ == "__main__":

    ## preprocess data
    # where to save the data
    if not os.path.isdir(f"../{params['ResultDir']}/extractedData/"):
        os.makedirs(f"../{params['ResultDir']}/extractedData/")

    print("extract eye position")
    extract_eyepos()

    print("get trials per attention position")
    trialsPerAP = getTrialsPerAP()

    print("get all bar onset and position combinations")
    getBarposAndOnsetCombinations()

    print("extract LIP data")
    extract_LIPdata()

    print(f"extract rates of layer V4L4")
    extract_onsetPos('r_V4L4')
    # additional evaluation for sub results
    # for layer in ['r_V1', 'E_V4L4', 'ALIP_V4L4', 'ASP_V4L4', 'AFEAT_V4L4', 'sum(A_LIPpc)_V4L4', 'sum(A_LIPcd)_V4L4']:
    #    print(f"extract data of layer {layer}")
    #    extract_onsetPos(layer)


    ## plotting
    # where to save the figures
    if not os.path.isdir(f"../{params['ResultDir']}/figs/"):
        os.makedirs(f"../{params['ResultDir']}/figs/")

    # id of trial to plot
    if len(sys.argv) > 1:
        trial = sys.argv[1]
    else:
        trial = 1

    # Figure 2
    plot_setup(trial)

    # Figure 3 and 6
    plot_actLIP()

    # Figure 4
    plot_revCorrelation('r_V4L4')
    # additional plots for sub results
    # for layer in ['r_V1', 'E_V4L4', 'ALIP_V4L4', 'ASP_V4L4', 'AFEAT_V4L4', 'sum(A_LIPpc)_V4L4', 'sum(A_LIPcd)_V4L4']:
    #     print(f"plot reverse correlation of layer {layer}")
    #     plot_revCorrelation(layer)

    # Figure 5 and 7
    plot_actV4L4()
    # additional plots for sub results
    # plot_actAll()

    # Figure 9
    plot_ratesOverTime(trial)
    plot_ratesOverTime(trial, timesteps=[0, 50, 300, 350, 417, 500, 700])
