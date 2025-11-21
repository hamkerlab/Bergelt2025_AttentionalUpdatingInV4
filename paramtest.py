import sys
import os
import numpy as np
import pandas as pd

from parameters.params_model import params


def createParamVariation(filename, perc, amount):
    '''
    create sets of parameter variations

    params: filename -- name of file for saving parameter values    
            perc     -- range of change (plus/minus 5% of original value)
            amount   -- number of parameter sets


    '''

    # list of parameters, that should not be varied
    skip = [# general parameters
            'runInParallel', 'var_num', 'ResultDir', 'SetupDir', 'ConnDir', 'numOfTrials',
            # spatial and temporal design of setup
            'VF_Deg', 'tEnd', 'range_v', 'range_sup', 'range_h', 'flash_duration', 'saccOnset', 'range_t', 'saccade', 'attType', 'att_turnoff',
            'pc_t_update', 'pc_sigma', 'pc_strength', 'pc_sigma_decay', 'attention_sigma', 'attention_strength', 'V1_sigma', 'V1_strength', 'cd_t_peak', 'cd_sigma', 'cd_strength', 'cd_sigma_rise', 'cd_sigma_decay',
            'v_FEFv-FEFvm' , 'v_FEFv-FEFvm_E', 'v_FEFv-FEFvm_S', 'v_FEFv-FEFvm,scale', 'v_FEFv-FEFvm,factor', 'v_FEFvm-FEFm_E', 'v_FEFvm-FEFm_S', 'v_FEFfix-FEFm', 'SaccadeThreshold',
            # model layout and general model parameters
            'resVisual', 'resSpatial_2d', 'resSpatial_4d', 'num_neurons_2d', 'num_neurons_4d', 'V1_shape', 'V4L4_shape', 'V4L23_shape', 'FEF_shape', 'FEFvm_shape',
            'viewfield', 'degPerCell_V4L4', 'degPerCell_V4L23', 'rfsize_V4p', 'RFsigma_V4L23-V4L4', 'RFsize_V4L23-V4L4', 'RFsize_V4L4-V4L23', 'RFsigma_V4L4-V4L23', 'RFsize_FEFv-FEFvm', 'RFsigma_FEFv-FEFvm',
            'tau_Xh_dep', 'tau']

    # list of parameters, that should be varied
    for k in skip:
        params.pop(k)
    # for k, v in params.items():
    #     print(k, v)
    print(len(params))


    ## create parameter set
    
    params_variations = {}
    for k, v in params.items():
        changes = np.random.uniform(-perc, perc, amount)
        v_variations = v *  (1 + changes / 100)
        params_variations[k] = v_variations
    # save
    df = pd.DataFrame(params_variations)
    df.to_json(filename, orient="records", lines=True)


def changeVariationRun(newNum):
    '''
    change number of current parameter variation globally
    i.e. change the parameter value in params_general.py

    params: newNum -- number of parameter set
    '''

    filename = "parameters/params_general.py"
    f = open(filename, 'r')
    content = f.readlines()
    f.close()

    # new content for parameter file = old content, despite var_num
    content_new = []
    for line in content:
        if line.startswith("params['var_num']"):
            content_new.append(f"params['var_num'] = {newNum}\n")
        else:
            content_new.append(line)

    # write new content
    f = open(filename, 'w')
    f.writelines(content_new)
    f.close()


######################
#### Main program ####
######################
if __name__ == "__main__":

    ## Create sets of parameter variations
    fn_paramVar = "parameters/param_variations.json"
    if not os.path.isfile(fn_paramVar):
        print('Create new sets of parameters')
        createParamVariation(fn_paramVar, perc=5, amount=1000)


    ## set number of variation set
    # number of parameter variation
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    else:
        num = 0
    changeVariationRun(num)
