# -*- coding: utf-8 -*-
"""
@author: juschu

Parameters needed to define neuro-computational model

 - defining model including layers, neurons and connections
 - creating input signals
 - saved in dictionary
"""
import numpy as np

from parameters.params_general import params

# params = {}

###################################
#### neuro-computational model ####
###################################
## Neurons ##
## V4 Parameters
params['sigma_V4L4'] = 0.4
params['g_V4L4'] = 1.066
params['v_V1-V4L4'] = 1.0
params['p_V1-V4L4'] = 1
params['delay_V1_V4L4'] = 5
params['v_V4L23-V4L4'] = 0.5
params['p_V4L23-V4L4'] = 1
params['v_FEFvm-V4L4'] = 4.0
params['v_LIP-V4L4'] = 3.0
params['sigma_V4L23'] = 1.0
params['g_V4L23'] = 1.625
params['v_V4L4-V4L23'] = 1.0
params['p_V4L4-V4L23'] = 0.25
params['p_V4L4-V4L23_ws'] = 4

## FEF Parameters
params['v_FEFv-FEFvm'] = 0.2
params['v_FEFv-FEFvm_E'] = 0.6
params['v_FEFv-FEFvm_S'] = 0.6
params['v_FEFv-FEFvm_scale'] = 0.35
params['v_FEFv-FEFvm_factor'] = 0.93
params['v_FEFvm-FEFm_E'] = 1.3
params['v_FEFvm-FEFm_S'] = 0.3
params['v_FEFfix-FEFm'] = 0.3

## LIP parameters
# saturation
params['A_LIPcd'] = 0.5
params['A_LIPpc'] = 1.0
# inhibition
params['D_LIPcd'] = 0.1
params['D_LIPpc'] = 0.1
params['D_Xh'] = 0.6
# divisive normalization
params['sigma_LIPcd'] = 0.5
params['sigma_LIPpc'] = 1.0
params['sigma_Xh'] = 0.5
# synaptic suppression for Xh
params['d_dep_Xh'] = 2.2
params['tau_dep_Xh'] = 10000

### Universal Parameters
params['tau'] = 10


## Populations ##
## Numbers of Neurons in the different areas
params['resVisual'] = tuple(params['VF_Deg']*2 + 1)
params['resSpatial_2d'] = tuple(params['VF_Deg']//2 + 1)
params['resSpatial_4d'] = params['resSpatial_2d']*2
params['num_neurons_2d'] = np.prod(params['resSpatial_2d'])
params['num_neurons_4d'] = np.prod(params['resSpatial_4d'])

params['V1_shape'] = params['resVisual'] + (1, 1)
params['V4L4_shape'] = params['resVisual'] + (1,)
params['V4L23_shape'] = ((params['V4L4_shape'][0]-1)//2+1, (params['V4L4_shape'][1]-1)//2+1, params['V4L4_shape'][2])
params['FEF_shape'] = params['resVisual']
params['FEFvm_shape'] = params['FEF_shape'] + (6,)

## Projections ##
## Convolutions
params['viewfield'] = np.array([10.3, 7.8])
params['degPerCell_V4L4'] = params['viewfield'] / params['resVisual']
params['degPerCell_V4L23'] = params['viewfield'] / params['V4L23_shape'][:2]
params['rfsize_V4p'] = [5, 5] * (params['degPerCell_V4L4']/params['degPerCell_V4L23'])
params['sigma_RF_A_Feat'] = params['rfsize_V4p'] / 3
params['RFsize23_4'] = [3, 3]
params['FBA_delay'] = 2
params['RFsize4_23'] = [5, 5]
params['RFsigma4_23'] = [5. / 3, 5. / 3]

params['RFsizev_vm'] = [41, 31]
params['RFsigmav_vm'] = [4, 3]

## own-defined Gaussian Connection Pattern
# highest value of Gaussian
params['K_FEFm-CD'] = 1.0
params['K_PC-CD'] = 10.0
params['K_FEFvm-LIPcd'] = 0.015
params['K_FEFvm-LIPpc'] = 0.1
params['K_V4L4-LIPcd'] = 0.5
params['K_V4L4-LIPpc'] = 2.0
params['K_CD-LIPcd'] = 4.0
params['K_PC-LIPpc'] = 1.0
params['K_Xh-LIPcd'] = 7.0
params['K_Xh-LIPpc'] = 2.0
params['K_LIPcd-Xh'] = 0.03
params['K_LIPpc-Xh'] = 0.05
params['K_LIPcd-V4L4'] = 0.5
params['K_LIPpc-V4L4'] = 0.2

params['w_CD-inh'] = 0.05
params['w_LIPcd-inh'] = 0.05
params['w_LIPpc-exc'] = 0.6
params['w_LIPpc-inh'] = 0.2
params['w_Xh-exc'] = 1.0
params['w_Xh-inh'] = 1.0

# width of Gaussian
params['sigma_FEFm-CD'] = 1.0
params['sigma_PC-CD'] = 0.5
params['sigma_FEFvm-LIPcd'] = 1.0
params['sigma_FEFvm-LIPpc'] = 1.0
params['sigma_V4L4-LIPcd'] = 0.5
params['sigma_V4L4-LIPpc'] = 0.5
params['sigma_CD-LIPcd'] = 0.25
params['sigma_PC-LIPpc'] = 1.0
params['sigma_Xh-LIPcd'] = 1.0
params['sigma_Xh-LIPpc'] = 1.0
params['sigma_LIPcd-Xh'] = 1.0
params['sigma_LIPpc-Xh'] = 1.0
params['sigma_LIPcd-V4L4'] = 0.5
params['sigma_LIPpc-V4L4'] = 0.5

params['sigma_LIPpc-exc'] = 0.25
params['sigma_Xh-exc'] = 0.25


#######################
#### input signals ####
#######################
## corollary discharge (CD)
# strength
params['cd_strength'] = 0.375
# width
params['cd_sigma'] = 1.0
# rise and decay
params['cd_rise'] = 60
params['cd_decay'] = 40
params['cd_peak'] = 29 # correcting for real saccadeOnset

## proprioceptive eye position (PC)
# strength
params['pc_strength'] = 0.3
# width
params['pc_sigma'] = 1.0
# update
params['pc_update'] = 60
params['pc_off_decay'] = 35

## visual input
# strength
params['V1_strength'] = 0.2
# width
params['V1_sigma'] = 1.0

## top-down attention
# strength
params['attention_strength'] = 0.3
# width
params['attention_sigma'] = 1.0

## saccade control
params['SaccadeThreshold'] = 0.9
