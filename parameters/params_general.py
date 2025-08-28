#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juschu

Parameters needed to define general things
 - setup and layout
 - saving
 - saved in dictionary
"""


##############################
#### imports and settings ####
##############################
import numpy as np


####################
#### parameters ####
####################
params = {}

# how many things should be run in parallel (for simulation and evaluation)
params['runInParallel'] = 50

################
#### saving ####
################
params['ResultDir'] = 'Results/attType/'
params['SetupDir'] = 'Results/setup/'
params['ConnDir'] = 'network/connections/'


######################
#### visual field ####
######################
params['VF_Deg'] = np.array([40, 30])


###############
#### setup ####
###############
params['numOfTrials'] = 4000       # total number of trials
params['tEnd'] = 700               # duration of each simulation

# stimuli (=bars) above/below horizontal center line over whole horizontal space with random onset aligned to saccade onset and random duration
params['range_v'] = 7                               # distance of "above" and "below" in degree
params['range_sup'] = 2                             # minimum distance between two simultaneously shown bars in degree
params['range_h'] = 0.5                             # minimum horizontal distance between two bars in degree
params['flash_duration'] = [5, 20]                  # flash duration for bars in ms
params['saccOnset'] = 350                           # saccade onset
params['range_t'] = [params['saccOnset']-300,       # start of stimulus presentation
                     params['saccOnset']+300,       # end of stimulus presentation
                     params['flash_duration'][1],   # minimum time between two bars
                     100]                           # minimum time between two bars at the same position

# saccade 5 degree to the right
params['saccade'] = [5, 0]

# top-down attention (tonic=active for entire simulation, phasic=deactivated after 250 ms)
params['attType'] = 'tonicAttention' #'tonicAttention' 'phasicAttention'
if 'tonic' in params['attType']:
  # attention for entire simulation
  params['att_turnoff'] = params['tEnd']+1
else:
  # turn off attention after simulation start
  params['att_turnoff'] = 250
params['ResultDir'] = params['ResultDir'].replace('attType', params['attType'])
