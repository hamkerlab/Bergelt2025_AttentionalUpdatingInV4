# -*- coding: utf-8 -*-
"""
@author: juschu

generating set of experimental setups

Following experiment of Marino and Mazer (2018):
 - 1 fixation point (FP), 1 saccade target (ST)
 - 1 attention position (AP) out of 4 possible positions
 - multiple random bars as stimuli at different positions above and below FP/ST

 - store generated setups
"""


##############################
#### imports and settings ####
##############################
# import standard libraries
import os
os.system('clear')

import multiprocessing as mp
import numpy as np

# import files from this folder
from saving import saveBarPos, saveOutput

# import files from subfolders
# parameters
from parameters.params_general import params


####################
#### simulation ####
####################
def run(number):
    '''
    main function to generate one setup

    params: number -- current number of setup
    '''

    # create directory for saving
    saveDir = params['SetupDir'] + str(number) + '/'
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    print("save everything at %s" % saveDir)

    ## setup
    np.random.seed()

    # simulation time
    duration = params['tEnd'] + 1

    # fixation point (=center of visual field) in degree and horizontal x vertical
    FP = np.array([0., 0.])
    # saccade target in degree and horizontal x vertical
    ST = FP + np.array(params['saccade'])
    

    # attention position in degree and horizontal x vertical
    # random out of 4 possible positions: above/below FP/ST
    # [0, 0] --> below ST, [0, 1] --> below FP, [1, 0] --> above ST, [1, 1] --> above FP
    above_FP = np.random.randint(0, 2, 2)
    dist_v = -params['range_v'] if above_FP[0] else params['range_v']
    AP = (FP if above_FP[1] else ST) + np.array([0, dist_v])


    # init barpos_over_time
    # bar positions: random position horizontally random every 0.5 degree and vertically above/below fixation point
    # list of bar positions in degree and horizontal x vertical
    barpos = {}
    stim = []
    barpos_over_time = {} # bar positions over time in degree and horizontal x vertical
    # extend matrix and cut afterwards to avoid border effects
    ext = 5
    # possible horizontal positions
    pos_h = np.arange(-params['VF_Deg'][0]/2-ext*params['range_h'], params['VF_Deg'][0]/2+params['range_h']+ext*params['range_h'], params['range_h'])
    # possible onsets
    onsets = np.arange(params['range_t'][0]-ext*params['range_t'][2], params['range_t'][1]+1+ext*params['range_t'][2], params['range_t'][2])
    # shape of matrix
    bar_onset_shape = (len(pos_h), len(onsets))
    # block surroundings
    # 2deg to left and right --> 4 entries to left and right
    # 100 ms before and after --> 100 entries to top and bottom
    block = [int(params['range_sup']/params['range_h']),        # horizontal position
             int(params['range_t'][3]/params['range_t'][2])]    # time
    counter = 0
    for i in range(2):
        bars_v = FP[1]-params['range_v'] if i else FP[1]+params['range_v']
        ## init
        bar_onset = np.zeros(bar_onset_shape)
        # print(bar_onset)
        # all possible entries
        pos = np.argwhere(np.ravel(bar_onset)==0)[:, 0]
        while len(pos) > 0:
            (idx_bars_h, idx_start) = np.unravel_index(np.random.choice(pos), bar_onset_shape)
            # block surroundings in matrix
            bar_onset[max(0, idx_bars_h-block[0]):min(bar_onset_shape[0]-1, idx_bars_h+block[0])+1,
                        max(0, idx_start-block[1]):min(bar_onset_shape[1]-1, idx_start+block[1])+1] = -1
            # add bar to matrix
            bar_onset[idx_bars_h, idx_start] = 1
            # update all possible entries
            pos = np.argwhere(np.ravel(bar_onset)==0)[:, 0]
            # complete stimulus with position, on- and offset (random onset, random flash duration)
            bars_h = pos_h[idx_bars_h]
            start = onsets[idx_start]
            # only add bar, if in correct range
            if (-params['VF_Deg'][0]/2 <= bars_h <= params['VF_Deg'][0]/2) and (params['range_t'][0] <= start <= params['range_t'][1]):
                barpos[str(counter)] = [bars_h, bars_v]
                dur = np.random.randint(params['flash_duration'][0], params['flash_duration'][1]+1)
                stim.append({'position':barpos[str(counter)], 'starttime':start, 'endtime':start+dur})            
                # init with current positions
                barpos_over_time[str(counter)] = np.ones((params['tEnd'] + 1, 1)) * [np.inf, np.inf]
                barpos_over_time[str(counter)][start:start+dur+1] = barpos[str(counter)]
                counter +=1


    output = {'t_start': 0, 'FP': FP, 'ST': ST, 'AP': AP, 't_sacStart': params['saccOnset'], 't_end': params['tEnd']}
    
    
    #####################
    ## save everything ##
    #####################
    print("\n\n")
    # save bar positions over time
    saveBarPos(barpos_over_time, duration, saveDir + '0_')

    # save output
    print(output)
    saveOutput(output, saveDir)


######################
#### Main program ####
######################
if __name__ == "__main__":

    # id of trials to simulate
    trials = range(params['numOfTrials'])

    ## simulate in parallel
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(trials)))
    # Step 2: `pool.apply` the `add()`
    results = pool.map(run, trials)
    # Step 3: Don't forget to close
    pool.close()


    ############
    ## finish ##
    ############
