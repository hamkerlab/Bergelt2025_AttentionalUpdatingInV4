# -*- coding: utf-8 -*-
"""
@author: juschu

auxiliary functions

 - readout eye and stimulus position, and output data
 - map neurons to corresponding positions in visual space and vice versa
"""


##############################
#### imports and settings ####
##############################
import numpy as np


######################
#### readout data ####
######################
def getEyepos(inputfile, duration):
    '''
    get eye position over time

    params: inputfile -- txt file of saved rates
            duration  -- simulation time

    return: ep        -- numpy array with eye positions of shape duration x 2
    '''

    # read file
    f = open(inputfile)
    content = f.readlines()
    f.close()

    # init eye position
    ep = np.zeros((duration, 2))

    # readout data from file and set to dictionary
    numLines = len(content)
    for i in range(numLines):
        splittedLine = content[i].split()
        if ',' in content[i]:
            # eye position was saved as list
            x = splittedLine[1].replace('[', '').replace(',', '')
            y = splittedLine[2].replace(']', '')
        else:
            # eye position was saved as numpy.array
            x = splittedLine[-2].replace('[', '')
            y = splittedLine[-1].replace(']', '')
        ep[i][0] = float(x)
        ep[i][1] = float(y)

    # return
    return ep

def getBarpos(inputfile, duration):
    '''
    get bar position(s) over time

    params: inputfile -- txt file of saved rates
            duration  -- simulation time

    return: bars      -- dictionary of numpy arrays with bar positions of shape duration x 2
    '''

    # read file
    f = open(inputfile)
    content = f.readlines()
    f.close()

    # init dictionary
    bars = {}

    # readout data from file and set to dictionary
    for line in content:
        # skip empty lines
        if len(line) > 1:
            if line.startswith('bar'):
                # new bar
                barname = line.split(': ')[1].replace('\n', '')
                # init bar position
                bars[barname] = np.zeros((duration, 2))
            else:
                # remove distracting spaces
                splittedLine = line.replace('[ ', '[').replace(' ]', ']').split()
                # bar position looks like:
                # t: [x y]
                t = splittedLine[0].replace(':', '')
                x = splittedLine[1].replace('[', '')
                y = splittedLine[2].replace(']', '')
                bars[barname][int(t)][0] = float(x)
                bars[barname][int(t)][1] = float(y)

    # return
    return bars

def getBarposAndOnset(dirname, duration, alignToSacOnset=False):
    '''
    get all presented bars with position and time of onset for given simulation

    params: dirname         -- directory of saved results
            duration        -- duration of simulation
            alignToSacOnset -- align time of stimulus onset with respect to saccade onset

    return: barposOnsets    -- directory with key=bar position and value=time of onset
    '''

    # bar positions over time
    bars = getBarpos(dirname + '0_barpos.txt', duration)

    if alignToSacOnset:
        outputData = getOutputData(dirname + 'output.txt')
        sacOnset = outputData['sac_start'][0]
    else:
        sacOnset = 0

    # dictionary of bar positions and their onsets
    barposOnsets = {}

    for b, posOverTime in bars.items():
        # get time of onset
        t = 0
        while (posOverTime[t] == np.array([np.inf, np.inf])).all():
            t += 1
        barposOnsets[b] = {'onset': t-sacOnset, 'pos': posOverTime[t]}

    return barposOnsets

def getOutputData(inputfile):
    '''
    get simulation data:
    start and end of simulation as well as start and end of saccade

    params: inputfile  -- txt file of saved rates

    return: outputData -- dictionary with data
    '''

    # read file
    f = open(inputfile)
    content = f.readlines()
    f.close()

    # init dictionary
    outputData = {}

    # readout data from file and set to dictionary
    for line in content:
        [name, value] = line.split(': ')
        if name.startswith('t_'):
            # timesteps (start, end, saccade onset)
            # value = t
            outputData[name] = int(value)
        else:
            # spatial points (fixation point, saccade target, attention position)
            # value = [x y]\n or [ x y]\n
            outputData[name] = np.fromstring(value[1:-2], dtype=float, sep=' ')

    # return
    return outputData


#################
#### mapping ####
#################
def idx_to_deg(idxs, size, vf):
    '''
    Maps neurons to corresponding positions in visual space (horizontal or vertical)
    spatial layout organized in horizontal x vertical

    params: idxs -- numpy array of indices of neurons
            size -- geometry of dimension
            vf   -- visual field in this dimension

    return: numpy array of positions (in degree)
    '''

    return vf*(idxs/(size-1)-0.5)

def deg_to_idx(degs, size, vf):
    '''
    Maps positions in visual space (horizontal or vertical) to corresponding neurons
    spatial layout organized in horizontal x vertical

    params: degs -- numpy array of positions (in degree)
            size -- geometry of dimension
            vf   -- visual field in this dimension

    return: numpy array of indices of neurons
    '''

    return np.round((degs/vf+0.5)*(size-1)).astype(int)