# -*- coding: utf-8 -*-
"""
@author: juschu

generate input signals for different layers

 - proprioceptive (PC) eye position for PC signal
 - top-down attention for Xh
 - retinal input for V1
 - corollary discharge for FEFv
"""


##############################
#### imports and settings ####
##############################
# import standard libraries
import sys
import math
import numpy as np

# import files from this folder
from helper import idx_to_deg

# Loading parameters for dorsal and ventral stream
from parameters.params_model import params as params_model

NO_STIM = sys.float_info.max

###############################################
#### main functions for generating signals ####
###############################################
def generatePCsignal(SaccStart, SaccTarget, SaccOnset, SaccDur, duration):
    '''
    2D eye position (PC) signal (input for PC signal)
    signal is gauss blob centered at current eye position
    spatial layout organized in horizontal x vertical, measured in degree

    params: SaccStart, SaccTarget -- starting point and target of saccade
            SaccOnset, SaccDur    -- onset time and duration of saccade
            duration              -- duration of whole simulation

    return: pc_sig                -- array of size (duration, horizontal, vertical)
                                     containing for each timestep activity of input neurons
                                     organizes as horizontal x vertical map
    '''

    ##########################
    ## predefine parameters ##
    ##########################

    ## timing of update relative to saccade offset
    SaccOffset = SaccOnset + SaccDur
    # update, i.e. offset of old PC signal and onset of new PC signal
    t_pc_update = SaccOffset + params_model['pc_t_update']
    t_pc_update = min(max(0, t_pc_update), duration) # bounded between 0 and duration


    #####################
    ## generate signal ##
    #####################

    ## initialize signal
    pc_sig = np.zeros((duration,) + params_model['resSpatial_2d'])

    ## eye position before saccade (at fixation point)
    # activity = gauss blob centered at given point
    pc_pre = esig2d(params_model['resSpatial_2d'], SaccStart, params_model['pc_sigma'], params_model['pc_strength'])
    pc_sig[:t_pc_update] = pc_pre

    ## eye position after saccade (at saccade target)
    # activity = gauss blob centered at given point
    pc_post = esig2d(params_model['resSpatial_2d'], SaccTarget, params_model['pc_sigma'], params_model['pc_strength'])
    pc_sig[t_pc_update:] = pc_post

    ## add gaussian decay to presaccadic position
    t_pc_decay = np.arange(t_pc_update, duration)
    factor = np.exp(-(t_pc_decay-t_pc_update)*(t_pc_decay-t_pc_update)/(2.0*params_model['pc_sigma_decay']*params_model['pc_sigma_decay']))
    pc_decay = pc_pre * (factor * np.ones(params_model['resSpatial_2d'][::-1] + (1,))).T
    pc_sig[t_pc_update:] = np.maximum(pc_sig[t_pc_update:], pc_decay)


    ############
    ## return ##
    ############
    return pc_sig


def generateAttentionSignal(attention, duration):
    '''
    2D top-down attention signal (input for Xh)
    signal are gauss blobs centered at different attention positions
    spatial layout organized in horizontal x vertical, measured in degree

    params: attention -- list of dictionaries of attentions of shape:
                         [{'name', 'position', 'starttime'[, 'endtime']}, {...}, ...]
            duration  -- duration of whole simulation

    return: att_sig   -- array of size (duration, horizontal, vertical)
                         containing for each timestep activity of input neurons
                         organizes as horizontal x vertical map
    '''

    #####################
    ## generate signal ##
    #####################

    ## initialize signal
    att_sig = np.zeros((duration,) + params_model['resSpatial_2d'])

    ## calculation of attention signal for each attention position separately
    for att in attention:
        # onset of attention
        startAtt = att['starttime']
        # offset of attention
        if 'endtime' in att:
            endAtt = min(duration+1, att['endtime']) # defined offset
        else:
            endAtt = duration+1 # attention until end

        # activity = gauss blob centered at given point
        att_sig[startAtt:endAtt] += esig2d(params_model['resSpatial_2d'], att['position'],
                                           params_model['attention_sigma'], params_model['attention_strength'])


    ############
    ## return ##
    ############
    return att_sig


def generateRetinalSignal(stimuli, duration):
    '''
    4D retinal signal (input for V1)
    signal are gauss blobs centered at different stimulus position
    independent of channel and feature plane (equal along those dimensions)
    spatial layout organized in horizontal x vertical, measured in degree

    params: stimuli  -- list of dictionaries of stimuli of shape:
                        [{'position', 'starttime'[, 'endtime']}, {...}, ...]
            duration -- duration of whole simulation

    return: ret_sig  -- array of size (duration, horizontal, vertical, channels, features)
                        containing for each timestep activity of input neurons
                        organizes as horizontal x vertical x channels x features map
    '''

    #####################
    ## generate signal ##
    #####################

    ## initialize signal
    ret_sig = np.zeros((duration,) + params_model['V1_shape'])

    ## calculation of retinal signal for each stimulus separately
    for stim in stimuli:
        # onset of stimulus
        startStim = stim['starttime']
        # offset of stimulus
        if 'endtime' in stim:
            endStim = min(duration, stim['endtime']) # defined offset
        else:
            endStim = duration # stimulus until end

        # activity = gauss blob centered at given point
        xr = rsig2d(params_model['V1_shape'][:2], stim['position'], params_model['V1_sigma'], params_model['V1_strength'])
        # independent of channel and feature
        for c in range(params_model['V1_shape'][2]):
            for f in range(params_model['V1_shape'][3]):
                ret_sig[startStim:endStim, :, :, c, f] += xr


    ############
    ## return ##
    ############
    return ret_sig


def generateCDSignal(SaccStart, SaccTarget, SaccOnset, duration):
    '''
    2D corollary discharge signal (input for FEFv)
    signal is gauss blob centered at eye-centered saccade target
    spatial layout organized in horizontal x vertical, measured in degree

    params: SaccStart, SaccTarget -- starting point and target of saccade
            SaccOnset             -- onset time of saccade
            duration              -- duration of whole simulation

    return: cd_sig                -- array of size (duration, horizontal, vertical)
                                     containing for each timestep activity of input neurons
                                     organizes as horizontal x vertical map
    '''

    ##########################
    ## predefine parameters ##
    ##########################

    # peak time of CD signal
    t_peak = SaccOnset + params_model['cd_t_peak']
    # CD signal is retinotopic, thus calculate eye-centered saccade target
    ST = SaccTarget - SaccStart


    #####################
    ## generate signal ##
    #####################

    ## initialize signal
    # activity = gauss blob centered at given point
    cd = esig2d(params_model['FEF_shape'], ST, params_model['cd_sigma'], params_model['cd_strength'])

    ## time course of CD signal
    tc = np.zeros(duration)
    # rise of CD signal until peak time
    rise = np.arange(0, t_peak+1)
    tc[rise] = np.exp(-((rise-t_peak)*(rise-t_peak))/(2.0*params_model['cd_sigma_rise']*params_model['cd_sigma_rise']))
    # decay of CD signal from peak time until end of simulation
    decay = np.arange(t_peak, duration)
    tc[decay] = np.exp(-((decay-t_peak)*(decay-t_peak))/(2.0*params_model['cd_sigma_decay']*params_model['cd_sigma_decay']))
    # add time course to cd signal
    cd_sig = cd * (tc * np.ones(params_model['FEF_shape'][::-1] + (1,))).T

    ############
    ## return ##
    ############
    return cd_sig


#############################
#### auxiliary functions ####
#############################
def esig2d(geometry, pos, sigma, strength):
    '''
    Returns an internal eye position / attention signal given the position in degree
    spatial layout organized in horizontal x vertical

    params: geometry -- geometry of input (horizontal x vertical)
            pos      -- eye / attention position (in deg)
            sigma    -- width of signal (in deg)
            strength -- strength of signal

    return: xe       -- input signal
    '''

    idxs = np.arange(float(geometry[0]))
    deg_x = idx_to_deg(idxs, geometry[0], params_model['VF_Deg'][0])
    idys = np.arange(float(geometry[1]))
    deg_y = idx_to_deg(idys, geometry[1], params_model['VF_Deg'][1])
    summand1 = ((pos[0]-deg_x)*(pos[0]-deg_x)).reshape(geometry[0], 1)
    summand2 = ((pos[1]-deg_y)*(pos[1]-deg_y)).reshape(1, geometry[1])
    xe = strength * np.exp(-(summand1 + summand2)/(2.0*sigma*sigma))

    return xe

def rsig2d(geometry, pos, sigma, strength):
    '''
    Returns an internal stimulus signal given the position of a stimulus in degree
    spatial layout organized in horizontal x vertical

    params: geometry -- geometry of input (horizontal x vertical)
            pos      -- stimulus position (in deg)
            sigma    -- width of signal (in deg)
            strength -- strength of signal

    return: xr       -- input signal
    '''

    idxs = np.arange(float(geometry[0]))
    deg_x = idx_to_deg(idxs, geometry[0], params_model['VF_Deg'][0])
    idys = np.arange(float(geometry[1]))
    deg_y = idx_to_deg(idys, geometry[1], params_model['VF_Deg'][1])
    summand1 = ((pos[0]-deg_x)*(pos[0]-deg_x)/(2.0*sigma*sigma)).reshape(geometry[0], 1)
    # double sigma for vertical dimension to get bars instead of circles
    summand2 = ((pos[1]-deg_y)*(pos[1]-deg_y)/(2.0*sigma*2*sigma*2)).reshape(1, geometry[1])
    xr = strength * np.exp(-(summand1 + summand2))

    return xr
