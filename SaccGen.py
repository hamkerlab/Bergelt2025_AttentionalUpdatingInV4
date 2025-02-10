# -*- coding: utf-8 -*-
"""
@author: juschu

saccade generator

 - saccades are calculated according to VanWetter&VanOpstal-Model (VanWetter & VanOpstal, 2008)
 - calculate coordinates of eye position during saccades for each timestep
"""


##############################
#### imports and settings ####
##############################
import math
import numpy as np

def CoG(A):
    '''
    calculate Center of Gravity (CoG) to indicate the location of an eye movement

    params: A       -- matrix which its center of gravity is desired

    return: Centers -- numpy array which contains the coordinates of Center of Gravity in pixel.
    '''
    rc, cc = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    Mt = sum(sum(A))
    Centers = np.zeros(2)
    Centers[0] = sum(sum(A * rc)) / Mt
    Centers[1] = sum(sum(A * cc)) / Mt

    return Centers


###########################
#### saccade generator ####
###########################
def SaccadeGenerator(eye0, eye1, MinSpeed=22, MinDistance=0.05, saccadeEndBySpeed=False):
    '''
    generate saccade for coordinates given in degree

    params: eye0, eye1             -- coordinates of start and end point of saccade (in degree)
            MinSpeed, MinDistance  -- minimum speed (in degree/s) and distance (in degree) to
                                      saccade target as termination condition
            saccadeEndBySpeed      -- flag, which termination condition should be used:
                                      False --> position, True --> speed

    return: eyePos                 -- numpy array with eye position during saccade
                                      shape: saccade duration x 2 (current eye position in degree at timestep)
    '''

    ##########################
    ## predefine parameters ##
    ##########################
    ## Eye position over time during saccade
    eyePos = np.zeros((1, 2))
    ## saccade amplitude in degree
    SaccAmp = np.linalg.norm(eye1 - eye0)
    ## parameters for VanWetter&VanOpstal-model
    m0 = 7.0
    vpk = 0.525 # original value used in VanWetter&VanOpstal-Paper
    # vpk = (750 * (1 - math.exp(-SaccAmp / 16)) + 50) / 1000. # value calculated based on experimental data
    print(f"Saccade Amplitude = {SaccAmp:.2f} deg")
    ## saccade duration
    saccDur = 0
    ## flag, if saccade has ended
    sac_has_ended = False


    ###########################
    ## generate eye movement ##
    ###########################
    if SaccAmp == 0:
        # fixation point and saccade target are equal
        sac_has_ended = True
    else:
        direction = 1 / np.linalg.norm(eye1 - eye0) * (eye1 - eye0) # direction of saccade
        A = 1.0 / (1.0 - math.exp(-SaccAmp / m0))

    ## initialization with t=0 and eyepos[t]=saccade start
    currentTimestep = 0
    currentEyepos = eye0
    eyePos[currentTimestep] = eye0

    ## loop until termination condition is reached
    while not sac_has_ended:

        currentTimestep += 1
        previousEyepos = currentEyepos
        currentEyepos = eye0 + direction*\
                        (m0 * math.log((A * math.exp(vpk * currentTimestep / m0)) /\
                                       (1.0 + A * math.exp((vpk * currentTimestep - SaccAmp)/m0))))

        # detect saccade end
        if saccadeEndBySpeed:
            # saccade end is calculated by eye speed
            # one time step is one ms, so now "current_eye_speed" is in deg/sec
            current_eye_speed = np.linalg.norm(currentEyepos - previousEyepos)*1000
            if current_eye_speed < MinSpeed:
                sac_has_ended = True
        else:
            # saccade end is calculated by position
            if np.linalg.norm(currentEyepos - eye1) < MinDistance:
                sac_has_ended = True

        # finish?
        if sac_has_ended:
            # saccade has ended
            eyePos = np.insert(eyePos, currentTimestep, [eye1], axis=0)
            saccDur = currentTimestep
        else:
            # saccade still ongoing
            eyePos = np.insert(eyePos, currentTimestep, [currentEyepos], axis=0)


    ############
    ## return ##
    ############
    print(f"Saccade ended after {saccDur} ms and goes from {np.array2string(eyePos[0], precision=2, floatmode='fixed')} deg " +
          f"to {np.array2string(eyePos[currentTimestep], precision=2, floatmode='fixed')} deg")

    return eyePos
