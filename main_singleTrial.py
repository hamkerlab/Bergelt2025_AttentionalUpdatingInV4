# -*- coding: utf-8 -*-
"""
@author: juschu

Main script to run model and simulate experiment

 - Run pre-defined experimental setups in parallel
 - Record activity of V4 Layer 4 as well as its inputs over time and save as hdf5
"""


##############################
#### imports and settings ####
##############################
# import standard libraries
import sys
import timeit
First_Start = timeit.default_timer()
import os
os.system('clear')
import numpy as np

import ANNarchy as ANN

# import files from this folder
from SaccGen import SaccadeGenerator, CoG
from generateSignals import generatePCsignal, generateAttentionSignal, generateRetinalSignal, generateCDSignal
from saving import saveEyePos, saveBarPos, saveOutput, saveRates, save_dict_to_hdf5
from helper import getOutputData, getBarpos, idx_to_deg

# import files from subfolders
# network
import network.model as net
# parameters
from parameters.params_general import params
from parameters.params_model import params as params_model
params.update(params_model)


####################
#### simulation ####
####################
def run(number):
    '''
    main function to run simulation

    params: number -- current number of trial
    '''

    Sim_Start = timeit.default_timer()

    # create directory for saving
    saveDir = f"{params['ResultDir']}/trials/{number}/"
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)
    saveDirRates = f"{saveDir}/Rates/"
    if not os.path.isdir(saveDirRates):
        os.makedirs(saveDirRates)

    print(f"save everything at {saveDir}")

    # load previous setup
    loadDir = f"{params['SetupDir']}/{number}/"
    loadedSetup = getOutputData(f"{loadDir}/output.txt")

    ## setup
    # simulation time
    duration = loadedSetup['t_end'] + 1

    # fixation point (=center of visual field) in degree and horizontal x vertical
    FP = loadedSetup['FP']
    # saccade target in degree and horizontal x vertical
    ST = loadedSetup['ST']

    # attention position in degree and horizontal x vertical
    AP = loadedSetup['AP']

    ## initialization of signals
    # eye position over time in degree and horizontal x vertical
    # init with current fixation point
    eyepos_over_time = np.ones((duration, 2)) * FP

    # init pc signal(s) at center of visual field and with no saccade
    # using generatePCsignal(SaccStart, SaccTarget, SaccOnset, SaccDur, duration)
    # NOTE: coordinates needed in degree and horizontal x vertical and relative to head direction
    print(f"init pc signal at {np.array2string(FP, precision=2, floatmode='fixed')} [deg]")
    pc_sig = generatePCsignal(FP, FP, duration, 0, duration)
    inputs = {'PC_pre': pc_sig}

    # init FEF signal representing saccade target
    # using generateFEFSignal(SaccTarget, duration)
    # NOTE: position needed in degree and horizontal x vertical
    FEF_sig = generateCDSignal(FP, ST, params['saccOnset'], duration)
    inputs['FEF'] = FEF_sig

    # init attention signal at top-down attention position
    # using generateAttentionSignal(Attention, duration) with
    # Attention = [{'name', 'position', 'starttime'[, 'endtime']},{...},...]
    # NOTE: position needed in degree and horizontal x vertical
    print(f"top-down attention at {np.array2string(AP, precision=2, floatmode='fixed')} [deg]")
    Attention = [{'name': 'top-down attention before saccade', 'position': AP,
                  'starttime': 0, 'endtime': params['att_turnoff']}]
    att_sig = generateAttentionSignal(Attention, duration)
    inputs['attention_pre'] = att_sig

    # init retinal signal representing bars
    # using generateRetinalSignal(stim, duration) with
    # stim = [{'position', 'starttime'[, 'endtime']},{...},...]
    # NOTE: position needed in degree and horizontal x vertical
    # init simultaneously barpos_over_time
    barpos_over_time = getBarpos(f"{loadDir}/0_barpos.txt", duration)
    # list of bar positions in degree and horizontal x vertical
    barpos = {}
    # complete stimulus with position, on- and offset (random onset, fixed flash duration)
    stim = []
    for b, pos_over_time in barpos_over_time.items():
        [start, end] = np.where(pos_over_time != np.inf)[0][[0, -1]]
        pos = list(pos_over_time[start])
        barpos[b] = pos
        stim.append({'position': pos, 'starttime': start, 'endtime': end})

    ret_sig = generateRetinalSignal(stim, duration)
    inputs['retinal_pre'] = ret_sig

    # save bar positions over time
    saveBarPos(barpos_over_time, duration, f"{saveDirRates}/0_")

    #############################
    ## Starting the simulation ##
    #############################
    output = {'t_start': 0, 'FP': FP, 'ST': ST, 'AP': AP}

    ## set signals
    # set baseline of XePC to pc signal
    net.PC_Input.rates = pc_sig
    # set baseline of Xh to attention signal
    net.Xh_Input.rates = att_sig
    # set baseline of V1 to retinal signal
    net.V1_Input.rates = ret_sig
    # set rate of FEFv to FEF signal
    net.FEFv_Pop.rates = FEF_sig

    ## Simulate until saccade
    print('\nSimulate until saccade is triggered.')
    Step = int(ANN.simulate_until(max_duration=duration, population=net.FEFm_Pop))-1
    if Step < duration-1:

        print(f"After step {Step} max(FEFm) = {np.max(net.FEFm_Pop.r)} at {np.unravel_index(np.argmax(net.FEFm_Pop.r), params['FEF_shape'])}")

        # find saccade target
        # saccade target is center of gravity of FEFm rates
        CurSac_neurons = CoG(net.FEFm_Pop.r)
        print(f"Saccade Target encoded by neuron {np.array2string(CurSac_neurons, precision=2, floatmode='fixed')}")
        # saccade target in degree and horizontal x vertical
        CurSac = [idx_to_deg(CurSac_neurons[0], params['FEF_shape'][0], params['VF_Deg'][0]),
                  idx_to_deg(CurSac_neurons[1], params['FEF_shape'][1], params['VF_Deg'][1])]

        # Saccade generator
        EyePos = SaccadeGenerator(np.asarray([0, 0]), np.asarray(CurSac))
        sacDur = len(EyePos) - 1

        eyepos_over_time[Step:Step+sacDur+1] = EyePos
        # now eyes are at a new fixation point
        eyepos_over_time[Step+sacDur+1:] = EyePos[sacDur]

        output['t_sacStart'] = Step
        output['t_sacEnd'] = Step+sacDur

        # update pc signal
        # using generatePCsignal(SaccStart, SaccTarget, SaccOnset, SaccDur, duration)
        # NOTE: coordinates needed in degree and horizontal x vertical and relative to head direction
        # saccade target in degree and horizontal x vertical
        ST = EyePos[sacDur]
        print(f"update pc signal from {np.array2string(FP, precision=2, floatmode='fixed')} deg " + 
              f"to {np.array2string(ST, precision=2, floatmode='fixed')} deg")
        pc_sig = generatePCsignal(FP, ST, Step, sacDur, duration)
        inputs['PC_post'] = pc_sig
        net.PC_Input.rates = pc_sig

        # update retinal signal: VF is moving --> new stimulus positions
        # using generateRetinalSignal(stim, duration) with
        # stim = [{'position', 'starttime'[, 'endtime']},{...},...]
        # NOTE: position needed in degree and horizontal x vertical
        # update simultaneously barpos_over_time
        stim2 = []
        for s in stim:
            b = list(barpos.keys())[list(barpos.values()).index(s['position'])]
            # during saccade
            for (t, eyepos) in enumerate(EyePos):
                if s['starttime'] <= Step+t <= s['endtime']:
                    pos = s['position']-(eyepos-FP)
                    stim2.append({'position': pos, 'starttime': Step+t, 'endtime': Step+t+1})
                    barpos_over_time[b][Step+t] = np.array(pos)
            # now eyes are at a new fixation point
            sacEnd = sacDur+Step
            pos = s['position']-(EyePos[sacDur]-FP)
            if s['starttime'] <= sacEnd <= s['endtime']:
                stim2.append({'position': pos, 'starttime': sacEnd, 'endtime': s['endtime']})
                barpos_over_time[b][sacEnd:s['endtime']+1] = np.array(pos)
            if s['starttime'] > sacEnd:
                stim2.append({'position': pos, 'starttime': s['starttime'], 'endtime': s['endtime']})
                barpos_over_time[b][s['starttime']:s['endtime']+1] = np.array(pos)

        ret_sig = generateRetinalSignal(stim2, duration)
        inputs['retinal_post'] = ret_sig
        net.V1_Input.rates = ret_sig

        # FEFfix controls the starting of the next object localization task. Until
        # it, an execution of a novel saccade is not allowed which ensures that the current
        # saccade is completely executed. New saccade execution is suppressed for the whole eye movement.
        fix = np.zeros((duration, 1))
        fix[Step:Step+sacDur+2] = 1
        net.FEFfix.rates = fix

        ## Simulate remaining time
        print("\nSimulate until end.")
        ANN.simulate(duration-1-Step)


    ## End for
    output['t_end'] = duration-1
    print(output)

    Sim_Stop = timeit.default_timer()
    print(f"One simulation needs {Sim_Stop - Sim_Start} seconds.")


    #####################
    ## save everything ##
    #####################
    print("\n")
    # save eye position over time
    saveEyePos(eyepos_over_time, duration, saveDirRates)

    # save bar positions over time
    saveBarPos(barpos_over_time, duration, saveDirRates)

    # save output
    saveOutput(output, saveDirRates)

    # save inputs
    print("Saving Inputs")
    save_dict_to_hdf5(inputs, f"{saveDirRates}/dict_inputs.hdf5")

    # save recorded rates
    print("Saving Rates")
    saveRates(monitors, f"{saveDirRates}/dict_rates_full.hdf5")


######################
#### Main program ####
######################
if __name__ == "__main__":

    Stop01 = timeit.default_timer()
    print(f"Precompile process has been finished in {Stop01 - First_Start} seconds.")


    #####################
    ## compile network ##
    #####################
    ANN.compile('annarchy_single')
    Stop02 = timeit.default_timer()
    print(f"Compile process has been finished in {Stop02 - Stop01} seconds.")
    # ANN.report('description/model.tex')
    # input('wrote tex file')


    ######################
    ## save projections ##
    ######################
    if not os.path.isdir(params['ConnDir']):
        # if folder does not already exist, save connections for further usage
        print("save connections")
        if not os.path.exists(params['ConnDir']):
            os.makedirs(params['ConnDir'])
        for proj in ANN.projections():
            projName = proj.pre.name + '-' + proj.post.name + '-' + proj.target
            print(f" - save {projName}")
            proj.save_connectivity(filename=params['ConnDir']+projName+'.data')


    ##############################
    ## Preparing the simulation ##
    ##############################
    ## Record some rates using Monitor function of ANNarchy
    monitors = {'V1': ANN.Monitor(net.V1_Pop, 'r'),
                'V4L23': ANN.Monitor(net.V4L23_Pop, ['r', 'E', 'sum(exc)']),
                'V4L4': ANN.Monitor(net.V4L4_Pop, ['sum(A_LIPpc)', 'sum(A_LIPcd)', 'ALIP', 'ASP', 'AFEAT', 'r', 'E']),
                'FEFvm': ANN.Monitor(net.FEFvm_Pop, ['r', 'ESv', 'sum(E_v)', 'sum(S_v)']),
                'FEFm': ANN.Monitor(net.FEFm_Pop, ['r', 'sum(vm)', 'sum(fix)']),
                'Xh': ANN.Monitor(net.Xh_Pop, 'r'),
                'PC signal': ANN.Monitor(net.PC_Pop, 'r'),
                'CD signal': ANN.Monitor(net.CD_Pop, ['r', 'sum(FF_FEF)', 'sum(FF_PC)', 'inh']),
                'LIP PC': ANN.Monitor(net.LIPpc_Pop, ['r', 'sum(FF_FEF)', 'sum(FF)', 'sum(FF_PC)', 'sum(FB)', 'sum(exc)', 'inh']),
                'LIP CD': ANN.Monitor(net.LIPcd_Pop, ['r', 'sum(FF_FEF)', 'sum(FF)', 'sum(FF_CD)', 'sum(FB)', 'inh'])
                }


    #########################
    ## Run the simulations ##
    #########################
    # create saving directory
    if not os.path.isdir(f"{params['ResultDir']}/trials/"):
        os.makedirs(f"{params['ResultDir']}/trials/")

    # id of trial to simulate
    if len(sys.argv) > 1:
        trial = sys.argv[1]
    else:
        trial = 1

    ## simulate one trial
    run(trial)


    ############
    ## finish ##
    ############
    Last_Stop = timeit.default_timer()
    print('\n----------------------------------------------------------------------------------------------------')
    print(f"The whole simulation has been finished in {Last_Stop - First_Start} seconds.")
