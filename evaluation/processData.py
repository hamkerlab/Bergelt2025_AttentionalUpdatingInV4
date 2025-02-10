"""
@author: juschu

Main script for processing data from simulations
needed for plotting
"""

##############################
#### imports and settings ####
##############################
# import standard libraries
import sys
import os
import multiprocessing as mp
import numpy as np

# import files from parent folder
sys.path.append('../')
from helper import deg_to_idx, getOutputData, getBarposAndOnset, getEyepos
from saving import load_dict_from_hdf5, save_dict_to_hdf5
# parameters
from parameters.params_general import params
from parameters.params_model import params as params_model
params.update(params_model)

def getData():
    '''
    get neurons of interest as well as attention positions
    (neurons of interest are above and below fixation point, saccade target and mirrored saccade target)
    define attention (AU, UA) and control task (UU)
    '''

    ## spatial points
    # fixation point (=center of visual field) in degree and horizontal x vertical
    FP = np.array([0., 0.])
    # saccade target in degree and horizontal x vertical
    ST = FP + np.array(params['saccade'])
    # mirrored saccade target in degree and horizontal x vertical
    STm = FP - np.array(params['saccade'])


    ## neurons
    neurons = {}
    for above in [0, 1]:
        dist_v = -params['range_v'] if above else params['range_v']
        for pos in [STm, FP, ST]:
            name = f"{'above' if above else 'below'} {'STm' if (pos==STm).all() else ('FP' if (pos==FP).all() else 'ST')}"
            n = [deg_to_idx(pos[0], params['V4L4_shape'][0], params['VF_Deg'][0]),
                 deg_to_idx(pos[1]+dist_v, params['V4L4_shape'][1], params['VF_Deg'][1])]
            neurons[name] = n
    

    ## attention positions and tasks
    # attention position in degree and horizontal x vertical
    # 4 possible postions: above/below FP/ST
    APs = {}
    # tasks: AU, UA and corresponding control (UU)
    tasks = {'AU': {}, 'UA': {}}
    control = {'AU': {}, 'UA': {}}
    # [above, p] can be: [0, FP] --> below FP, [0, ST] --> below ST, [1, FP] --> above FP, [1, ST] --> above ST
    for above in [0, 1]:
        for p in [FP, ST]:
            name = f"{'above' if above else 'below'} {'FP' if (p==FP).all() else 'ST'}"
            dist_v = -params['range_v'] if above else params['range_v']
            AP = p + np.array([0, dist_v])
            APs[name] = AP

            # attended->unattended (AU)
            # neuron with RF at AP
            n = [deg_to_idx(AP[0], params['V4L4_shape'][0], params['VF_Deg'][0]),
                 deg_to_idx(AP[1], params['V4L4_shape'][1], params['VF_Deg'][1])]
            # neuron with RF at vertically opposite AP
            n_control = [deg_to_idx(AP[0], params['V4L4_shape'][0], params['VF_Deg'][0]),
                         deg_to_idx(-AP[1], params['V4L4_shape'][1], params['VF_Deg'][1])]
            tasks['AU'].update({str(AP): n})
            control['AU'].update({str(AP): n_control})

            # unattended->attended (UA)
            # neuron with RF at remapped AP
            n = [deg_to_idx(AP[0]-params['saccade'][0], params['V4L4_shape'][0], params['VF_Deg'][0]),
                 deg_to_idx(AP[1]-params['saccade'][1], params['V4L4_shape'][1], params['VF_Deg'][1])]
            # neuron with RF at vertically opposite AP
            n_control = [deg_to_idx(AP[0]-params['saccade'][0], params['V4L4_shape'][0], params['VF_Deg'][0]),
                         deg_to_idx(-(AP[1]-params['saccade'][1]), params['V4L4_shape'][1], params['VF_Deg'][1])]
            tasks['UA'].update({str(AP): n})
            control['UA'].update({str(AP): n_control})


    return neurons, APs, tasks, control


def getTrialsPerAP():
    '''
    get all trial ids separated by attention position

    return: trialsPerAP -- dictionary with trial ids according to attention position
    '''

    # dictionary with trial ids according to attention position
    trialsPerAP = {}
    for t in next(os.walk(f"../{params['SetupDir']}"))[1]:
        filename = f"../{params['SetupDir']}/{t}/output.txt"
        outputData = getOutputData(filename)
        AP = str(outputData['AP'])
        if AP not in trialsPerAP:
            trialsPerAP[AP] = [t]
        else:
            trialsPerAP[AP].append(t)

    ## save for further usage
    save_dict_to_hdf5(trialsPerAP, f"../{params['SetupDir']}/trialsPerAP.hdf5")

    return trialsPerAP


def extract_eyepos():
    '''
    mean over all eye trajectories generated in simulations
    '''

    # go through all trials in parallel
    runParallel = [(f"../{params['ResultDir']}/trials/{t}/Rates/eyepos.txt", params['tEnd']+1)
                    for t in next(os.walk(f"../{params['ResultDir']}/trials/"))[1]]
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
    # Step 2: `pool.apply` the `add()`
    eyepos = pool.starmap(getEyepos, runParallel)
    # Step 3: Don't forget to close
    pool.close()

    # save mean
    np.save(f"../{params['ResultDir']}/extractedData/eyepos.npy", np.mean(eyepos, axis=0))


def extract_LIPdata_single(AP_str, trial, layers):
    '''
    extract data for given attention position and trial
    can be run in parallel

    params: AP     -- given attention position
            trials -- given trial
            layers -- LIP layers to be extracted
    '''
    
    # convert string into array
    AP = np.fromstring(AP_str [1:-1], dtype=float, sep=' ')

    # get rates of current trial
    fn = f"../{params['ResultDir']}/trials/{trial}/Rates/dict_rates.hdf5"
    rates = load_dict_from_hdf5(fn)        
    # neuron is retionotopic, thus dependent on actual eye position
    # eye position over time
    ep = getEyepos(f"../{params['ResultDir']}/trials/{trial}/Rates/eyepos.txt", params['tEnd'] + 1)
    # fixed vertical neuron
    neuron_v = int(np.mean(deg_to_idx(AP[1]-ep[:, 1], params['V4L4_shape'][1], params['VF_Deg'][1])))
    # extract LIP rates
    rates_LIP = {}
    for layer in layers:
        layer_name = layer[layer.rfind('_')+1:]
        layer_stuff = layer[:layer.rfind('_')]
        rates_LIP[layer] = rates[layer_name][layer_stuff][:, :, neuron_v, 0]

    return rates_LIP

def extract_LIPdata():
    '''
    extract and summarize simulated LIP results from all trials
    create LIPdata
    '''
    
    # trialsPerAP = getTrialsPerAP()
    trialsPerAP = load_dict_from_hdf5(f"../{params['SetupDir']}/trialsPerAP.hdf5")
    
    # LIP data to be extracted    
    layers = {'sum(A_LIPpc)_V4L4': 'spatial attention from LIP PC',
              'sum(A_LIPcd)_V4L4': 'spatial attention from LIP CD',
              'ALIP_V4L4':'spatial attention from LIP'}

    LIPdata = {}
    # for each attention position seperately
    for AP, subtrials in trialsPerAP.items():
        ## extract data in parallel
        print(f"extract data for AP {AP}")
        runParallel = [(AP, int(trial), list(layers.keys())) for trial in subtrials]    
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
        # Step 2: `pool.apply` the `add()`
        subresults = pool.starmap(extract_LIPdata_single, runParallel)
        # Step 3: Don't forget to close
        pool.close()

        ## get mean over all results
        print(f"summarize data for AP {AP}")
        # init
        LIPdata[str(AP)] = {}
        for exp in layers.values():
            LIPdata[str(AP)][exp] = np.zeros((params['tEnd']+1, params['V4L4_shape'][0]))
        # sum up all subresults
        for res in subresults:
            for layer, exp in layers.items():    
                LIPdata[str(AP)][exp] += res[layer]
        # normalize
        for exp in layers.values():
            LIPdata[str(AP)][exp] /= len(runParallel)

    ## save extracted results for further usage
    save_dict_to_hdf5(LIPdata, f"../{params['ResultDir']}/extractedData/LIPdata.hdf5")


def extract_onsetPos_single(trial, neurons, layer):
    '''
    extract data for given trial and layers and all neurons
    can be run in parallel

    params: trial   -- given trial
            neurons -- list of neurons, whose rates should be extracted
            layer   -- name of layer/stuff that should be extracted
    '''

    layer_name = layer[layer.rfind('_')+1:]
    layer_stuff = layer[:layer.rfind('_')]

    ## get firing rate over time for given neuron in this trial
    fn = f"../{params['ResultDir']}/trials/{trial}/Rates/dict_rates.hdf5"
    rate = load_dict_from_hdf5(fn)[layer_name][layer_stuff]
    if layer_name == 'V1':
        rate = rate[:, :, :, 0, 0]
    else:
        rate = rate[:, :, :, 0]

    ## get position and onset of presented bars in this trial
    barOnsets = getBarposAndOnset(f"../{params['ResultDir']}/trials/{trial}/Rates/", params['tEnd']+1, False)


    ## create matrix of firing rate caused for each possible bar positions and onsets for each neuron
    # 2 separate matrices for bars presented above and below the horizontal center line  
    
    # init matrix
    # all possible horizontal positions of bar
    pos_h = np.arange(-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2+params['range_h'], params['range_h'])
    # all possible onsets of bar
    onsets = np.arange(params['range_t'][0], params['range_t'][1]+1, params['range_t'][2])  
    onsetPos = {}
    counter_onsetPos = {}
    for n in neurons:
        onsetPos[str(n)] = {'above': np.zeros((len(pos_h), len(onsets))), 'below': np.zeros((len(pos_h), len(onsets)))}
        counter_onsetPos[str(n)] = {'above': np.zeros((len(pos_h), len(onsets))), 'below': np.zeros((len(pos_h), len(onsets)))}
    
    # fill matrix
    tw = 20 # time window
    for data in barOnsets.values():

        # find index in onsetPos-matrix
        idx_pos = np.where(pos_h==data['pos'][0])[0][0]
        idx_onset = np.where(onsets==data['onset'])[0][0]

        # get firing rate caused by presented bar for each neuron
        t_start = data['onset']# - tw//2
        t_end = t_start + tw
        for n in neurons:
            r_max = np.max(rate[t_start:t_end+1, n[0], n[1]])

            if data['pos'][1] == params['range_v']:
                # below horizontal center line
                onsetPos[str(n)]['below'][idx_pos, idx_onset] += r_max
                counter_onsetPos[str(n)]['below'][idx_pos, idx_onset] += 1
            else:
                # above horizontal center line
                onsetPos[str(n)]['above'][idx_pos, idx_onset] += r_max
                counter_onsetPos[str(n)]['above'][idx_pos, idx_onset] += 1

    # summarize
    return {'onsetPos': onsetPos, 'counter': counter_onsetPos}
    
def extract_onsetPos(layer):
    '''
    extract and summarize simulation results from all trials
    create onsetPos-matrix

    params: layer -- name of layer/stuff whose data should be extracted
    '''

    neurons, _, _, _ = getData()
    # trialsPerAP = getTrialsPerAP()
    trialsPerAP = load_dict_from_hdf5(f"../{params['SetupDir']}/trialsPerAP.hdf5")

    results = {}
    # for each attention position seperately
    for AP, subtrials in trialsPerAP.items():
        ## extract data in parallel
        print(f"extract data for AP {AP}")    
        runParallel = [(int(trial), list(neurons.values()), layer) for trial in subtrials]
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
        # Step 2: `pool.apply` the `add()`
        subresults = pool.starmap(extract_onsetPos_single, runParallel)
        # Step 3: Don't forget to close
        pool.close()

        ## get mean over all results
        # print(f"summarize data for AP {AP}")
        # sum up all subresults
        onsetPos = subresults[0]['onsetPos']
        counter = subresults[0]['counter']
        for res in subresults[1:]:
            for n, res_n in res['onsetPos'].items():
                for pos, res_n_pos in res_n.items():
                    onsetPos[str(n)][pos] += res_n_pos
                    counter[str(n)][pos] += res['counter'][str(n)][pos]
        # normalize
        for n, res_n in onsetPos.items():
            for pos, res_n_pos in res_n.items():
                res_n_pos /= counter[str(n)][pos]
        
        ## add results
        results[str(AP)] = onsetPos
                

    ## save extracted results for further usage
    save_dict_to_hdf5(results, f"../{params['ResultDir']}/extractedData/{layer}.hdf5")
