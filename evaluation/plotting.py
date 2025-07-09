"""
@author: juschu

Main script for plotting
"""

##############################
#### imports and settings ####
##############################
# import standard libraries
import os
import sys
import numpy as np
import multiprocessing as mp
import pylab as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

# import files from this folder
from processData import getData

# import files from parent folders
sys.path.append('../')
from helper import getEyepos, getBarpos, getOutputData, idx_to_deg
from saving import load_dict_from_hdf5
# parameters
from parameters.params_general import params
from parameters.params_model import params as params_model
from parameters.params_plotting import params as params_plotting
params.update(params_model)
params.update(params_plotting)


def plot_setup(trial):
    '''
    Figure 1:
    spatial and temporal layout of given trial

    params: trial -- number of trial
    '''

    ## get data
    # relevant time steps to plot
    timesteps = [-300, 0, 300]

    # setup including points and timesteps
    loadedSetup = getOutputData(f"../{params['ResultDir']}/trials/{trial}/Rates/output.txt")
    FP = loadedSetup['FP']                      # fixation point
    ST = loadedSetup['ST']                      # saccade target
    AP = loadedSetup['AP']                      # attention position
    RAP = AP - np.array(params['saccade'])      # remapped attenion position
    start_timestep = loadedSetup['t_start']     # start of simulation
    end_timestep = loadedSetup['t_end']         # end of simulation
    duration = end_timestep-start_timestep+1    # total duration of simulation
    sacOnset = loadedSetup['t_sacStart']        # saccade onset
    sacDur = loadedSetup['t_sacEnd'] - sacOnset # saccade duration

    # eye position over time
    ep = getEyepos(f"../{params['ResultDir']}/trials/{trial}/Rates/eyepos.txt", duration)
    
    # bar positions over time
    bars = getBarpos(f"../{params['ResultDir']}/trials/{trial}/Rates/0_barpos.txt", duration)

    ## plotting
    ## spatial    
    for t in timesteps:
        timestep = t + sacOnset

        fig = plt.figure(figsize=(16, 12))

        # eye position
        if t < 0:
            # fixate at fixation point
            plt.scatter(FP[0], FP[1], marker='o', facecolor='black', s=700)
            plt.text(FP[0], FP[1]-1, 'FP', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='bottom')
        elif 0 <= t <= sacDur:
            # onging saccade towards ST
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='white', s=700)
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='black', s=70)
            plt.text(ST[0], ST[1]-1, 'ST', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='bottom')
            plt.arrow(FP[0], FP[1], params['saccade'][0]-1, params['saccade'][1],
                      head_width=0.5, head_length=1, fc='black', ec='black', linewidth=5)
            plt.text(0.5*(FP[0]+ST[0]-0.5), 0.5*(FP[1]+ST[1])+0.5,
                     'saccade', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='top')
        else:
            # fixate at saccade target
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='white', s=700)
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='black', s=70)
            plt.text(ST[0], ST[1]-1, 'ST', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='bottom')
        
        # (remapped) attention position
        plt.scatter(AP[0], AP[1], marker='o', edgecolor='black', facecolor='none', s=700, linewidth=5)
        plt.text(AP[0], AP[1]-1, 'AP', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='bottom')
        plt.scatter(RAP[0], RAP[1], marker='o', edgecolor='gray', facecolor='none', s=700, linewidth=5)
        plt.text(RAP[0], RAP[1]-1, 'RAP', color='gray', fontsize=params['fontsizes']['setup_text'], horizontalalignment='center', verticalalignment='bottom')

        # bars
        for b in bars:
            plt.scatter(bars[b][timestep][0], bars[b][timestep][1], marker='|', s=2000, linewidth=5, facecolor='black')

        # arrange plot
        ax = plt.gca()
        plt.xlim([-params['VF_Deg'][0]/2.0, params['VF_Deg'][0]/2.0])
        plt.ylim([params['VF_Deg'][1]/2.0, -params['VF_Deg'][1]/2.0])
        plt.xticks([])
        plt.yticks([])
        # plt.xticks(np.linspace(-params['VF_Deg'][0]/2.0, params['VF_Deg'][0]/2.0, 5))
        # plt.yticks([params['VF_Deg'][1]/2.0, params['range_v'], 0,  -params['range_v'], -params['VF_Deg'][1]/2.0])
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\\circ$'))
        # ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\\circ$'))
        # plt.xlabel('\nHorizontal', fontsize=params['fontsizes']['setup_axes'])
        # plt.ylabel('Vertical', fontsize=params['fontsizes']['setup_axes'])
        # for label in ax.get_xticklabels() + ax.get_yticklabels():
        #     label.set_fontsize(params['fontsizes']['setup_axes'])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../{params['ResultDir']}/figs/Fig1A_trial={trial}_t={t:0>3d}.svg", dpi=300)
        plt.close(fig)


    ## temporal
    fig = plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # eye position
    plt.plot(ep[:, 0], color='black', linewidth=2, label='eyepos x')
    plt.text(start_timestep-end_timestep/10, ep[0, 0], 'Eye x', fontsize=params['fontsizes']['setup_axes'], horizontalalignment='left')

    # bars
    for b, bar in bars.items():
        t_start = np.argwhere(bar!=np.inf)[0][0]
        # print(t_start, bar[t_start])
        plt.plot([t_start, t_start], [-params['range_v'], -params['range_v']+params['saccade'][0]], color='gray', linewidth=2)
    plt.text(start_timestep-end_timestep/10, -params['range_v'], 'Probes', fontsize=params['fontsizes']['setup_axes'], horizontalalignment='left')
    # bar limits
    frame = plt.Rectangle([params['range_t'][0], -params['range_v']], params['range_t'][1]-params['range_t'][0], params['saccade'][0],
                          fill=False, linewidth=2, color='black')
    ax.add_patch(frame)
    plt.plot([start_timestep, end_timestep], [-params['range_v'], -params['range_v']], color='black', linewidth=2)
    
    # saccade
    frame = plt.Rectangle([sacOnset, -params['range_v']-0.5], sacDur, params['range_v']+params['saccade'][0]+1.0,
                          linewidth=2, color='gray', alpha=0.5)
    ax.add_patch(frame)

    # adjust plot
    plt.xlim(start_timestep, end_timestep)
    plt.xticks(np.linspace(start_timestep, end_timestep, 5, dtype='int'),
               np.linspace(start_timestep, end_timestep, 5, dtype='int')-sacOnset)
    plt.xlabel('Time relative to saccade onset (ms)', fontsize=params['fontsizes']['setup_axes'], labelpad=15)
    for label in ax.get_xticklabels():
        label.set_fontsize(params['fontsizes']['setup_axes'])
    plt.ylim(-params['range_v']-0.5, params['saccade'][0]+0.5)
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    
    # plt.show()
    plt.savefig(f"../{params['ResultDir']}/figs/Fig1B_trial={trial}.svg")
    plt.close(fig)

def plot_setup_simple(attTask, AP, neuron, trick_colorbar=False):
    '''
    plot setup including fixation point (FP), saccade target (ST), attention position (AP)
    as well as (future) receptive field ((F)RF) of neurons encoding given attention task
    used in evaluation plots (Fig. 4,5,6,8)

    params: attTask        -- given attention task (AU, UA or UU)
            AP             -- given attention position
            neuron         -- corresponding neuron (according to AP and attTask)
            trick_colorbar -- size must be adapted in Figure 4 due to colorbars in other subplots
    '''


    # fixation point
    FP = np.array([0., 0.])
    plt.scatter(FP[0], FP[1], marker='o', facecolor='black', s=100)

    # saccade target
    ST = FP + np.array(params['saccade'])
    plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='white', s=100)
    plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='black', s=20)

    # saccade
    plt.arrow(FP[0], FP[1], params['saccade'][0]-0.7, params['saccade'][1],
                head_width=1, head_length=2, fc='black', ec='black', linewidth=1, length_includes_head=True)

    # # eye position
    # plt.scatter(FP[0], FP[1], marker='x', facecolor='red', s=200, linewidth=2)

    # attention position
    plt.scatter(AP[0], AP[1], marker='o', edgecolor='black', facecolor='none', s=100, linewidth=2, linestyle='-')

    # (F)RF of neuron encoding attention task
    RF = [idx_to_deg(neuron[0], params['V4L4_shape'][0], params['VF_Deg'][0]),
            idx_to_deg(neuron[1], params['V4L4_shape'][1], params['VF_Deg'][1])]
    FRF = RF + np.array(params['saccade'])
    plt.scatter(RF[0], RF[1], marker='o', edgecolor='red', facecolor='none', s=400, linewidth=2, linestyle='-')
    plt.scatter(FRF[0], FRF[1], marker='o', edgecolor='red', facecolor='none', s=400, linewidth=2, linestyle='--')

    # frame color-coded for attention condition
    for p in ["top", "left", "right", "bottom"]:
        ax = plt.gca()
        ax.spines[p].set_color(params[attTask]['color'])
        ax.spines[p].set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # trick to shrink the subplot to the same size as the others subplots (due to colorbar)
    # need only for Figure 4
    if trick_colorbar:
        cbar = plt.colorbar(shrink=0)
        cbar.set_ticks([])

    # adjust plot
    plt.title(attTask, color=params[attTask]['color'], fontsize=params['fontsizes']['title'])
    plt.xlim([-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2])
    plt.ylim([params['VF_Deg'][1]/2, -params['VF_Deg'][1]/2])
    plt.gca().set_aspect('equal')

def plot_actLIP():
    '''
    Figure 3, 7:
    pot activity of projection from both LIP maps to V4L4 individually as well as their sum
    normalized over all runs, for each AP separately
    '''

    # important points
    FP = np.array([0., 0.])                 # fixation point
    saccade = np.array(params['saccade'])
    ST = FP + saccade                       # saccade target
    # timesteps aligned to saccade onset
    timesteps = np.linspace(0, params['tEnd'], params['tEnd']+1, dtype='int') - params['saccOnset']
    ext_rates = [-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2, timesteps[-1], timesteps[0]]

    # extract LIP data (see processData.extract_LIPdata())
    ratesPerAP = load_dict_from_hdf5(f"../{params['ResultDir']}/extractedData/LIPdata.hdf5")

    # eye position over time
    ep = np.load(f"../{params['ResultDir']}/extractedData/eyepos.npy")

    # one figure for each attention position
    for AP_str, rates in ratesPerAP.items():
        fig = plt.figure(figsize=(24, 9))
        gs = GridSpec(4, 9)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.1)

        # (remapped) attention position
        # convert string into array
        AP = np.fromstring(AP_str[1:-1], dtype=float, sep=' ')
        RAP = AP - saccade
        # axes limits for setup plot
        ymin = min(params['VF_Deg'][1], FP[1], ST[1], AP[1], RAP[1])
        ymax = max(0, FP[1], ST[1], AP[1], RAP[1])
        
        for i, (name, rates_mean) in enumerate(rates.items()):

            ## setup
            ax = plt.subplot(gs[3, i*3:3+i*3])
            # fixation point
            plt.scatter(FP[0], FP[1], marker='o', s=200, color='black')
            plt.text(FP[0], FP[1]+2, 'FP', fontsize=params['fontsizes']['text'], horizontalalignment='center', verticalalignment='top')
            # saccade target
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='white', s=200)
            plt.scatter(ST[0], ST[1], marker='o', edgecolor='black', facecolor='black', s=20)
            plt.text(ST[0], ST[1]+2, 'ST', fontsize=params['fontsizes']['text'], horizontalalignment='center', verticalalignment='top')
            # attention position            
            plt.scatter(AP[0], AP[1], marker='o', s=200, color='black')
            plt.text(AP[0], AP[1]-2, 'AP', fontsize=params['fontsizes']['text'], horizontalalignment='center', verticalalignment='bottom')
            # remapped attention position
            plt.scatter(RAP[0], RAP[1], marker='o', s=200, color='gray')
            plt.text(RAP[0], RAP[1]-2, 'RAP', color='gray', fontsize=params['fontsizes']['text'], horizontalalignment='center', verticalalignment='bottom')
            # saccade
            plt.arrow(FP[0], FP[1], saccade[0], saccade[1], color='black', head_width=2, head_length=1.5, length_includes_head=True)
            # ax.set_aspect(1/3)
            ax.set_xlim((ext_rates[0], ext_rates[1]))
            ax.set_ylim((ymax+10, ymin-10))
            ax.get_yaxis().set_visible(False)
            for pos in ['top', 'right', 'left']:
                ax.spines[pos].set_visible(False)
            plt.xlabel('Horizontal position (deg)', fontsize=params['fontsizes']['axes'], labelpad=10)
            for label in ax.get_xticklabels():
                label.set_fontsize(params['fontsizes']['axes'])
            
            ## activity from LIP to V4, Layer 4
            ax = plt.subplot(gs[:3, i*3:(i+1)*3])
            plt.title(f'normalized\n{name}', fontsize=params['fontsizes']['title'])
            # plot spatial attention of V4 for fixed vertical neuron
            # normalize and remove noise (=rate from outer neurons)
            r = (rates_mean/np.max(rates_mean)).T
            noise = np.max(r[np.append(range(10), range(-10,0))], axis=0)
            r[np.where(r<=noise)] = 0
            plt.imshow(r.T, aspect='auto', cmap='myReds', extent=ext_rates)
            plt.grid()
            ax.get_xaxis().set_visible(False)
            plt.clim(0, 1)
            # encoded (R)AP over time
            plt.plot((AP-ep)[:, 0], timesteps, color='black')
            plt.plot((RAP-ep)[:, 0], timesteps, color='gray', linestyle='--')
            # adjust plot
            if i==0:
                plt.ylabel('Time relative to saccade onset (ms)', fontsize=params['fontsizes']['axes'], labelpad=15)
                for label in ax.get_yticklabels():
                    label.set_fontsize(params['fontsizes']['axes'])
            else:                
                ax.set_yticklabels([])

        # add colorbar
        cbaxes = fig.add_axes([0.97, 0.3, 0.01, 0.6])  # position for the colorbar
        cbar = plt.colorbar(cax=cbaxes)
        cbar.ax.tick_params(labelsize=params['fontsizes']['axes'])

        # plt.show()
        plt.savefig(f"../{params['ResultDir']}/figs/Fig3_{AP}.png", dpi=300)
        plt.close(fig)


def plot_revCorrelation(layer):
    '''
    Figure 4:
    plot for one attention position firing rates for all possible combinations of bar positions and onsets
    for neurons encoding AU and UA task, respectively, for AU/UA task, UU task as well as difference between AU/UA and UU

    params: layer -- layer whose data should be plotted
    '''

    gs = GridSpec(4, 2)

    # all possible horizontal positions of bar
    pos_h = np.arange(-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2+params['range_h'], params['range_h'])
    # all possible onsets of bar
    onsets = np.arange(params['range_t'][0], params['range_t'][1]+1, params['range_t'][2])
    # saccade onset
    sacOnset = params['saccOnset']

    neurons, APs, tasks, control = getData()
    
    onsetPos = load_dict_from_hdf5(f"../{params['ResultDir']}/extractedData/{layer}.hdf5")


    # one figure for each attention position
    for AP in APs.values():

        fig = plt.figure(figsize=(12, 13.8))
        plt.subplots_adjust(hspace=0.3)

        for i, (cond, neuronPerAP) in enumerate(tasks.items()):
            # corresponding neuron in attention task (AU or UA) and control neuron
            neuron_att = neuronPerAP[str(AP)]
            neuron_control = control[cond][str(AP)]
            # is this neurons above or below?
            pos_v_att = list(neurons.keys())[list(neurons.values()).index(neuron_att)].split(' ')[0]
            pos_v_control = list(neurons.keys())[list(neurons.values()).index(neuron_control)].split(' ')[0]

            # firing rates for all possible combinations of bar positions and onsets for this neuron
            onsetPos_att = onsetPos[str(AP)][str(neuron_att)][pos_v_att]
            onsetPos_control = onsetPos[str(AP)][str(neuron_control)][pos_v_control]
            # reduce noise and bound to 0 (lower bound), no upper bound
            onsetPos_att = np.clip(onsetPos_att - np.max(onsetPos_control)/2, 0, np.inf)
            onsetPos_control = np.clip(onsetPos_control - np.max(onsetPos_control)/2, 0, np.inf)

            ## plot
            # setup
            plt.subplot(gs[0, i])
            plot_setup_simple(cond, AP, neuron_att, True)

            # rates of neuron encoding attention task
            ax = plt.subplot(gs[1, i])
            plt.title(cond, fontsize=params['fontsizes']['title'])
            # plot onsetPos
            plt.imshow(onsetPos_att.T, cmap='myReds',
                       extent=[pos_h[0], pos_h[-1], onsets[-1]-sacOnset, onsets[0]-sacOnset])
            plt.clim(0, params[cond]['clim'])
            cbar = plt.colorbar(ticks=mticker.MultipleLocator(params[cond]['clim']/2))
            cbar.ax.tick_params(labelsize=params['fontsizes']['axes'])

            plt.plot([pos_h[0], pos_h[-1]], [0, 0], 'black', linestyle='--')
            plt.plot([AP[0], AP[0]], [onsets[0]-sacOnset, onsets[-1]-sacOnset],
                        'black', linestyle='--')
            plt.xlim(-params['VF_Deg'][0]/2.0, params['VF_Deg'][0]/2.0)
            plt.yticks(np.array(onsets[0::5])-sacOnset)
            if i==0:
                plt.ylabel('bar onset relative\nto saccade onset (ms)', fontsize=params['fontsizes']['axes'])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(params['fontsizes']['axes'])
            plt.gca().set_aspect(params['VF_Deg'][1]/(onsets[-1]-onsets[0]))
            
            # rates of control neuron
            ax = plt.subplot(gs[2, i])
            plt.title('UU', fontsize=params['fontsizes']['title'])
            # plot onsetPos
            plt.imshow(onsetPos_control.T, cmap='myReds',
                       extent=[pos_h[0], pos_h[-1], onsets[-1]-sacOnset, onsets[0]-sacOnset])
            plt.clim(0, params[cond]['clim'])
            cbar = plt.colorbar(ticks=mticker.MultipleLocator(params[cond]['clim']/2))
            cbar.ax.tick_params(labelsize=params['fontsizes']['axes'])

            plt.plot([pos_h[0], pos_h[-1]], [0, 0], 'black', linestyle='--')
            plt.plot([AP[0], AP[0]], [onsets[0]-sacOnset, onsets[-1]-sacOnset],
                        'black', linestyle='--')
            plt.xlim(-params['VF_Deg'][0]/2.0, params['VF_Deg'][0]/2.0)
            plt.yticks(np.array(onsets[0::5])-sacOnset)
            if i==0:
                plt.ylabel('bar onset relative\nto saccade onset (ms)', fontsize=params['fontsizes']['axes'])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(params['fontsizes']['axes'])
            plt.gca().set_aspect(params['VF_Deg'][1]/(onsets[-1]-onsets[0]))
            
            # difference of rates
            ax = plt.subplot(gs[3, i])
            plt.title(f"{cond}-UU", fontsize=params['fontsizes']['title'])
            # plot onsetPos
            plt.imshow((onsetPos_att-onsetPos_control).T, cmap='seismic',
                       extent=[pos_h[0], pos_h[-1], onsets[-1]-sacOnset, onsets[0]-sacOnset])
            plt.clim(-params[cond]['clim']/2, params[cond]['clim']/2)
            cbar = plt.colorbar(ticks=mticker.MultipleLocator(params[cond]['clim']/2))
            cbar.ax.tick_params(labelsize=params['fontsizes']['axes'])

            plt.plot([pos_h[0], pos_h[-1]], [0, 0], 'black', linestyle='--')
            plt.plot([AP[0], AP[0]], [onsets[0]-sacOnset, onsets[-1]-sacOnset],
                        'black', linestyle='--')
            plt.xlim(-params['VF_Deg'][0]/2.0, params['VF_Deg'][0]/2.0)
            plt.xlabel('horizontal bar position (in deg)', fontsize=params['fontsizes']['axes'])
            plt.yticks(np.array(onsets[0::5])-sacOnset)
            if i==0:
                plt.ylabel('bar onset relative\nto saccade onset (ms)', fontsize=params['fontsizes']['axes'])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(params['fontsizes']['axes'])
            plt.gca().set_aspect(params['VF_Deg'][1]/(onsets[-1]-onsets[0]))

        # plt.show()
        plt.savefig(f"../{params['ResultDir']}/figs/Fig4_{AP}_{layer}.png", dpi=300)
        plt.close(fig)


def plot_actV4L4():
    '''
    Figure 5, 6, 8:
    averaged activity of V4, L4 neurons for the three different tasks (AU, UA, UU)
    '''

    # all possible horizontal positions of bar
    pos_h = np.arange(-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2+params['range_h'], params['range_h'])
    # all possible onsets of bar
    onsets = np.arange(params['range_t'][0], params['range_t'][1]+1, params['range_t'][2])
    # saccade onset
    sacOnset = params['saccOnset']

    neurons, APs, tasks, control = getData()

    ## y-axis for both subplots
    ylim = [[0, 10], [-0.5, 4.5]]
    

    ## get averaged activity
    onsetPos = load_dict_from_hdf5(f"../{params['ResultDir']}/extractedData/r_V4L4.hdf5")
    avgAct = {'AU': np.zeros(len(onsets)), 'UA': np.zeros(len(onsets)), 'UU': np.zeros(len(onsets))}
    for cond, neuronPerAP in tasks.items():
        # AU and UA
        for AP, neuron in neuronPerAP.items():
            # is this neurons above or below?
            pos_v = list(neurons.keys())[list(neurons.values()).index(neuron)].split(' ')[0]
            
            # firing rates for all possible combinations of bar positions and onsets for this neuron
            # summed over all positions
            avgAct[cond] += onsetPos[str(AP)][str(neuron)][pos_v].sum(axis=0)
        # normalize
        avgAct[cond] /= len(neuronPerAP)
    # UU
    count = 0
    for cond, neuronPerAP in control.items():
        for AP, neuron in neuronPerAP.items():
            # is this neurons above or below?
            pos_v = list(neurons.keys())[list(neurons.values()).index(neuron)].split(' ')[0]
            
            # firing rates for all possible combinations of bar positions and onsets for this neuron
            # summed over all positions
            avgAct['UU'] += onsetPos[str(AP)][str(neuron)][pos_v].sum(axis=0)
            count += 1
    # normalize
    avgAct['UU'] /= count


    ## plot
    fig = plt.figure(figsize=(24, 13.8))
    plt.subplots_adjust(hspace=1.0)
    gs = GridSpec(6, 5)

    lineSacOnset = (sacOnset-onsets[0])/(onsets[-1]-onsets[0]) * (len(onsets)-1)

    # subplot of averaged responses for all three conditions
    ax = plt.subplot(gs[:3, :-1])
    for cond, v in avgAct.items():
        # averaged V4 responses
        plt.plot(v, color=params[cond]['color'], label=cond, linewidth=3)
    # saccade onset
    plt.plot([lineSacOnset, lineSacOnset], ylim[0], 'black')
    # adjust plot
    plt.title('average over neurons', fontsize=params['fontsizes']['title'])
    plt.legend(fontsize=params['fontsizes']['legend'])
    # x-axis (bar onsets)
    plt.xlim(0, len(onsets)-1)
    plt.xticks([])
    # y-axis (response of V4)
    plt.ylim(ylim[0])
    plt.ylabel('average response of V4 neurons\nover all bar positions', fontsize=params['fontsizes']['axes'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(params['fontsizes']['axes'])

    # subplot of normalized averaged responses for all three conditions
    ax = plt.subplot(gs[3:, :-1])
    for cond in ['AU', 'UA']:
        plt.plot(avgAct[cond]-avgAct['UU'], color=params[cond]['color'], linestyle='--', label=cond, linewidth=3)
    # saccade onset
    plt.plot([lineSacOnset, lineSacOnset], ylim[1], 'black')
    # zero
    plt.plot([0, len(onsets)-1], [0, 0], 'black')
    # adjust plot
    plt.title('normalized average over neurons', fontsize=params['fontsizes']['title'])
    plt.legend(fontsize=params['fontsizes']['legend'])
    # x-axis (bar onsets)
    plt.xlim(0, len(onsets)-1)
    min_t = onsets[0]-sacOnset
    plt.xticks([0, 0.5*lineSacOnset, lineSacOnset, 1.5*lineSacOnset, 2*lineSacOnset],
               np.array([min_t, min_t/2, 0, -min_t/2, -min_t], dtype=int))
    plt.xlabel('bar onset relative to saccade onset (ms)', fontsize=params['fontsizes']['axes'])
    # y-axis (response of V4)
    plt.ylim(ylim[1])
    plt.ylabel('normalized average response of V4 neurons\nover all bar positions',
               fontsize=params['fontsizes']['axes'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(params['fontsizes']['axes'])

    # setup for 3 different attention conditions (AU, UA, UU)
    # define fixed AP for all three conditions
    AP = APs['above FP']
    plt.subplot(gs[:2, -1])
    plot_setup_simple('AU', AP, tasks['AU'][str(AP)])
    plt.subplot(gs[2:4, -1])
    plot_setup_simple('UA', AP, tasks['UA'][str(AP)])
    plt.subplot(gs[4:, -1])
    plot_setup_simple('UU', AP, control['AU'][str(AP)])

    # plt.show()
    plt.savefig(f"../{params['ResultDir']}/figs/Fig5.png", dpi=300)
    plt.close(fig)

def plot_actAll():
    '''
    modification of plot_actV4L4()
    averaged activity of all rates and stuff for the three different tasks (AU, UA, UU)
    '''

    # all possible horizontal positions of bar
    pos_h = np.arange(-params['VF_Deg'][0]/2, params['VF_Deg'][0]/2+params['range_h'], params['range_h'])
    # all possible onsets of bar
    onsets = np.arange(params['range_t'][0], params['range_t'][1]+1, params['range_t'][2])
    # saccade onset
    sacOnset = params['saccOnset']

    neurons, APs, tasks, control = getData()

    fig = plt.figure(figsize=(24, 13.8))

    for i, layer in enumerate(['r_V1', 'E_V4L4', 'ASP_V4L4', 'AFEAT_V4L4', 'sum(A_LIPpc)_V4L4', 'sum(A_LIPcd)_V4L4', 'ALIP_V4L4', 'r_V4L4']):
    
        onsetPos = load_dict_from_hdf5(f"../{params['ResultDir']}/extractedData/{layer}.hdf5")

        ## get averaged activity
        avgAct = {'AU': np.zeros(len(onsets)), 'UA': np.zeros(len(onsets)), 'UU': np.zeros(len(onsets))}
        for cond, neuronPerAP in tasks.items():
            # AU and UA
            for AP, neuron in neuronPerAP.items():
                # is this neurons above or below?
                pos_v = list(neurons.keys())[list(neurons.values()).index(neuron)].split(' ')[0]
                
                # firing rates for all possible combinations of bar positions and onsets for this neuron
                # summed over all positions
                avgAct[cond] += onsetPos[str(AP)][str(neuron)][pos_v].sum(axis=0)
            # normalize
            avgAct[cond] /= len(neuronPerAP)
        # UU
        count = 0
        for cond, neuronPerAP in control.items():
            for AP, neuron in neuronPerAP.items():
                # is this neurons above or below?
                pos_v = list(neurons.keys())[list(neurons.values()).index(neuron)].split(' ')[0]
                
                # firing rates for all possible combinations of bar positions and onsets for this neuron
                # summed over all positions
                avgAct['UU'] += onsetPos[str(AP)][str(neuron)][pos_v].sum(axis=0)
                count += 1
        # normalize
        avgAct['UU'] /= count


        ## plot
        ax = plt.subplot(4, 2, i+1)
        lineSacOnset = (sacOnset-onsets[0])/(onsets[-1]-onsets[0]) * (len(onsets)-1)

        # subplot of averaged responses for all three conditions
        for cond, v in avgAct.items():
            # averaged V4 responses
            plt.plot(v, color=params[cond]['color'], label=cond, linewidth=2)
        # saccade onset
        plt.plot([lineSacOnset, lineSacOnset], ax.get_ylim(), 'black')
        # adjust plot
        plt.title(layer, fontsize=params['fontsizes']['title'])
        plt.legend(fontsize=params['fontsizes']['legend'])
        # x-axis (bar onsets)
        plt.xlim(0, len(onsets)-1)
        min_t = onsets[0]-sacOnset
        plt.xticks([0, 0.5*lineSacOnset, lineSacOnset, 1.5*lineSacOnset, 2*lineSacOnset],
                [min_t, min_t/2, 0, -min_t/2, -min_t])
        if i>=6:
            plt.xlabel('bar onset relative to saccade onset (ms)', fontsize=params['fontsizes']['axes'])
        # y-axis (response of V4)
        if not i%2:
            plt.ylabel('average response', fontsize=params['fontsizes']['axes'])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(params['fontsizes']['axes'])

    # plt.show()
    plt.savefig(f"../{params['ResultDir']}/figs/Fig5_all.png", dpi=300)
    plt.close(fig)


def generate_setupPlotsOneTimestep(ep, bars):
    '''
    generate setup plot for given timestep and return it as numpy array
    can be run in parallel

    params: ep -- current eye position
            bars -- current bar position
    '''

    fig = plt.figure(figsize=(4, 3), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # eye position
    plt.scatter(ep[0]+params['VF_Deg'][0]/2, ep[1]+params['VF_Deg'][1]/2,
                marker='x', s=100, linewidth=3, facecolor='red')

    # bars
    for b in bars:
        plt.scatter(bars[b][0]+params['VF_Deg'][0]/2, bars[b][1]+params['VF_Deg'][1]/2,
                    marker='|', s=200, linewidth=4, facecolor='green')

    plt.xlim([0, params['VF_Deg'][0]])
    plt.ylim([params['VF_Deg'][1], 0])

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.tight_layout()

    ## convert figure into numpy array
    # draw the renderer
    fig.canvas.draw()
    # transform figure into np.array
    plot_setup = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot_setup = plot_setup.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close(fig)

    ## normalize plot_setup
    plot_setup = plot_setup / float(plot_setup.max())

    return plot_setup


def generate_setupPlots(ep, bars):
    '''
    generate setup plots over time and save them as numpy array
    can be used in big plot    

    params: ep -- eye position over time
            bars -- bar position over time
    '''

    print("Pre-plot setup over time")

    ## generate setup plots in parallel
    runParallel = [(ep_, {b: bars[b][t] for b in bars}) for t, ep_ in enumerate(ep)]    
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
    # Step 2: `pool.apply` the `add()`
    results = pool.starmap(generate_setupPlotsOneTimestep, runParallel)
    # Step 3: Don't forget to close
    pool.close()
    
    return results

def plot_ratesOneTimestep(timestep, rates, layers, sacOnset, sacOffset, gs, saveForMovie=True):
    '''
    plot activities of given time step

    params: timestep     -- current timestep
            rates        -- current activities for all layers
            layers       -- name of all layers with corresponding plotting data (title, panel where to plot)
            sacOnset     -- timestep of saccade onset
            sacOffset    -- timestep of saccade offset
            gs           -- grid spec used for plotting
            saveForMovie -- if True, save figure of current timestep for further movie generation
                            else, save figure of current timestep as stand-alone plot
                            dafault: True
    '''
    
    # transformation to make plots slanted
    transform = mtransforms.Affine2D().skew_deg(30, 20).rotate_deg(-20)

    fig = plt.figure(figsize=(30, 16))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.6)
    plt.suptitle(f"\n\n\n\n\n{timestep-sacOnset:>4d}ms", fontsize=params['fontsizes']['title'])

    ## plot setup and activities of current timestep
    for l, data in layers.items():
        ax = fig.add_subplot(data['panel'])
        plt.title(data['title'], loc='right', fontsize=params['fontsizes']['title'])
        if l == 'setup':
            # plot setup
            im = plt.imshow(rates[l]['r'], interpolation='hanning')
            im.set_transform(transform + ax.transData)

            plt.xlim(-params['VF_Deg'][0], params['VF_Deg'][0]*10+params['VF_Deg'][1]*10*np.cos(np.radians(20)))
            plt.ylim(-params['VF_Deg'][1], params['VF_Deg'][1]*10*np.sin(np.radians(50)))
            plt.axis('off')

            if sacOnset <= timestep <= sacOffset:
                plt.text(params['VF_Deg'][0]*20, params['VF_Deg'][1]*10*np.sin(np.radians(50))/2, 'saccade',
                         color='red', fontsize=params['fontsizes']['title'], horizontalalignment='right', verticalalignment='top')

        else:
            # plot activities
            r = rates[l]['r']
            im = plt.imshow(r.T, cmap='hot')
            im.set_transform(transform + ax.transData)

            plt.xlim(-r.shape[0]/10, r.shape[0]+r.shape[1]*np.cos(np.radians(20)))
            plt.ylim(-r.shape[1]/10, r.shape[1]*np.sin(np.radians(50)))
            plt.clim(0, rates[l]['max_r'])
            plt.axis('off')

    ## add brackground (frames and arrows), created externally
    ax = fig.add_subplot(gs[:,:])
    bg = plt.imread("movie_rates_bg.png")
    plt.imshow(bg)
    plt.axis('off')

    if saveForMovie:
        plt.savefig(f"../{params['ResultDir']}/figs/Fig9/{timestep:0>3d}.png", bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(f"../{params['ResultDir']}/figs/Fig9_t={timestep-sacOnset}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def plot_ratesOverTime(trial, **kwargs):
    '''
    Figure 9:
    plot firing rates of important layers over time and generate movie out of sequence
    if timesteps are given, plot only given timesteps

    params: trial     -- number of trial

    kwargs: timesteps -- list of given timestep(s)
    '''

    gs = GridSpec(5, 13)
    # layes to plot with corresponding panel
    layers = {'FEF': {'panel': gs[1, 1:3], 'title': 'FEFv'},
              'FEFvm': {'panel': gs[1, 3:5], 'title': 'FEFvm'},
              'FEFm': {'panel': gs[1, 5:7], 'title': 'FEFm'},
              'V1': {'panel': gs[3, 3:5], 'title': 'V1'},
              'V4L4': {'panel': gs[2, 3:5], 'title': 'V4, L4'},
              'LIP PC': {'panel': gs[2, 10:12], 'title': 'LIP PC'},
              'LIP CD': {'panel': gs[2, 8:10], 'title': 'LIP CD'},
              'PC signal': {'panel': gs[3, 10:12], 'title': 'PC signal'},
              'Xh': {'panel': gs[1, 10:12], 'title': 'top-down\nattention signal'},
              'setup': {'panel': gs[4, 3:5], 'title': 'visual\npresentation'}}

    ## get data
    print("preprocess data")
    # output data
    loadedSetup = getOutputData(f"../{params['ResultDir']}/trials/{trial}/Rates/output.txt")
    start_timestep = loadedSetup['t_start']     # start of simulation
    end_timestep = loadedSetup['t_end']         # end of simulation
    duration = end_timestep-start_timestep+1    # total duration of simulation
    sacOnset = loadedSetup['t_sacStart']        # saccade onset
    sacOffset = loadedSetup['t_sacEnd']         # saccade offset
    # eye position over time
    ep = getEyepos(f"../{params['ResultDir']}/trials/{trial}/Rates/eyepos.txt", duration)
    # bar positions over time
    bars = getBarpos(f"../{params['ResultDir']}/trials/{trial}/Rates/0_barpos.txt", duration)
    # activities
    rates_full = load_dict_from_hdf5(f"../{params['ResultDir']}/trials/{trial}/Rates/dict_rates_full.hdf5")
    rates_full.update(load_dict_from_hdf5(f"../{params['ResultDir']}/trials/{trial}/Rates/dict_inputs.hdf5"))
    
    ## extract relevant data for plotting
    rates = {}
    for l in layers.keys():
        # generate setup plots over time
        if l == 'setup':
            r = generate_setupPlots(ep, bars)
        # bring activities in 3d shape (time x horizontal x vertical)
        elif l == 'FEF':
            r = rates_full[l]
        elif l == 'FEFvm':
            # maximum over third dimension
            r = np.max(rates_full[l]['r'], axis=3)
        elif l.startswith('V4'):
            r = rates_full[l]['r'][:, :, :, 0]
        elif l == 'V1':
            r = rates_full[l]['r'][:, :, :, 0, 0]
        elif l.startswith('LIP'):
            # project to visual space
            r = np.max(np.max(rates_full[l]['r'], axis=4), axis=3)
        else:
            r = rates_full[l]['r']
        rates[l] = {'r': r, 'max_r': np.max(r)}

    ## plotting
    if 'timesteps' in kwargs:
        ## one figure for each given timestep
        # generate plots in parallel
        runParallel = [(t, {l: {'r': r['r'][t], 'max_r': r['max_r']} for l, r in rates.items()}, layers, sacOnset, sacOffset, gs, False) for t in kwargs['timesteps']]
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
        # Step 2: `pool.apply` the `add()`
        pool.starmap(plot_ratesOneTimestep, runParallel)
        # Step 3: Don't forget to close
        pool.close()
    else:
        ## sequence of figures over time
        # where to save the figures
        if not os.path.isdir(f"../{params['ResultDir']}/figs/Fig9/"):
            os.makedirs(f"../{params['ResultDir']}/figs/Fig9/")
        print(f"Single plots are saved at ../{params['ResultDir']}/figs/Fig9/")
        # generate plots in parallel
        runParallel = [(t, {l: {'r': r['r'][t], 'max_r': r['max_r']} for l, r in rates.items()}, layers, sacOnset, sacOffset, gs) for t in range(duration)]
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(min(mp.cpu_count()-1, params['runInParallel'], len(runParallel)))
        # Step 2: `pool.apply` the `add()`
        pool.starmap(plot_ratesOneTimestep, runParallel)
        # Step 3: Don't forget to close
        pool.close()

        ## create movie out of png sequence
        os.system(f"ffmpeg -framerate 20 -i ../{params['ResultDir']}/figs/Fig9/%03d.png ../{params['ResultDir']}/figs/Fig9/movie.mp4")
