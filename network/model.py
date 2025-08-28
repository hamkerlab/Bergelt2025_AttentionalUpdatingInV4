"""
@author: juschu

Definition of neuro-computational model with ANNarchy

 - ODE of neurons
 - populations consisting of neurons
 - connections between populations
"""

##############################
#### imports and settings ####
##############################
import os
import numpy as np

from ANNarchy import Neuron, Population, TimedArray, Projection
from ANNarchy.extensions.convolution import Pooling, Convolution

from network.helper import rangeX, Gaussian2D, positive
# get own-defined connection pattern
from network.connections import gaussian2dTo4d_h, gaussian2dTo4d_v, gaussian2dTo4d_diag, gaussian4dTo2d_diag,\
                                gaussian4d_diagTo4d_v, all2all_exp2d, all2all_exp4d,\
                                gaussian3dTo4d, gaussian4dTo3d

from parameters.params_model import params as params_model
from parameters.params_general import params
params.update(params_model)


##########################################
##########  NEURON DEFINITION   ##########
##########################################
## Basic Auxiliary Neuron is transmitting an unmodified input
Aux_Neuron = Neuron(
    name='Aux Neuron',
    equations="""
        r = sum(exc)
    """
)

## Basic Input Neuron receiving pre-defined signal
Inp_Neuron = Neuron(
    name='Input Neuron',
    parameters="""
        tau = 'tau' : population
    """,
    equations="""
        tau * dr/dt + r = pos(sum(inp)) : min=0.0, max=1.0
    """,
    extra_values=params
)

## Neuron of Layer 4 in Area V4: Receives Input from V1, V4L23, and FEFvm
# See Eq 4.30-4.32 / 4.35-4.37 / 4.41
V4L4_Neuron = Neuron(
    name='V4L4 Neuron',
    parameters="""
        tau = 'tau' : population
        vV1 = 'v_V1-V4L4' : population
        pV1 = 'p_V1-V4L4' : population
        vV4L23 = 'v_V4L23-V4L4' : population
        pV4L23 = 'p_V4L23-V4L4' : population
        vFEFvm = 'v_FEFvm-V4L4' : population
        vLIP = 'v_LIP-V4L4' : population
        sigma = 'sigma_V4L4' : population
        g = 'g_V4L4' : population
    """,
    equations="""
        E = pow(vV1*clip(sum(exc), 0, 1), pV1)
        AFEAT = vV4L23*pow(sum(A_FEAT), pV4L23)
        ASP = vFEFvm*sum(A_SP)
        ALIP = vLIP*(sum(A_LIPpc)+sum(A_LIPcd))
        A = 1 + ASP + AFEAT + ALIP
        tau * dr/dt + r = g * E * A / (sigma + E * A)
    """,
    extra_values=params
)

## Neuron of Layer 2/3 in Area V4: Receives Input from V4L4
# See Eq 4.48-4.50, input from PFC removed
V4L23_Neuron = Neuron(
    name='V4L23 Neuron',
    parameters="""
        tau = 'tau' : population
        vV4L4 = 'v_V4L4-V4L23' : population
        pV4L4 = 'p_V4L4-V4L23' : population
        sigma = 'sigma_V4L23' : population
        g = 'g_V4L23' : population
    """,
    equations="""
        E = pow(vV4L4*sum(exc), pV4L4)
        tau * dr/dt + r = g * E / (sigma + E) : max = 1.0
    """,
    extra_values=params
)

## Neuron of visuo-motoric Layer in FEF: Receives Input from FEFv and FEFm
# See Eq 4.61-4.64
# vFEFvm will be adapted after creating the population
FEFvm_Neuron = Neuron(
    name='FEFvm Neuron',
    parameters="""
        tau = 'tau' : population
        vFEFv = 'v_FEFv-FEFvm' : population
        vFEFv_E = 'v_FEFv-FEFvm_E' : population
        vFEFv_S = 'v_FEFv-FEFvm_S' : population
        vFEFvm = 1.0
    """,
    equations="""
        Ev = vFEFv_E*sum(E_v)
        Sv = vFEFv_S*sum(S_v)
        ESv = vFEFv * pos(Ev) + (1-vFEFv) * clip(Ev-Sv,0,1)
        tau * dr/dt + r = vFEFvm * ESv + (1-vFEFvm) * sum(E_m): min=0.0
    """,
    extra_values=params
)

## Neuron of motoric Layer in FEF: Receives Input from FEFvm and FEFfix
# See Eq 4.68-4.71
FEFm_Neuron = Neuron(
    name='FEFm Neuron',
    parameters="""
        tau = 'tau' : population
        vFEFvm_E = 'v_FEFvm-FEFm_E' : population
        vFEFvm_S = 'v_FEFvm-FEFm_S' : population
        vFEFfix = 'v_FEFfix-FEFm' : population
        SaccadeThreshold = 'SaccadeThreshold' : population
    """,
    equations="""
        svm = sum(vm)
        tau * dr/dt + r = vFEFvm_E*svm - vFEFvm_S*max(svm) - vFEFfix*sum(fix) : min=0.0, max=1.0
    """,
    extra_values=params
)

## Neuron of (head-centered) CD signal
CD_Neuron = Neuron(
    name='CD Neuron',
    parameters="""
        tau = 'tau' : population
        w_inh = 'w_CD-inh' : population
        num_neurons = 'num_neurons_4d' : population
    """,
    equations="""
        inh = r * w_inh * num_neurons * mean(r)
        tau * dr/dt + r = sum(FF_FEF) * sum(FF_PC) - inh : min = 0.0, max = 1.0
    """,
    extra_values=params
)

## Neuron of LIP PC
LIPpc_Neuron = Neuron(
    name='LIP_PC Neuron',
    parameters="""
        tau = 'tau' : population
        A = 'A_LIPpc' : population
        D = 'D_LIPpc' : population
        w_inh = 'w_LIPpc-inh' : population
        num_neurons = 'num_neurons_4d' : population
        sigma = 'sigma_LIPpc' : population
    """,
    equations="""
        full_FF = (sum(FF)+sum(FF_FEF)) * pos(A - max(r))*sum(FF_PC)
        inh = (r + D) * w_inh * num_neurons * mean(r)
        tau * dr/dt + r = full_FF + sum(FB)/(sigma+sum(FB))*sum(FF_PC) + sum(exc) - inh : min = 0.0, max = 1.0
    """,
    extra_values=params
)

## Neuron of LIP CD
LIPcd_Neuron = Neuron(
    name='LIP_CD Neuron',
    parameters="""
        A = 'A_LIPcd' : population
        D = 'D_LIPcd : population
        tau = 'tau' : population
        w_inh = 'w_LIPcd-inh' : population
        num_neurons = 'num_neurons_4d' : population
        sigma = 'sigma_LIPcd' : population
    """,
    equations="""
        full_FF = (sum(FF)+sum(FF_FEF)) * (0.1 + pos(A - r)*sum(FF_CD))
        inh = (r + D) * w_inh *  num_neurons * mean(r)
        tau * dr/dt + r = full_FF + sum(FB)/(sigma+sum(FB))*sum(FF_CD) - inh : min = 0.0, max = 1.0
    """,
    extra_values=params
)

## Neuron of Xh
Xh_Neuron = Neuron(
    name='Xh Neuron',
    parameters="""
        D = 'D_Xh' : population
        tau = 'tau' : population
        tau_dep = 'tau_dep_Xh' : population
        d_dep = 'd_dep_Xh' : population
        w_inh = 'w_Xh-inh' : population
        num_neurons = 'num_neurons_2d' : population
        sigma = 'sigma_Xh' : population
    """,
    equations="""
        input = sum(FF_PC) + sum(FF_CD) + sum(inp)
        tau_dep * ds/dt + s = input
        S2 = 1-d_dep*s : min = 0.0, max = 1.0
        inh = (r + D) * w_inh * num_neurons * mean(r)
        tau * dr/dt + r = input/(sigma+input) * S2 + sum(exc) - inh : min = 0.0, max = 1.0
    """,
    extra_values=params
)


##########################################
######### POPULATION DEFINITION  #########
##########################################
AuxA_Pop = Population(name='AuxA', geometry=params['resVisual'], neuron=Aux_Neuron)
V1_Pop = Population(name='V1', geometry=params['V1_shape'], neuron=Inp_Neuron)
V4L4_Pop = Population(name='V4L4', geometry=params['V4L4_shape'], neuron=V4L4_Neuron)
V4L23_Pop = Population(name='V4L23', geometry=params['V4L23_shape'], neuron=V4L23_Neuron)
FEFvm_Pop = Population(name='FEFvm', geometry=params['FEFvm_shape'], neuron=FEFvm_Neuron)
FEFm_Pop = Population(name='FEFm', geometry=params['FEF_shape'], neuron=FEFm_Neuron, stop_condition = "r > SaccadeThreshold : any")
PC_Pop = Population(name='PC signal', geometry=params['resSpatial_2d'], neuron=Inp_Neuron)
CD_Pop = Population(name='CD signal', geometry=params['resSpatial_4d'], neuron=CD_Neuron)
LIPpc_Pop = Population(name='LIP PC', geometry=params['resSpatial_4d'], neuron=LIPpc_Neuron)
LIPcd_Pop = Population(name='LIP CD', geometry=params['resSpatial_4d'], neuron=LIPcd_Neuron)
Xh_Pop = Population(name='Xh', geometry=params['resSpatial_2d'], neuron=Xh_Neuron)

# input populations
V1_Input = TimedArray(name='V1_input', rates=np.zeros((params['tEnd'],)+params['V1_shape']))
FEFv_Pop = TimedArray(name='FEFv', rates=np.zeros((params['tEnd'],)+params['FEF_shape']))
FEFfix = TimedArray(name='FEFfix', rates=np.zeros((params['tEnd'], 1)))
PC_Input = TimedArray(name='PC_input', rates=np.zeros((params['tEnd'],)+params['resSpatial_2d']))
Xh_Input = TimedArray(name='Xh_input', rates=np.zeros((params['tEnd'],)+params['resSpatial_2d']))

## set population parameters
FEFvm_Pop.vFEFvm *= np.linspace(1, 0, params['FEFvm_shape'][-1])[None, None]

##########################################
######### CONNECTION DEFINITION  #########
##########################################

## Input connections
# - input to V1
V1_inp = Projection(V1_Input, V1_Pop, 'inp')
fn = f"{params['ConnDir']}{V1_inp.pre.name}-{V1_inp.post.name}-{V1_inp.target}.data"
if os.path.exists(fn):
    V1_inp.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    V1_inp.connect_one_to_one(1.0)
# - input to PC signal
PC_inp = Projection(PC_Input, PC_Pop, 'inp')
fn = f"{params['ConnDir']}{PC_inp.pre.name}-{PC_inp.post.name}-{PC_inp.target}.data"
if os.path.exists(fn):
    PC_inp.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    PC_inp.connect_one_to_one(1.0)
# - input to Xh
Xh_inp = Projection(Xh_Input, Xh_Pop, 'inp')
fn = f"{params['ConnDir']}{Xh_inp.pre.name}-{Xh_inp.post.name}-{Xh_inp.target}.data"
if os.path.exists(fn):
    Xh_inp.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    Xh_inp.connect_one_to_one(1.0)

## Connections to V4 L4
# - excitation from V1
w = np.ones((params['V4L4_shape'][-1],)+(1,1,)+params['V1_shape'][2:])
V1_V4L4 = Convolution(V1_Pop, V4L4_Pop, 'exc')
V1_V4L4.connect_filters(weights=w, delays=params['delay_V1_V4L4'])
# - feature-based amplification from V4 L2/3
w = Gaussian2D(1.0, params['RFsize23_4'], params['sigma_RF_A_Feat'])[:, :, None]
ssList = []
for Row, Col, Plane in rangeX(params['V4L4_shape']):
    ssList.append([Row // 2, Col // 2, Plane])
V4L23_V4L4A = Convolution(V4L23_Pop, V4L4_Pop, 'A_FEAT', operation='max')
V4L23_V4L4A.connect_filter(weights=w, delays=params['FBA_delay'], keep_last_dimension=True, subsampling=ssList)
# - amplification from FEFvm
# The auxiliary population is used to pool FEFvm activities over different layers. Then a one to many connectivity is used.
# This combination is currently not possible in one step in ANNarchy.
FEFvm_AuxA = Pooling(FEFvm_Pop, AuxA_Pop, 'exc', operation='mean')
FEFvm_AuxA.connect_pooling(extent=(1, 1, params['FEFvm_shape'][-1]))
w = np.ones(params['V4L4_shape'][-1])[:, None, None]
AuxA_V4L4A = Convolution(AuxA_Pop, V4L4_Pop, 'A_SP')
AuxA_V4L4A.connect_filters(weights=w)
# - amplification from LIP PC
LIPpc_V4L4 = Projection(LIPpc_Pop, V4L4_Pop, 'A_LIPpc')
fn = f"{params['ConnDir']}{LIPpc_V4L4.pre.name}-{LIPpc_V4L4.post.name}-{LIPpc_V4L4.target}.data"
if os.path.exists(fn):
    LIPpc_V4L4.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    LIPpc_V4L4.connect_with_func(method=gaussian4dTo3d, mv=params['K_LIPpc-V4L4'], sigma=params['sigma_LIPpc-V4L4'])
# - amplification from LIP CD
LIPcd_V4L4 = Projection(LIPcd_Pop, V4L4_Pop, 'A_LIPcd')
fn = f"{params['ConnDir']}{LIPcd_V4L4.pre.name}-{LIPcd_V4L4.post.name}-{LIPcd_V4L4.target}.data"
if os.path.exists(fn):
    LIPcd_V4L4.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    LIPcd_V4L4.connect_with_func(method=gaussian4dTo3d, mv=params['K_LIPcd-V4L4'], sigma=params['sigma_LIPcd-V4L4'])

## Connections to V4 L2/3
# - excitation from V4 L4
w = Gaussian2D(1.0, params['RFsize4_23'], params['RFsigma4_23'])[:, :, None]
w /= w.sum()
pspText = 'w*pow(pre.r, {p_V4L4-V4L23_ws})'.format(**params)
ssList = []
for Row, Col, Plane in rangeX(params['V4L23_shape']):
    ssList.append([Row * 2 + 1, Col * 2 + 1, Plane])
V4L4_V4L23 = Convolution(V4L4_Pop, V4L23_Pop, 'exc', psp=pspText)
V4L4_V4L23.connect_filter(weights=w, keep_last_dimension=True, subsampling=ssList)

## Connections to FEFvm
# - excitation and suppression from FEFv
# A lowered Gaussian is used to simulate the combined responses
G = Gaussian2D(1.0, params['RFsizev_vm'], params['RFsigmav_vm'])
w = np.tile((G - params['v_FEFv-FEFvm_scale'])[None, :, :], (params['FEFvm_shape'][-1], 1, 1))
w *= params['v_FEFv-FEFvm_factor']**np.arange(6)[:, None, None]
# The plus sign(+) is needed, so that w will not be overwritten
FEFv_FEFvmE = Convolution(FEFv_Pop, FEFvm_Pop, 'E_v')
FEFv_FEFvmE.connect_filters(weights=positive(+w))
FEFv_FEFvmS = Convolution(FEFv_Pop, FEFvm_Pop, 'S_v')
FEFv_FEFvmS.connect_filters(weights=positive(-w))
# - excitation from FEFm, distributing the activity
w = np.ones(params['FEFvm_shape'][-1])[:, None, None]
FEFm_FEFvm = Convolution(FEFm_Pop, FEFvm_Pop, 'E_m')
FEFm_FEFvm.connect_filters(weights=w)

## Connections to FEFm
# - mean pooling from FEFvm
FEFvm_FEFm = Pooling(FEFvm_Pop, FEFm_Pop, 'vm', operation='mean')
FEFvm_FEFm.connect_pooling(extent=(1, 1, params['FEFvm_shape'][-1]))
## - fix from FEFfix
FEFfix_FEFm = Projection(FEFfix, FEFm_Pop, 'fix')
fn = f"{params['ConnDir']}{FEFfix_FEFm.pre.name}-{FEFfix_FEFm.post.name}-{FEFfix_FEFm.target}.data"
if os.path.exists(fn):
    FEFfix_FEFm.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    FEFfix_FEFm.connect_all_to_all(weights=1.0)

## Connections to CD signal
# - FF from PC signal
PCsignal_CDsignal = Projection(PC_Pop, CD_Pop, 'FF_PC')
fn = f"{params['ConnDir']}{PCsignal_CDsignal.pre.name}-{PCsignal_CDsignal.post.name}-{PCsignal_CDsignal.target}.data"
if os.path.exists(fn):
    PCsignal_CDsignal.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    PCsignal_CDsignal.connect_with_func(method=gaussian2dTo4d_v, mv=params['K_PC-CD'], sigma=params['sigma_PC-CD'])
# - FF from FEFm
FEFm_CDsignal = Projection(FEFm_Pop, CD_Pop, 'FF_FEF')
fn = f"{params['ConnDir']}{FEFm_CDsignal.pre.name}-{FEFm_CDsignal.post.name}-{FEFm_CDsignal.target}.data"
if os.path.exists(fn):
    FEFm_CDsignal.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    FEFm_CDsignal.connect_with_func(method=gaussian2dTo4d_h, mv=params['K_FEFm-CD'], sigma=params['sigma_FEFm-CD'])

## Connections to LIP PC
# - FF from V4 L4
V4L4_LIPpc = Projection(V4L4_Pop, LIPpc_Pop, 'FF')
fn = f"{params['ConnDir']}{V4L4_LIPpc.pre.name}-{V4L4_LIPpc.post.name}-{V4L4_LIPpc.target}.data"
if os.path.exists(fn):
    V4L4_LIPpc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    V4L4_LIPpc.connect_with_func(method=gaussian3dTo4d, mv=params['K_V4L4-LIPpc'], sigma=params['sigma_V4L4-LIPpc'])
# - FF from PC signal
PCsignal_LIPpc = Projection(PC_Pop, LIPpc_Pop, 'FF_PC')
fn = f"{params['ConnDir']}{PCsignal_LIPpc.pre.name}-{PCsignal_LIPpc.post.name}-{PCsignal_LIPpc.target}.data"
if os.path.exists(fn):
    PCsignal_LIPpc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    PCsignal_LIPpc.connect_with_func(method=gaussian2dTo4d_v, mv=params['K_PC-LIPpc'], sigma=params['sigma_PC-LIPpc'])
# - FF from FEFvm
# via auxiliary population used already for FEFvm -> V4 L4
FEFvm_LIPpc = Projection(AuxA_Pop, LIPpc_Pop, 'FF_FEF')
fn = f"{params['ConnDir']}{FEFvm_LIPpc.pre.name}-{FEFvm_LIPpc.post.name}-{FEFvm_LIPpc.target}.data"
if os.path.exists(fn):
    FEFvm_LIPpc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    FEFvm_LIPpc.connect_with_func(method=gaussian2dTo4d_h, mv=params['K_FEFvm-LIPpc'], sigma=params['sigma_FEFvm-LIPpc'])
# - FB from Xh
Xh_LIPpc = Projection(Xh_Pop, LIPpc_Pop, 'FB')
fn = f"{params['ConnDir']}{Xh_LIPpc.pre.name}-{Xh_LIPpc.post.name}-{Xh_LIPpc.target}.data"
if os.path.exists(fn):
    Xh_LIPpc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    Xh_LIPpc.connect_with_func(method=gaussian2dTo4d_diag, mv=params['K_Xh-LIPpc'], sigma=params['sigma_Xh-LIPpc'])
# - lateral excitation
LIPpc_exc = Projection(LIPpc_Pop, LIPpc_Pop, 'exc')
fn = f"{params['ConnDir']}{LIPpc_exc.pre.name}-{LIPpc_exc.post.name}-{LIPpc_exc.target}.data"
if os.path.exists(fn):
    LIPpc_exc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    LIPpc_exc.connect_with_func(method=all2all_exp4d, mv=params['w_LIPpc-exc'], sigma=params['sigma_LIPpc-exc'])

## Connections to LIP CD
# - FF from V4 L4
V4L4_LIPcd = Projection(V4L4_Pop, LIPcd_Pop, 'FF')
fn = f"{params['ConnDir']}{V4L4_LIPcd.pre.name}-{V4L4_LIPcd.post.name}-{V4L4_LIPcd.target}.data"
if os.path.exists(fn):
    V4L4_LIPcd.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    V4L4_LIPcd.connect_with_func(method=gaussian3dTo4d, mv=params['K_V4L4-LIPcd'], sigma=params['sigma_V4L4-LIPcd'])
# - FF from CD signal
CDsignal_LIPcd = Projection(CD_Pop, LIPcd_Pop, 'FF_CD')
fn = f"{params['ConnDir']}{CDsignal_LIPcd.pre.name}-{CDsignal_LIPcd.post.name}-{CDsignal_LIPcd.target}.data"
if os.path.exists(fn):
    CDsignal_LIPcd.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    CDsignal_LIPcd.connect_with_func(method=gaussian4d_diagTo4d_v, mv=params['K_CD-LIPcd'], sigma=params['sigma_CD-LIPcd'])
# - FF from FEFvm
# via auxiliary population used already for FEFvm -> V4 L4
FEFvm_LIPcd = Projection(AuxA_Pop, LIPcd_Pop, 'FF_FEF')
fn = f"{params['ConnDir']}{FEFvm_LIPcd.pre.name}-{FEFvm_LIPcd.post.name}-{FEFvm_LIPcd.target}.data"
if os.path.exists(fn):
    FEFvm_LIPcd.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    FEFvm_LIPcd.connect_with_func(method=gaussian2dTo4d_h, mv=params['K_FEFvm-LIPcd'], sigma=params['sigma_FEFvm-LIPcd'])
# - FB from Xh
Xh_LIPcd = Projection(Xh_Pop, LIPcd_Pop, 'FB')
fn = f"{params['ConnDir']}{Xh_LIPcd.pre.name}-{Xh_LIPcd.post.name}-{Xh_LIPcd.target}.data"
if os.path.exists(fn):
    Xh_LIPcd.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    Xh_LIPcd.connect_with_func(method=gaussian2dTo4d_diag, mv=params['K_Xh-LIPcd'], sigma=params['sigma_Xh-LIPcd'])

## to Xh
# - FF from LIP PC
LIPpc_Xh = Projection(LIPpc_Pop, Xh_Pop, 'FF_PC')
fn = f"{params['ConnDir']}{LIPpc_Xh.pre.name}-{LIPpc_Xh.post.name}-{LIPpc_Xh.target}.data"
if os.path.exists(fn):
    LIPpc_Xh.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    LIPpc_Xh.connect_with_func(method=gaussian4dTo2d_diag, mv=params['K_LIPpc-Xh'], sigma=params['sigma_LIPpc-Xh'])
# - FF from LIP CD
LIPcd_Xh = Projection(LIPcd_Pop, Xh_Pop, 'FF_CD')
fn = f"{params['ConnDir']}{LIPcd_Xh.pre.name}-{LIPcd_Xh.post.name}-{LIPcd_Xh.target}.data"
if os.path.exists(fn):
    LIPcd_Xh.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    LIPcd_Xh.connect_with_func(method=gaussian4dTo2d_diag, mv=params['K_LIPcd-Xh'], sigma=params['sigma_LIPcd-Xh'])
# - lateral excitation
Xh_exc = Projection(Xh_Pop, Xh_Pop, 'exc')
fn = f"{params['ConnDir']}{Xh_exc.pre.name}-{Xh_exc.post.name}-{Xh_exc.target}.data"
if os.path.exists(fn):
    Xh_exc.connect_from_file(filename=fn)
    print(" - loaded %s" % fn)
else:
    # could not load connection, therefore create it
    Xh_exc.connect_with_func(method=all2all_exp2d, mv=params['w_Xh-exc'], sigma=params['sigma_Xh-exc'])
