
## import libraries
import nest

from fna.networks.snn import SpikingNetwork
from fna.tools.network_architect.connectivity import NESTConnector
from fna.decoders.extractors import set_recording_device

import numpy as np

## neuron parameters

N = 2000 * ctx_increase_scale  # used to be 2000 in the original destexhe model
nMGN = 100 * thl_increase_scale
nTRN = 100 * thl_increase_scale
nEA1 = int(0.8 * N)
nIA1 = int(0.2 * N)

epsilon = 0.02 # Destexhe 2009
scale_eps_intratrn = 2
epsilon_th = epsilon / thl_increase_scale

ctx_epsilon_decrease = 1. # decrease epsilon to 80pc and then grow the 20pc connections with structural plasticity
epsilon_aone = (epsilon / ctx_increase_scale)

# we have distributed the mgn and ea1 into subpopulations
# therefore, their epsilon value has to be raised by the number of subpopulations (in this case 5)
num_subpopulations = 5
epsilon_mgn_ctx = num_subpopulations * (epsilon / ctx_increase_scale)
epsilon_ctx_mgn = num_subpopulations * (epsilon / thl_increase_scale)

epsilon_ctx_trn = epsilon / thl_increase_scale

stim_amplitude = 4.
nuX_th = stim_amplitude * (nMGN * epsilon_th)
wX_TRN = 1.
nuX_aone = stim_amplitude * (nEA1 * epsilon_aone)

gamma = 10. # one gamma for both CTX and THL

w_exc = 4. # unit: nano Siemens (nS)

wMGN = w_exc
w_input_th = w_exc
gamma_th = gamma   # relative inhibitory to excitatory synaptic weight - gamma

w_aone = w_exc
w_input_aone = w_exc
gamma_aone = gamma

w_ctx_trn = w_exc
w_ctx_mgn = w_exc
w_mgn_ctx = w_exc

d = 1.5  # synaptic transmission delay (ms)

MGN_params = {
    'model': 'aeif_cond_exp',
    "a": 40.,
    "b": 0.,
    'tau_w': 600.,
    'Delta_T': 2.5,

    'C_m': 200.,
    'g_L': 10.,
    'V_reset': -60.,
    'V_th': -50.,
    'E_L': -60.,
    'E_in': -80.0,
    'V_m': -60.,
    'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
    'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
    "t_ref": 2.5
}

TRN_params = {
    'model': 'aeif_cond_exp',
    "a": 80.,
    "b": 30.,
    'tau_w': 600.,
    'Delta_T': 2.5,

    'C_m': 200.,
    'g_L': 10.,
    'V_reset': -60.,
    'V_th': -50.,
    'E_L': -60.,
    'E_in': -80.0,
    'V_m': -60.,
    'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
    'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
    "t_ref": 2.5
}

# according to RS neuron parameters in Destexhe, A. (2009). https://doi.org/10.1007/s10827-009-0164-4
# with "strong adaptation"
eA1_params = {
    'model': 'aeif_cond_exp',
    "a": 1.,
    "b": 5.,
    'tau_w': 600.,
    'Delta_T': 2.5,

    'C_m': 200.,
    'g_L': 10.,
    'V_reset': -60.,
    'V_th': -50.,
    'E_L': -60.,
    'E_in': -80.0,
    'V_m': -60.,
    'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
    'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
    "t_ref": 2.5
}

# according to fast-spiking inh neuron parameters in Destexhe, A. (2009). https://doi.org/10.1007/s10827-009-0164-4
iA1_params = {
    'model': 'aeif_cond_exp',
    "a": 1.,
    "b": 0.,
    'tau_w': 600.,
    'Delta_T': 2.5,

    'C_m': 200.,
    'g_L': 10.,
    'V_reset': -60.,
    'V_th': -50.,
    'E_L': -60.,
    'E_in': -80.0,
    'V_m': -60.,
    'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
    'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
    "t_ref": 2.5
}

snn_parameters = {
 'populations': ['MGN', 'TRN', 'eA1', 'iA1'],
 'population_size': [nMGN, nTRN, nEA1, nIA1],
 'neurons': [MGN_params, TRN_params, eA1_params, iA1_params],
 'randomize': [
     {'V_m': (np.random.uniform, {'low': MGN_params['E_L'], 'high': MGN_params['V_th']})},
     {'V_m': (np.random.uniform, {'low': TRN_params['E_L'], 'high': TRN_params['V_th']})},
     {'V_m': (np.random.uniform, {'low': neuron_exc_params_aone['E_L'], 'high': neuron_exc_params_aone['V_th']})},
     {'V_m': (np.random.uniform, {'low': neuron_inh_params_aone['E_L'], 'high': neuron_inh_params_aone['V_th']})},
     ]
}

## connectivity paramters

## structural plasticity parameters

## instantiate network

# TGT <- SRC
topology_snn_synapses = {
    'connect_populations': [('TRN', 'TRN'), ('MGN', 'TRN'), ('TRN', 'MGN'), # within thalamus
                            ('eA1', 'eA1'),
                            ('iA1', 'eA1'), ('iA1', 'iA1'), ('eA1', 'iA1'),  # recurrent A1
                            #('eA1', 'MGN'),
                            ('iA1', 'MGN'),  # thalamocortical
                            #('MGN', 'eA1'),
                            ('TRN', 'eA1'),  # cortico-thalamic
                            ],
    'weight_matrix': [None, None, None,
                    None,
                    None, None, None,
                    #None,
                    None,
                    #None,
                    None
                        ],
    # TODO add rest / update
    'conn_specs': [
                {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': scale_eps_intratrn * epsilon_th}
        ],
    'syn_specs': [syn_inh_mgn, syn_inh_mgn, syn_exc_mgn,
                    syn_exc_aone,
                    syn_exc_aone, syn_inh_aone, syn_inh_aone,
                    #syn_exc_mgn_ctx,
                    syn_exc_mgn_ctx,
                    #syn_exc_ctx_mgn,
                    syn_exc_ctx_trn
                  ]
}

## stimulate

## demyelinate

## enable structural plasticity

## stimulate

