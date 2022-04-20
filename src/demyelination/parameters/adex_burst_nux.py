"""

"""
import numpy as np
import sys
from defaults.paths import set_project_paths
from fna.tools.utils.system import set_kernel_defaults
from fna.decoders.extractors import set_recording_device

from utils.system import set_system_parameters

# ######################################################################################
# experiments parameters
project_label = 'demyelination'

# experiment_label = 'exp3'
experiment_label = 'adex_burst_test'

# ######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
    # 'nuX': np.arange(3, 9, 1),      # rate of background noise (Hz)
    # 'nuX': np.arange(8, 16),      # rate of background noise (Hz)
    # 'gamma': np.arange(2, 11, 1),     # E/I weight ratio
    # 'wMGN': np.arange(1., 3.1, 0.5),
    #'nuX_stim': np.arange(5., 50.1, 5)
    # 'nuX_stim': [250., 500., 750., 1000.]

    # 'nuX': [15.],

    # 'alpha': [1.],
    # 'beta': [0.7],
    # 'nuX_TRN': [6.5],
    # 'nuX_MGN': [15.],
    # 'gamma': [8.],
    # 'wMGN': [0.25],
    # 'nuX_stim': [400.]

    'nuX_TRN': np.arange(5., 8.1, 0.5),
    'nuX_MGN': np.arange(8., 16.1, 2.),
    'gamma': np.arange(5., 12.1, 1.),
    'wMGN': np.arange(0.2, 0.41, 0.05),
    'nuX_stim': np.arange(300., 801., 100.)
}


################################
#def build_parameters(NE, T):
# def build_parameters(nuX, gamma, alpha, beta, wMGN, nuX_stim):
def build_parameters(nuX_TRN, nuX_MGN, gamma, wMGN, nuX_stim):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=512000)
    # system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=32, mem=64000, queue="blaustein,hamstein")

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    # Specify network parameters
    N_MGN = 1000
    N_TRN = 1000

    # synapse parameters
    w_input_th = 0.5  # excitatory synaptic weight of background noise onto thalamus (mV)
    wMGN = wMGN  # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    #g = 5.  # relative inhibitory to excitatory synaptic weight - gamma
    d = 1.5  # synaptic transmission delay (ms)

    neuron_params = {
        'model': 'aeif_cond_exp',
        'E_L': -60.,  # resting membrane potential (mV)
        'C_m': 25.0,      # membrane capacity (pF)
        'g_L': 5.0,      # leak conductance
        'V_reset': -51.,  # reset membrane potential after a spike (mV)
        'V_th': -50.,  # spike threshold (mV)
        'tau_syn_ex': 2.5, # exc. synaptic time constant
        'tau_syn_in': 10., # exc. synaptic time constant

        # initial burst + adaptation
        "a": 0.5,
        "b": 7.,
        'tau_w':100.,

        # # rebound burst
        # "a": 80.,
        # "b": 10.,
        # 'tau_w': 720.,
    }

    snn_parameters = {
        'populations': ['MGN', 'TRN'],
        'population_size': [N_MGN, N_TRN],
        'neurons': [neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            ]
        }

    # for simplicity all other parameters are the same, only topology is added
    layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params['model']}

    epsilon = 0.01
    # Connectivity
    # E synapses
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    syn_exc = {'synapse_model': 'static_synapse', 'delay': d, 'weight': wMGN}
    # conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
    conn_exc = {'rule': 'pairwise_bernoulli', 'p': epsilon}
    # I synapses
    #syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma * w_input}
    syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma * wMGN}
    # conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}
    conn_inh = {'rule': 'pairwise_bernoulli', 'p': epsilon}

    # conn_dict = {'rule': 'pairwise_bernoulli',
    #              'mask': {'circular': {'radius': 20.}},
    #              'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.25)
    #              }

    # TGT <- SRC
    topology_snn_synapses = {
        'connect_populations': [('MGN', 'TRN'), ('TRN', 'MGN')],  # within thalamus
        'weight_matrix': [None, None],
        'conn_specs': [conn_inh, conn_exc],
        'syn_specs': [syn_inh, syn_exc]
    }

    noise_pars = {
        # 'nuX': nuX * N_MGN * 0.1,  # amplitude
        'nuX_TRN': nuX_TRN * N_MGN * 0.1,  # amplitude
        'nuX_MGN': nuX_MGN * N_MGN * 0.1,  # amplitude
        'w_noise_thalamus': w_input_th, # in the paper it's about 3*w
        # 'w_noise_mgn': w_input_th * alpha, # in the paper it's about 3*w
        # 'w_noise_trn': w_input_th * beta, # in the paper it's about 3*w
        'w_noise_mgn': w_input_th,  # in the paper it's about 3*w
        'w_noise_trn': w_input_th,  # in the paper it's about 3*w
        'wMGN' : wMGN,
        'nuX_stim' : nuX_stim
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 ('layer_pars', layer_properties),
                 ('noise_pars', noise_pars)
    ])

