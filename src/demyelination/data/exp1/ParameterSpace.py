"""

"""
import numpy as np
import sys
from defaults.paths import set_project_paths
from fna.tools.utils.system import set_kernel_defaults
from fna.decoders.extractors import set_recording_device

from utils.system import set_system_parameters

N = 1000

# ######################################################################################
# experiments parameters
project_label = 'demyelination'

experiment_label = 'exp1'

# ######################################################################################
# system parameters
system_label = 'Hambach'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
    # 'nuX': np.arange(3, 9, 1),      # rate of background noise (Hz)
    'nuX': np.arange(8, 16),      # rate of background noise (Hz)
    'gamma': np.arange(2, 11, 1),     # E/I weight ratio
    'nTRN': [500, 1000]
}


################################
#def build_parameters(NE, T):
def build_parameters(nuX, gamma, nTRN):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=512000)
    # system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=32, mem=64000, queue="blaustein,hamstein")

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    # Specify network parameters
    N_MGN = 1000
    N_TRN = nTRN

    # synapse parameters
    w = 1.  # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    #g = 5.  # relative inhibitory to excitatory synaptic weight - gamma
    d = 1.5  # synaptic transmission delay (ms)

    neuron_params = {
        'model': 'aeif_cond_exp',
        # 'C_m': 1.0,      # membrane capacity (pF)
        'E_L': -70.,  # resting membrane potential (mV)
        # 'I_e': 0.,       # external input current (pA)
        # 'V_m': 0.,       # membrane potential (mV) generally the resting potential is -70
        # 'V_reset': 10.,  # reset membrane potential after a spike (mV)
        'V_th': -55.,  # spike threshold (mV)
        # 't_ref': 2.0,    # refractory period (ms)
        # 'tau_m': 20.,    # membrane time constant (ms)
        'tau_syn_ex': 2.5, # exc. synaptic time constant
        'tau_syn_in': 10., # exc. synaptic time constant
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
    syn_exc = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w}
    # conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
    conn_exc = {'rule': 'pairwise_bernoulli', 'p': epsilon}
    # I synapses
    syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma * w}
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
        'nuX': nuX * N_MGN * 0.1,  # amplitude
        # 'w_thalamus': w  # in the paper it's about 3*w
        'w_thalamus': 3*w  # in the paper it's about 3*w
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 ('layer_pars', layer_properties),
                 ('noise_pars', noise_pars)
    ])

