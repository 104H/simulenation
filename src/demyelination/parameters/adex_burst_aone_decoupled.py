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
experiment_label = 'fullmodel_decoupled'

 ######################################################################################
# system parameters
system_label = 'local'
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
    # 'nuX_aone': np.arange(3., 10.1, 1.),
    # 'gamma_aone': np.arange(5., 10.1, 1.),
    'nuX_aone': [3.],
    'gamma_aone': [6.]
}


################################
def build_parameters(nuX_aone, gamma_aone):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=512000)
    # system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=32, mem=64000, queue="blaustein,hamstein")

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    nuX_th = 3.5
    nuX_stim = 700.
    gamma_th = 8.   # relative inhibitory to excitatory synaptic weight - gamma
    wMGN = 0.6      # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    sigma_MGN = 0.3
    sigma_TRN = 0.3
    wX_TRN = 0.5

    w_aone = 0.7
    w_ctx_trn = 0.     # TODO not set in stone
    w_ctx_mgn = 0.0    # TODO not set in stone
    w_mgn_ctx = 0.      # TODO not set in stone
    # gamma_aone = make_Variable
    # nuX_aone =  make_Variable

    # Specify network parameters
    N_MGN = 1000
    N_TRN = 1000
    nEA1 = 2000
    nIA1 = 500

    # synapse parameters
    w_input_th = 0.9  # excitatory synaptic weight of background noise onto thalamus (mV)
    w_input_a1 = 1.  # excitatory synaptic weight of background noise onto A1 (mV)  ?
    d = 1.5  # synaptic transmission delay (ms)

    neuron_params_aone = {
        'model': 'aeif_cond_exp',
        'E_L': -70.,  # resting membrane potential (mV)
        'C_m': 150.0,  # membrane capacity (pF)
        'g_L': 10.0,  # leak conductance  - in combo with C_m you get tau_m = ~15 ms
        'V_reset': -60.,  # reset membrane potential after a spike (mV)
        'V_th': -55.,  # spike threshold (mV)
        'tau_syn_ex': 5.,  # exc. synaptic time constant
        'tau_syn_in': 10.,  # exc. synaptic time constant

        # initial burst + adaptation
        "a": 2.,
        "b": 60.,
        'tau_w': 200.,
    }

    neuron_params_thl = {
        'model': 'aeif_cond_exp',
        'E_L': -60.,  # resting membrane potential (mV) - see refs
        'C_m': 50.0,      # membrane capacity (pF)
        'g_L': 5.0,      # leak conductance  - see refs
        'V_reset': -52.,  # reset membrane potential after a spike (mV)  - for bustiness
        'V_th': -50.,  # spike threshold (mV)
        'tau_syn_ex': 2.5, # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10., # exc. synaptic time constant  - mit paper

        # initial burst + adaptation
        "a": 0.5,
        "b": 10.,
        'tau_w': 150.,
    }


    snn_parameters = {
        'populations': ['MGN', 'TRN', 'eA1', 'iA1'],
        'population_size': [N_MGN, N_TRN, nEA1, nIA1],
        'neurons': [neuron_params_thl, neuron_params_thl, neuron_params_aone, neuron_params_aone],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params_thl['E_L'], 'high': neuron_params_thl['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params_thl['E_L'], 'high': neuron_params_thl['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params_aone['E_L'], 'high': neuron_params_aone['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params_aone['E_L'], 'high': neuron_params_aone['V_th']})},
            ]
        }

    # for simplicity all other parameters are the same, only topology is added
    # TODO we may use this later, but not now
    layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params_aone['model']}

    # Connectivity
    epsilon_th = 0.01
    epsilon_aone = 0.1
    epsilon_mgn_ctx = 0.05  # TODO look up literature ?
    epsilon_ctx_trn = 0.05  # TODO look up literature ?
    epsilon_ctx_mgn = 0.05  # TODO look up literature ?

    # E synapses
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    syn_exc_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': wMGN}
    conn_exc_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}

    # I synapses
    syn_inh_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_th * wMGN}
    conn_inh_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}

    syn_inh_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_aone * w_aone}
    conn_inh_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    syn_exc_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_aone}
    conn_exc_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    # thalamocortical projections: to both eA1 and iA1
    syn_exc_mgn_ctx = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_mgn_ctx}
    conn_exc_mgn_ctx = {'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx}

    # CORTICO-THALAMIC projections
    syn_exc_ctx_trn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_trn}
    syn_exc_ctx_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_mgn}
    conn_exc_ctx_trn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_trn}
    conn_exc_ctx_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_mgn}

    # conn_dict = {'rule': 'pairwise_bernoulli',
    #              'mask': {'circular': {'radius': 20.}},
    #              'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.25)
    #              }

    # TGT <- SRC
    topology_snn_synapses = {
        'connect_populations': [('MGN', 'TRN'), ('TRN', 'MGN'),  # within thalamus
                                ('eA1', 'MGN'), ('iA1', 'MGN'),  # thalamocortical
                                ('eA1', 'eA1'), ('iA1', 'eA1'), ('iA1', 'iA1'), ('eA1', 'iA1'),  # recurrent A1
                                ('MGN', 'eA1'), ('TRN', 'eA1'),  # cortico-thalamic
                                ],
        'weight_matrix': [None, None, None, None, None, None, None, None, None, None],
        # TODO add rest / update
        'conn_specs': [conn_inh_mgn, conn_exc_mgn,
                        conn_exc_mgn_ctx, conn_exc_mgn_ctx,
                        conn_exc_aone, conn_exc_aone, conn_inh_aone, conn_inh_aone,
                        conn_exc_ctx_mgn, conn_exc_ctx_trn],
        'syn_specs': [syn_inh_mgn, syn_exc_mgn,
                        syn_exc_mgn_ctx, syn_exc_mgn_ctx,
                        syn_exc_aone, syn_exc_aone, syn_inh_aone, syn_inh_aone,
                        syn_exc_ctx_mgn, syn_exc_ctx_trn]
    }


    noise_pars = {
        'nuX_th': nuX_th * N_MGN * 0.1,  # amplitude
        'w_noise_stim': w_input_th,  # in the paper it's about 3*w
        'w_noise_mgn': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, N_MGN),
        'w_noise_trn': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, N_TRN),
        'w_noise_ctx' : w_input_a1,
        'nuX_aone' : nuX_aone * nEA1 * 0.1,
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

