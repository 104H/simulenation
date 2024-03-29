"""
Aug 2, 2022

Updated synaptic time constants.
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

experiment_label = 'aone_lowergamma-syntau'

 ######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
        'nuX_aone' : np.arange(5, 21, 1),
        'gamma_aone' : np.arange(2, 10, 1),
}


################################
def build_parameters(nuX_aone, gamma_aone):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=8, mem=512000)

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    ''' Thalamus Params
    nuX_th = 3.5
    nuX_stim = 700.
    gamma_th = 9.   # relative inhibitory to excitatory synaptic weight - gamma
    wMGN = 0.6      # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    sigma_MGN = 0.3
    sigma_TRN = 0.3
    wX_TRN = 0.05
    '''

    #nuX_aone = 10. # this gives ~3Hz
    #nuX_aone = 1. # used to be 10.
    #gamma_aone = 2.
    w_aone = 3. #1.

    w_input_aone = 15. # used to be 1. # excitatory synaptic weight of background noise onto A1 (mV)  ?
    #w_ctx_trn = 0.08
    #w_ctx_mgn = 0.04
    #w_mgn_ctx = 0.5

    # Specify network parameters
    #N_MGN = 1000
    #N_TRN = 1000
    nEA1 = 2000
    nIA1 = 500

    # synapse parameters
    #w_input_th = 0.9  # excitatory synaptic weight of background noise onto thalamus (mV)
    d = 1.5  # synaptic transmission delay (ms)

    # previously
    # neuron_params_aone = {
    #     'model': 'aeif_cond_exp',
    #     'E_L': -70.,  # resting membrane potential (mV)
    #     'C_m': 150.0,  # membrane capacity (pF)
    #     'g_L': 10.0,  # leak conductance  - in combo with C_m you get tau_m = ~15 ms
    #     'V_reset': -60.,  # reset membrane potential after a spike (mV)
    #     'V_th': -55.,  # spike threshold (mV)
    #     'tau_syn_ex': 5.,  # exc. synaptic time constant
    #     'tau_syn_in': 10.,  # exc. synaptic time constant
    #
    #     # initial burst + adaptation
    #     "a": 2.,
    #     "b": 60.,
    #     'tau_w': 200.,
    #
    #     "beta_Ca" : 0.0001
    # }

    # according to RS neuron parameters in Destexhe, A. (2009). https://doi.org/10.1007/s10827-009-0164-4
    # with "strong adaptation"
    neuron_exc_params_aone = {
        'model': 'aeif_cond_exp',
        "a": 2.,
        "b": 40.,
        'tau_w': 150.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,
        'V_m': -60.,
    }

    # according to fast-spiking inh neuron parameters in Destexhe, A. (2009). https://doi.org/10.1007/s10827-009-0164-4
    neuron_inh_params_aone = {
        'model': 'aeif_cond_exp',
        "a": 2.,
        "b": 0.,            # no spike-based adaptation
        'tau_w': 600.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,  # this is changed
        'V_m': -60.,
    }

    #''' Plasticity Parameters
    '''
    update_interval = 100.

    gr_scaling = .00001
    # Excitatory synaptic elements of excitatory neurons
    growth_curve_e_e = {
        'growth_curve': "gaussian",
        'growth_rate': 100 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': .01,  # Ca2+
        'eps': 0.095,  # Ca2+
    }

    # Inhibitory synaptic elements of excitatory neurons
    growth_curve_e_i = {
        'growth_curve': "gaussian",
        'growth_rate': 100 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': .01,  # Ca2+
        #'eps': growth_curve_e_e['eps'],  # Ca2+
        'eps': growth_curve_e_e['eps'],  # Ca2+
    }

    # Excitatory synaptic elements of inhibitory neurons
    growth_curve_i_e = {
        'growth_curve': "gaussian",
        'growth_rate': 1 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': 0.175,  # Ca2+
    }

    # Inhibitory synaptic elements of inhibitory neurons
    growth_curve_i_i = {
        'growth_curve': "gaussian",
        'growth_rate': 1 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': growth_curve_i_e['eps']  # Ca2+
    }

    growth_curves = {
                'growth_curve_e_e' : growth_curve_e_e,
                'growth_curve_e_i' : growth_curve_e_i,
                'growth_curve_i_e' : growth_curve_i_e,
                'growth_curve_i_i' : growth_curve_i_i
            }
    '''

    snn_parameters = {
    'populations': ['eA1', 'iA1'],
    'population_size': [nEA1, nIA1],
    'neurons': [neuron_exc_params_aone, neuron_inh_params_aone],
    'randomize': [
        {'V_m': (np.random.uniform, {'low': neuron_exc_params_aone['E_L'], 'high': neuron_exc_params_aone['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_inh_params_aone['E_L'], 'high': neuron_inh_params_aone['V_th']})},
        ]
    }

    # for simplicity all other parameters are the same, only topology is added
    # TODO we may use this later, but not now
    #layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params_aone['model']}

    # Connectivity
    epsilon_aone = 0.1
    '''
    epsilon_th = 0.01
    epsilon_mgn_ctx = 0.05  # TODO look up literature ?
    epsilon_ctx_trn = 0.03  # TODO look up literature ?
    epsilon_ctx_mgn = 0.05  # TODO look up literature ?
    '''

    # E synapses
    '''
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    syn_exc_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': wMGN}
    conn_exc_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}

    # I synapses
    syn_inh_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_th * wMGN}
    conn_inh_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}
    '''

    syn_inh_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_aone * w_aone}
    conn_inh_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    syn_exc_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_aone}
    conn_exc_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    '''
    # thalamocortical projections: to both eA1 and iA1
    syn_exc_mgn_ctx = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_mgn_ctx}
    conn_exc_mgn_ctx = {'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx}

    # CORTICO-THALAMIC projections
    syn_exc_ctx_trn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_trn}
    syn_exc_ctx_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_mgn}
    conn_exc_ctx_trn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_trn}
    conn_exc_ctx_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_mgn}
    '''

    # TGT <- SRC
    topology_snn_synapses = {
        'connect_populations': [('eA1', 'eA1'), ('iA1', 'eA1'), ('iA1', 'iA1'), ('eA1', 'iA1'),  # recurrent A1
                                ],
        'weight_matrix': [None, None, None, None],
        # TODO add rest / update
        'conn_specs': [conn_exc_aone, conn_exc_aone, conn_inh_aone, conn_inh_aone,
        #'conn_specs': [None, None, None, None,
                        ],
        'syn_specs': [syn_exc_aone, syn_exc_aone, syn_inh_aone, syn_inh_aone,
        #'syn_specs': [None, None, None, None,
                      ]
    }


    noise_pars = {
        #'nuX_th': nuX_th * N_MGN * 0.1,  # amplitude
        #'w_noise_stim': w_input_th,  # in the paper it's about 3*w
        #'w_noise_mgn': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, N_MGN),
        #'w_noise_trn': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, N_TRN),
        'w_noise_ctx' : w_input_aone,
        'nuX_aone' : nuX_aone * nEA1 * 0.1,
        #'wMGN' : wMGN,
        #'nuX_stim' : nuX_stim
        'gamma_aone' : gamma_aone,
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 #('layer_pars', layer_properties),
                 ('noise_pars', noise_pars),
                 #('growth_pars', growth_curves)
    ])
