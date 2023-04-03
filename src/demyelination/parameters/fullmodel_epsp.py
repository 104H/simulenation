"""
June 30, 2022
The weights of the model lead to very small EPSPs. This series of experiments is done to explore weights which lead to high EPSPs so that the activity in the model is not dominated by the background noise.
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

experiment_label = 'fullmodel-sameepsp'

 ######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
        'T' : np.arange(0, 1., 1.),
        'nuX_th' : np.arange(5., 26., 1.),
        'w_input_aone' : np.arange(5., 26., 1.)
}


################################
def build_parameters(T, nuX_th, w_input_aone):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=1, mem=512000)

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    # Specify network parameters
    nMGN = 500
    nTRN = 500
    nEA1 = 2000
    nIA1 = 500

    #''' Thalamus Params
    #nuX_th = 15
    nuX_stim = 45
    gamma_th = 1.   # relative inhibitory to excitatory synaptic weight - gamma
    wMGN = .35      # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    sigma_MGN = 0.2
    sigma_TRN = 0.2
    wX_TRN = 1.3

    nuX_aone = 13.
    gamma_aone = 1.
    w_aone = 6.

    #w_input_aone = 15. # used to be 1. # excitatory synaptic weight of background noise onto A1 (mV)  ?
    w_ctx_trn = 0.35
    w_ctx_mgn = 0.35
    w_mgn_ctx = 6.

    # synapse parameters
    w_input_th = 1.  # excitatory synaptic weight of background noise onto thalamus (mV)
    d = 1.5  # synaptic transmission delay (ms)

    MGN_params = {
        'model': 'aeif_cond_exp',
        'E_L': -60.,  # resting membrane potential (mV) - see refs
        'C_m': 150.0,      # membrane capacity (pF)
        'g_L': 10.0,      # leak conductance  - see refs
        'V_reset': -55.,  # reset membrane potential after a spike (mV)  - for burstiness
        'V_th': -50.,  # spike threshold (mV)
        'tau_syn_ex': 2.5, # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10., # exc. synaptic time constant  - mit paper

        "a": 40.,
        "b": 0.,
        'tau_w': 150.,
    }

    TRN_params = {
        'model': 'aeif_cond_exp',
        'E_L': -60.,  # resting membrane potential (mV) - see refs
        'C_m': 150.0,      # membrane capacity (pF)
        'g_L': 10.0,      # leak conductance  - see refs
        'V_reset': -55.,  # reset membrane potential after a spike (mV)  - for burstiness
        'V_th': -50.,  # spike threshold (mV)
        'tau_syn_ex': 2.5, # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10., # exc. synaptic time constant  - mit paper
        "a": 30.,
        "b": 80.,
        'tau_w': 300.
    }

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

    ''' Plasticity Parameters
    update_interval = 100.

    gr_scaling = .001
    #g_curve = 'linear'
    # Excitatory synaptic elements of excitatory neurons
    growth_curve_e_e_den = {
        'growth_curve': "gaussian",
        'growth_rate': 10 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': eCa,  # Ca2+
    }

   # Excitatory synaptic elements of excitatory neurons
    growth_curve_e_e = {
        'growth_curve': "gaussian",
        'growth_rate': 20 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': eCa,  # Ca2+
    }

    # Inhibitory synaptic elements of excitatory neurons
    growth_curve_e_i = {
        'growth_curve': "gaussian",
        'growth_rate': 20 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        #'eps': growth_curve_e_e['eps'],  # Ca2+
        'eps': 1.0,  # Ca2+
    }

    # Excitatory synaptic elements of inhibitory neurons
    growth_curve_i_e = {
        'growth_curve': "gaussian",
        'growth_rate': 10 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': iCa,  # Ca2+
    }

    # Inhibitory synaptic elements of inhibitory neurons
    growth_curve_i_i = {
        'growth_curve': "gaussian",
        'growth_rate': 20 * gr_scaling,  # (elements/ms)
        'continuous': False,
        'eta': 0.0,  # Ca2+
        'eps': growth_curve_i_e['eps']  # Ca2+
    }

    growth_curves = {
                'growth_curve_e_e_den' : growth_curve_e_e_den,
                'growth_curve_e_e' : growth_curve_e_e,
                'growth_curve_e_i' : growth_curve_e_i,
                'growth_curve_i_e' : growth_curve_i_e,
                'growth_curve_i_i' : growth_curve_i_i
            }
    '''

    snn_parameters = {
    'populations': ['MGN', 'TRN', 'eA1', 'iA1'],
    'population_size': [nMGN, nTRN, nEA1, nIA1],
    'neurons': [MGN_params, TRN_params, neuron_exc_params_aone, neuron_inh_params_aone],
    'randomize': [
        {'V_m': (np.random.uniform, {'low': MGN_params['E_L'], 'high': MGN_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': TRN_params['E_L'], 'high': TRN_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_exc_params_aone['E_L'], 'high': neuron_exc_params_aone['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_inh_params_aone['E_L'], 'high': neuron_inh_params_aone['V_th']})},
        ]
    }

    # for simplicity all other parameters are the same, only topology is added
    # TODO we may use this later, but not now
    #layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params_aone['model']}

    # Connectivity
    epsilon_th = 0.02 # Destexhe 2009
    epsilon_aone = 0.1
    
    epsilon_mgn_ctx = 0.026  # TODO look up literature ?
    epsilon_ctx_trn = 0.03  # TODO look up literature ?
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

    # TGT <- SRC
    topology_snn_synapses = {
        'connect_populations': [('MGN', 'TRN'), ('TRN', 'MGN'), # within thalamus
                                ('eA1', 'eA1'), ('iA1', 'eA1'), ('iA1', 'iA1'), ('eA1', 'iA1'),  # recurrent A1
                                ('eA1', 'MGN'), ('iA1', 'MGN'),  # thalamocortical
                                ('MGN', 'eA1'), ('TRN', 'eA1'),  # cortico-thalamic
                                ],
        'weight_matrix': [None, None,
                        None, None, None, None,
                        None, None,
                        None, None
                            ],
        # TODO add rest / update
        'conn_specs': [conn_inh_mgn, conn_exc_mgn,
                        conn_exc_aone, conn_exc_aone, conn_inh_aone, conn_inh_aone,
                        conn_exc_mgn_ctx, conn_exc_mgn_ctx,
                        conn_exc_ctx_mgn, conn_exc_ctx_trn,
                        ],
        'syn_specs': [syn_inh_mgn, syn_exc_mgn,
                        syn_exc_aone, syn_exc_aone, syn_inh_aone, syn_inh_aone,
                        syn_exc_mgn_ctx, syn_exc_mgn_ctx,
                        syn_exc_ctx_mgn, syn_exc_ctx_trn
                      ]
    }


    noise_pars = {
        'nuX_th': nuX_th * nMGN * 0.1,  # amplitude
        'w_noise_stim': w_input_th,  # in the paper it's about 3*w
        'w_noise_mgn': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, nMGN),
        'w_noise_trn': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, nTRN),
        'w_noise_ctx' : w_input_aone,
        'nuX_aone' : nuX_aone * nEA1 * 0.1,
        'nuX_stim' : nuX_stim * nMGN * 0.1
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 #('layer_pars', layer_properties),
                 ('noise_pars', noise_pars),
                 #('growth_pars', growth_curves)
    ])

