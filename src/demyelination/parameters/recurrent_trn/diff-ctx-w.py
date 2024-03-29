"""
Aug 15, 2022

A different approach which is connectivity TRN->TRN and denser TRN->MGN connectivity.
The connectivity between the MGN and the eA1 in this file is tonotopic.
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

experiment_label = 'recurr-trn-tono-diff-ctx-w'

 ######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
    "nuX_stim": [10., 20., 50.],
    "b": [40., 20., 10., 5.],
    # "b": [40.],
    # "nu_x": np.arange(2, 6, 2),
    'w_mgn_ectx' : [0., 3., 4., 5., 6., 15.],
}

################################
def build_parameters(nuX_stim, b, w_mgn_ectx):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=1, mem=512000)

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    stim_dur = 100.

    # Specify network parameters
    nMGN = 500
    nTRN = 500
    nEA1 = 2000
    nIA1 = 500

    #''' Thalamus Params
    nuX_th = 10.
    gamma_th = 10.   # relative inhibitory to excitatory synaptic weight - gamma
    wMGN = 2.2 # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    sigma_MGN = 0.2
    sigma_TRN = 0.2
    wX_TRN = 1.3

    nuX_aone = 5.
    gamma_aone = 9.
    w_aone = 3. # epsp 0.26

    w_input_aone = 3. # used to be 1. # excitatory synaptic weight of background noise onto A1 (mV)  ?
    w_ctx_trn = 0.3
    w_ctx_mgn = 0.3
    #w_mgn_ectx = 3.
    w_mgn_ictx = 3.

    w_input_th = 1. # excitatory synaptic weight of background noise onto thalamus (mV)
    d = 1.5  # synaptic transmission delay (ms)

    MGN_params = {
        'model': 'aeif_cond_exp',
        "a": 40.,
        "b": 0.,
        'tau_w': 150.,
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -55.,
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
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -55.,
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
    neuron_exc_params_aone = {
        'model': 'aeif_cond_exp',
        "a": 1.,
        "b": b,
        'tau_w': 150., # this was part of a search
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,
        'E_in': -80.0, # inhibitory reverse potential
        'V_m': -60.,
        'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
        "t_ref": 2.5
    }

    # according to fast-spiking inh neuron parameters in Destexhe, A. (2009). https://doi.org/10.1007/s10827-009-0164-4
    neuron_inh_params_aone = {
        'model': 'aeif_cond_exp',
        "a": 1.,
        "b": 0.,            # no spike-based adaptation
        'tau_w': 600.,
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,  # this is changed
        'E_in': -80.0,
        'V_m': -60.,
        'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
        "t_ref": 2.5
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
    
    epsilon_mgn_ctx = 0.026 # TODO look up literature ?
    epsilon_ctx_trn = 0.03  # TODO look up literature ?
    epsilon_ctx_mgn = 0.03 # TODO look up literature ?

    # E synapses
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    syn_exc_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': wMGN}
    conn_exc_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}

    # I synapses
    syn_inh_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_th * wMGN}
    conn_inh_trn_trn = {'rule': 'pairwise_bernoulli', 'p': epsilon_th}
    conn_inh_trn_mgn = {'rule': 'pairwise_bernoulli', 'p': 2 * epsilon_th}

    syn_inh_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_aone * w_aone}
    conn_inh_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    syn_exc_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_aone}
    conn_exc_aone = {'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    # thalamocortical projections: to eA1 and iA1
    syn_exc_mgn_ictx = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_mgn_ictx}
    syn_exc_mgn_ectx = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_mgn_ectx}
    conn_exc_mgn_ctx = {'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx}

    # CORTICO-THALAMIC projections
    syn_exc_ctx_trn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_trn}
    syn_exc_ctx_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_mgn}
    conn_exc_ctx_trn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_trn}
    conn_exc_ctx_mgn = {'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_mgn}

    # TGT <- SRC
    topology_snn_synapses = {
        'connect_populations': [('TRN', 'TRN'), ('MGN', 'TRN'), ('TRN', 'MGN'), # within thalamus
                                ('eA1', 'eA1'), ('iA1', 'eA1'), ('iA1', 'iA1'), ('eA1', 'iA1'),  # recurrent A1
                                #('eA1', 'MGN'),
                                ('iA1', 'MGN'),  # thalamocortical
                                #('MGN', 'eA1'),
                                ('TRN', 'eA1'),  # cortico-thalamic
                                ],
        'weight_matrix': [None, None, None,
                        None, None, None, None,
                        None,
                        None
                            ],
        # TODO add rest / update
        'conn_specs': [conn_inh_trn_trn, conn_inh_trn_mgn, conn_exc_mgn,
                        conn_exc_aone, conn_exc_aone, conn_inh_aone, conn_inh_aone,
                        conn_exc_mgn_ctx,
                        conn_exc_ctx_trn,
                        ],
        'syn_specs': [syn_inh_mgn, syn_inh_mgn, syn_exc_mgn,
                        syn_exc_aone, syn_exc_aone, syn_inh_aone, syn_inh_aone,
                        syn_exc_mgn_ictx,
                        syn_exc_ctx_trn
                      ]
    }


    noise_pars = {
        'nuX_th': nuX_th * nMGN * 0.1,  # amplitude
        'w_noise_stim': w_input_th,  # in the paper it's about 3*w
        'w_noise_mgn': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, nMGN),
        'w_noise_trn': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, nTRN),
        'w_noise_ctx' : w_input_aone,
        'nuX_aone' : nuX_aone * nEA1 * 0.1,
        'nuX_stim' : nuX_stim * nMGN * 0.1,
        'stim_duration' : stim_dur
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 #('layer_pars', layer_properties),
                 ('noise_pars', noise_pars),
                 #('growth_pars', growth_curves)
                 ('mgn_ctx_pars', {'conn' : conn_exc_mgn_ctx, 'syn' : syn_exc_mgn_ectx}),
                 ('ctx_mgn_pars', {'conn' : conn_exc_ctx_mgn, 'syn' : syn_exc_ctx_mgn})
    ])

