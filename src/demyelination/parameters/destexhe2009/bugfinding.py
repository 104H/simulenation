"""
Nov 3, 2022

There is a difference between the plasticity file and the exp. This file is used to determine that.

This experiment loads a network with connections grown in the eA1.

Scaling up the size of the model.
The exp builds upon trnconn_hetdelays.py

Population Name, Former Size, New Size
TRN, 100, 200
MGN, 100, 200
eA1, 1600, 2000
iA1, 400, 500

In this experiment we apply tonotopic stimimulation from the MGN to the eA1.
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

experiment_label = 'destexhe-bugfinding'


######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

ParameterRange = {
    'T' : [0]
}

################################
def build_parameters(T):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=8, mem=512000)

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    hetdelay_thl = 0
    hetdelay_ctx = 0

    # We selected the following params from the scaleup search
    b = 5.
    nu_x = 4.
    w_exc = 4.


    nuX_stim = 15.
    w_th_stimscale = 15.

    #w_th_stimscale = 0. #20. no stimulation
    #nuX_stim = 1e-12
    stim_dur = 100.

    thl_increase_scale = 2
    ctx_increase_scale = 1.25

    # Specify network parameters
    N = 2000 * ctx_increase_scale  # used to be 2000 in the original destexhe model
    nMGN = 100 * thl_increase_scale
    nTRN = 100 * thl_increase_scale
    nEA1 = int(0.8 * N)
    nIA1 = int(0.2 * N)

    # Connectivity
    # the model has been scaled up, so the global epsilon is not used anymore
    # however the stim_amplitude var is dependent on it, so we will keep it uncommented
    # the network will be connected based on a loaded experiment therefore we make epsilon zero here
    epsilon = 0.02 # Destexhe 2009

    scale_eps_intratrn = 2

    epsilon_th = epsilon / thl_increase_scale
    epsilon_aone = epsilon / ctx_increase_scale

    epsilon_mgn_ctx = epsilon / ctx_increase_scale
    epsilon_ctx_trn = epsilon / thl_increase_scale
    epsilon_ctx_mgn = epsilon / thl_increase_scale

    stim_amplitude = nu_x
    nuX_th = stim_amplitude * (nMGN * epsilon_th)
    wX_TRN = 1.
    nuX_aone = stim_amplitude * (nEA1 * epsilon_aone)

    # we set epsilon for recurrent conn to 0 because the conn will be loaded
    # this is intentionaly done here because epsilon is used in the computation of nux above
    #epsilon_th = 0
    #epsilon_aone = 0

    #w_exc = 6. # excitatory weight

    gamma = 10. # one gamma for both CTX and THL

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
    neuron_exc_params_aone = {
        'model': 'aeif_cond_exp',
        "a": 1.,
        # "b": 40.,
        "b": b,
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
    neuron_inh_params_aone = {
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

    ''' Plasticity Parameters
    # for an unknown reason it overshoots the Ca level by .025 in the gaussian case and .05 in the linear case
    eCa = (13 / 100) # target calium level = target activity / 100
    iCa = (17 / 100) / 2
    update_interval = 1.

    gr_scaling = .001
    g_curve = 'linear'
    # Excitatory synaptic elements of excitatory neurons
    growth_curve_e_e = {
        'growth_curve': g_curve,
        'growth_rate': 1 * gr_scaling,  # (elements/ms)
        'continuous': False,
        #'eta': 0.0,  # Ca2+
        'eps': eCa,  # Ca2+
    }

    # Inhibitory synaptic elements of excitatory neurons
    growth_curve_e_i = {
        'growth_curve': g_curve,
        'growth_rate': 0 * gr_scaling,  # (elements/ms)
        'continuous': False,
        #'eta': 0.0,  # Ca2+
        #'eps': growth_curve_e_e['eps'],  # Ca2+
        'eps': growth_curve_e_e['eps'],  # Ca2+
    }

    # Excitatory synaptic elements of inhibitory neurons
    growth_curve_i_e = {
        'growth_curve': g_curve,
        'growth_rate': 0 * gr_scaling,  # (elements/ms)
        'continuous': False,
        #'eta': 0.0,  # Ca2+
        'eps': iCa,  # Ca2+
    }

    # Inhibitory synaptic elements of inhibitory neurons
    growth_curve_i_i = {
        'growth_curve': g_curve,
        'growth_rate': 0 * gr_scaling,  # (elements/ms)
        'continuous': False,
        #'eta': 0.0,  # Ca2+
        'eps': growth_curve_i_e['eps']  # Ca2+
    }

    growth_curves = {
                'growth_curve_e_e' : growth_curve_e_e,
                'growth_curve_e_i' : growth_curve_e_i,
                'growth_curve_i_e' : growth_curve_i_e,
                'growth_curve_i_i' : growth_curve_i_i
            }
    #'''
    
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

    # E synapses
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    '''
    if (hetdelay_thl):
        syn_exc_mgn = {'synapse_model': 'static_synapse', 'delay': nest.random.uniform(min=1., max=3.), 'weight': wMGN}
        syn_inh_mgn = {'synapse_model': 'static_synapse', 'delay': nest.random.uniform(min=1., max=3.), 'weight': - gamma_th * wMGN}
    if not(hetdelay_thl):
    '''
    syn_exc_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': wMGN}
    syn_inh_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_th * wMGN}

    conn_exc_mgn = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_th}

    conn_inh_mgn = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': 2 * epsilon_th}
    conn_intra_trn = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': scale_eps_intratrn * epsilon_th}

    syn_inh_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - gamma_aone * w_aone}
    syn_exc_aone = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_aone}

    conn_inh_aone = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    '''
    conn_exc_aone_to_iA1 = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_aone}
    conn_exc_aone_intra_eA1 = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_aone * ctx_epsilon_decrease}
    '''
    conn_exc_aone_recurrant = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0}
    conn_exc_aone = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_aone}

    # thalamocortical projections: to both eA1 and iA1
    syn_exc_mgn_ctx = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_mgn_ctx}
    conn_exc_mgn_ctx = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx}

    # CORTICO-THALAMIC projections
    syn_exc_ctx_trn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_trn}
    syn_exc_ctx_mgn = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w_ctx_mgn}
    conn_exc_ctx_trn = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_trn}
    conn_exc_ctx_mgn = {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_mgn}

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
        'conn_specs': [conn_intra_trn, conn_inh_mgn, conn_exc_mgn,
                        conn_exc_aone_intra_eA1,
                        conn_exc_aone_to_iA1, conn_inh_aone, conn_inh_aone,
                        #conn_exc_mgn_ctx,
                        conn_exc_mgn_ctx,
                        #conn_exc_ctx_mgn,
                        conn_exc_ctx_trn,
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


    noise_pars = {
        'nuX_th': nuX_th,  # amplitude
        'w_noise_stim': w_input_th * w_th_stimscale,  # in the paper it's about 3*w
        'w_noise_mgn': w_input_th,
        'w_noise_trn': w_input_th * wX_TRN,
        'w_noise_ctx' : w_input_aone,
        'nuX_aone' : nuX_aone,
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
                 ('hetdelay_pars', {'hetdelay_ctx':hetdelay_ctx, 'hetdelay_thl':hetdelay_thl}),
                 ('mgn_ctx_pars', {'conn' : conn_exc_mgn_ctx, 'syn' : syn_exc_mgn_ctx}),
                 ('ctx_mgn_pars', {'conn' : conn_exc_ctx_mgn, 'syn' : syn_exc_ctx_mgn})
    ])

