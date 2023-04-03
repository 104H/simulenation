import numpy as np
from scipy import stats
from defaults.paths import set_project_paths
from fna.tools.utils.system import set_kernel_defaults

from utils.system import set_system_parameters
from utils.helper import get_weight
from parameters.defaults import set_network_defaults, default_spiking_encoder, \
    default_inputs, set_neuron_defaults

# ######################################################################################
# experiments parameters
project_label = 'tdl_csi'
experiment_label = 'hunaid-full-fb-stim'
system_label = 'local'

# ################################
ParameterRange = {
    "nuX_stim": [10., 20., 50.],
    "b": [40., 20., 10., 5.],
    # "b": [40.],
    # "nu_x": np.arange(2, 6, 2),
    "tau_w_a1": [150.],
    'T': np.arange(1, 2)
}


# ################################
def build_parameters(nuX_stim, b, tau_w_a1, T):
    system_params = set_system_parameters(cluster=system_label, queue='blaustein', walltime='05-00:00:00', nodes=1,
                                          ppn=12)
    paths = set_project_paths(system=system_label, project_label=project_label)

    # ############################################################
    # Simulation parameters
    sim_dt = 0.1
    kernel_pars = set_kernel_defaults(resolution=sim_dt,
                                      run_type=system_label,
                                      data_label=experiment_label,
                                      data_paths=paths,
                                      **system_params)

    # ###########################################################
    # Network parameters
    N = 2500
    nE = int(0.8 * N)
    nI = int(0.2 * N)
    nMGN = 500
    nTRN = 500
    delay = 1.5

    epsilon_th = 0.02
    epsilon_ctx = 0.1
    # N/D - 3% is somewhat larger than Destexhe, on avg. each thalamic cell gets K=nE*0.03 synapses from eA1
    epsilon_ctx_trn = 0.03
    epsilon_ctx_mgn = 0.03  # TODO look up literature ?
    epsilon_mgn_ctx = 0.026  # should be fine for these numbers

    ######################
    # ''' Thalamus Params
    # nuX_th = 19  # rates in TH of 23 Hz, too high
    nuX_th = 10.

    gamma_th = 10.  # relative inhibitory to excitatory synaptic weight - gamma
    # gamma_th = 20.  # now that we have TRN-TRN connections, a value of 20. is too high.
    w_mgn_trn = 2.2  # excitatory synaptic weight (mV)  - we keep this fixed now, but can change later on
    sigma_MGN = 0.2
    sigma_TRN = 0.2
    wX_TRN = 1.3  # bit manually tuned to achieve the desired rates..

    nuX_aone = 5.  # LOWERED TO ACHIEVE DESIRED RATES
    gamma_aone = 9.
    w_aone = 3.

    w_input_th = 1.  # excitatory synaptic weight of background noise onto thalamus (mV) -- THIS CAN REMAIN FIXED,
    # IF NEED BE, WE ADJUST THE LOGNORMAL WIDTH
    w_input_aone = 3.  # excitatory synaptic weight of background noise onto A1 (mV)
    w_ctx_trn = 0.3  # THIS CAUSES A ~0.21 EPSP IN THALAMIC CELLS (NOT SURE AT WHICH VM, -70 OR -60)... MAYBE TOO LOW?
    w_ctx_mgn = 0.3
    w_mgn_ctx = 3.  # LET IT BE SAME AS W_AONE, SO THE SAME AS RECURRENT CORTICAL EXC. WEIGHTS

    # ######
    connections = [
        ('E1', 'E1'),  # E1 ---> E1 (recurrent E)
        ('I1', 'E1'),  # E1 ---> I1
        ('E1', 'I1'),  # I1 ---> E1
        ('I1', 'I1'),  # I1 ---> I1

        ('MGN', 'E1'),
        ('TRN', 'E1'),

        ('E1', 'MGN'),
        ('I1', 'MGN'),
        ('TRN', 'MGN'),

        ('MGN', 'TRN'),
        ('TRN', 'TRN'),
    ]

    conn_specs = [
        # module 1 internal
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx},

        # E1 -> MGN
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_mgn},
        # E1 -> TRN
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_ctx_trn},

        # MGN -> E1
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx},
        # MGN -> I1
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_mgn_ctx},

        # MGN -> TRN
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_th},

        # TRN -> MGN
        # APPARENTLY, THERE'S EVIDENCE THAT TRN->MGN CONNECTIONS ARE DENSER THAN MGN->TRN: SEE REF IN DESTEXHE 2009,
        # "the excitatory projection from TC to RE cells had a connection probability of 2%, as in cortex, while the RE to TC inhibitory projection was more dense (Kim et al. 1997), and was here of a probability of about 8%."
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': 2 * epsilon_th},

        # TRN -> TRN
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon_th},
    ]

    syn_specs = [
        # module 1 internal
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_aone},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_aone},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma_aone * w_aone},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma_aone * w_aone},

        # E1 -> MGN
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_ctx_mgn},
        # E1 -> TRN
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_ctx_trn},

        # MGN ->
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_mgn_ctx},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_mgn_ctx},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': w_mgn_trn},

        # TRN -> MGN
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma_th * w_mgn_trn},
        # TRN -> TRN
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma_th * w_mgn_trn},
    ]

    syn_pars_dict = dict(
        connected_populations=connections,
        weight_matrix=[None for _ in range(len(connections))],
        conn_specs=conn_specs,
        syn_specs=syn_specs)

    _, net_params, connection_params = set_network_defaults(default_set='Destexhe-full',
                                                            N=N, **syn_pars_dict)
    net_params['population_size'] = [nE, nI, nMGN, nTRN]

    net_params['neurons'][0] = {
        'model': 'aeif_cond_exp',
        "a": 1.,
        "b": b,
        'tau_w': tau_w_a1,
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,
        'E_in': -80.0,
        'V_m': -60.,
        'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
        "t_ref": 2.5
    }
    net_params['neurons'][1] = {
        'model': 'aeif_cond_exp',
        "a": 1.,
        "b": 0.,
        'tau_w': 600.,
        'Delta_T': 2.,

        'C_m': 150.,
        'g_L': 10.,
        'V_reset': -60.,
        'V_th': -50.,
        'E_L': -70.,
        'E_in': -80.0,
        'V_m': -60.,
        'tau_syn_ex': 5.,  # exc. synaptic time constant  - mit paper
        'tau_syn_in': 10.,  # exc. synaptic time constant  - mit paper
        "t_ref": 2.5
    }
    net_params['neurons'][2] = {
        # MGN ###################
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
    net_params['neurons'][3] = {
        # TRN ###################
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

    # ############################################################
    # Input parameters
    T = 1
    continuous = True  # True process each batch as whole batch
    stim_duration = 10.  # total sim_time == T * stim_duration
    stim_amplitude = nuX_stim * (nMGN * 0.1)
    # stim_amplitude = 1e-12

    input_type = 'static'
    distribution_pars = {'loc': 1.}
    input_parameters = {'batch_size': T,
                        'dist_pars': distribution_pars,
                        'stim_duration': stim_duration,
                        'stim_amplitude': stim_amplitude,
                        'continuous': continuous}

    input_params, task_params = default_inputs(input_type, **input_parameters)

    noise_pars = {
        'nuX_th': nuX_th * nMGN * 0.1,  # amplitude
        'w_noise_stim': w_input_th,  # in the paper it's about 3*w
        'w_noise_mgn': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, nMGN),
        'w_noise_trn': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, nTRN),
        'w_noise_ctx': w_input_aone,
        'nuX_aone': nuX_aone * nE * 0.1,
        # 'nuX_stim' : nuX_stim * nMGN * 0.1
    }

    # #############################################################
    # #############################################################
    # #############################################################
    # IGNORE STUFF BELOW
    # #############################################################
    # #############################################################
    # Encoder parameters
    # encoding device
    encoder = 'inhomogeneous_poisson_generator'
    enc_label = 'poisson-input'

    # input synapses
    inp_epsilon = 1.
    input_connection_params = {
        'connect_populations': [
            ('E1', 'poisson-input'),
            # ('I1', 'poisson-input'),
            # ('MGN', 'poisson-input'),
            # ('TRN', 'poisson-input'),
        ],
        'weight_matrix': [None for _ in range(1)],
        'conn_specs': [
            {'rule': 'pairwise_bernoulli', 'p': 1e-12},
            # {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
            # {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
            # {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
        ],
        'syn_specs': [
            {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': 0.},
            # {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': w_input_aone},
            # {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': np.random.lognormal(w_input_th, np.sqrt(w_input_th) * sigma_MGN, nMGN)},
            # {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': np.random.lognormal(w_input_th * wX_TRN, np.sqrt(w_input_th * wX_TRN) * sigma_TRN, nTRN)},
        ]
    }

    encoder_params = {'device': encoder, 'name': enc_label, 'connections': input_connection_params}

    # ######################################################
    # decoding parameters
    extractor_parameters = {}

    for i in [1]:
        extractor_parameters.update(
            {
                "spk_E{}".format(i):
                    {
                        'population': "E{}".format(i),
                        'variable': "spikes",
                        'sampling_times':['stim_offset'],
                        'standardize': False,
                        'save': True
                    }
            }
        )

    decoding_parameters = {
        '{}-decoder'.format(k.split('_')[0]): {'algorithm': "ridge-sgd", 'save': True, 'extractor': "{}".format(k)}
        for k in extractor_parameters.keys()}

    # ######################################################
    # activity analysis parameters
    analysis_params = {
        # analysis depth
        'depth': 2,  # 1: save only summary of data, use only fastest measures
        # 2: save all data, use only fastest measures
        # 3: save only summary of data, use all available measures
        # 4: save all data, use all available measures
        'ai_scan': True,

        'population_activity': {
            'time_bin': 2.,  # bin width for spike counts, fano factors and correlation coefficients
            'n_pairs': 500,  # number of spike train pairs to consider in correlation coefficient
            'tau': 20.,  # time constant of exponential filter (van Rossum distance)
            'window_len': 100,  # length of sliding time window (for time_resolved analysis)
            'n_bins': 100,  # for histograms
            'time_resolved': False,  # perform time-resolved analysis
        }
    }

    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', net_params),
                 ('connection_pars', connection_params),
                 ('task_pars', task_params),
                 ('input_pars', input_params),
                 ('encoder_pars', encoder_params),
                 ('extractor_pars', extractor_parameters),
                 ('decoding_pars', decoding_parameters),
                 ('analysis_pars', analysis_params),
                 ('noise_pars', noise_pars)
                 ])
