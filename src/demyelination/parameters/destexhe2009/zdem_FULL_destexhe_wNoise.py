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
experiment_label = 'destexhe-full-wnoise'
system_label = 'local'

# ################################
ParameterRange = {
    # "gamma": np.arange(10, 11, 1),
    "b": [40., 20., 10., 5.],
    "nu_x": np.arange(2, 6, 2),
    'T': np.arange(1, 2)
}


# ################################
# def build_parameters(gamma, nu_x, T):
def build_parameters(b, nu_x, T):
# def build_parameters(b, T):
    gamma = 10. # same gamma in both the ctx and thl
    # nu_x = 10.

    system_params = set_system_parameters(cluster=system_label, queue='blaustein', walltime='05-00:00:00', nodes=1,
                                          ppn=6)
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
    N = 2000
    nE = int(0.8 * N)
    nI = int(0.2 * N)
    nMGN = 100
    nTRN = 100
    delay = 1.5
    epsilon = 0.02

    wE = 6.

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
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},

        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},

        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},

        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': 4 * epsilon},
        {'allow_autapses': False, 'allow_multapses': False, 'rule': 'pairwise_bernoulli', 'p': 4 * epsilon},

    ]

    syn_specs = [
        # module 1 internal
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma * wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma * wE},

        # E1 -> Th
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},

        # MGN ->
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': wE},

        # TRN ->
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma * wE},
        {'synapse_model': 'static_synapse', 'delay': delay, 'weight': -gamma * wE},
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
    net_params['neurons'][1] = {
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
    net_params['neurons'][2] = {
        # MGN ###################
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
    net_params['neurons'][3] = {
        # TRN ###################
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

    # Interesting stuff above only
    # ############################################################
    # Input parameters
    T = 1
    continuous = True  # True process each batch as whole batch
    stim_duration = 6000.  # total sim_time == T * stim_duration
    stim_amplitude = nu_x * (nE * epsilon)

    input_type = 'static'
    distribution_pars = {'loc': 1.}
    input_parameters = {'batch_size': T,
                        'dist_pars': distribution_pars,
                        'stim_duration': stim_duration,
                        'stim_amplitude': stim_amplitude,
                        'continuous': continuous}

    input_params, task_params = default_inputs(input_type, **input_parameters)

    # #############################################################
    # Encoder parameters
    # encoding device
    encoder = 'inhomogeneous_poisson_generator'
    enc_label = 'poisson-input'

    # input synapses
    inp_epsilon = 1.
    input_connection_params = {
        'connect_populations': [('E1', 'poisson-input'),
                                ('I1', 'poisson-input'),
                                ('MGN', 'poisson-input'),
                                ('TRN', 'poisson-input'),
                                ],
        'weight_matrix': [None for _ in range(4)],
        'conn_specs': [
            {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
            {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
            {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
            {'rule': 'pairwise_bernoulli', 'p': inp_epsilon},
        ],
        'syn_specs': [
            {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': wE},
            {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': wE},
            {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': wE},
            {'synapse_model': 'static_synapse', 'delay': 0.1, 'weight': wE},
        ]
    }

    encoder_params = {'device': encoder, 'name': enc_label, 'connections': input_connection_params}

    # ######################################################
    # decoding parameters
    extractor_parameters = {}

    for i in [1, 2]:
        extractor_parameters.update(
            {
                "spk_E{}".format(i):
                    {
                        'population': "E{}".format(i),
                        'variable': "spikes",
                        'sampling_rate': 1000.,
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
                 ('analysis_pars', analysis_params)
                 ])
