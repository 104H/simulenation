"""

"""
import numpy as np
import sys
from defaults.paths import set_project_paths
from fna.tools.utils.system import set_kernel_defaults
from fna.decoders.extractors import set_recording_device

from utils.system import set_system_parameters

N = 100

# ######################################################################################
# experiments parameters
project_label = 'demyelination'

experiment_label = 'test_run'

# ######################################################################################
# system parameters
system_label = 'local'
# system_params = set_system_parameters(cluster=system_label)
paths = set_project_paths(system=system_label, project_label=project_label)

# ################################
ParameterRange = {
    'NE': [1000],
    'T': np.arange(1, 2)
}


# ################################
def build_parameters(NE, T):
    # system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=512000)
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=200000)

    # ############################################################
    # Simulation parameters
    resolution = .1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, **system_params)

    # Specify network parameters
    gamma = 0.25  # relative number of inhibitory connections
    # NE = 40  # number of excitatory neurons (10.000 in [1])
    NI = int(gamma * NE)  # number of inhibitory neurons
    CE = 10  # indegree from excitatory neurons
    CI = int(gamma * CE)  # indegree from inhibitory neurons
    N_MGN = 1000
    N_TRN = 1000

    # synapse parameters
    w = 1.  # excitatory synaptic weight (mV)
    g = 5.  # relative inhibitory to excitatory synaptic weight
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
    }

    snn_parameters = {
        'populations': ['MGN', 'TRN', 'eA1', 'iA1'],
        'population_size': [N_MGN, N_TRN, NE, NI],
        'neurons': [neuron_params, neuron_params, neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}

    # spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=resolution, record_to='memory',
    #                                       device_type='spike_recorder')
    # spike_recorders = [spike_recorder for _ in snn_parameters['populations']]

    # for simplicity all other parameters are the same, only topology is added
    layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params['model']}

    # pos_exc = set_positions(N=NE, dim=2, topology='random', specs=layer_properties)
    # pos_inh = set_positions(N=NI, dim=2, topology='random', specs=layer_properties)
    # E_layer_properties = copy_dict(layer_properties, {'positions': pos_exc})
    # I_layer_properties = copy_dict(layer_properties, {'positions': pos_inh})

    # topology_snn = SpikingNetwork(snn_parameters, label='AdEx with spatial topology',
    #                               topologies=[E_layer_properties, E_layer_properties, E_layer_properties,
    #                                           I_layer_properties],
    #                               spike_recorders=spike_recorders)

    # Connectivity
    # E synapses
    # synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
    syn_exc = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w}
    conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
    # conn_exc = {'rule': 'pairwise_bernoulli', 'p': 0.1}
    # I synapses
    syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - g * w}
    conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}

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
        'conn_specs': [conn_inh, conn_exc,
                       conn_exc, conn_exc,
                       conn_exc, conn_exc, conn_inh, conn_inh,
                       conn_exc, conn_exc],
        'syn_specs': [syn_inh, syn_exc,
                      syn_exc, syn_exc,
                      syn_exc, syn_exc, syn_inh, syn_inh,
                      syn_exc, syn_exc]
    }

    # keys need to end with _pars
    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', snn_parameters),
                 ('connection_pars', topology_snn_synapses),
                 ('layer_pars', layer_properties)
    ])
