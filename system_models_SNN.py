#!/usr/bin/env python
# coding: utf-8

import sys
import nest
import numpy as np
import matplotlib.pyplot as plt

from fna.tools.network_architect import Network
[func for func in dir(Network) if callable(getattr(Network, func)) and not func.startswith("__")]

from fna.networks.snn import SpikingNetwork
from fna.decoders.extractors import set_recording_device
from fna.tools.utils.system import set_kernel_defaults, reset_nest_kernel
from fna.tools.network_architect.topology import set_positions
from fna.tools.visualization.plotting import plot_spectral_radius, plot_histograms, plot_spatial_connectivity, plot_network_topology

from fna.tools.network_architect.connectivity import NESTConnector

# Specify system and simulation parameters
resolution = 0.1
data_label = 'test_network'
system = 'local'
system_params = {
    'nodes': 1,
    'ppn': 16,
    'mem': 8,
    'walltime': '01-00:00:00',
    'queue': 'batch'}
paths = {
    'local': {  # storage label
        'data_path': '../data/',  # path for data (in/out)
        'jdf_template': 	None, # TODO remove
        'matplotlib_rc': 	None, # 
        'remote_directory': None,
        'queueing_system':  None}}

# initialize NEST kernel
kernel_pars = set_kernel_defaults(run_type=system, data_label=data_label, data_paths=paths, **system_params)
reset_nest_kernel(kernel_pars)

# Specify network parameters
gamma = 0.25               # relative number of inhibitory connections
NE = 40                  # number of excitatory neurons (10.000 in [1])
NI = int(gamma * NE)       # number of inhibitory neurons
CE = 10                  # indegree from excitatory neurons
CI = int(gamma * CE)       # indegree from inhibitory neurons
N_MGN = 20
N_TRN = 20

# synapse parameters
w = 0.1                    # excitatory synaptic weight (mV)
g = 5.                     # relative inhibitory to excitatory synaptic weight
d = 1.5                    # synaptic transmission delay (ms)

neuron_params = {
            'model': 'aeif_cond_exp',
            # 'C_m': 1.0,      # membrane capacity (pF)
            'E_L': -70.,       # resting membrane potential (mV)
            # 'I_e': 0.,       # external input current (pA)
            # 'V_m': 0.,       # membrane potential (mV) generally the resting potential is -70
            # 'V_reset': 10.,  # reset membrane potential after a spike (mV)
            'V_th': -55.,     # spike threshold (mV)
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

spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=resolution, record_to='memory',
                                      device_type='spike_recorder')
spike_recorders = [spike_recorder for _ in snn_parameters['populations']]

# Generate SNN instance

# C) SNN with topology
# #####################################################################################
from fna.tools.network_architect.topology import set_positions
from fna.tools.utils.operations import copy_dict

N = NE + NI

# for simplicity all other parameters are the same, only topology is added
layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params['model']}
pos_exc = set_positions(N=NE, dim=2, topology='random', specs=layer_properties)
pos_inh = set_positions(N=NI, dim=2, topology='random', specs=layer_properties)
E_layer_properties = copy_dict(layer_properties, {'positions': pos_exc})
I_layer_properties = copy_dict(layer_properties, {'positions': pos_inh})

topology_snn = SpikingNetwork(snn_parameters, label='AdEx with spatial topology',
                              topologies=[E_layer_properties, E_layer_properties, E_layer_properties, I_layer_properties],
                              spike_recorders=spike_recorders)


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

topology_connections = NESTConnector(source_network=topology_snn, target_network=topology_snn,
                                     connection_parameters=topology_snn_synapses)
w_rec = topology_connections.compile_weights()

# possion generator
num_nodes = 1
pg = nest.Create('poisson_generator', n=num_nodes, params={'rate': [50000.0]})

#m = nest.Create('multimeter', num_nodes, {'interval': 0.1, 'record_from': ['rate']})

# nest.Connect(m, pg, 'one_to_one') # multimeter to poisson generator
[nest.Connect(pg, _.nodes, 'all_to_all', syn_spec={'weight': 5.}) for _ in topology_snn.populations.values()] # poisson generator to snn
#import pdb; pdb.se
#[nest.Connect(pg, _) for _ in spike_recorders] # poisson generator to spike recorders

nest.Simulate(200)
topology_snn.extract_activity(flush=False)  # this reads out the recordings

# print(topology_snn.spiking_activity[0].spike_counts(10))  # histogram

#####
# plot spatial positions of neurons

# fig, ax = plt.subplots(len(topology_snn.populations), 1)
# plot_network_topology(topology_snn, ax=ax, display=False)
# for idx, (pop_name, pop_obj) in enumerate(topology_snn.populations.items()):
#     positions = nest.GetPosition(pop_obj.nodes)
#     pos_x = [x[0] for x in positions]
#     pos_y = [x[1] for x in positions]
#     ax[idx].scatter(pos_x, pos_y, color='k')
#     ax[idx].set_title(pop_name)
#plot_spatial_connectivity(topology_snn, kernel=conn_dict['kernel'], mask=conn_dict['mask'], ax=ax)
# plt.tight_layout()
# plt.show()


#####
# plot spatial positions of neurons
fig, ax = plt.subplots(len(topology_snn.populations), 1)
for idx, pop_name in enumerate(topology_snn.population_names):
    # topology_snn.spiking_activity[idx].raster_plot(with_rate=False, ax=ax[idx], display=False)
    topology_snn.populations[pop_name].spiking_activity.raster_plot(with_rate=False, ax=ax[idx], display=False)
    ax[idx].set_title(f'Population {pop_name}')
plt.tight_layout()
plt.show()