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
NE = 400                  # number of excitatory neurons (10.000 in [1])
NI = int(gamma * NE)       # number of inhibitory neurons
CE = 100                  # indegree from excitatory neurons
CI = int(gamma * CE)       # indegree from inhibitory neurons

# synapse parameters
w = 0.1                    # excitatory synaptic weight (mV)
g = 5.                     # relative inhibitory to excitatory synaptic weight
d = 1.5                    # synaptic transmission delay (ms)

neuron_params = {
            'model': 'aeif_psc_delta_clopath',
            'C_m': 1.0,      # membrane capacity (pF)
            'E_L': 0.,       # resting membrane potential (mV)
            'I_e': 0.,       # external input current (pA)
            'V_m': 0.,       # membrane potential (mV)
            'V_reset': 10.,  # reset membrane potential after a spike (mV)
            'V_th': 20.,     # spike threshold (mV)
            't_ref': 2.0,    # refractory period (ms)
            'tau_m': 20.,    # membrane time constant (ms)
        }

snn_parameters = {
    'populations': ['MGN', 'RE', 'eA1', 'iA1'],
    'population_size': [NE, NE, NE, NI],
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


topology_snn_parameters = {
    'populations': ['MGN', 'RE', 'eA1', 'iA1'],
    'population_size': [NE, NI, NE, NI],
    'neurons': [neuron_params, neuron_params, neuron_params, neuron_params],
    'randomize': [
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}

topology_snn = SpikingNetwork(topology_snn_parameters, label='AdEx with spatial topology', 
                              topologies=[E_layer_properties, E_layer_properties, E_layer_properties, I_layer_properties])


fig, ax = plt.subplots()
plot_network_topology(topology_snn, ax=ax, display=False)

from fna.tools.network_architect.connectivity import NESTConnector

# Connectivity
# E synapses
# synapse_model is a bernoulli synapse https://nest-simulator.readthedocs.io/en/v2.20.1/models/static.html
syn_exc = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w}
conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
# I synapses
syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': - g * w}
conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}

conn_dict = {'rule': 'pairwise_bernoulli',
             'mask': {'circular': {'radius': 20.}},
             'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.25)
             }

topology_snn_synapses = {
    'connect_populations': [('MGN', 'RE'), ('RE', 'MGN'), ('eA1', 'MGN'), ('eA1', 'iA1')],
    'weight_matrix': [None, None, None, None],
    'conn_specs': [conn_exc, conn_inh, conn_exc, conn_inh],
    'syn_specs': [syn_exc, syn_inh, syn_exc, syn_inh]
}

topology_connections = NESTConnector(source_network=topology_snn, target_network=topology_snn,
                                    connection_parameters=topology_snn_synapses)
w_rec = topology_connections.compile_weights()

fig, ax = plt.subplots()
plot_network_topology(topology_snn, ax=ax, display=False)
#plot_spatial_connectivity(topology_snn, kernel=conn_dict['kernel'], mask=conn_dict['mask'], ax=ax)
plt.show()

"""
# ### 10.2.1. Topological connections

# E synapses
syn_exc = {'synapse_model': 'static_synapse', 'delay': d, 'weight': w}
# I synapses
syn_inh = {'synapse_model': 'static_synapse', 'delay': d, 'weight': -g * w}

conn_dict = {'rule': 'pairwise_bernoulli',
             'mask': {'circular': {'radius': 20.}},
             'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.25)
             }
topology_snn_synapses = {
    'connect_populations': [('tpE', 'tpE'), ('tpE', 'tpI'), ('tpI', 'tpE'), ('tpI', 'tpI')],
    'weight_matrix': [None, None, None, None],
    'conn_specs': [conn_dict, conn_dict, conn_dict, conn_dict],
    'syn_specs': [syn_exc, syn_inh, syn_exc, syn_inh],
}

w_rec = topology_connections.compile_weights()
"""

