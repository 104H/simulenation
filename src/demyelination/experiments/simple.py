'''
8 Nov, 2022

An exponential distribution with rate 13 is used to define the Ca levels.

Save the network frequently to assess the evolution of the connections.

This file is supposed to be run in the conda env nust-noautapses.
The nest installation in this env is built using sandra's branch which implements functionality for not having autapses with structural plasticity.

We try to grow connections using structural plasticity.
This file saves the network and restores it for future use.
'''

import os
import sys
import re
import nest
import pickle
from matplotlib import pyplot as plt
import numpy as np

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations
from fna.tools.network_architect.topology import set_positions
from fna.networks.snn import SpikingNetwork
from fna.tools.network_architect.connectivity import NESTConnector
from fna.tools.utils.operations import copy_dict
from fna.decoders.extractors import set_recording_device
from fna.tools.parameters import extract_nestvalid_dict

from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)


class spikesandparams:
    def __init__(self, paramdict, spikeobj, metrics):
        self.params = paramdict
        self.spikeobj = spikeobj
        self.metrics = metrics
        self.calcium = {'eA1': [], 'MGN': []}
        '''
        self.connectivity = {
                "z" : {
                    "Den" : {
                            "ex" : { 'eA1' : [], 'MGN' : [] },
                            "in" : { 'eA1' : [], 'MGN' : [] }
                            },
                    "Axon" : {
                            "ex" : { 'eA1' : [], 'MGN' : [] },
                            "in" : { 'eA1' : [], 'MGN' : [] }
                            }
                    },
                "z_connected" : {
                    "Den" : {
                            "ex" : { 'eA1' : [], 'MGN' : [] },
                            "in" : { 'eA1' : [], 'MGN' : [] }
                            },
                    "Axon" : {
                            "ex" : { 'eA1' : [], 'MGN' : [] },
                            "in" : { 'eA1' : [], 'MGN' : [] }
                            }
                    }
                }
        '''
        # parent dict is the source, child dict is the target
        self.nestconnectivity = {
            'MGN': {'eA1': []}
        }
        self.neuron_states = None


def store_net(topology_snn, rank, path, time):
    network = {}
    # topology_snn.populations['eA1'].nodes
    network["synapse_ex"] = nest.GetConnections().get(("source", "target", "delay", "weight"), output="pandas")

    p = os.path.join(path, f"{time}_net_{rank}")
    with open(p, "wb") as f:
        pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)


def record_neuron_state(topology_snn):
    states = []
    for population in ["eA1", "iA1", "MGN", "TRN"]:
        for idx, node in enumerate(topology_snn.populations[population].nodes):
            state = {
                "population": population,
                "id": idx,
                "V_m": node.V_m,
                "g_ex": node.g_ex,
                "g_in": node.g_in,
                "w": node.w
            }
            states.append(state)
    return states


def record_ca(population):
    ca = [c for c in population.nodes.Ca if c != None]
    return np.mean(ca)


def record_connectivity(population, connType, synType, metric):
    syn_elems = population.nodes.synaptic_elements
    return np.sum(list(neuron[connType + '_' + synType][metric] for neuron in syn_elems if neuron != None))


def record_connectivity_fromnest(source, target):
    return len(nest.GetConnections(source, target).get('target'))


def record_data(o, topology_snn):
    o.calcium['eA1'].append(record_ca(topology_snn.find_population('eA1')))
    o.calcium['MGN'].append(record_ca(topology_snn.find_population('MGN')))

    o.nestconnectivity['MGN']['eA1'].append(
        record_connectivity_fromnest(topology_snn.populations['MGN'].nodes, topology_snn.populations['eA1'].nodes))
    '''
    for z in ['z_connected', 'z']:
        for c in ['Axon', 'Den']:
            for t in ['ex', 'in']:
                for p in ['eA1', 'MGN']:
                    try:
                        o.connectivity[z][c][t][p].append( record_connectivity(topology_snn.find_population(p), c, t, z) ) 
                    except:
                        pass
    '''
    o.neuron_states = record_neuron_state(topology_snn)


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)
    # set kernel parameters after reset
    parameters.kernel_pars['local_num_threads'] = 1  # can't run plasticity with multiple threads
    parameters.kernel_pars['data_path'] = storage_paths['other']  # store the spikerecorder readings to the other folder
    nest.SetKernelStatus(extract_nestvalid_dict(parameters.kernel_pars.as_dict(), param_type='kernel'))

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    '''
    # now we build the network
    pos_exc = set_positions(N=parameters.net_pars.population_size[0], dim=2, topology='random',
                            specs=parameters.layer_pars)
    pos_inh = set_positions(N=parameters.net_pars.population_size[1], dim=2, topology='random',
                            specs=parameters.layer_pars)

    #E_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_exc})
    #I_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_inh})
    '''

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max,
                                          resolution=parameters.kernel_pars.resolution,
                                          record_to='memory', device_type='spike_recorder')
    spike_recorders = [spike_recorder for _ in parameters.net_pars.populations]

    topology_snn = SpikingNetwork(parameters.net_pars, label='AdEx with spatial topology',
                                  # topologies=[E_layer_properties, I_layer_properties],
                                  spike_recorders=spike_recorders)

    # connect network
    NESTConnector(source_network=topology_snn, target_network=topology_snn,
                  connection_parameters=parameters.connection_pars)

    # possion generator
    pg_th = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_th})
    nest.Connect(pg_th, topology_snn.find_population('MGN').nodes, 'all_to_all',
                 syn_spec={'weight': parameters.noise_pars.w_noise_mgn})
    nest.Connect(pg_th, topology_snn.find_population('TRN').nodes, 'all_to_all',
                 syn_spec={'weight': parameters.noise_pars.w_noise_trn})

    pg_aone = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_aone})
    nest.Connect(pg_aone, topology_snn.find_population('eA1').nodes, 'all_to_all',
                 syn_spec={'weight': parameters.noise_pars.w_noise_ctx})
    nest.Connect(pg_aone, topology_snn.find_population('iA1').nodes, 'all_to_all',
                 syn_spec={'weight': parameters.noise_pars.w_noise_ctx})

    if parameters.hetdelay_pars.hetdelay_ctx:
        assert False
        # connect the ctx with heterogenous delays
        s = nest.GetConnections(topology_snn.find_population('eA1').nodes, topology_snn.find_population('eA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('eA1').nodes, topology_snn.find_population('iA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('iA1').nodes, topology_snn.find_population('eA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('iA1').nodes, topology_snn.find_population('iA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))

    if parameters.hetdelay_pars.hetdelay_thl:
        assert False
        # connect the thl with heterogenous delays
        s = nest.GetConnections(topology_snn.find_population('MGN').nodes, topology_snn.find_population('TRN').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('TRN').nodes, topology_snn.find_population('MGN').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))

    # ''' tonotopic MGN to eA1 connection
    # the populations of MGN and eA1 was split into 5 equally sized subpopulations
    # each was these was only connected to a corresponding subpopulation
    # e.g Neurons 1-100 MGN are connected to neurons 1-400 in the eA1
    for m, c in zip(range(0, 200, 40), range(0, 2000, 400)):
        # connection from mgn to eA1
        nest.Connect(topology_snn.find_population('MGN').nodes[m: m + 40], \
                     topology_snn.find_population('eA1').nodes[c: c + 400], \
                     parameters.mgn_ctx_pars.conn, \
                     parameters.mgn_ctx_pars.syn)
        nest.Connect(topology_snn.find_population('eA1').nodes[c: c + 400], \
                     topology_snn.find_population('MGN').nodes[m: m + 40], \
                     parameters.ctx_mgn_pars.conn, \
                     parameters.ctx_mgn_pars.syn)
    # '''


    nest.structural_plasticity_update_interval = parameters.sp_pars['update_interval'] * 1000.

    ''' Stimulus generator removed for now
    # stimulus generator
    ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 2000., 'stop' : 2025.})
    # connecting stimulus !!! generator to snn
    nest.Connect(ng, topology_snn.populations['TRN'].nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})
    '''

    # preparing file path to save data
    rank = str(nest.Rank())
    filepath = os.path.join(storage_paths['activity'], f'spk_{parameters.label}_{rank}')
    o = spikesandparams(parameters.label, None, None)

    #'''setting the synatic elements
    for idx in range(200):
        synaptic_elements = {
            'Axon_ex': parameters.growth_pars['growth_curve_mgn_axon'][idx],
        }
        nest.SetStatus(topology_snn.find_population('MGN').nodes[idx], 'synaptic_elements', synaptic_elements)

    # setting the synatic elements to the ctx
    for idx in range(2000):
        synaptic_elements = {
            'Den_ex': parameters.growth_pars['growth_curve_eaone_exc_dendrite'][idx],
        }
        nest.SetStatus(topology_snn.find_population('eA1').nodes[idx], 'synaptic_elements', synaptic_elements)
    #'''
    #'''
    nest.CopyModel('static_synapse', 'synapse_ex')
    nest.SetDefaults('synapse_ex', {'weight': 4, 'delay': 1.5})  # add w_aone
    nest.structural_plasticity_synapses = {
        'synapse_ex': {
            'synapse_model': 'synapse_ex',
            'post_synaptic_element': 'Den_ex',
            'pre_synaptic_element': 'Axon_ex',
            'allow_autapses': False,
        }
    }
    #'''

    # store the network state at time zero
    store_net(topology_snn, rank, storage_paths['other'], 0)
    record_data(o, topology_snn)

    # Simulate for 5 seconds to remove initial transients
    nest.Simulate(50. * 1000)
    record_data(o, topology_snn)

    #'''
    # assign currect Ca level as target Ca level to the MGN Neuron
    for idx in range(200):
        calevel = topology_snn.find_population('MGN').nodes[idx].get("Ca")
        if calevel == 0: calevel = .129
        parameters.growth_pars['growth_curve_mgn_axon'][idx]['eps'] = calevel
        synaptic_elements = {
            'Axon_ex': parameters.growth_pars['growth_curve_mgn_axon'][idx],
        }
        nest.SetStatus(topology_snn.find_population('MGN').nodes[idx], 'synaptic_elements_param', synaptic_elements)
        # crash if the ca level is not successfully set to the neuron
        assert calevel == nest.GetStatus(topology_snn.find_population('MGN').nodes[idx])[0]['synaptic_elements']['Axon_ex']['eps']
    #'''
    #'''
    # assign currect Ca level as target Ca level to the eA1 Neuron
    for idx in range(2000):
        calevel = topology_snn.find_population('eA1').nodes[idx].get("Ca")
        if calevel == 0: calevel = .104
        parameters.growth_pars['growth_curve_eaone_exc_dendrite'][idx]['eps'] = calevel
        synaptic_elements = {
            'Den_ex': parameters.growth_pars['growth_curve_eaone_exc_dendrite'][idx],
        }
        nest.SetStatus(topology_snn.find_population('eA1').nodes[idx], 'synaptic_elements_param', synaptic_elements)
        # crash if the ca level is not successfully set to the neuron
        assert calevel == nest.GetStatus(topology_snn.find_population('eA1').nodes[idx])[0]['synaptic_elements']['Den_ex']['eps']
    #'''

    # Disable some connection from the MGN to eA1
    c = nest.GetConnections(topology_snn.find_population('MGN').nodes, topology_snn.find_population('eA1').nodes)
    num_disabled_conn = round(parameters.disabled_conn_ratio_pars * len(c))
    disabled_conn_ids = np.random.randint(low=0., high=len(c) - 1, size=num_disabled_conn)
    for idx in disabled_conn_ids:
        c[idx].weight = 0

    # store the network state after demyelination
    store_net(topology_snn, rank, storage_paths['other'], 5000)

    # Enable SP
    nest.EnableStructuralPlasticity()

    with nest.RunManager():
        record_interval = 50. * 1000
        for total_time in np.arange(0, 1000. * 1000, record_interval):
            record_data(o, topology_snn)
            nest.Run(record_interval)
            store_net(topology_snn, rank, storage_paths['other'], total_time)

    record_data(o, topology_snn)

    topology_snn.extract_activity(flush=False)  # this reads out the recordings

    ''' DUMP ALL POPULATIONS INTO A PICKLE FILE '''
    o.spikeobj = dict(
        zip(topology_snn.population_names, [_.spiking_activity for _ in topology_snn.populations.values()]))

    # now also saving the spiking info
    with open(filepath, 'wb') as f:
        pickle.dump(o, f)

    ''' Pearson Coeff Not Needed Now
    # temp spike objects to not include the first second in the computation
    temp_mgn = topology_snn.populations['MGN'].spiking_activity.time_slice(2000, 5000)
    temp_trn = topology_snn.populations['TRN'].spiking_activity.time_slice(2000, 5000)
    temp_eaone = topology_snn.populations['eA1'].spiking_activity.time_slice(2000, 5000)
    temp_iaone = topology_snn.populations['iA1'].spiking_activity.time_slice(2000, 5000)
    precomputed = { "pearsoncoeff" : {
                        "MGN" : temp_mgn.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0],
                        "TRN" : temp_trn.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0],
                        "eA1" : temp_eaone.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0],
                        "iA1" : temp_iaone.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0]
                    }
                }

    print(precomputed, flush=True)
    '''
