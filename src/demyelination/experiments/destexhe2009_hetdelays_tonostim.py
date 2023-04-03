"""
Sep 2, 2022

Implementation of Destexhe 2009 in which there is bg noise applied throughout the simulation.
The synaptic delays are heterogenous in this experiment.
The connectivity between the MGN and the eA1 is tonotopic.
"""

import os
import sys
import re
import nest
import pickle
from matplotlib import pyplot as plt
from numpy.random import uniform

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

def store_net (topology_snn, rank, path):
    network = {}
    #("source", "target", "delay", "weight"), 
    network["synapse_ex"] = nest.GetConnections(topology_snn.populations['eA1'].nodes, topology_snn.populations['eA1'].nodes).get(output="pandas")

    p = os.path.join(path, f"net_{rank}")
    with open(p, "wb") as f:
        pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

class spikesandparams:
    def __init__ (self, paramdict, spikeobj, metrics):
        self.params = paramdict
        self.spikeobj = spikeobj
        self.metrics = metrics

def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    print("[EXP FILE LOG]", parameters)

    nest.ResetKernel()
    #nest.rng_seed = 12345

    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)
    # set kernel parameters after reset
    nest.SetKernelStatus(extract_nestvalid_dict(parameters.kernel_pars.as_dict(), param_type='kernel'))

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    '''
    # now we build the network
    pos_exc = set_positions(N=parameters.net_pars.population_size[0], dim=2, topology='random',
                            specs=parameters.layer_pars)
    pos_inh = set_positions(N=parameters.net_pars.population_size[1], dim=2, topology='random',
                            specs=parameters.layer_pars)

    E_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_exc})
    I_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_inh})
    '''

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=parameters.kernel_pars.resolution,
                                          record_to='memory', device_type='spike_recorder')
    spike_recorders = [spike_recorder for _ in parameters.net_pars.populations]

    topology_snn = SpikingNetwork(parameters.net_pars, label='AdEx with spatial topology',
                                  #topologies=[E_layer_properties, I_layer_properties, E_layer_properties, I_layer_properties],
                                  spike_recorders=spike_recorders)

    # connect network
    NESTConnector(source_network=topology_snn, target_network=topology_snn, connection_parameters=parameters.connection_pars)

    # possion generator
    pg_th = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_th})
    # can be done without a loop
    for idx in range(topology_snn.find_population('MGN').size):
        nest.Connect(pg_th, topology_snn.find_population('MGN').nodes[idx], 'one_to_one', syn_spec={'weight': parameters.noise_pars.w_noise_mgn})
        nest.Connect(pg_th, topology_snn.find_population('TRN').nodes[idx], 'one_to_one', syn_spec={'weight': parameters.noise_pars.w_noise_trn})

    pg_aone = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_aone})
    nest.Connect(pg_aone, topology_snn.find_population('eA1').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_ctx})
    nest.Connect(pg_aone, topology_snn.find_population('iA1').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_ctx})

    if (parameters.hetdelay_pars.hetdelay_ctx):
        # connect the ctx with heterogenous delays
        s = nest.GetConnections(topology_snn.find_population('eA1').nodes, topology_snn.find_population('eA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('eA1').nodes, topology_snn.find_population('iA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('iA1').nodes, topology_snn.find_population('eA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('iA1').nodes, topology_snn.find_population('iA1').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))

    if (parameters.hetdelay_pars.hetdelay_thl):
        # connect the thl with heterogenous delays
        s = nest.GetConnections(topology_snn.find_population('MGN').nodes, topology_snn.find_population('TRN').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))
        s = nest.GetConnections(topology_snn.find_population('TRN').nodes, topology_snn.find_population('MGN').nodes)
        s.delay = uniform(low=1., high=3., size=len(s))

    #''' tonotopic MGN to eA1 connection
    # the populations of MGN and eA1 was split into 5 equally sized subpopulations
    # each was these was only connected to a corresponding subpopulation
    # e.g Neurons 1-100 MGN are connected to neurons 1-400 in the eA1
    for m, c in zip( range(0, 200, 40), range(0, 2000, 400)):
        # connection from mgn to eA1
        nest.Connect(topology_snn.find_population('MGN').nodes[m : m+39], \
                topology_snn.find_population('eA1').nodes[c : c+399], \
                parameters.mgn_ctx_pars.conn, \
                parameters.mgn_ctx_pars.syn)
        nest.Connect(topology_snn.find_population('eA1').nodes[c : c+399], \
                topology_snn.find_population('MGN').nodes[m : m+39], \
                parameters.ctx_mgn_pars.conn, \
                parameters.ctx_mgn_pars.syn)
    #'''

    #''' Stimulus generator removed for now
    # stimulus generator
    ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 2000., 'stop' : 2000.+parameters.noise_pars.stim_duration})
    # connecting stimulus !!! generator to snn
    nest.Connect(ng, topology_snn.populations['MGN'].nodes[:39], 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})
    #'''

    store_net(topology_snn, 0, storage_paths['other'])
    print(nest.GetDefaults("static_synapse"))
    exit()

    nest.Simulate(5000.)
    topology_snn.extract_activity(flush=False)  # this reads out the recordings

    ''' DUMP ALL POPULATIONS INTO A PICKLE FILE '''
    activitylist = dict( zip( topology_snn.population_names, [_.spiking_activity for _ in topology_snn.populations.values()] ) )

    print("[EXP FILE LOG] Activity List Prepared", flush=True)
    print("[EXP FILE LOG] Computing Pearson Coefficient", flush=True)

    '''
    # temp spike objects to not include the first second in the computation
    temp_mgn = topology_snn.populations['MGN'].spiking_activity.time_slice(3000, 5000)
    temp_trn = topology_snn.populations['TRN'].spiking_activity.time_slice(3000, 5000)
    temp_eaone = topology_snn.populations['eA1'].spiking_activity.time_slice(3000, 5000)
    temp_iaone = topology_snn.populations['iA1'].spiking_activity.time_slice(3000, 5000)
    precomputed = { "pearsoncoeff" : {
                        "MGN" : temp_mgn.pairwise_pearson_corrcoeff(nb_pairs=100, time_bin=10)[0],
                        "TRN" : temp_trn.pairwise_pearson_corrcoeff(nb_pairs=100, time_bin=10)[0],
                        "eA1" : temp_eaone.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0],
                        "iA1" : temp_iaone.pairwise_pearson_corrcoeff(nb_pairs=400, time_bin=10)[0]
                    }
                }
    '''
    
    print("[EXP FILE LOG] Pearson Coefficient Computed", flush=True)

    o = spikesandparams( parameters.label, activitylist, None )

    filepath = os.path.join(storage_paths['activity'], f'spk_{parameters.label}')

    with open(filepath, 'wb') as f:
        pickle.dump(o, f)

    print("[EXP FILE LOG] Pickle File Saved", flush=True)
