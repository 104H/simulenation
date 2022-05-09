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
    def __init__ (self, paramdict, spikeobj, metrics):
        self.params = paramdict
        self.spikeobj = spikeobj
        self.metrics = metrics
        self.calcium = { 'eA1' : [], 'iA1' : [] }
        self.connectivity = { 'eA1' : [], 'iA1' : [] }

def record_ca(population):
        return np.mean(population.nodes.Ca)

def record_connectivity(population, synType):
        t = "ex" if synType == "excitatory" else "in"

        syn_elems_e = nest.GetStatus(population.nodes, 'synaptic_elements')
        return sum(neuron['Axon_' + t]['z_connected'] for neuron in syn_elems_e)

def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    nest.ResetKernel()

    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)
    # set kernel parameters after reset
    nest.SetKernelStatus(extract_nestvalid_dict(parameters.kernel_pars.as_dict(), param_type='kernel'))
    nest.local_num_threads = 1 # can't run plasticity with multiple threads

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    # now we build the network
    pos_exc = set_positions(N=parameters.net_pars.population_size[0], dim=2, topology='random',
                            specs=parameters.layer_pars)
    pos_inh = set_positions(N=parameters.net_pars.population_size[1], dim=2, topology='random',
                            specs=parameters.layer_pars)

    E_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_exc})
    I_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_inh})

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=parameters.kernel_pars.resolution,
                                          record_to='memory', device_type='spike_recorder')
    spike_recorders = [spike_recorder for _ in parameters.net_pars.populations]

    topology_snn = SpikingNetwork(parameters.net_pars, label='AdEx with spatial topology',
                                  topologies=[E_layer_properties, I_layer_properties],
                                  spike_recorders=spike_recorders)

    synaptic_elements = {
        'Den_ex': parameters.growth_pars['growth_curve_e_e'],
        'Den_in': parameters.growth_pars['growth_curve_e_i'],
        'Axon_ex': parameters.growth_pars['growth_curve_e_e'],
    }

    synaptic_elements_i = {
        'Den_ex': parameters.growth_pars['growth_curve_i_e'],
        'Den_in': parameters.growth_pars['growth_curve_i_i'],
        'Axon_in': parameters.growth_pars['growth_curve_i_i'],
    }

    nest.SetStatus(topology_snn.find_population('eA1').nodes, 'synaptic_elements', synaptic_elements)
    nest.SetStatus(topology_snn.find_population('iA1').nodes, 'synaptic_elements', synaptic_elements_i)

    # connect network
    NESTConnector(source_network=topology_snn, target_network=topology_snn, connection_parameters=parameters.connection_pars)

    ''' Thalamus Removed
    # possion generator
    pg_th = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_th})
    for idx in range(topology_snn.find_population('MGN').size):
        nest.Connect(pg_th, topology_snn.find_population('MGN').nodes[idx], 'one_to_one', syn_spec={'weight': parameters.noise_pars.w_noise_mgn[idx]})
        nest.Connect(pg_th, topology_snn.find_population('TRN').nodes[idx], 'one_to_one', syn_spec={'weight': parameters.noise_pars.w_noise_trn[idx]})
    '''

    pg_aone = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_aone})
    nest.Connect(pg_aone, topology_snn.find_population('eA1').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_ctx})
    nest.Connect(pg_aone, topology_snn.find_population('iA1').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_ctx})

    nest.structural_plasticity_update_interval = 100.

    nest.CopyModel('static_synapse', 'synapse_ex')
    nest.SetDefaults('synapse_ex', {'weight': 585.0, 'delay': .1})
    nest.CopyModel('static_synapse', 'synapse_in')
    nest.SetDefaults('synapse_in', {'weight': -585.0, 'delay': .1})
    nest.structural_plasticity_synapses = {
            'synapse_ex': {
                'synapse_model': 'synapse_ex',
                'post_synaptic_element': 'Den_ex',
                'pre_synaptic_element': 'Axon_ex',
            },
            'synapse_in': {
                'synapse_model': 'synapse_in',
                'post_synaptic_element': 'Den_in',
                'pre_synaptic_element': 'Axon_in',
            },
        }

    ''' Stimulus generator removed for now
    # stimulus generator
    ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 1000., 'stop' : 1025.})
    # connecting stimulus !!! generator to snn
    nest.Connect(ng, topology_snn.populations['TRN'].nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})
    '''

    nest.EnableStructuralPlasticity()

    o = spikesandparams(parameters.label, None, None)

    record_interval = 50.
    for _ in np.arange(0., 5000., record_interval):
            nest.Simulate(record_interval)

            o.calcium['eA1'].append( record_ca(topology_snn.find_population('eA1')) )
            o.calcium['iA1'].append( record_ca(topology_snn.find_population('iA1')) )

            o.connectivity['eA1'].append( record_connectivity(topology_snn.find_population('eA1'), "excitatory") )
            o.connectivity['iA1'].append( record_connectivity(topology_snn.find_population('iA1'), "inhibitory") )

    topology_snn.extract_activity(flush=False)  # this reads out the recordings

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

    ''' DUMP ALL POPULATIONS INTO A PICKLE FILE '''
    o.spikeobj = dict( zip( topology_snn.population_names, [_.spiking_activity for _ in topology_snn.populations.values()] ) )

    filepath = os.path.join(storage_paths['activity'], f'spk_{parameters.label}')

    with open(filepath, 'wb') as f:
        pickle.dump(o, f)

