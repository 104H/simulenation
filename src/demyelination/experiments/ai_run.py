import os
import sys
import re
import nest
import pickle
from matplotlib import pyplot as plt

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
                                  topologies=[E_layer_properties, I_layer_properties, E_layer_properties, I_layer_properties],
                                  spike_recorders=spike_recorders)

    # connect network
    NESTConnector(source_network=topology_snn, target_network=topology_snn,
                  connection_parameters=parameters.connection_pars)

    # possion generator
    #num_nodes = 1
    # pg = nest.Create('poisson_generator', n=1, params={'rate': [parameters.noise_pars.nuX]})
    pg_trn = nest.Create('poisson_generator', n=1, params={'rate': [parameters.noise_pars.nuX_TRN]})
    pg_mgn = nest.Create('poisson_generator', n=1, params={'rate': [parameters.noise_pars.nuX_MGN]})
    # connecting noise generator to snn
    # [nest.Connect(pg, _.nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_thalamus}) for _ in
    #  topology_snn.populations.values()]
    nest.Connect(pg_mgn, topology_snn.find_population('MGN').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_mgn})
    nest.Connect(pg_trn, topology_snn.find_population('TRN').nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_trn})

    # stimulus generator
    ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 500., 'stop' : 530.})
    # connecting stimulus !!! generator to snn
    nest.Connect(ng, topology_snn.populations['TRN'].nodes, 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_thalamus})

    nest.Simulate(1000.)
    topology_snn.extract_activity(flush=False)  # this reads out the recordings
    #topology_snn.populations['MGN'].spiking_activity.raster_plot(ms=2.)
    #topology_snn.populations['TRN'].spiking_activity.raster_plot(ms=2., color='r')

    print("preparing pickle file", flush=True)
    ''' DUMP ALL POPULATIONS INTO A PICKLE FILE '''
    activitylist = dict( zip( topology_snn.population_names, [_.spiking_activity for _ in topology_snn.populations.values()] ) )
    print("activity list prepared", flush=True)
    precomputed = { "pearsoncoeff" : {
                        # "MGN" : topology_snn.populations['MGN'].spiking_activity.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0],
                        # "TRN" : topology_snn.populations['TRN'].spiking_activity.pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=10)[0]
                        0., 0.,
                    }
                }

    print(precomputed, flush=True)

    '''
    r = re.findall("_*?(\w+)=(\d+)_", parameters.label)
    paramsfromfilename = {}
    [ paramsfromfilename.update( {p[0] : int(p[1])} ) for p in r ]
    '''

    o = spikesandparams( parameters.label, activitylist, precomputed )

    filepath = os.path.join(storage_paths['activity'], f'spk_{parameters.label}')

    with open(filepath, 'wb') as f:
        pickle.dump(o, f)

    print("pickle file saved", flush=True)

    #for idx, pop_name in enumerate(topology_snn.population_names):
    #    topology_snn.populations[pop_name].spiking_activity.save( os.path.join(storage_paths['activity'], f'spk_{pop_name}_{parameters.label}') )

    ''' MAKE A RASTER PLOT
    fig, ax = plt.subplots(len(topology_snn.populations), 1)
    for idx, pop_name in enumerate(topology_snn.population_names):
        # topology_snn.spiking_activity[idx].raster_plot(with_rate=False, ax=ax[idx], display=False)
        try:
            topology_snn.populations[pop_name].spiking_activity.raster_plot(with_rate=False, ax=ax[idx], display=False)
        except:
            print(pop_name, topology_snn.populations[pop_name].spiking_activity.mean_rate(), "spks/sec")
            pass
        ax[idx].set_title(f'Population {pop_name}')
    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], f'myfig_{parameters.label}'))

    if save:
        # save parameters
        parameters.save(storage_paths['parameters'] + parameters.label)
    '''

