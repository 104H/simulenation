import os
import sys
import nest
from matplotlib import pyplot as plt

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations
from fna.tools.network_architect.topology import set_positions
from fna.networks.snn import SpikingNetwork
from fna.tools.utils.operations import copy_dict
from fna.decoders.extractors import set_recording_device
from fna.tools.parameters import extract_nestvalid_dict

from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)


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
    # layer_properties = {'extent': [2500., 1000.], 'elements': neuron_params['model']}

    pos_exc = set_positions(N=parameters.net_pars.population_size[2], dim=2, topology='random',
                            specs=parameters.layer_pars)
    pos_inh = set_positions(N=parameters.net_pars.population_size[3], dim=2, topology='random',
                            specs=parameters.layer_pars)
    E_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_exc})
    I_layer_properties = copy_dict(parameters.layer_pars, {'positions': pos_inh})

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=parameters.kernel_pars.resolution,
                                          record_to='memory', device_type='spike_recorder')
    spike_recorders = [spike_recorder for _ in parameters.net_pars.populations]

    topology_snn = SpikingNetwork(parameters.net_pars, label='AdEx with spatial topology',
                                  topologies=[E_layer_properties, E_layer_properties, E_layer_properties,
                                              I_layer_properties],
                                  spike_recorders=spike_recorders)

    # possion generator
    num_nodes = 1
    pg = nest.Create('poisson_generator', n=num_nodes, params={'rate': [10000.0]})

    # m = nest.Create('multimeter', num_nodes, {'interval': 0.1, 'record_from': ['rate']})

    # nest.Connect(m, pg, 'one_to_one') # multimeter to poisson generator
    [nest.Connect(pg, _.nodes, 'all_to_all', syn_spec={'weight': 5.}) for _ in
     topology_snn.populations.values()]  # poisson generator to snn
    # import pdb; pdb.se
    # [nest.Connect(pg, _) for _ in spike_recorders] # poisson generator to spike recorders

    nest.Simulate(1000)
    topology_snn.extract_activity(flush=False)  # this reads out the recordings

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
