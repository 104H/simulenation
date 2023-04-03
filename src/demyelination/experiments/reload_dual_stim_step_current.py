"""
Oct 10, 2022

This experiment loads a network with connections grown in the eA1.

Implementation of Destexhe 2009 in which there is bg noise applied throughout the simulation.
The MGN is stimulated.
"""
import copy
import os
import sys
import re
import nest
import pickle
import numpy as np

from debugpy.common.timestamp import current
from matplotlib import pyplot as plt

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.visualization.plotting import plot_raster
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

def restore_recurreaone():
    import pickle

    import pandas as pd
    df = pd.DataFrame()

    dump_filename = 'demyelination/data/20pcdemy-15000s-eaone/other/spk_20pcdemy-15000s-eaone_disabled_conn_ratio=0.2_gr_scale=1e-05_update_interval=50_0/100000.0_net_'

    for threadid in range(4):
        with open(dump_filename + str(threadid), "rb") as f:
            network = pickle.load(f)

            net = network['synapse_ex']

            df = pd.concat([df, net])

    df = df[df.source.between(400, 2400) & df.target.between(400, 2400)]

    #nest.Connect(df.source, df.target, "one_to_one",\
    #        {"synapse_model" : "static_synapse", "delay" : df.delay.values, "weight" : df.weight.values})

    df = df.reset_index()

    return df


def connect_sample(df, quantity):
    if df.shape[0] < quantity: quantity = df.shape[0]

    sub_df = df.sample(quantity)

    nest.Connect(sub_df.source, sub_df.target, "one_to_one",\
            {"synapse_model" : "static_synapse", "delay" : sub_df.delay.values, "weight" : sub_df.weight.values})

    return sub_df.index


def restore_network(topology_snn):
    import pickle

    import pandas as pd
    df = pd.DataFrame()

    dump_filename = 'demyelination/data/eaone-demy-recurr/other/spk_eaone-demy-recurr_gr_scale=1e-05_update_interval=50_'
    networktime = 10000 * 1000.0

    for threadid in range(48):
        # filepath = folder _ threadid / networktime _ threadid
        filepth = dump_filename + str(threadid) + '/' + str(networktime) + '_net_' + str(threadid)
        with open(filepth, "rb") as f:
            network = pickle.load(f)
            net = network['synapse_ex']
            df = pd.concat([df, net])

    # remove connections in DF to spikerecorder
    #df = df[df.target != 2901]
    #df = df[df.target != 2902]
    #df = df[df.target != 2903]
    #df = df[df.target != 2904]

    #df = df[df.source != 2905]
    #df = df[df.source != 2906]

    # don't reload the recurrant exc ea1 conn
    #df = df[~df.source.between(400, 2400) | ~df.target.between(400, 2400)]
    syn_dict = {
        "synapse_model" : "static_synapse",
        "delay" : df.delay.values,
        "weight" : df.weight.values
    }
    nest.Connect(df.source, df.target, "one_to_one", syn_dict)

def restore_network_state(topology_snn):
    assert False
    import pandas as pd
    dump_filename = 'demyelination/data/destexhe-tonotopic-cadist-allconn/activity/spk_destexhe-tonotopic-cadist-allconn_T=0_0'
    net_state = pd.read_pickle(dump_filename).neuron_states

    for st in net_state:
        topology_snn.populations[ st['population'] ].nodes[ st['id'] ].set(V_m = st['V_m'])
        topology_snn.populations[ st['population'] ].nodes[ st['id'] ].set(g_ex = st['g_ex'])
        topology_snn.populations[ st['population'] ].nodes[ st['id'] ].set(g_in = st['g_in'])
        topology_snn.populations[ st['population'] ].nodes[ st['id'] ].set(w = st['w'])


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)
    #parameters.kernel_pars['rng_seed'] = 3
    # set kernel parameters after reset
    nest.SetKernelStatus(extract_nestvalid_dict(parameters.kernel_pars.as_dict(), param_type='kernel'))

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=parameters.kernel_pars.resolution,
                                          record_to='memory', device_type='spike_recorder')
    spike_recorders = [spike_recorder for _ in parameters.net_pars.populations]

    topology_snn = SpikingNetwork(parameters.net_pars, label='AdEx with spatial topology',
                                  spike_recorders=spike_recorders)

    NESTConnector(source_network=topology_snn, target_network=topology_snn,
                  connection_parameters=parameters.connection_pars)

    # possion generator
    pg_th = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_th})
    #nest.Connect(pg_th, topology_snn.find_population('MGN').nodes, 'all_to_all',
    #             syn_spec={'weight': parameters.noise_pars.w_noise_mgn})
    #nest.Connect(pg_th, topology_snn.find_population('TRN').nodes, 'all_to_all',
    #             syn_spec={'weight': parameters.noise_pars.w_noise_trn})

    pg_aone = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_aone})
    #nest.Connect(pg_aone, topology_snn.find_population('eA1').nodes, 'all_to_all',
    #             syn_spec={'weight': parameters.noise_pars.w_noise_ctx})
    #nest.Connect(pg_aone, topology_snn.find_population('iA1').nodes, 'all_to_all',
    #             syn_spec={'weight': parameters.noise_pars.w_noise_ctx})


    restore_network(topology_snn)

    '''
    # first stimulus generator
    first_ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 2000., 'stop' : 2000.+parameters.noise_pars.stim_duration})
    # connecting stimulus !!! generator to snn
    nest.Connect(first_ng, topology_snn.populations['MGN'].nodes[:40], 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})

    # second stimulus generator
    second_ng = nest.Create('poisson_generator', n=1, params={'rate': parameters.noise_pars.nuX_stim, 'start' : 6000., 'stop' : 6000.+parameters.noise_pars.stim_duration})
    # connecting stimulus !!! generator to snn
    nest.Connect(second_ng, topology_snn.populations['MGN'].nodes[41:80], 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})
    '''

    current_amp = 1000.
    tstart = 2000.
    step_current_attr = {
        "amplitude_times" : [tstart+0., tstart+300., tstart+400., tstart+700., tstart+800., tstart+1100., tstart+1200., tstart+1500., tstart+1600., tstart+1900.],
        "amplitude_values" : [current_amp, 0.] * 5
    }
    first_step_current_gen = nest.Create("step_current_generator")
    nest.SetStatus(first_step_current_gen, step_current_attr)
    nest.Connect(first_step_current_gen, topology_snn.populations['MGN'].nodes[:40], 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})

    tstart = 6000.
    step_current_attr = {
        "amplitude_times" : [tstart+0., tstart+300., tstart+400., tstart+700., tstart+800., tstart+1100., tstart+1200., tstart+1500., tstart+1600., tstart+1900.],
        "amplitude_values" : [current_amp, 0.] * 5
    }
    second_step_current_gen = nest.Create("step_current_generator")
    nest.SetStatus(second_step_current_gen, step_current_attr)
    nest.Connect(second_step_current_gen, topology_snn.populations['MGN'].nodes[41:80], 'all_to_all', syn_spec={'weight': parameters.noise_pars.w_noise_stim})

    sim_time = 10.
    nest.Simulate(sim_time * 1000)

    topology_snn.extract_activity(flush=False, t_start=0., t_stop=sim_time * 1000)  # this reads out the recordings

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
    precomputed = None
    
    print("[EXP FILE LOG] Pearson Coefficient Computed", flush=True)

    o = spikesandparams( parameters.label, activitylist, precomputed )

    filepath = os.path.join(storage_paths['activity'], f'spk_{parameters.label}')

    with open(filepath, 'wb') as f:
        pickle.dump(o, f)

    print("[EXP FILE LOG] Pickle File Saved", flush=True)