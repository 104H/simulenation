import os
import glob
from functools import reduce
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from itertools import compress
from pandas import DataFrame
import shutil
import sqlite3
from IPython.display import display

from fna.tools.utils import operations

from utils.parameters import ParameterSpace


def harvest_results(par_space, results_path, key_sets=None, operations=None, indices=None, flat=True, as_dict=False,
                    verbose=True):
    """

    :return:
    """
    table = {}

    if key_sets is None:
        key_sets = par_space.get_stored_keys(results_path)
        operations = [None for _ in range(len(key_sets))]
    if operations is None:
        operations = [None for _ in range(len(key_sets))]
    if indices is None:
        indices = [None for _ in range(len(key_sets))]

    for idx, (k, op, ind) in enumerate(zip(key_sets, operations, indices)):
        experiment_results = par_space.harvest(data_path=results_path, key_set=k, operation=op, verbose=verbose)
        if idx == 0:
            table['label'] = experiment_results[0]
        if op is not None:
            k += '-{}'.format(op.__name__)
        if ind is not None:
            if not isinstance(experiment_results[1], np.ndarray):
                exp_result = np.array(experiment_results[1])
            else:
                exp_result = experiment_results[1]
            exp_res = np.empty(shape=exp_result.shape)
            for index, x in np.ndenumerate(exp_result):
                if x is not None:
                    try:
                        exp_res[index] = x[ind]
                    except:
                        exp_res[index] = None
                else:
                    exp_res[index] = x
            table[k] = exp_res
        else:
            table[k] = experiment_results[1]

    if flat:
        for k, v in table.items():
            table[k] = v.flatten()
    if not as_dict:
        pars_info = get_parameter_values(par_space)
        dataset = pd.DataFrame(table)
        dataset['label'] = [x.lower() for x in dataset['label']]
        dataset = reduce(lambda x, y: pd.merge(x, y, how='outer', on='label'), [dataset, pars_info])
        dataset.set_index('label')
    else:
        dataset = table

    return dataset


def harvest_logs(pars, path, label, as_dict=False, verbose=True):
    """
    Extract relevant log information
    :param par_space:
    :param logs_path:
    :return:
    """
    log_dict = {'label': [], 'start-time': [], 'stop-time': [], 'total-duration': [], 'total-mem': [], 'peak-mem': [],
                'prepare-duration': [], 'simulate-duration': [], 'process-duration': []}
    for idx, par in enumerate(pars.parameter_sets):
        p_label = par.label
        log_dict['label'].append(p_label.lower())

        # harvest timers
        try:
            with open(path+label+'/logs/{}_timers.pkl'.format(p_label), 'rb') as fp:
                dd = pickle.load(fp)
            if verbose:
                print("Loading logs for {0}".format(p_label))
        except:
            if verbose:
                print("Dataset {0} Not Found, skipping".format(p_label))
            for k in ['start-time', 'stop-time', 'total-duration', 'peak-mem', 'total-mem', 'prepare-duration',
                      'simulate-duration', 'process-duration']:
                log_dict[k].append(np.nan)
            continue

        # main timer
        sim_key = 'simulation {}'.format(p_label)
        log_dict['start-time'].append(datetime.fromtimestamp(dd.timers[sim_key]['start']).strftime("%Y-%m-%d %H:%M"))
        log_dict['stop-time'].append(datetime.fromtimestamp(dd.timers[sim_key]['stop']).strftime("%Y-%m-%d %H:%M"))
        log_dict['total-duration'].append(dd.timers[sim_key]['duration'])

        # detailed timers
        timers_keys = ['prepare', 'simulate', 'process']
        for k in timers_keys:
            if '{} {}'.format(k, p_label) in dd.timers.keys():
                log_dict['{}-duration'.format(k)].append(dd.timers['{} {}'.format(k, p_label)]['duration'])
            else:
                log_dict['{}-duration'.format(k)].append(np.nan)

        # harvest memory consumption logs
        with open(path+label+'/logs/{}_memory.pkl'.format(p_label), 'rb') as fp:
            dd = pickle.load(fp)

        log_dict['peak-mem'].append(dd['peak-total-resource'])
        log_dict['total-mem'].append(dd['psutil'])

    if not as_dict:
        logs = pd.DataFrame(log_dict)
        logs.set_index('label')
    else:
        logs = log_dict
    return logs


def get_experiment_logs(path, label, description=None, status='CMPL'):
    # harvest submission logs and info
    with open(path+label+'/logs/{}_submission.log'.format(label), 'rb') as fp:
        dd = fp.readlines()
    ll = []
    for l in dd:
        if b"Calling command:" in l:
            ll.append(l)
    comm = ll[0].split()
    ddd = {'project_label': comm[comm.index(b"--project")+1].decode(),
           'system_label': comm[comm.index(b"--system")+1].decode(),
           'experiment_type': comm[comm.index(b"-c")+1].decode()}

    with open(path+label+'/logs/{}_submission_timers.pkl'.format(label), 'rb') as fp:
        dd = pickle.load(fp)
    submission_time = datetime.fromtimestamp(dd.timers['parameters']['start']).strftime("%Y-%m-%d %H:%M")

    paths = sorted(Path(path+label+'/logs/').iterdir(), key=os.path.getmtime)
    last_log = [str(x.resolve()) for x in paths[-3:] if "timers.pkl" in str(x.resolve()).split('_')][0]
    with open(last_log, 'rb') as fp:
        dd = pickle.load(fp)

    log_end = [datetime.fromtimestamp(v['stop']).strftime("%Y-%m-%d %H:%M") for k, v in dd.timers.items()
                    if 'simulation' in k.split(' ')]
    if not operations.empty(log_end):
        last_log_end = log_end[0]
    else:
        last_log_end = np.nan

    if ddd['system_label'] == 'local':
        executed_runs = len(next(os.walk(path+label+'/results/'))[2])
    else:
        assert os.path.isdir(path+label+'/cluster/'), "No job info available"
        executed_runs = len([file for file in Path(path+label+'/cluster/').rglob('*slurm-*')])

    exp = {
        'label': label.lower(),
        'type': ddd['experiment_type'],
        'project': ddd['project_label'],
        'num_runs': len(next(os.walk(path+label+'/parameters/'))[2]),
        'completed_runs': executed_runs,
        'data_size': sum(file.stat().st_size for file in Path(path+label).rglob('*')) / 1024**3,
        'system': ddd['system_label'],
        'start_time': submission_time,
        'end_time': last_log_end,
        'status': status,
        'description': description}
    return pd.DataFrame([exp])


def gather_metadata(pars, path, label):
    log_info = harvest_logs(pars, path, label, as_dict=False, verbose=False)
    desc = gather_main_descriptors(pars)
    with open(path+label+'/logs/{}_submission.log'.format(label), 'rb') as fp:
        dd = fp.readlines()
    ll = []
    for l in dd:
        if b"Calling command:" in l:
            ll.append(l)
    comm = ll[0].split()
    system = comm[comm.index(b"--system")+1].decode(),

    if system not in ['local']:
        jobs = path + label + '/cluster/slurm-*.out'
        j_list = glob.glob(jobs)

        job_df = {'label': [], 'job_id': [], 'error': []}

        for job in j_list:
            job_id = int(job.split('/')[-1].split('-')[-1].split('.')[0])
            with open(job, 'rb') as fp:
                cl_job_info = fp.readlines()
            try:
                sim_label = [x.split()[-3][:-1].decode() for x in cl_job_info if b"Simulation " in x][0]
                job_df['label'].append(sim_label)
                job_df['error'].append(None)
            except:
                error_msg = [x.split(b":")[-1].decode() for x in cl_job_info if b"error: " in x]#[0]
                if not operations.empty(error_msg):
                    error_msg = error_msg[0]
                job_df['error'].append(error_msg[:-2])
                job_df['label'].append(None)
            job_df['job_id'].append(int(job_id))

        job_df = pd.DataFrame(job_df)
        job_df = job_df.astype({"job_id": int})
        meta_data = reduce(lambda x, y: pd.merge(x, y, how='outer', on='label'), [log_info, job_df, desc])
    else:
        meta_data = reduce(lambda x, y: pd.merge(x, y, how='outer', on='label'), [log_info, desc])
    return meta_data


def gather_main_descriptors(pars):
    params = {'label': [], 'experiment-label': [], 'grng_seed': [], 'np_seed': [], 'nodes': [], 'ppn': [],
              'network-type': [], 'input-type': [], 'sequencer': [],
              'embedding': [], 'encoder': [], 'decoder': [], 'analysis-depth': []}

    for p in pars:
        params['label'].append(p.label.lower())
        params['experiment-label'].append(p.kernel_pars.data_prefix.lower())
        if "grng_seed" in p.kernel_pars.keys():
            params['grng_seed'].append(p.kernel_pars.grng_seed)
        else:
            params['grng_seed'].append(None)
        params['np_seed'].append(p.kernel_pars.np_seed)
        params['nodes'].append(int(p.kernel_pars.system.jdf_fields['{{ nodes }}']))
        params['ppn'].append(int(p.kernel_pars.system.jdf_fields['{{ ppn }}']))
        if hasattr(p.net_pars, "type"):
            params['network-type'].append(p.net_pars.type)
        else:
            params['network-type'].append(None)
        if hasattr(p.task_pars, "input_type"):
            params['input-type'].append(p.task_pars.input_type)
        else:
            params['input-type'].append(None)
        params['sequencer'].append(p.input_pars.discrete_sequencer[0])
        params['embedding'].append(p.input_pars.embedding[0]+'-'+p.input_pars.embedding[1])
        if "encoder_pars" in p.keys():
            params['encoder'].append(p.encoder_pars.device)
        else:
            params['encoder'].append(None)
        if "decoding_pars" in p.keys():
            params['decoder'].append(True)
        else:
            params['decoder'].append(False)
        if hasattr(p, "analysis_pars"):
            params['analysis-depth'].append(p.analysis_pars.depth)
        else:
            params['analysis-depth'].append(np.nan)
    return pd.DataFrame(params)


def get_parameter_values(pars, as_dict=False):
    range_vars = list(pars.parameter_ranges.keys())
    par_ranges = {'label': []}
    par_ranges.update({x: [] for x in range_vars})

    for idx, par in enumerate(pars.parameter_sets):
        p_label = par.label.split('_')
        par_ranges['label'].append(par.label.lower())

        for var_name in range_vars:
            var_indices = [var_name+'=' in x for x in p_label]
            variables = list(compress(p_label, var_indices))[0]
            try:
                value = float(variables.split('=')[1])
            except:
                value = variables.split('=')[1]

            par_ranges[var_name].append(value)

    if not as_dict:
        range_dict = pd.DataFrame(par_ranges)
        range_dict.set_index('label')
    else:
        range_dict = par_ranges
    return range_dict


def gather_all_metadata(data_path, label, description, status, clean_up_routine=None, to_db=False, verbose=False):
    # decompress
    dcmp = False
    if not os.path.isdir(data_path+label):
        dcmp = True
        shutil.unpack_archive(data_path+label+'.tar.gz', data_path+label)

    # read results
    parameters = ParameterSpace(data_path+label+'/ParameterSpace.py')

    # experiment metadata
    exp = get_experiment_logs(data_path, label, description=description, status=status)

    # gather all metadata
    md = gather_metadata(parameters, data_path, label)

    # correct experiment metadata
    exp['num_runs'] = len(get_parameter_values(parameters))
    exp.insert(5, 'canceled_runs', md['error'].count())#.loc[md['error']]
    exp.insert(6, 'missing_runs', len(md.loc[~md['label'].isnull() & md['start-time'].isnull()]))

    # clean dataset (you need to know exactly how)
    if clean_up_routine is not None:
        if isinstance(clean_up_routine, list):
            for comm in clean_up_routine:
                exec(comm)
        else:
            exec(clean_up_routine)
        # code = compile(clean_up_routine, 'data-clean', "exec")
        # exec(code)

    if verbose:
        display(exp)
        display(md)

    if dcmp:
        # remove decompressed files
        remove_dir_tree(data_path+label, verbose=verbose)
    if to_db:
        assert isinstance(to_db, str) and os.path.exists(to_db), "Please provide DB filename"
        with sqlite3.connect('/home/neuro/Desktop/org/computing_resources/fna-projects.db') as conn:
            exp.to_sql(name='experiments', con=conn, if_exists="append")
            md.to_sql(name='metadata', con=conn, if_exists="append")
    else:
        return exp, md


def unpack_data(path, label):
    """
    Extract compressed dataset
    :param path:
    :param label:
    :return:
    """
    if not os.path.isdir(path+label):
        shutil.unpack_archive(path+label+'.tar.gz', path+label)
    else:
        print("Directory {} already exists".format(path+label))


def remove_dir_tree(path, verbose=True):
    """
    Delete a given directory and its subdirectories.

    :param target: The directory to delete
    :param only_if_empty: Raise RuntimeError if any file is found in the tree
    """
    out = os.system("rm -rf {}".format(path))

    if not out and verbose:
        print("Successfully deleted {}".format(path))
    else:
        if verbose:
            print("{} was not successfully removed".format(path))
        remove_dir_tree(path)


def clean_array(x):
    """
    Remove None entries from an array and replace with np.nan
    :return:
    """
    for idx, val in np.ndenumerate(x):
        if val is None:
            x[idx] = np.nan
        elif operations.empty(val):
            x[idx] = np.nan
    return x


def convert_performance(readout, performance_dict):
    perf_dict = {readout.label+'-'+readout.task: performance_dict}
    df_dict = {}
    for k, v in perf_dict.items():
        df_dict.update({k: {}})
        for k1, v1 in v.items():
            if k1 in ['raw', 'max', 'label']:
                for k2, v2 in v1.items():
                    df_dict[k].update({k1+'-'+k2: v2})
    perf_data = DataFrame(df_dict)
    print("Decoding performance: \n{0}".format(perf_data))
    return perf_data


def un_pickle(pkl_file):
    try:
        with open(pkl_file, 'rb') as fp:
            data = pickle.load(fp)
        return data
    except:
        print("Pickle file {} not found".format(pkl_file))
