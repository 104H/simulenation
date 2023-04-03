import nest
import sys
import os
import pandas as pd
import fna
import numpy as np
from pathos.multiprocessing import ProcessingPool as PathosPool
from scipy.signal import argrelmin, argrelmax
import pathos

sys.path.append("demyelination/") # we need this to successfully read pickle files

dirname = "demyelination/data/adaptation-thl-widerange/activity/"

lim = 2000

lowlim = 1000

t_start = 100.
t_stop = 1000.
tau = 10.

from fna.tools.signals import make_simple_kernel, pad_array
kernel = make_simple_kernel('exp', tau=tau, width=100., height=1./tau, resolution=1.)

files = sorted(os.listdir(dirname))

def process_files(arg_dict):
    results = []
    cnt = 0

    for filename in arg_dict['files_dict']:
        ok = True

        p = pd.read_pickle(f"{dirname}/{filename}")
        spikes_trn = p.spikeobj['TRN'].time_slice(t_start, t_stop)
        spikes_mgn = p.spikeobj['MGN'].time_slice(t_start, t_stop)

        mean_rate_mgn = p.spikeobj['MGN'].mean_rate(t_start=800., t_stop=2000.)
        mean_rate_trn = p.spikeobj['TRN'].mean_rate(t_start=800., t_stop=2000.)

        axis_, rates_trn = spikes_trn.averaged_instantaneous_rate(resolution=1., kernel=kernel, norm=1000)
        _, rates_mgn = spikes_mgn.averaged_instantaneous_rate(resolution=1., kernel=kernel, norm=1000)

        peak_mgn = max(rates_mgn)
        peak_trn = max(rates_trn)
        #print("PEAK | ", str(peak_mgn), str(peak_trn))

        t_idx_max_trn = np.where(rates_trn == peak_trn)[0][0]  # idx of peak
        #print("TIME IDX | ", str(t_idx_max_trn))

        #order = len(rates_trn[t_idx_max_trn + 1:])
        order = 100

        try:
            t_idx_pmin1_trn = argrelmin(rates_trn[t_idx_max_trn + 1:], order=order)[0][
                                  0] + t_idx_max_trn + 1  # first min after peak
            #print("ARGRELMIN | ", str(argrelmin(rates_trn[t_idx_max_trn + 1:], order=order)))

            t_idx_ppeak2_trn = argrelmax(rates_trn[t_idx_pmin1_trn + 1:], order=order)[0][
                                   0] + t_idx_pmin1_trn + 1  # second peak
        except:
            print(filename)

        min_1_trn = rates_trn[t_idx_pmin1_trn]
        #print("1 MIN AFTER PEAK | ", min_1_trn)

        peak_2_trn = rates_trn[t_idx_ppeak2_trn]
        #print("2 PEAK | ", peak_2_trn)

        if (mean_rate_trn < 9. or mean_rate_trn > 15.) \
                or (mean_rate_trn - mean_rate_mgn > 5.) \
                or (mean_rate_trn < 9. or mean_rate_trn > 15.) \
                or (mean_rate_mgn < 7. or mean_rate_mgn > 14.) \
                or (peak_mgn < 30.) \
                or (peak_mgn - peak_2_trn < 5.) \
                :
            ok = False

        '''
        if (mean_rate_trn - mean_rate_mgn < 3.) \
                or (mean_rate_trn - mean_rate_mgn > 5.) \
                or (mean_rate_trn < 9. or mean_rate_trn > 15.) \
                or (mean_rate_mgn < 7. or mean_rate_mgn > 14.) \
                or (peak_mgn < 30.) \
                or (peak_2_trn < 20.) \
                or (peak_mgn - peak_2_trn < 5.) \
                :
            ok = False

        if ok:
            results.append(filename)

        # identify the time and amplitude of the next n peaks
        # first min after peak
        t_idx_peak_trn = argrelmax(rates_trn[t_idx_max_trn + 1:], order=order)[0] + t_idx_max_trn + 1

        # identify the time and amplitude of the next n troughs
        t_idx_trgh_trn = argrelmin(rates_trn[t_idx_max_trn + 1:], order=order)[0] + t_idx_max_trn + 1

        # combine time array of peak and trough such that they alternate
        t_combine = []
        for x in zip(t_idx_peak_trn, t_idx_trgh_trn):
            t_combine.append(x[0])
            t_combine.append(x[1])

        t_combine = t_combine[1:] # remove the first
        print("t_combine | ", str(t_combine))

        # verify if the array is monotonically increasing
        ok = all(x<y for x, y in zip(t_combine, t_combine[1:]))

        if (mean_rate_trn > 18. or mean_rate_trn < 15.) \
            or (mean_rate_mgn > 13. or mean_rate_mgn < 8.):
            ok = False
        '''

        if ok:
            results.append(filename)

    return results


n_cpu = 6
n_files = 100
stepsize = int(n_files / n_cpu)

thread_args_dict = [{'files_dict': files[x:x + stepsize]} for x in np.arange(0, n_files, stepsize)]
pool = PathosPool(len(thread_args_dict))
pool_results = pool.map(process_files, thread_args_dict)

print(pool_results)

# file to write the valid (according to the filtering) data to
# with open('okres_adex_burst_weights_test3.data', 'w') as f:
'''
with open('okres_adex_burst_weights_wIn=0.9.data', 'w') as f:
    for r in pool_results:
        for l in sorted(r):
            f.write(l + "\n")
'''

