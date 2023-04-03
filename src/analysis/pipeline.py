
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import os
import sys
import re
import sys

sys.path.append("demyelination/")

'''
class spikesandparams:
    def __init__ (self, paramdict, spikeobj):
        self.params = paramdict
        self.spikeobj = spikeobj
'''

def readfiles (path):
    # read pickle files
    #print("[PIPELINE readfiles] fetching data from folder ", path)
    data = []

    for dirpath, _, filenames in os.walk(path):
        #print("[PIPELINE readfiles] fetched filenames")
        if len(filenames) == 0:
            raise Exception("No activity files found.")
   
        for fl in filenames:
            path = os.path.join(dirpath, fl)
            #print("[PIPELINE readfiles] reading file ", path)
            d = pd.read_pickle( path )
            #print("[PIPELINE readfiles] read file ")
            
            r = re.findall("_(\w+)=([0-9]*[.]*[0-9]*)", fl)
            paramsfromfilename = {}
            [ paramsfromfilename.update( {p[0] : float(p[1])} ) for p in r ]
            d.params = paramsfromfilename
            
            data.append( d )
            #print("[PIPELINE readfiles] reading file ")
    return data

'''
def preparedf (dataobjs):
    # input dataframe
    # output organized data
    if len(dataobjs) == 0:
        raise Exception("No activity files found.")

    parts = dataobjs[0].spikeobj.keys()

    df = pd.DataFrame()

    for part in parts:
        for data in dataobjs:
            data.spikeobj[part].time_offset(250)
            df = df.append({
                "nux" : data.params['exp3_nuX'],
                "gamma" : data.params['gamma'],
                "brainPart" : part,
                "meanSpkRate" : data.spikeobj[part].mean_rate(),
                "pearsonCoeff" : data.metrics['pearsoncoeff'][part],
                "cvIsi" : data.spikeobj[part].cv_isi().mean()
                }, ignore_index=True)

    sortingParams = ["nux", "gamma"]
    df = df.sort_values(by=sortingParams)

    return df


def makesubplot (data, fig, ax, attr):
    for axis, part in zip([0, 1], ["MGN", "TRN"]):
        d = data[ data["brainPart"] == part ].pivot_table(attr, "nux", "gamma")

        p = ax[axis].imshow(d, interpolation="nearest")
        fig.colorbar(p, ax=ax[axis])

        ax[axis].set_title(attr + " " + part)

        ax[axis].set_yticklabels(d.index.values)
        ax[axis].set_xticklabels(d.columns.values)

        ax[axis].set_ylabel("nuX")
        ax[axis].set_xlabel("gamma")

'''

def preparedf (dataobjs, timeframe):
    '''
        given pickle files of spike objects, prepare a dataframe
        Args:
            dataobjs: iterable of pickle files
            timeframe: iterable of the time values of the simulation during which the metrics should be computed
    '''
    parts = dataobjs[0].spikeobj.keys()

    df = pd.DataFrame()

    for part in parts:
        for data in dataobjs:
            #data.spikeobj[part].time_offset(250)
            tmp = data.params
            tmp = tmp.update({
                "brainPart" : part,
                "meanSpkRate" : data.spikeobj[part].mean_rate(),
                "pearsonCoeff" : data.metrics['pearsoncoeff'][part],
                #"pearsonCoeffHundren" : data.metrics['pearsoncoeffHundred'][part],
                "cvIsi" : data.spikeobj[part].cv_isi().mean(),
                "meanBurstSpikePercentageCerina" : np.mean(computePopulationBurstSpikeRatio( data.spikeobj[part].time_slice(timeframe[0], timeframe[1]), "cerina" )),
                "meanBurstSpikePercentageMIT" : np.mean(computePopulationBurstSpikeRatio( data.spikeobj[part].time_slice(timeframe[0], timeframe[1]), "mit" )),
            })
            df = df.append(data.params, ignore_index=True)

    return df

def preparestimdf (dataobjs, stimstart=2000, stimdur=100):
    '''
        given pickle files of spike objects, prepare a dataframe for pre, during and post stim analysis
        Args:
            dataobjs: iterable of pickle files
            stimstart: integer. start time of stimulation. default 2000
            stimdur: integer. duration of the stimulation. default 100
    '''
    parts = dataobjs[0].spikeobj.keys()
    predur = 1000 # number of seconds before the stim from which to analyse
    postdur = 1000 # number of seconds before the stim from which to analyse

    df = pd.DataFrame()

    for part in parts:
        for data in dataobjs:
            #data.spikeobj[part].time_offset(250)
            tmp = data.params
            tmp = tmp.update({
                "brainPart" : part,
                "pre_meanSpkRate" : data.spikeobj[part].time_slice(stimstart-predur, stimstart).mean_rate(),
                "peri_meanSpkRate" : data.spikeobj[part].time_slice(stimstart, stimstart+stimdur).mean_rate(),
                "post_meanSpkRate" : data.spikeobj[part].time_slice(stimstart+stimdur, stimstart+stimdur+postdur).mean_rate(),
                
                
                "pearsonCoeff" : data.metrics['pearsoncoeff'][part],
                "cvIsi" : data.spikeobj[part].cv_isi().mean(),
                "meanBurstSpikePercentageCerina" : np.mean(computePopulationBurstSpikeRatio(data.spikeobj[part].time_slice(1000, 2000), "cerina")),
                "meanBurstSpikePercentageMIT" : np.mean(computePopulationBurstSpikeRatio(data.spikeobj[part].time_slice(1000, 2000), "mit")),
            })
            df = df.append(data.params, ignore_index=True)

    return df

def makesubplot (data, fig, ax, attr, varone, vartwo, brainParts, title="", unit=""):
    #varone, vartwo = "nuX_aone", "nuX_th"
    #for axis, part in zip([0, 1], ["MGN", "TRN"]):
    for axis, part in enumerate(brainParts):
        d = data[ data["brainPart"] == part ].pivot_table(attr, varone, vartwo)

        p = ax[axis].imshow(d, interpolation="nearest")
        
        cb = fig.colorbar(p, ax=ax[axis])
        cb.set_label(unit)

        ax[axis].set_title(title + " " + attr + " " + part)

        j = 2
        
        ax[axis].set_yticks(range(0, len(d.index.values), j))
        ax[axis].set_yticklabels(d.index.values[::j])
        
        ax[axis].set_xticks(range(0, len(d.columns.values), j))
        ax[axis].set_xticklabels(d.columns.values[::j])
        
        ax[axis].set_ylabel(varone)
        ax[axis].set_xlabel(vartwo)

def makeplot (rawdata):
    # input matplotlib plot object
    # output saved image as png
    fig, ax = plt.subplots(nrows=3, ncols=2)

    fig.set_size_inches(13, 9)
    plt.subplots_adjust(left=0.01, right=0.03, top=0.03, bottom=0.02)

    data = preparedf(rawdata)
    print(data)

    makesubplot(data, fig, ax[0], "cvIsi")

    makesubplot(data, fig, ax[1], "meanSpkRate")

    makesubplot(data, fig, ax[2], "pearsonCoeff")

    fig.tight_layout()
#     fig.savefig("/users/hameed/simulenation/src/demyelination/data/test_run/figures/fig.png")

def computeBurstSpikeRatio (spiketrain, criterion="cerina"):
    if criterion == "cerina":
        quiettime = 8.
        minspikes = 4.
        preceding_silence = 50.
    elif criterion == "mit":
        quiettime = 20.
        minspikes = 2.
        preceding_silence = 100.
    else:
        raise Exception("Criterion not implemented")
        
    spkdiff = np.diff(spiketrain)

    count = 0
    idx = 1

    while idx < len(spkdiff):
        # was the last spike less than 50ms/100ms ago
        if spkdiff[idx-1] <= preceding_silence:
            idx += 1
            continue

        # how many more burst spikes can we find?
        c = 1
        
        # is the next spike within 4ms
        if spkdiff[idx] <= quiettime:
            # how many of the following spikes are within 8ms/20ms
            # we need at least 3 more for this to be a burst
            
            try:
                while spkdiff[idx+c] <= quiettime:
                    c += 1

                if c >= minspikes:
                    count += c + 1
                    idx += c
                    continue # restart the loop
            except:
                pass
        
        idx += 1

    return 0 if len(spiketrain) == 0 else 100 * count / len(spiketrain)

def computePopulationBurstSpikeRatio (population, criterion="cerina"):
    k = population.spiketrains.keys()
    
    return [computeBurstSpikeRatio(population.spiketrains[_].spike_times, criterion) for _ in k]

def makerasterplot (fls, path, fig, ax):
    for idx, fl in enumerate(fls):
        d = pd.read_pickle(path + fl)

        fig.set_size_inches(50, 9)
        #plt.subplots_adjust(left=0.01, right=0.03, top=0.03, bottom=0.02)

        for axis, part in zip([0, 1], d.spikeobj.keys()):
            clr = 'red' if part == 'MGN' else 'blue'

            d.spikeobj[part].raster_plot(ax=ax[idx][axis], dt=10, display=False, color=clr)
            ax[idx][axis].set_ylabel(part)
            ax[idx][axis].set_title(fl)

    fig.tight_layout()

def smoothedspikes(spikes):
    tau = 10
    
    from fna.tools.signals import make_simple_kernel, pad_array
    kernel = make_simple_kernel('exp', tau=tau, width=100., height=1./tau, resolution=1.)
    
    #spikes = spikes.time_slice(0, l)

    axis_, rates = spikes.averaged_instantaneous_rate(resolution=1., kernel=kernel, norm=1000)
    
    return axis_, rates

def plotPowerSpectra (picklefile, mode="meanfiringrate"):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    ax = ax.flatten()

    for brainPart, axis in zip(['MGN', 'TRN', 'eA1', 'iA1'], ax):
        if mode == "meanfiringrate":
            data = p.spikeobj[brainPart].firing_rate(10, average=True)
        elif mode == 'kernelsmoothing':
            data = pipeline.smoothedspikes(p.spikeobj[brainPart])[1]
        else:
            raise Exception("Available modes are `meanfiringrate` and `kernelsmoothing`")

        ps = np.abs(np.fft.fft(data))**2

        time_step = 1 / 30
        freqs = np.fft.fftfreq(data.size, time_step)
        idx = np.argsort(freqs)
        
        dB = 10 * np.log( ps[idx] / max(ps[idx]) )

        axis.plot(freqs[idx], dB, label=brainPart)
        
        axis.legend()
        axis.set_xlabel("")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power (dB)")
        axis.set_ylim(-150, 0)

def preparestim_tonotopicdf(dataobjs, parts=('MGN', 'eA1')):
    timeframes = ()
    
    df = pd.DataFrame()

    for part in parts:
        for data in dataobjs:
            neuron_idx_start = int (list(data.spikeobj[part].spiketrains.keys())[0])
            stim_neuron_idx = int (len(data.spikeobj[part].spiketrains.keys()) * 1/5) +neuron_idx_start
            neuron_idx_end = int (list(data.spikeobj[part].spiketrains.keys())[-1])
            
            tmp = data.params
            tmp = tmp.update({
                "brainPart" : part,
                "pre_stim_meanrate" : data.spikeobj[part].time_slice(1800, 2000).id_slice(np.arange(neuron_idx_start, stim_neuron_idx+1)).mean_rate(),
                "peri_stim_meanrate" : data.spikeobj[part].time_slice(2000, 2100).id_slice(np.arange(neuron_idx_start, stim_neuron_idx+1)).mean_rate(),
                "post_stim_meanrate" : data.spikeobj[part].time_slice(2200, 2600).id_slice(np.arange(neuron_idx_start, stim_neuron_idx+1)).mean_rate(),
                
                "pre_nonstim_meanrate" : data.spikeobj[part].time_slice(1800, 2000).id_slice(np.arange(stim_neuron_idx, neuron_idx_end+1)).mean_rate(),
                "peri_nonstim_meanrate" : data.spikeobj[part].time_slice(2000, 2100).id_slice(np.arange(stim_neuron_idx, neuron_idx_end+1)).mean_rate(),
                "post_nonstim_meanrate" : data.spikeobj[part].time_slice(2200, 2600).id_slice(np.arange(stim_neuron_idx, neuron_idx_end+1)).mean_rate()
            })
            df = df.append(data.params, ignore_index=True)

    return df

def firingRatePlot (pth, f, mode="meanfiringrate"):
    p = pd.read_pickle(pth + f[0])
    fig, ax = plt.subplots(nrows=5, ncols=1)
    fig.set_size_inches(20, 24)

    for x in [0, 1, 2, 3]:
        ax[x].set_frame_on(False)
        ax[x].tick_params(labelbottom=False)
        ax[x].get_yaxis().set_ticks([])

    tstart = 1800
    tstop = 2300

    p.spikeobj['TRN'].time_slice(tstart, tstop).raster_plot(ax=ax[0], dt=10, display=False, color='lightblue')
    p.spikeobj['MGN'].time_slice(tstart, tstop).raster_plot(ax=ax[1], dt=10, display=False, color='pink')
    p.spikeobj['eA1'].time_slice(tstart, tstop).raster_plot(ax=ax[2], dt=10, display=False, color='brown')
    p.spikeobj['iA1'].time_slice(tstart, tstop).raster_plot(ax=ax[3], dt=10, display=False, color='orange')
    
    colours = {'TRN' : 'lightblue', 'MGN' : 'pink', 'eA1' : 'brown', 'iA1' : 'orange'}

    for idx, brainPart in enumerate(['TRN', 'MGN', 'eA1', 'iA1']):
        if mode == 'meanfiringrate':
            ax[4].plot(p.spikeobj[brainPart].time_slice(tstart, tstop).firing_rate(10, average=True), label=brainPart, c=colours[brainPart])
            
        elif mode == 'kernelsmoothing':
            x, y = pipeline.smoothedspikes(p.spikeobj[brainPart].time_slice(tstart, tstop))
            ax[4].plot(x, y, label=brainPart, c=colours[brainPart])

    plt.xlabel("Time (ms)")
    plt.ylabel("Mean Spiking Rate")
    #plt.axhline(25, c='brown', label="y=25")

    #plt.xticks(range(0, 60+1, 10), range(tstart, tstop+1, 100))
    ax[2].spines.right.set_visible(False)
    ax[2].spines.top.set_visible(False)

    ax[4].legend()

def main():
    # read sys arg as path
    path = sys.argv[1]

    df = readfiles(path)

    makeplot(df)

if __name__ == "__main__":
    main()

