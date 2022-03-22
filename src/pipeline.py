
import pandas as pd
from matplotlib import pyplot as plt

import os
import sys
import re

class spikesandparams:
    def __init__ (self, paramdict, spikeobj):
        self.params = paramdict
        self.spikeobj = spikeobj

def readfiles (path):
    # read pickle files
    data = []

    for dirpath, _, filenames in os.walk(path):
        for fl in filenames:
            r = re.findall("_*?(\w+)=(\d+)_", fl)
            params = {}
            [ params.update( {p[0] : int(p[1])} ) for p in r ]

            d = spikesandparams( params, pd.read_pickle( os.path.join(dirpath, fl) ) )
            data.append( d )

    return data

def prepareplot (dataobjs, attr):
    # input dataframe
    # output plot object
    parts = dataobjs[0].spikeobj.keys()

    datapoints = {}
    for part in parts:
        datapointspart = []
        for data in dataobjs:
            if attr == 'meanfiringrate':
                datapointspart.append( (data.params['spk_test_run_nuX'], data.params['gamma'], data.spikeobj[part].mean_rate()) )

            if attr == 'pairwise_pearson_corrcoeff':
                datapointspart.append( (data.params['spk_test_run_nuX'], data.params['gamma'], data.spikeobj[part].pairwise_pearson_corrcoeff(nb_pairs=500, time_bin=2)[0]) )

            if attr == 'cv_isi':
                datapointspart.append( (data.params['spk_test_run_nuX'], data.params['gamma'], data.spikeobj[part].cv_isi().mean() ) )

        datapoints.update( {part : datapointspart} )

    return datapoints

def makesubplot (data, fig, ax, attr):
    p = ax[0].scatter([x[0] for x in data['MGN']], [x[1] for x in data['MGN']], c=[x[2] for x in data['MGN']])
    fig.colorbar(p, ax=ax[0])

    p = ax[1].scatter([x[0] for x in data['TRN']], [x[1] for x in data['TRN']], c=[x[2] for x in data['TRN']])
    fig.colorbar(p, ax=ax[1])

    for axis in ax:
        axis.set_xlabel("nuX")
        axis.set_ylabel("gamma")

    ax[0].set_title(attr + " MGN")
    ax[1].set_title(attr + " TRN")


def makeplot (rawdata):
    # input matplotlib plot object
    # output saved image as png
    fig, ax = plt.subplots(nrows=3, ncols=2)
    plt.subplots_adjust(left=0.01, right=0.03, top=0.03, bottom=0.02)

    attr = 'meanfiringrate'
    data = prepareplot(rawdata, attr)
    makesubplot(data, fig, ax[0], attr)

    attr = 'cv_isi'
    data = prepareplot(rawdata, attr)
    makesubplot(data, fig, ax[1], attr)

    attr = 'pairwise_pearson_corrcoeff'
    data = prepareplot(rawdata, attr)
    makesubplot(data, fig, ax[2], attr)

    fig.tight_layout()
    fig.savefig("/users/hameed/simulenation/src/demyelination/data/test_run/figures/fig.png")

def main():
    # read sys arg as path
    path = sys.argv[1]

    df = readfiles(path)

    makeplot(df)

if __name__ == "__main__":
    main()

