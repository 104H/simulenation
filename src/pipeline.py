
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
    data = []

    for dirpath, _, filenames in os.walk(path):
        for fl in filenames:
            '''
            r = re.findall("_*?(\w+)=(\d+)_", fl)
            params = {}
            [ params.update( {p[0] : int(p[1])} ) for p in r ]

            d = spikesandparams( params, pd.read_pickle( os.path.join(dirpath, fl) ) )
            data.append( d )
            '''
            path = os.path.join(dirpath, fl)
            d = pd.read_pickle( os.path.join(dirpath, fl) )
            data.append( d )

    return data

def preparedf (dataobjs):
    # input dataframe
    # output organized data
    if len(dataobjs) == 0:
        raise Exception("No activity files found.")

    parts = dataobjs[0].spikeobj.keys()

    df = pd.DataFrame()

    for part in parts:
        for data in dataobjs:
            df = df.append({
                "nux" : data.params['test_run_nuX'],
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

def makeplot (rawdata):
    # input matplotlib plot object
    # output saved image as png
    fig, ax = plt.subplots(nrows=3, ncols=2)

    fig.set_size_inches(13, 9)
    plt.subplots_adjust(left=0.01, right=0.03, top=0.03, bottom=0.02)

    data = preparedf(rawdata)

    makesubplot(data, fig, ax[0], "cvIsi")

    makesubplot(data, fig, ax[1], "meanSpkRate")

    makesubplot(data, fig, ax[2], "pearsonCoeff")

    fig.tight_layout()
    fig.savefig("/users/hameed/simulenation/src/demyelination/data/test_run/figures/fig.png")

def main():
    # read sys arg as path
    path = sys.argv[1]

    df = readfiles(path)

    makeplot(df)

if __name__ == "__main__":
    main()

