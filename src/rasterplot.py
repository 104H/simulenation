
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("demyelination/")

def main():
    #fl = sys.argv[1]
    fls = [
            "spk_test_run_nuX=14_gamma=4_T=1",
            "spk_test_run_nuX=14_gamma=5_T=1",
            "spk_test_run_nuX=12_gamma=4_T=1",
            ]

    fig, ax = plt.subplots(nrows=len(fls), ncols=2)

    for idx, fl in enumerate(fls):
        d = pd.read_pickle("demyelination/data/test_run/activity/" + fl)

        fig.set_size_inches(50, 9)
        #plt.subplots_adjust(left=0.01, right=0.03, top=0.03, bottom=0.02)

        for axis, part in zip([0, 1], d.spikeobj.keys()):
            d.spikeobj[part].raster_plot(ax=ax[idx][axis], dt=10, display=False)
            ax[idx][axis].set_ylabel(part)
            ax[idx][axis].set_title(fl)

    fig.tight_layout()
    fig.savefig("/users/hameed/simulenation/src/demyelination/data/test_run/figures/rasterplots.png")


if __name__ == "__main__":
    main()

