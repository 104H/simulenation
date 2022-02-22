import os
import numpy as np
import sys


def hold(init, end):
    job_list = np.arange(init, end)
    for i in job_list:
        os.system("scontrol hold {}".format(i))


if __name__ == '__main__':
    hold(int(sys.argv[1]), int(sys.argv[2]))
