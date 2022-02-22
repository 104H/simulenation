import os
import numpy as np
import sys


def release(init, end):
    job_list = np.arange(init, end)
    for i in job_list:
        os.system("scontrol release {}".format(i))


if __name__ == '__main__':
    release(int(sys.argv[1]), int(sys.argv[2]))
