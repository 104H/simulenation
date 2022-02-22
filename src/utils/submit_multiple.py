import os
import sys


#def submit_jobs(start_idx=0, stop_idx=None):
def submit_jobs(njobs, offset_start):
    with open('job_list.txt') as fp:
        n_lines = len(list(enumerate(fp)))
        for offset in range(offset_start, (n_lines+1) // njobs + 1):
            os.system('sbatch ../../../../utils/master_dev.sh {0} {1}'.format(offset, njobs))


if __name__ == '__main__':
    submit_jobs(int(sys.argv[1]), int(sys.argv[2]))
