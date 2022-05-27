"""
Submits multiple jobs, each containing multiple experiments/simulations.
Call:
    python submit_multijobs_multisim_main.py offset_start n_sim_per_job
"""

import os
import sys


def submit_jobs(offset_start, n_sim_per_job):
    with open('job_list.txt') as fp:
        n_lines = len(list(enumerate(fp)))
        for offset in range(offset_start, (n_lines+1) // n_sim_per_job):
            os.system(f'sbatch --ntasks={n_sim_per_job} --cpus-per-task={int(256 / n_sim_per_job)} ../../../../utils/submit_multijobs_multisim_main.sh {offset} {n_sim_per_job}')


if __name__ == '__main__':
    submit_jobs(int(sys.argv[1]), int(sys.argv[2]))
