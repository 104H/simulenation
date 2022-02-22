import os
import sys


def submit_jobs(job_list_file, start_idx=0, stop_idx=None):
	with open(job_list_file) as fp:
		for idx, line in enumerate(fp):
			if stop_idx is not None:
				if (idx >= start_idx) and (idx <= stop_idx):
					os.system('sbatch {0}'.format(line))
					# os.system('qsub {0}'.format(line)) SGE
			else:
				os.system('sbatch {0}'.format(line))
				# os.system('qsub {0}'.format(line)) SGE


if __name__ == '__main__':
	if len(sys.argv) > 2:
		submit_jobs(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
	else:
		submit_jobs(str(sys.argv[1]))