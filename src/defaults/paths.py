import os
from pathlib import Path

"""
Defines paths to various directories and files specific to the running mode (locally or on a cluster).
Should be modified by users according to their particular needs.
"""


def set_project_paths(system, project_label):
	wd = os.path.abspath(Path(__file__).parent.parent)
	paths = {
		'local': {
			'data_path': 		wd + '/' + project_label + '/data/',  # output directory, must be created before running
			'jdf_template': 	None,								  # cluster template not needed
			'matplotlib_rc': 	wd + '/defaults/matplotlibrc',	      # custom matplotlib configuration
			'remote_directory': wd + '/' + project_label + '/data/export/',	# directory for export scripts to be run on cluster
			'queueing_system':  None},									# only when running on clusters

		'Hambach': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/Hambach.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlibrc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'slurm'},

		'Jureca': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/Jureca_jdf.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlibrc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'slurm'},

		'Jureca-CPU': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/Jureca-CPU.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlibrc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'slurm'},

		'Jureca-DC-CPU': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/Jureca-DC-CPU.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlibrc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'slurm'},
		'Jureca-DC-CPU-DEV': {
			'data_path': wd + '/' + project_label + '/data/',
			'jdf_template': wd + '/defaults/cluster_templates/Jureca-DC-CPU.sh',
			'matplotlib_rc': wd + '/defaults/matplotlibrc',
			'remote_directory': wd + '/' + project_label + '/data/export/',
			'queueing_system': 'slurm'},
		'Jureca-GPU': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/Jureca-GPU.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlibrc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'slurm'},

		'MPI': {
			'data_path':            wd + '/' + project_label + '/data/',
			'jdf_template':         wd + '/defaults/cluster_templates/MPI.sh',
			'matplotlib_rc':        wd + '/defaults/matplotlib_rc',
			'remote_directory':     wd + '/' + project_label + '/data/export/',
			'queueing_system':      'sge'},
	}
	return {system: paths[system]}

