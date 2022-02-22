import shutil
import sys
import os


def compress(project_label, experiment_label, storage_path=None, rmtree=False):
    if storage_path is None:
        results_path = "{}/data/".format(project_label)
        export_path = "{}/data/export/".format(project_label)
    else:
        results_path = "{}/{}/".format(storage_path, project_label)
        export_path = "{}/{}/export/".format(storage_path, project_label)
    if os.path.exists(export_path+experiment_label) and not os.path.exists(results_path+experiment_label+'/cluster/'):
        shutil.copytree(export_path+experiment_label, results_path+experiment_label+'/cluster/')
        if rmtree:
            shutil.rmtree(export_path+experiment_label)
    shutil.make_archive(results_path+experiment_label, 'gztar', results_path+experiment_label)
    if rmtree:
        shutil.rmtree(results_path+experiment_label)


if __name__ == '__main__':
    compress(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
