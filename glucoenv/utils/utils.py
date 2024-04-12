import os
import yaml
import shutil


class Yaml2Args(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def update(self, dict):
        self.__dict__.update(dict)


def load_yaml(FILE):
    stream = open(FILE, 'r')
    return yaml.safe_load(stream)


def setup_experiment_folders(folder_id):
    LOG_DIR = 'results/' + folder_id
    CHECK_FOLDER = os.path.isdir(LOG_DIR)
    if CHECK_FOLDER:
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)
    return LOG_DIR


def load_args(FILE, folder_id):
    d = load_yaml(FILE=FILE)
    ppo_args = Yaml2Args(d)
    dir = setup_experiment_folders(folder_id)
    ppo_args.update({'experiment_dir': dir})
    print('\nExperiment parameters:')
    print(vars(ppo_args))
    return ppo_args
