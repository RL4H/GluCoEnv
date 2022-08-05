import yaml
import numpy as np
import pkg_resources
from glucoenv.env.env import T1DSimEnv
from glucoenv.agent.sbb.benchmark import SBB


def get_env_params(FILE):
    stream = open(FILE, 'r')
    dictionary = yaml.safe_load(stream)
    patients, n_env, scenario = [], 0, {}
    for p in dictionary['patients']:
        n_env += dictionary['patients'][p]['n_env']
        for i in range(0, dictionary['patients'][p]['n_env']):
            patients.append(dictionary['patients'][p]['name'])

    if not dictionary['use_individual_meal_protocol']: # a common meal protocol for all subjects
        scenario = {
            'prob': np.repeat(np.array([dictionary['meal_protocol']['probability']]), n_env, axis=0),
            'time_lb': np.repeat(np.array([dictionary['meal_protocol']['time_lower_bound']]), n_env, axis=0),
            'time_ub': np.repeat(np.array([dictionary['meal_protocol']['time_upper_bound']]), n_env, axis=0),
            'time_mu': np.repeat(np.array([dictionary['meal_protocol']['time_mu']]), n_env, axis=0),
            'time_sigma': np.repeat(np.array([dictionary['meal_protocol']['time_sigma']]), n_env, axis=0),
            'amount_mu': np.repeat(np.array([dictionary['meal_protocol']['amount_mu']]), n_env, axis=0),
            'amount_sigma': np.repeat(np.array([dictionary['meal_protocol']['amount_sigma']]), n_env, axis=0)
        }
    # todo: implement the individual scenarios

    sensor = dictionary['sensor']
    pump = dictionary['pump']
    device = dictionary['device']
    seed = dictionary['seed']
    env_type = dictionary['env_type']
    obs_type = dictionary['obs_type']
    reward_fun = None

    # get start time info.
    if dictionary['use_start_time']:
        start_time = np.repeat(np.array([dictionary['start_time']]), n_env, axis=0)
    else:
        if dictionary['start_midnight']:
            start_time = np.repeat(np.array([[0]]), n_env, axis=0)
        else:  # random start times
            start_time = np.random.randint(low=0, high=1439, size=(n_env,1))

    return device, patients, sensor, pump, scenario, n_env, reward_fun, start_time, seed, env_type, obs_type


def get_scenario(scenario_name, n_env):
    FILE = pkg_resources.resource_filename('glucoenv', 'settings/scenarios/' + scenario_name + '.yaml')
    stream = open(FILE, 'r')
    dictionary = yaml.safe_load(stream)
    scenario = {
        'prob': np.repeat(np.array([dictionary['meal_protocol']['probability']]), n_env, axis=0),
        'time_lb': np.repeat(np.array([dictionary['meal_protocol']['time_lower_bound']]), n_env, axis=0),
        'time_ub': np.repeat(np.array([dictionary['meal_protocol']['time_upper_bound']]), n_env, axis=0),
        'time_mu': np.repeat(np.array([dictionary['meal_protocol']['time_mu']]), n_env, axis=0),
        'time_sigma': np.repeat(np.array([dictionary['meal_protocol']['time_sigma']]), n_env, axis=0),
        'amount_mu': np.repeat(np.array([dictionary['meal_protocol']['amount_mu']]), n_env, axis=0),
        'amount_sigma': np.repeat(np.array([dictionary['meal_protocol']['amount_sigma']]), n_env, axis=0)
    }
    return scenario


def make(settings=None, env=None, n_env=None, env_type=None, obs_type=None, scenario=None, device=None, **kwargs):
    # setting or (env, n_env) is a must!
    if (settings is None) and (env is None):
        print('\n#### GluCoEnv ####')
        print('ERROR: Please provide settings yaml or define env.')
        print('A sample settings file is located in: settings/settings.yaml.')
        print('Or choose from possible enviroments: adolescent#001, adolescent#002,..., adult#001,..., child#001,...')
        print('Documentation: ')
        exit()

    if settings is not None:
        device, patients, sensor, pump, scenario, n_env, reward_fun, start_time, seed, env_type, obs_type = get_env_params(
            FILE=settings)
    else:
        n_env = 1 if n_env is None else n_env
        patients = [env] * n_env
        scenario_name = 'moderate' if scenario is None else scenario
        scenario = get_scenario(scenario_name=scenario_name, n_env=n_env)
        device = 'cpu' if device is None else device
        # optional params
        pump = 'Insulet'
        sensor = 'GuardianRT'
        seed = 0
        env_type = 'train' if env_type is None else env_type
        obs_type = 'current' if obs_type is None else obs_type
        if env_type == 'train':
            start_time = np.random.randint(low=0, high=1439, size=(n_env, 1))
        else:  # testing starts at midnight
            start_time = np.repeat(np.array([[0]]), n_env, axis=0)

        for key, value in kwargs.items():
            if key == 'pump':
                pump = value
            if key == 'sensor':
                sensor = value
            if key == 'seed':
                seed = value
            if key == 'start_time':
                start_time = value

    # device, patients, sensor, pump, scenario, n_env, reward_fun, start_time, seed = get_env_params(
    #     FILE="RLGluCon/settings/settings.yaml")
    env = T1DSimEnv(device=device, patients=patients, sensor=sensor, pump=pump, scenario=scenario, n_env=n_env,
                    reward_fun=None, start_time=start_time, seed=seed, obs_type=obs_type)
    return env


def benchmark_controller(settings=None, env=None, n_env=None, mode=None, sample_time=None, device=None, env_device=None):
    if (settings is None) and (env is None):
        print('\n#### GluCoEnv ####')
        print('ERROR: Please provide settings yaml or define env, corresponding to the created environment.')
        exit()
    if sample_time is None:
        print('\n#### GluCoEnv ####')
        print('ERROR: Please provide sample time for the environment.')
        print('HINT: Try env.sample_time.')
        exit()

    if settings is not None:
        _, patients, _, _, _, n_env, _, _, _, _, _ = get_env_params(FILE=settings)
    else:
        n_env = n_env
        patients = [env] * n_env
    device = 'cpu' if device is None else device
    env_device = 'cpu' if env_device is None else env_device
    controller = SBB(device, patients, n_env, mode, sample_time, env_device)
    return controller
