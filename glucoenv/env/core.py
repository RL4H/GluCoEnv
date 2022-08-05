import torch
import torch.nn as nn
import pandas as pd
import pkg_resources
from glucoenv.env.patient import T1DCohort
from glucoenv.env.sensor import Sensor
from glucoenv.env.pump import InsulinPump
from glucoenv.env.scenario import Scenario
torch.set_default_dtype(torch.float32)

SENSOR_PARA_FILE = pkg_resources.resource_filename('glucoenv', 'env/params/sensor_params.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('glucoenv', 'env/params/vpatient_params.csv')
INSULIN_PUMP_PARA_FILE = pkg_resources.resource_filename('glucoenv', 'env/params/pump_params.csv')
QUEST_FILE = pkg_resources.resource_filename('glucoenv', 'env/params/Quest.csv')


class Action(nn.Module):
    def __init__(self, n_env, device):
        super().__init__()
        self.n_env = n_env
        self.device = device
        self.action = nn.ParameterDict({'CHO': nn.Parameter(torch.ones(self.n_env, 1, device=self.device), requires_grad=False),
                                     'insulin': nn.Parameter(torch.ones(self.n_env, 1, device=self.device), requires_grad=False)})

    def set_action(self, ins=None, cho=None):
        if cho is not None:
            self.action['CHO'] = cho
        if ins is not None:
            self.action['insulin'] = ins

    def get_action(self):
        return self.action


class Info(nn.Module):
    def __init__(self, n_env, device):
        super().__init__()
        self.n_env = n_env
        self.device = device
        self.info = nn.ParameterDict({'sample_time': nn.Parameter(torch.ones(self.n_env, 1, device=self.device), requires_grad=False),
                                     'meal': nn.Parameter(torch.ones(self.n_env, 1, device=self.device), requires_grad=False)})

    def update(self, key, value):
        self.info[key] = value

    def add(self, key, value):
        self.info.update({key: nn.Parameter(torch.tensor(value, dtype=torch.float32, device=self.device), requires_grad=False)})

    def get(self):
        return self.info


def get_params_withID(patient_id):
    '''
    Construct patient by patient_id
    id are integers from 1 to 30.
    1  - 10: adolescent#001 - adolescent#010
    11 - 20: adult#001 - adult#001
    21 - 30: child#001 - child#010
    '''
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    params = patient_params.iloc[patient_id - 1, :]
    return params


def get_params_withName(name):
    '''
    Construct patient by name.
    Names can be
        adolescent#001 - adolescent#010
        adult#001 - adult#001
        child#001 - child#010
    '''
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    params = patient_params.loc[patient_params.Name == name].squeeze()
    return params


def get_quest_withName(name):
    '''
    Construct patient by name.
    Names can be
        adolescent#001 - adolescent#010
        adult#001 - adult#001
        child#001 - child#010
    '''
    patient_params = pd.read_csv(QUEST_FILE)
    params = patient_params.loc[patient_params.Name == name].squeeze()
    return params


def setup_sensor(name='Dexcom', n_env=None, device=None, seed=None):
    sensor_params = pd.read_csv(SENSOR_PARA_FILE)
    p = sensor_params.loc[sensor_params.Name == name].squeeze()
    param_torch_dict = nn.ParameterDict(
        {'Name': nn.Parameter(torch.tensor([0], dtype=torch.int32, device=device), requires_grad=False)})  # Name is passed as an int
    keys = p.index.values
    for i in range(1, len(keys)):
        param_torch_dict.update(
            {keys[i]: nn.Parameter(torch.tensor([p[keys[i]]], dtype=torch.float32, device=device), requires_grad=False)})
    sensor = Sensor(param_torch_dict, n_env, device, seed=seed)
    return sensor


def setup_pump(name='Insulet', device=None):
    pump_params = pd.read_csv(INSULIN_PUMP_PARA_FILE)
    p = pump_params.loc[pump_params.Name == name].squeeze()
    param_torch_dict = nn.ParameterDict(
        {'Name': nn.Parameter(torch.tensor([0], dtype=torch.int32, device=device), requires_grad=False)})  # Name is passed as an int
    keys = p.index.values
    for i in range(1, len(keys)):
        param_torch_dict.update(
            {keys[i]: nn.Parameter(torch.tensor([p[keys[i]]], dtype=torch.float32, device=device), requires_grad=False)})
    pump = InsulinPump(param_torch_dict)
    return pump


def setup_scenario(device, params, n_env, meal_type='normal', start_time = 1, seed=0):
    scenario = Scenario(device, params, n_env, meal_type='normal', start_time=1, seed=seed)
    return scenario


def setup_envs(device, patient_params, n_env, seed):
    n_patients = n_env
    params = []
    for ii in range(0, n_patients):
        patient_id = ii
        name = patient_params[ii]
        p = get_params_withName(name)
        param_torch_dict = nn.ParameterDict(
            {'Name': nn.Parameter(torch.tensor([patient_id], dtype=torch.int32, device=device), requires_grad=False)})
        keys = p.index.values
        for i in range(1, len(keys)):
            param_torch_dict.update(
                {keys[i]: nn.Parameter(torch.tensor([p[keys[i]]], dtype=torch.float32, device=device), requires_grad=False)})
        params.append(param_torch_dict)
    envs = T1DCohort(device, params, n_patients, init_state=None, random_init_bg=False, seed=seed, t0=0.0)
    return envs


def setup_benchmark_controller(device, patient_params, n_patients):
    params = []
    for ii in range(0, n_patients):
        patient_id = ii
        name = patient_params[ii]
        p = get_params_withName(name)
        q = get_quest_withName(name)
        param_torch_dict = nn.ParameterDict(
            {'Name': nn.Parameter(torch.tensor([patient_id], dtype=torch.int32, device=device), requires_grad=False)})
        keys = ['TDI', 'CR', 'CF']
        for i in keys:
            param_torch_dict.update(
                {i: nn.Parameter(torch.tensor([q[i]], dtype=torch.float32, device=device), requires_grad=False)})
        keys = ['u2ss', 'BW']
        for i in keys:
            param_torch_dict.update(
                {i: nn.Parameter(torch.tensor([p[i]], dtype=torch.float32, device=device), requires_grad=False)})
        params.append(param_torch_dict)
    return params

