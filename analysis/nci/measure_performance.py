import time
from glucoenv import T1DEnv
import gc
import argparse
from csv import writer
import json

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_env', type=int, default=3)
parser.add_argument('--d_env', type=str, default='cpu')
parser.add_argument('--d_agent', type=str, default='cpu')
parser.add_argument('--file', type=str, default='test')
args = parser.parse_args()
#python measure_performance.py --file cpu --d_env cpu --d_agent cpu
if __name__ == '__main__':
    d_agent = args.d_agent
    d_env = args.d_env
    file = args.file
    steps = 100

    names = ['adolescent#001']#, 'adolescent#002']
    n_envs = [1, 2, 8, 16, 64, 128, 256] #[1, 5, 10, 20, 100, 1000]
    name = 'adolescent#001'
    exec_times = []
    trials = 5
    for n_env in n_envs:
        trial_exec = []
        for i in range(0, trials):
            env = T1DEnv.make(env=name, n_env=n_env, scenario='moderate', device=d_env)
            SBB = T1DEnv.benchmark_controller(env=name, n_env=n_env, sample_time=env.sample_time,
                                              mode='perfect', device=d_agent)
            action = SBB.init_action()
            print('\nn_env:{} '.format(n_env))
            tstart = time.perf_counter()
            for i in range(0, steps):
                obs, rew, done, info = env.step(action)
                action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
            tstop = time.perf_counter()
            print((tstop- tstart)/60)
            trial_exec.append((tstop- tstart)/60)
            del env
            del SBB
            gc.collect()
        exec_times.append(trial_exec)

    print(n_envs)
    print(exec_times)
    data = {'n_envs': n_envs, 'times': exec_times}
    with open(file+'.json', 'w') as f:
        json.dump(data, f)
