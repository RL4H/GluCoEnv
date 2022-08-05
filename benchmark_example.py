import time
import torch
import argparse
from glucoenv import T1DEnv
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--agent', type=str, default='cuda')
parser.add_argument('--n_env', type=int, default=3)
parser.add_argument('--steps', type=int, default=5)
args = parser.parse_args()

# python 3.8.10
# pytorch 1.12.0 with cuda 11.3

if __name__ == '__main__':
    n_env = args.n_env
    steps = args.steps
    ENV_DEVICE = args.device
    AGENT_DEVICE = args.agent
    print('\n#### GluCoEnv ####')
    print('The Environment device: ' + ENV_DEVICE + ' , Agent device: ' + AGENT_DEVICE)
    print('Number of environments: ', n_env)
    print('simulation Steps: ', steps)

    # manual argument parsing.
    # env = T1DEnv.make(env='adolescent#001', n_env=n_env, scenario='moderate', device=ENV_DEVICE)
    # SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=n_env, sample_time=env.sample_time,
    #                                   mode='perfect', device=AGENT_DEVICE, env_device=ENV_DEVICE)

    # using settings yaml.
    env = T1DEnv.make(settings='example_settings.yaml')
    SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=env.n_env, sample_time=env.sample_time,
                                      mode='perfect', device=env.device, env_device=ENV_DEVICE)

    action = SBB.init_action()
    start = time.time()
    for _ in range(0, steps):
        obs, rew, done, info = env.step(action)
        action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
    end = time.time()
    print('total run time: ', end-start)
    print('Successful')









