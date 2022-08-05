import torch
import argparse
from glucoenv import T1DEnv
from glucoenv.env.memory import Memory
from glucoenv.visualiser.T1DVisu import render
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--agent', type=str, default='cuda')
parser.add_argument('--n_env', type=int, default=2)
parser.add_argument('--steps', type=int, default=5)
args = parser.parse_args()

if __name__ == '__main__':
    n_env = args.n_env
    steps = args.steps
    ENV_DEVICE = args.device
    AGENT_DEVICE = args.agent
    print('\n#### GluCoEnv ####')
    print('The Environment device: ' + ENV_DEVICE + ' , Agent device: ' + AGENT_DEVICE)
    print('Number of environments: ', n_env)
    print('simulation Steps: ', steps)

    # demonstrate visualisation and memory
    env = T1DEnv.make(env='adolescent#001', n_env=n_env, scenario='moderate', device=ENV_DEVICE)
    SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=n_env, sample_time=env.sample_time,
                                       mode='error_quadratic', device=AGENT_DEVICE)
    mem = Memory(device=ENV_DEVICE, env=env, n_steps=steps)
    action = SBB.init_action()
    for i in range(0, steps):
        obs, rew, done, info = env.step(action)
        mem.update(step=i, glucose=info['BG'], CGM=obs, insulin=action,
                   CHO=info['meal'], MA=info['meal_announcement'], t=info['time'])
        action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
    render(mem, env_ids=[0])  # provide the id's of the required graphs,
