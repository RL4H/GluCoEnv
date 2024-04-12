import torch
torch.set_default_dtype(torch.float32)
import argparse
from glucoenv import T1DEnv
from glucoenv.agent.ppo.ppo import PPO
from glucoenv.utils.utils import load_args

parser = argparse.ArgumentParser()
parser.add_argument('--folder_id', type=str, default='test3')
parser.add_argument('--env', type=str, default='adolescent#001')
parser.add_argument('--n_env', type=int, default=2)
parser.add_argument('--d_env', type=str, default='cpu')
parser.add_argument('--d_agent', type=str, default='cpu')
args = parser.parse_args()

#'adolescent#001'

if __name__ == '__main__':
    ppo_args = load_args(FILE='../glucoenv/agent/ppo/config.yaml', folder_id='test123')
    env = T1DEnv.make(env=args.env, n_env=args.n_env, env_type='train', obs_type='past_history', scenario='moderate', device=args.d_env)
    eval_env = T1DEnv.make(env=args.env, n_env=args.n_env, env_type='eval', obs_type='past_history', scenario='moderate', device=args.d_env)
    model = PPO(args=ppo_args, env=env, eval_env=eval_env, device=args.d_agent)
    model.learn(total_timesteps=2000)


# python ppo_example.py --env adolescent#001 --n_env 2 --d_env cpu --d_agent cpu






