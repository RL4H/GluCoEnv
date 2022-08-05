import torch
torch.set_default_dtype(torch.float32)
import argparse
from glucoenv import T1DEnv
from glucoenv.agent.ppo.ppo import PPO
from glucoenv.utils.utils import load_args

parser = argparse.ArgumentParser()
parser.add_argument('--folder_id', type=str, default='test3')
args = parser.parse_args()


if __name__ == '__main__':
    ppo_args = load_args(FILE='glucoenv/agent/ppo/config.yaml', folder_id='test123')
    env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='train', obs_type='past_history', scenario='moderate', device='cpu')
    eval_env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='eval', obs_type='past_history', scenario='moderate', device='cpu')
    model = PPO(args=ppo_args, env=env, eval_env=eval_env, device='cpu')
    model.learn(total_timesteps=500000)









