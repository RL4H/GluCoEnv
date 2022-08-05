import torch
from glucoenv import T1DEnv

# test custom settings.
env = T1DEnv.make(settings='custom_settings.yaml')
action = torch.ones(3, 1) * 0.01
for i in range(0, 30):
    obs, rew, done, info = env.step(action)
print('Successful1')

# test passing env params, hard.
env = T1DEnv.make(env='child#001', n_env=3, scenario='hard')
action = torch.ones(3, 1) * 0.01
for i in range(0, 30):
    obs, rew, done, info = env.step(action)
print('Successful2')

# test passing env params, moderate.
env = T1DEnv.make(env='adolescent#001', n_env=3, scenario='moderate')
action = torch.ones(3, 1) * 0.01
for i in range(0, 30):
    obs, rew, done, info = env.step(action)
print('Successful3')

# test passing env params, easy.
env = T1DEnv.make(env='adolescent#001', n_env=3, scenario='easy', sensor='Dexcom', pump='Cozmo')
action = torch.ones(3,1) * 0.01
for i in range(0, 30):
    obs, rew, done, info = env.step(action)
print('Successful4')


# test passing env params, benchmark, no n_env defined
env = T1DEnv.make(env='adult#001', scenario='benchmark')
action = torch.ones(1,1) * 0.01
for i in range(0, 30):
    obs, rew, done, info = env.step(action)
print('Successful5')
