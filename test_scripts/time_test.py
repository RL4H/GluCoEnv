import torch
from glucoenv import T1DEnv

device = 'cpu'
AGENT_DEVICE = device
ENV_DEVICE = device
steps = 20
n_env = 4
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    env = T1DEnv.make(env='adolescent#001', n_env=n_env, scenario='moderate', device=ENV_DEVICE)
    SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=n_env, sample_time=env.sample_time,
                                      mode='perfect', device=AGENT_DEVICE)
    action = SBB.init_action()
    for i in range(0, steps):
        obs, rew, done, info = env.step(action)
        action = SBB.get_action(meal=info['SBB_MA'], glucose=obs, t=info['time'], done=done)
print(prof.key_averages())