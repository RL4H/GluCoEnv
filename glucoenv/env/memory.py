import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)


class Memory(nn.Module):
    def __init__(self, device, env, n_steps):
        super().__init__()
        self.device = device
        self.n_env = env.n_env
        self.n_steps = n_steps
        self.sample_time = env.sample_time
        self.sensor_params = env.sensor_params
        self.pump_params = env.pump_params
        self.t = torch.zeros(self.n_steps, self.n_env, device=self.device, dtype=torch.float32)
        self.glucose = torch.zeros(self.n_steps, self.n_env,  device=self.device, dtype=torch.float32)
        self.cgm = torch.zeros(self.n_steps, self.n_env, device=self.device, dtype=torch.float32)
        self.insulin = torch.zeros(self.n_steps, self.n_env,  device=self.device, dtype=torch.float32)
        self.CHO = torch.zeros(self.n_steps, self.n_env, device=self.device, dtype=torch.float32)
        self.MA = torch.zeros(self.n_steps, self.n_env, device=self.device, dtype=torch.float32)
        self.const_ones = torch.ones(self.n_steps, self.n_env, device=self.device, dtype=torch.float32)

    def update(self, step, glucose, CGM, insulin, CHO, MA, t):
        self.glucose[step, :] = glucose[0]
        self.insulin[step, :] = insulin[0]
        self.CHO[step, :] = CHO[0]
        self.cgm[step, :] = CGM[0]
        self.t[step, :] = t[0]
        self.MA[step, :] = MA[0]

    def reset(self):
        self.t = self.const_ones * 0
        self.glucose = self.const_ones * 0
        self.cgm = self.const_ones * 0
        self.insulin = self.const_ones * 0
        self.CHO = self.const_ones * 0
        self.MA = self.const_ones * 0

    def get_simu_data(self):
        return self.glucose, self.cgm, self.t, self.CHO, self.insulin, self.MA