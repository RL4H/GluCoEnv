import torch
import torch.nn as nn
from glucoenv.env.core import setup_benchmark_controller
torch.set_default_dtype(torch.float32)


class SBB(nn.Module):
    def __init__(self, device=None, patients=None, n_env=None, mode=None, sample_time=None, env_device=None):
        super().__init__()
        self.device = device
        self.env_device = env_device
        self.n_envs = n_env
        self.params = setup_benchmark_controller(device, patients, n_env)
        self.sample_time = sample_time.to(self.device)
        self.mode = 'perfect' if mode is None else mode # perfect, error_linear, error_quadratic
        self.BW = torch.zeros(self.n_envs, 1, device=self.device)
        self.u2ss = torch.zeros(self.n_envs, 1, device=self.device)
        self.TDI = torch.zeros(self.n_envs, 1, device=self.device)
        self.CR = torch.zeros(self.n_envs, 1, device=self.device)
        self.CF = torch.zeros(self.n_envs, 1, device=self.device)
        for p in range(0, self.n_envs):
            self.BW[p] = self.params[p]['BW']
            self.u2ss[p] = self.params[p]['u2ss']
            self.TDI[p] = self.params[p]['TDI']
            self.CR[p] = self.params[p]['CR']
            self.CF[p] = self.params[p]['CF']
        self.basal = self.u2ss * self.BW / 6000
        self.target = torch.zeros(self.n_envs, 1, device=self.device) * 140
        self.cf_target = torch.zeros(self.n_envs, 1, device=self.device) * 150
        self.past_meal_memory = torch.zeros(self.n_envs, int(180/self.sample_time), device=self.device) # meal memory for 3 hours
        self.const_ones = torch.ones(self.n_envs, 1, device=self.device)
        self.const_zeros = torch.zeros(self.n_envs, 1, device=self.device)
        self.adjust_parameters()

    def adjust_parameters(self):
        #self.TDI = self.BW * 0.55
        self.CR = 500 / self.TDI
        self.CF = 1800 / self.TDI
        self.basal = (self.TDI * 0.48) / (24 * 60)

    def get_action(self, glucose, done, meal, t):
        meal = meal.to(self.device)
        glucose = glucose.to(self.device)
        t = t.to(self.device)
        done = done.to(self.device)
        self.past_meal_memory = self.past_meal_memory * torch.logical_not(done)  # reset if done
        self.f_meal = self.carb_calc(meal, t)
        self.f_cooldown = torch.sum(self.past_meal_memory, 1, keepdim=True) == 0
        self.bolus = self.f_meal / self.CR + (glucose > self.cf_target) * self.f_cooldown * (glucose - self.target) / self.CF
        self.bolus = (self.bolus * (self.f_meal > 0)) / self.sample_time
        self.past_meal_memory = torch.cat((self.past_meal_memory, self.f_meal), 1)
        return (self.basal + self.bolus).to(self.env_device)

    def init_action(self):
        return (self.const_zeros).to(self.env_device)

    def carb_calc(self, cho_real, t):
        # paper: https://www.liebertpub.com/doi/full/10.1089/dia.2019.0502
        # UVA /Padova Univesity.
        # CHO_estimate = CHO_real + CHO_estimate_error
        self.breakfast = ((6 <= (t / 60)) & ((t / 60) <= 9)) * self.const_ones
        self.lunch = ((12 <= (t / 60)) & ((t / 60) <= 15)) * self.const_ones
        self.dinner = ((18 <= (t / 60)) & ((t / 60) <= 21)) * self.const_ones
        self.snack = self.const_ones - self.breakfast - self.lunch - self.dinner
        self.cho_error = 0 * self.const_ones
        if self.mode == 'perfect':
            self.cho_error = 0 * self.const_ones
        elif self.mode == 'error_linear':
            self.cho_error = 9.22 - 0.34 * cho_real + (0.09 * self.BW) + (3.11 * self.lunch) + (0.68 * self.dinner) - (7.05 * self.snack)
        elif self.mode == 'error_quadratic':
            self.cho_error = 3.56 - 0.07 * cho_real - 0.0008 * cho_real * cho_real + (6.77 * self.lunch) + (18.01 * self.dinner) - (0.49 * self.snack) - \
                                 (0.08 * cho_real * self.lunch) - (0.25 * cho_real * self.dinner) - (0.06 * cho_real * self.snack)
        self.c_meal = (cho_real + self.cho_error) * (cho_real > 0)
        self.c_meal = torch.max(self.c_meal, self.const_zeros)
        return self.c_meal

    def reset(self):  # full reset todo
        self.past_meal_memory = torch.zeros(self.n_envs, int(180 / self.sample_time), device=self.device)  # meal memory for 3 hours

    def reset_env(self):  # reset given env todo
        self.past_meal_memory = torch.zeros(self.n_envs, int(180 / self.sample_time), device=self.device)  # meal memory for 3 hours
