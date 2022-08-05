import torch
import torch.nn as nn
from glucoenv.env.core import setup_sensor, setup_pump, setup_scenario, setup_envs, Action, Info
from glucoenv.env.reward import reward
torch.set_default_dtype(torch.float32)


class T1DSimEnv(nn.Module):
    def __init__(self, device, patients=None, sensor=None, pump=None, scenario=None, n_env=None,
                 reward_fun=None, start_time=None, seed=None, obs_type=None):
        super().__init__()
        self.n_env = n_env
        self.device = device
        self.const_ones = torch.ones(self.n_env, 1, device=self.device)
        self.patient_params = patients
        self.sensor_params = sensor
        self.pump_params = pump
        self.scenario_params = scenario
        self.seed = seed
        self.rand_gen = torch.Generator(device=device)
        self.rand_gen.manual_seed(self.seed)
        self.start_time = start_time
        self.obs_type = 'current' if obs_type is None else obs_type

        self.sensor = setup_sensor(name=sensor, n_env=self.n_env, device=self.device, seed=seed)
        self.env = setup_envs(device=self.device, patient_params=patients, n_env=self.n_env, seed=seed)
        self.scenario = setup_scenario(device=device, params=scenario, n_env=self.n_env, seed=seed)
        self.pump = setup_pump(name=pump, device=self.device)
        self.t0 = torch.tensor(start_time, dtype=torch.float32, device=self.device)
        self.counter = torch.zeros(1, device=self.device)

        self.t_init = torch.tensor(start_time, dtype=torch.float32, device=self.device)
        self.sample_time = self.sensor.sample_time
        #self.SAMPLE_TIME = torch.ones(1, device=self.device)  # min

        self.Action = Action(n_env=n_env, device=device)
        self.info = Info(n_env=n_env, device=device)
        self.info.update('sample_time', self.sample_time)
        if reward_fun is None:  # todo add custom rew func
            self.reward_fun = reward

        # catastrophic failure BG < 40 or BG > 600
        self.MIN_GLUCOSE = torch.ones(self.n_env, 1, device=self.device) * 40
        self.MAX_GLUCOSE = torch.ones(self.n_env, 1, device=self.device) * 600

        self.window = int(60 / self.sample_time)
        self.BG_hist = torch.zeros(self.n_env, self.window, device=self.device)
        self.CGM_hist = 40 * torch.ones(self.n_env, self.window, dtype=torch.float32, device=self.device, requires_grad=False)  # -1 => denotes start.
        self.ins_hist = 0 * torch.ones(self.n_env, self.window, dtype=torch.float32, device=self.device, requires_grad=False)

        self.MA = 20 * torch.ones(1, 1, device=self.device)  # meal announcement is 20minutes before.
        self.prev_ma = torch.zeros(self.n_env, 1, device=self.device)

    def t(self):
        return self.t0

    def reset_env(self, envs_done):
        self.env.reset(envs_done)

    def reset(self):  # full reset
        self.counter = torch.zeros(1, device=self.device)
        self.sensor.reset(seed=self.rand_gen.seed())
        self.env.full_reset()
        self.scenario.reset(seed=self.rand_gen.seed())
        self.t0 = torch.tensor(self.start_time, dtype=torch.float32, device=self.device)
        self.BG_hist = torch.zeros(self.n_env, self.window, dtype=torch.float32, device=self.device)
        self.CGM_hist = 40 * torch.ones(self.n_env, self.window, dtype=torch.float32, device=self.device, requires_grad=False)  # -1 => denotes start.
        self.ins_hist = 0 * torch.ones(self.n_env, self.window, dtype=torch.float32, device=self.device, requires_grad=False)
        rl_obs = torch.stack((self.CGM_hist, self.ins_hist), dim=2)
        dones = torch.zeros(self.n_env, 1, dtype=torch.bool, device=self.device)
        return rl_obs, dones

    def action_space(self):
        return self.const_ones * self.pump.min(), self.const_ones * self.pump.max()

    def observation_space(self):
        if self.obs_type == 'current':
            obs_space = self.const_ones * self.sensor.min(), self.const_ones * self.sensor.max()
        elif self.obs_type == 'past_history':
            obs_space = torch.ones(self.n_env, self.window, 2, device=self.device) * self.sensor.min(),  \
                         torch.ones(self.n_env, self.window, 2, device=self.device) * self.sensor.max()
        else:
            obs_space = None
        return obs_space

    def mini_step(self, action):  # the real simulation steps. i.e., sample rate 1 minute (real resolution).
        #  return (cho, insulin, next bg, next cgm)
        self.mini_CHO = self.scenario.get_action(self.t0)
        self.Action.set_action(ins=action, cho=self.mini_CHO)
        self.env.step(self.Action.get_action())  # State update
        return self.mini_CHO, action, self.env.observation(), self.sensor.measure(self.env)

    def step(self, action):  # action is a tensor (n_env x 1)print('\nMemory ODE')
        #action = action.to(self.device)
        if action.shape[0] != self.n_env:
            print('\n#### GluConRL ####')
            print('ERROR: The dimension of the action ', action.shape[0], ' doesnt match n_env ', self.n_env)
            exit()
        self.s_CHO = 0 * self.const_ones
        self.s_insulin = 0 * self.const_ones
        self.s_BG = 0 * self.const_ones
        self.s_CGM = 0 * self.const_ones
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            self.s_tmp_CHO, self.s_tmp_insulin, self.s_tmp_BG, self.s_tmp_CGM = self.mini_step(action)
            self.s_CHO += self.s_tmp_CHO #/ self.sample_time
            self.s_insulin += self.s_tmp_insulin / self.sample_time
            self.s_BG += self.s_tmp_BG / self.sample_time
            self.s_CGM += self.s_tmp_CGM / self.sample_time
            self.t0 = self.t0 + 1
            self.counter = self.counter + 1

        self.BG_hist = torch.cat((self.BG_hist[:, 1:], self.s_BG), 1)
        self.CGM_hist = torch.cat((self.CGM_hist[:, 1:], self.s_CGM), 1)
        self.ins_hist = torch.cat((self.ins_hist[:, 1:], self.s_insulin), 1)

        self.s_reward = self.reward_fun(self.BG_hist)  # Compute risk index & reward
        self.s_done = (self.s_BG < self.MIN_GLUCOSE) + (self.s_BG > self.MAX_GLUCOSE)  # catastrophic failure check

        # if done reset corresponding environment and the env time.
        self.s_complete_patients = torch.squeeze(self.s_done, 1).nonzero().flatten()
        if self.s_complete_patients.numel() != 0:
            self.reset_env(self.s_complete_patients)
            for i in self.s_complete_patients:
                self.t0[i] = self.t_init[i]
                self.BG_hist[i, 0:self.window] = torch.zeros(1, self.window, device=self.device)
                self.CGM_hist[i, 0:self.window] = 40 * torch.ones(1, self.window, device=self.device)
                self.ins_hist[i, 0:self.window] = 0 * torch.ones(1, self.window, device=self.device)

        self.s_ma, self.s_t_delta = self.scenario.meal_announcement(self.t0, self.MA)
        self.s_new_meal = (self.prev_ma == 0) * self.s_ma

        self.info.update('meal', self.s_CHO)
        self.info.update('meal_announcement', self.s_ma)
        self.info.update('time2meal', self.s_t_delta)
        self.info.update('time', self.t0)
        self.info.update('SBB_MA', self.s_new_meal)
        self.info.update('BG', self.s_BG)
        self.prev_ma = self.s_ma

        rl_obs = torch.stack((self.CGM_hist, self.ins_hist), dim=2)
        obs = self.s_CGM if self.obs_type == 'current' else rl_obs

        if self.counter > 14400:  # force reset env if run for more than 10 days, or else might overflow patient ode solver!
            self.reset()

        return obs, self.s_reward, self.s_done, self.info.get()




