import torch
import torch.nn as nn
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
torch.set_default_dtype(torch.float32)


class Sensor(nn.Module):
    def __init__(self, params,  n_env, device, seed=None):
        super().__init__()
        self._params = params
        self.n_env = n_env
        self.device = device
        self.sample_time = self._params['sample_time']
        self._last_CGM = torch.zeros(self.n_env, 1, device=self.device)
        self.CGM = torch.zeros(self.n_env, 1, device=self.device)
        self.const_zero = torch.zeros(self.n_env, 1, device=self.device)
        self.reset(seed)

    def measure(self, env):
        if env.t() % self.sample_time == 0:
            self.CGM = env.observation() + next(self._noise_generator)  # next called every sample time of sensor.
            self.CGM = torch.max(self.CGM, self._params["min"])
            self.CGM = torch.min(self.CGM, self._params["max"])
            self._last_CGM = self.CGM
            return self.CGM
        return self._last_CGM  # Zero-Order Hold => used in the mini_steps

    def min(self):
        return self._params['min']

    def max(self):
        return self._params['max']

    def reset(self, seed):
        self.seed = seed
        self._noise_generator = CGMNoise(self._params, self.device, self.n_env, seed=self.seed)
        self._last_CGM = self.const_zero

    # def seed(self, seed):
    #     self.seed = seed
    #     self._noise_generator = CGMNoise(self._params, self.device, self.n_env, seed=seed)


def johnson_transform_SU(xi, lam, gamma, delta, x):
    return xi + lam * torch.sinh((x - gamma) / delta)


class noise15_iter(nn.Module):
    def __init__(self, params, device, n_env, seed=None):
        super().__init__()
        self.seed = seed
        self.n_env = n_env
        self.device = device
        self.rand_gen = torch.Generator(device=device)  # np.random.RandomState(self.seed)
        self.rand_gen.manual_seed(self.seed)
        self._params = params
        self.n = torch.tensor(float('inf'), device=self.device) #n
        self.e = torch.zeros(self.n_env, 1, device=self.device) #0
        self.count = torch.zeros(1, device=self.device) #0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count == 0:
            self.e = torch.randn(self.n_env, generator=self.rand_gen, device=self.device)  # self.rand_gen.randn()
        elif self.count < self.n:
            self.e = self._params["PACF"] * (self.e + torch.randn(self.n_env, generator=self.rand_gen, device=self.device))
        else:
            raise StopIteration()
        self.count += 1
        return johnson_transform_SU(self._params["xi"], self._params["lambda"], self._params["gamma"],
                                    self._params["delta"], self.e)


class CGMNoise(nn.Module):
    def __init__(self, params, device, n_env, seed=None):
        super().__init__()
        self.device = device
        self.n_env = n_env
        self.PRECOMPUTE = 10  #* torch.ones(1, device=self.device)  # length of pre-compute noise sequence
        self.MDL_SAMPLE_TIME = 15  #* torch.ones(1, device=self.device)
        self._params = params
        self.seed = seed
        self.nsample = (torch.floor(self.PRECOMPUTE * self.MDL_SAMPLE_TIME / self._params["sample_time"])).to(torch.int) + 1
        self.t = torch.tensor([self._params["sample_time"] * i for i in range(0, self.nsample)], device=self.device)
        self._noise15_gen = noise15_iter(params, device, n_env, seed=seed)
        self._noise_init = next(self._noise15_gen)
        self.n = torch.tensor(float('inf'), device=self.device)
        self.count = torch.zeros(1, device=self.device)
        self.noise = torch.tensor([[] for i in range(0, self.n_env)], device=self.device)  # deque()

    def _get_noise_seq(self):  # todo: refactor
        # To make the noise sequence continous, keep the last noise as the
        # beginning of the new sequence
        noise15 = [self._noise_init]
        noise15.extend([next(self._noise15_gen) for _ in range(self.PRECOMPUTE)])
        noise15 = torch.stack(noise15, 1)
        self._noise_init = noise15[:, -1]
        t15 = torch.tensor([torch.tensor(self.MDL_SAMPLE_TIME) * i for i in range(0, noise15.shape[1])], device=self.device, dtype=torch.float64)

        noise15 = torch.unsqueeze(noise15, 1)
        noise15 = torch.unsqueeze(noise15, 3)

        spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t15, noise15))
        noise = spline.evaluate(self.t)
        noise = torch.squeeze(noise, 3)
        noise = torch.squeeze(noise, 1)
        # interp_f = interp1d(t15, noise15, kind='cubic')
        # noise = interp_f(t)

        # noise dimensions = n_env * length
        noise2return = noise[:, 1:]
        return noise2return

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.n:
            if self.noise.shape[1] == 0:
                self.noise = self._get_noise_seq()
            self.count += 1
            self.n_ret_noise = torch.unsqueeze(self.noise[:, 0], 1)  # the output is shape - (n_env x 1)
            self.noise = self.noise[:, 1:]  # pop the left and return
            return self.n_ret_noise
        else:
            raise StopIteration()
