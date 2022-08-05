import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)


def approx_trunc_normal(a, b, mu, sigma, rand_gen):
    N = torch.normal(mu, sigma, generator=rand_gen)
    low_adjust = torch.fmod((N-mu), (a-mu)) + mu
    high_adjust = torch.fmod((N-mu), (b-mu)) + mu
    N = torch.where(N < a, low_adjust, N)
    N = torch.where(N > b, high_adjust, N)
    return N


class Scenario(nn.Module):
    def __init__(self, device, params, n_env, meal_type='normal', start_time=None, seed=None):
        super().__init__()
        # all times are in minutes
        self.device= device
        self.n_env = n_env
        self.prob = torch.tensor(params['prob'], dtype=torch.float32, device=self.device)
        self.time_lb = 60 * torch.tensor(params['time_lb'], dtype=torch.float32, device=self.device)
        self.time_ub = 60 * torch.tensor(params['time_ub'], dtype=torch.float32, device=self.device)
        self.time_mu = 60 * torch.tensor(params['time_mu'], dtype=torch.float32, device=self.device)
        self.time_sigma = torch.tensor(params['time_sigma'], dtype=torch.float32, device=self.device)
        self.amount_mu = torch.tensor(params['amount_mu'], dtype=torch.float32, device=self.device)
        self.amount_sigma = torch.tensor(params['amount_sigma'], dtype=torch.float32, device=self.device)
        self.n_meals = self.prob.shape[1]
        self.meal_type = meal_type
        self.meals = torch.zeros(self.n_env, self.n_meals, device=self.device)
        self.meal_times = torch.zeros(self.n_env, self.n_meals, device=self.device)
        self.const_neg_one = -1 * torch.ones(self.n_env, self.n_meals, device=self.device)
        self.cont_zeros = torch.zeros(self.n_env, 1, device=self.device)
        self.cont_zeros2 = torch.zeros(self.n_env, self.n_meals, device=self.device)
        self.reset(seed)

    def get_action(self, t):
        ''''t should be in 'minutes' time should update to reflect the actual time
        return the CHO associated for 't' for n_envs = (n_env, t) '''
        # t is a vector a different time fot each env.
        daily_t = torch.remainder(t, 1440)  # for multi day simulations
        time_filterd = (self.meal_times == daily_t) * self.meals
        cho, _ = torch.max(time_filterd, 1)
        cho = torch.unsqueeze(cho, 1)
        return cho

    def meal_announcement(self, t, ma):
        daily_t = torch.remainder(t, 1440)
        filtered = (((self.meal_times - ma) <= daily_t) & (daily_t <= self.meal_times)) * self.meals
        cho, _ = torch.max(filtered, 1)
        cho = torch.unsqueeze(cho, 1)
        # calc time to the meal
        filt_time = (((self.meal_times - ma) <= daily_t) & (daily_t <= self.meal_times)) * self.meal_times
        time, _ = torch.max(filt_time, 1)
        time = torch.unsqueeze(time, 1)
        time_delta = time - t
        time_delta = torch.max(time_delta, self.cont_zeros)
        return cho, time_delta

    def create_scenario(self):
        # create a random meal time, amount and store in meals, meal_time of size  -  (n_env, n_meals)
        self.rand_prob = torch.rand(self.n_env, self.n_meals, generator=self.rand_gen, device=self.device)
        if self.meal_type == 'normal':
            self.meals = torch.max(torch.round(torch.normal(self.amount_mu, self.amount_sigma, generator=self.rand_gen)),
                               self.cont_zeros2) * (self.rand_prob < self.prob)
        self.meal_times = torch.round(approx_trunc_normal(self.time_lb, self.time_ub, self.time_mu, self.time_sigma, self.rand_gen))
        self.meal_times = torch.where(self.rand_prob < self.prob, self.meal_times, self.const_neg_one)  # -1 where meal not considered.

        # todo: meal and time sample using uniform distribution.
        # elif self.meal_type == 'uniform':
        #     self.meals = torch.max(torch.round(torch.normal(self.amount_mu, self.amount_sigma, generator=rand_gen)),
        #                            torch.zeros(self.n_env, self.n_meals)) * (self.rand_prob < self.prob)

    def reset(self, seed):
        # reset the scenario, random meal events allocated.
        # todo: reset full cohort, vs individual subjects
        self.seed = seed
        self.rand_gen = torch.Generator(device=self.device)
        self.rand_gen.manual_seed(self.seed)
        self.create_scenario()


if __name__ == '__main__':
    import numpy as np
    n_env = 2
    device = 'cpu'
    params = {
        'prob': np.array([[0.95, 0.3, 0.95, 0.3, 0.95, 0.3],
                         [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]]), # probability [0,1]
        'time_lb': np.array([[5, 9, 10, 14, 16, 20],
                            [5, 9, 10, 14, 16, 20]]), # 24hr format
        'time_ub': np.array([[9, 10, 14, 16, 20, 23],
                            [9, 10, 14, 16, 20, 23]]), # 24hour format
        'time_mu' : np.array([[7, 9.5, 12, 15, 18, 21.5],
                             [7, 9.5, 12, 15, 18, 21.5]]), # 24 hour format 9:30 => 9.5
        'time_sigma' : np.array([[60, 30, 60, 30, 60, 30],
                                [60, 30, 60, 30, 60, 30]]), # mins
        'amount_mu' : np.array([[45, 10, 70, 10, 80, 10],
                               [45, 10, 70, 10, 80, 10]]), # CHO in grams
        'amount_sigma' : np.array([[10, 5, 10, 5, 10, 5],
                                  [10, 5, 10, 5, 10, 5]])  # CHO in grams
    }


    scenario = Scenario(device, params, n_env, meal_type='normal', start_time = 1, seed=0)
    print(scenario.meal_times)
    print(scenario.meals)

    print(scenario.get_action(643))
    print(scenario.get_action(1298))



