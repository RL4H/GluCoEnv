import torch
torch.set_default_dtype(torch.float64)

time = torch.tensor([[1,2,3],[1.5, 2, 3], [1.5, 1.75, 8]])
meals = torch.tensor([[10,20,30],[15, 20, 30], [15, 175, 20]], dtype=torch.float64)
std_meals = torch.tensor([[1,2,3],[1.5, 2, 3], [1.5, 1.75, 2]])
cur_time = 1.5

print('times')
print(time)

print('meals')
print(meals)

# meals at t==2
condition = (time==cur_time)
filter = condition * meals

m, _= torch.max(filter, 1)
m = torch.unsqueeze(m, 1)

print('meal at ', cur_time)
print(m)


# np.random.RandomState.randn => random floats sampled from normal dist, mean = 0 , variance = 1
# torch rand  => randon numbers from uniform dist [0, 1)
# torch randn - N(0,1)


# random scenrio generation
n_env = 3
rand_gen = torch.Generator()
rand_prob = torch.rand(n_env, 3, generator=rand_gen)
prob = torch.tensor([[0.1,0.2,0.3],[0.9, 0.2, 0.3], [0.5, 0.75, 0.8]])
print('randprob')
print(rand_prob)
print('prob')
print(prob)
print(rand_prob < prob)

print('\nmeal randomise')
print('mean')
print(meals)
final_meal = torch.normal(meals, std_meals, generator=rand_gen)
final_meal = torch.max(torch.round(final_meal), torch.zeros(n_env, 3))
print(final_meal)
ff = final_meal * (rand_prob < prob)
print(ff)

def trun_normal(a, b, mu, sigma):
    N = torch.normal(mu, sigma, generator=rand_gen)
    low_adjust = torch.fmod((N-mu), (a-mu)) + mu
    high_adjust = torch.fmod((N-mu), (b-mu)) + mu
    N = torch.where(N < a, low_adjust, N)
    N = torch.where(N > b, high_adjust, N)
    return N

print('\n sample time for the meal')
w = torch.empty(n_env, 3)
meals = torch.tensor([[10,20,30],[15, 20, 30], [15, 175, 20]], dtype=torch.float64)
std_meals = torch.tensor([[1,2,3],[1.5, 2, 3], [1.5, 1.75, 2]])
a =  torch.tensor([[9,15,25],[14, 15, 25], [8, 170, 15]], dtype=torch.float64)
b =  torch.tensor([[15,25,31],[20, 25, 31], [20, 180, 21]], dtype=torch.float64)
# x = torch.nn.init.trunc_normal(w, mean=meals, std=std_meals, a=a, b=b)

print('meals')
print(meals)

print('\nstd meals')
print(std_meals)

print('\nlowee')
print(a)

print('\nupper')
print(b)

print('\ntrunc normal')
tn = trun_normal(a, b, meals, std_meals)
print(tn)

x = (tn > a) & (tn < b)
print(x)


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
samples = 100000
muu = 15 * 60
sigmaa = 30
LOW = 14 * 60
HIGH = 16 * 60
mu = torch.ones(samples, 1) * muu
sigma = torch.ones(samples, 1) * sigmaa
a = torch.ones(samples,1) * LOW
b = torch.ones(samples, 1)* HIGH

res = trun_normal(a, b, mu, sigma)
res = torch.flatten(res)
fig, ax = plt.subplots(1,1)
ax.hist(res, density=True, histtype='stepfilled', alpha=0.2, bins=70)

x = np.linspace(muu-3*sigmaa, muu+3*sigmaa, 100)
ax.plot(x, stats.norm.pdf(x,muu, sigmaa))
plt.show()