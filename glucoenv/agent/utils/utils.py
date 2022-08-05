import torch
import torch.nn as nn


def f_kl(log_p, log_q):  # KL[q,p] = (r-1) - log(r) ;forward KL
    log_ratio = log_p - log_q
    return torch.mean((torch.exp(log_ratio) - 1) - log_ratio)


def r_kl(log_p, log_q):  # KL[p, q] = rlog(r) -(r-1)
    log_ratio = log_p - log_q
    return torch.mean(torch.exp(log_ratio)*log_ratio - (torch.exp(log_ratio) - 1))


def linear_scaling(x, x_min, x_max):  # scale to [-1, 1] range
    return ((x - x_min) * 2 / (x_max - x_min)) - 1


def inverse_linear_scaling(y, x_min, x_max):  # scale back to original
    return (y+1) * (x_max - x_min) * (1/2) + x_min


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out


def scale_observations(obs, MIN, MAX):
    obs[:, :, 0:1] = linear_scaling(obs[:, :, 0:1], MIN[0], MAX[0])
    obs[:, :, 1:2] = linear_scaling(obs[:, :, 1:2], MIN[1], MAX[1])
    return obs


def scale_actions(actions, translation_func='exponential', max=5):
    # todo: add other translation functions
    # todo implement safety, i.e., non negative and bounded
    if translation_func == 'exponential':
        agent_actions = max * (torch.exp((actions - 1) * 4))
    return agent_actions