import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)


class InsulinPump(nn.Module):
    def __init__(self, params, seed=None):
        super().__init__()
        self.U2PMOL = 6000
        self._params = params

    def bolus(self, amount):
        bol = amount * self.U2PMOL  # convert from U/min to pmol/min
        bol = torch.round(bol / self._params['inc_bolus']) * self._params['inc_bolus']
        bol = bol / self.U2PMOL     # convert from pmol/min to U/min
        bol = torch.min(bol, self._params['max_bolus'])
        bol = torch.max(bol, self._params['min_bolus'])
        return bol

    def basal(self, amount):
        bas = amount * self.U2PMOL  # convert from U/min to pmol/min
        bas = torch.round(bas / self._params['inc_basal']) * self._params['inc_basal']
        bas = bas / self.U2PMOL     # convert from pmol/min to U/min
        bas = torch.min(bas, self._params['max_basal'])
        bas = torch.max(bas, self._params['min_basal'])
        return bas

    def reset(self):
        pass

    def min(self):
        return self._params['min_basal']

    def max(self):
        return self._params['max_basal']


if __name__ == '__main__':
    from glucoenv.env.core import Action
    n_env = 2
    device= 'cpu'
    a = Action(n_env=n_env, device=device)
    cho = torch.rand(n_env, 1, device=device)
    ins = torch.rand(n_env, 1, device=device)
    a.set_action(ins=None, cho=cho)
    f = a.get_action()
    print(f['CHO'])
    print(f['insulin'])
