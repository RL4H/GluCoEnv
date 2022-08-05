import torch
import torch.nn as nn
from torchdiffeq import odeint
torch.set_default_dtype(torch.float64)

PATIENT_PARA_FILE = './params/vpatient_params.csv'

class T1DPatient(nn.Module):

    def __init__(self, device, params, init_state=None, random_init_bg=False, seed=None, t0=0.0):
        super().__init__()

        self.device = device
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = nn.Parameter(torch.tensor([seed]),requires_grad=False)
        self.t0 = torch.zeros(1, device=self.device) #nn.Parameter(torch.tensor([0.0]))
        self.name = self._params['Name']
        self.reset()
        self.SAMPLE_TIME = nn.Parameter(torch.tensor([1.0]), requires_grad=False)  # min
        self.EAT_RATE = 5 * torch.ones(1, device=self.device)  # g/min CHO
        self.const_zero = torch.zeros(1, device=self.device)  # g/min CHO
        self.odeint = odeint

    def forward(self, t, x):

        action = self.cur_action
        params = self._params
        last_Qsto = self._last_Qsto
        last_foodtaken = self._last_foodtaken

        dxdt = nn.Parameter(torch.zeros(13, device=self.device), requires_grad=False)

        d = action['CHO'] * 1000  # g -> mg
        insulin = action['insulin'] * 6000 / params['BW']  # U/min -> pmol/kg/min
        #basal = params['u2ss'] * params['BW'] / 6000  # U/min

        # Glucose in the stomach
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken



        # Stomach solid
        dxdt[0] = -params['kmax'] * x[0] + d


        if Dbar > 0:
            aa = 5 / 2 / (1 - params['b']) / Dbar
            cc = 5 / 2 / params['d'] / Dbar
            kgut = params['kmin'] + (params['kmax'] - params['kmin']) / 2 * (torch.tanh(
                aa * (qsto - params['b'] * Dbar)) - torch.tanh(cc * (qsto - params['d'] * Dbar)) + 2)
        else:
            kgut = params['kmax']



        # stomach liquid
        dxdt[1] = params['kmax'] * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params['kabs'] * x[2]

        # Rate of appearance
        Rat = params['f'] * params['kabs'] * x[2] / params['BW']
        # Glucose Production
        EGPt = params['kp1'] - params['kp2'] * x[3] - params['kp3'] * x[8]
        # Glucose Utilization
        Uiit = params['Fsnc']

        # renal excretion
        if x[3] > params['ke2']:
            Et = params['ke1'] * (x[3] - params['ke2'])
        else:
            Et = self.const_zero

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        dxdt[3] = torch.max(EGPt, self.const_zero) + Rat - Uiit - Et - \
            params['k1'] * x[3] + params['k2'] * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params['Vm0'] + params['Vmx'] * x[6]
        Kmt = params['Km0']
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params['k1'] * x[3] - params['k2'] * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # insulin kinetics
        dxdt[5] = -(params['m2'] + params['m4']) * x[5] + params['m1'] * x[9] + params['ka1'] * \
            x[10] + params['ka2'] * x[11]  # plus insulin IV injection u[3] if needed
        It = x[5] / params['Vi']
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        # insulin action on glucose utilization
        dxdt[6] = -params['p2u'] * x[6] + params['p2u'] * (It - params['Ib'])

        # insulin action on production
        dxdt[7] = -params['ki'] * (x[7] - It)

        dxdt[8] = -params['ki'] * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params['m1'] + params['m30']) * x[9] + params['m2'] * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params['ka1'] + params['kd']) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]

        dxdt[11] = params['kd'] * x[10] - params['ka2'] * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # subcutaneous glcuose
        dxdt[12] = (-params['ksc'] * x[12] + params['ksc'] * x[3])
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        # if action.insulin > basal:
        #     logger.debug('t = {}, injecting insulin: {}'.format(
        #         t, action.insulin))
        #

        return dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4], dxdt[5], dxdt[6], dxdt[7], dxdt[8], dxdt[9], dxdt[10], dxdt[11], dxdt[12]

    def reset(self):
        '''
        Reset the patient state to default intial state
        '''
        if self._init_state is None:
            self.init_state = (self._params['x0_ 1'], self._params['x0_ 2'], self._params['x0_ 3'],
                               self._params['x0_ 4'], self._params['x0_ 5'], self._params['x0_ 6'],
                               self._params['x0_ 7'], self._params['x0_ 8'], self._params['x0_ 9'],
                               self._params['x0_10'], self._params['x0_11'], self._params['x0_12'],
                               self._params['x0_13'])
        else:
            self.init_state = self._init_state
        self.state = self.init_state  # chirath addition

        # self.random_state = np.random.RandomState(self.seed)
        # if self.random_init_bg:
        #     # Only randomize glucose related states, x4, x5, and x13
        #     mean = [1.0 * self.init_state[3],
        #             1.0 * self.init_state[4],
        #             1.0 * self.init_state[12]]
        #     cov = np.diag([0.1 * self.init_state[3],
        #                    0.1 * self.init_state[4],
        #                    0.1 * self.init_state[12]])
        #     bg_init = self.random_state.multivariate_normal(mean, cov)
        #     self.init_state[3] = 1.0 * bg_init[0]
        #     self.init_state[4] = 1.0 * bg_init[1]
        #     self.init_state[12] = 1.0 * bg_init[2]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_action = nn.ParameterDict({'CHO': nn.Parameter(torch.tensor([0.0]), requires_grad=False),
                               'insulin': nn.Parameter(torch.tensor([0.0]), requires_grad=False)})
        self.is_eating = False
        self._last_foodtaken = torch.zeros(1, device=self.device)
        self.planned_meal = torch.zeros(1, device=self.device)
        self.to_eat = torch.zeros(1, device=self.device)

    def _announce_meal(self, meal):
        '''
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        '''

        self.planned_meal += meal
        if self.planned_meal > 0:
            self.to_eat = torch.min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= self.to_eat
            self.planned_meal = torch.max(self.const_zero, self.planned_meal)
        else:
            self.to_eat = torch.zeros(1, device=self.device)
        return self.to_eat

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action['CHO'])
        action.update({'CHO': nn.Parameter(to_eat, requires_grad=False)})  #action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action['CHO'] > 0 and self._last_action['CHO'] <= 0:
            #logger.info('t = {}, patient starts eating ...'.format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = torch.zeros(1, device=self.device)
            self.is_eating = True

        # if to_eat > 0:
        #     print('patient eats')

        if self.is_eating:
            self._last_foodtaken += action['CHO']   # g

        # Detect eating ended
        if action['CHO'] <= 0 and self._last_action['CHO'] > 0:
            #logger.info('t = {}, Patient finishes eating!'.format(self.t))
            print('finished eating')
            self.is_eating = False

        # Update last input
        self._last_action = action
        self.cur_action = action  # because i cant update some params in the forward

        t0 = self.t0
        event_t = t0 + self.SAMPLE_TIME
        tt = torch.linspace(float(t0), float(event_t), int((float(event_t) - float(t0)) * 500), device=self.device)[1:-1]
        tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
        solution = odeint(self, self.state, tt, atol=1e-8, rtol=1e-8)

        self.state = tuple(s[-1] for s in solution)
        self.t0 = event_t

    def observation(self):
        '''
        return the observation from patient - subcutaneous glucose level
        '''
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params['Vg']
        #observation = Observation(Gsub=Gsub)
        return Gsub

    def t(self):
        return self.t0


if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from core import get_params_withName, get_params_withID
    import pandas as pd
    device = 'cpu'
    patient_device = 'cpu'
    print('Simulator => ', patient_device)
    print('Controller => ', device)

    patient_id = 1
    name = 'adolescent#001'
    p = get_params_withID(patient_id)
    p = get_params_withName(name)

    param_torch_dict = nn.ParameterDict(
        {'Name': nn.Parameter(torch.tensor([patient_id], device=patient_device), requires_grad=False)})
    keys = p.index.values
    for i in range(1, len(keys)):
        param_torch_dict.update(
            {keys[i]: nn.Parameter(torch.tensor([p[keys[i]]], device=patient_device), requires_grad=False)})

    p = T1DPatient(device, param_torch_dict, init_state=None, random_init_bg=False, seed=0.0, t0=0.0)
    basal = p._params.u2ss * p._params.BW / 6000  # U/min

    t = []
    CHO = []
    insulin = []
    BG = []

    while p.t() < 400:
        ins = basal
        carb = 0
        if p.t() == 100:
            carb = 80
            ins = 80.0 / 6.0 + basal
        else:
            carb = 0
            ins = basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal

        # act = Action(insulin=ins, CHO=carb)
        act = nn.ParameterDict(
            {'CHO': nn.Parameter(carb * torch.ones(1, device='cpu'), requires_grad=False),
             'insulin': nn.Parameter(ins * torch.ones(1, device='cpu'), requires_grad=False)})

        t.append(p.t())
        CHO.append(act['CHO'])
        insulin.append(act['insulin'])
        BG.append(p.observation())
        p.step(act)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()

    print('done')