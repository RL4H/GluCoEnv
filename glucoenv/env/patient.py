import torch
import torch.nn as nn
from torchdiffeq import odeint
torch.set_default_dtype(torch.float64)


class T1DCohort(nn.Module):
    def __init__(self, device, params, n_patients, init_state=None, random_init_bg=False, seed=None, t0=None):
        super().__init__()
        self.device = device
        self.n_envs = n_patients
        self.n_states = 13
        self.t0 = torch.zeros(1, device=self.device)
        self.patients = params

        # dynamic parameters
        self.states = torch.zeros(self.n_envs, 13, device=self.device)
        self.dxdt = [torch.zeros(self.n_envs, 1) for i in range(0, self.n_states)]

        self.planned_meal = torch.zeros(self.n_envs, 1, device=self.device)
        self.to_eat = torch.zeros(self.n_envs, 1, device=self.device)

        self.is_eating = torch.zeros(self.n_envs, 1, dtype=torch.bool, device=self.device)
        self.false_bool_tensor = torch.zeros(self.n_envs, 1, dtype=torch.bool, device=self.device)
        self.true_bool_tensor = torch.ones(self.n_envs, 1, dtype=torch.bool, device=self.device)

        self._last_foodtaken = torch.zeros(self.n_envs, 1, device=self.device)
        self._last_Qsto = torch.zeros(self.n_envs, 1, device=self.device)
        self._last_action_ins = torch.zeros(self.n_envs, 1, device=self.device)
        self._last_action_CHO = torch.zeros(self.n_envs, 1, device=self.device)

        # current action
        self.action_ins = torch.zeros(self.n_envs, 1, device=self.device)
        self.action_CHO = torch.zeros(self.n_envs, 1, device=self.device)

        # CONST patient specific parameters for simulation
        self.BW = torch.zeros(self.n_envs, 1, device=self.device)
        self.KMAX = torch.zeros(self.n_envs, 1, device=self.device)
        self.B = torch.zeros(self.n_envs, 1, device=self.device)
        self.D = torch.zeros(self.n_envs, 1, device=self.device)
        self.KMIN = torch.zeros(self.n_envs, 1, device=self.device)
        self.KABS = torch.zeros(self.n_envs, 1, device=self.device)
        self.F = torch.zeros(self.n_envs, 1, device=self.device)
        self.KP1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KP2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KP3 = torch.zeros(self.n_envs, 1, device=self.device)
        self.FSNC = torch.zeros(self.n_envs, 1, device=self.device)
        self.KE1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KE2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.K1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.K2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.VM0 = torch.zeros(self.n_envs, 1, device=self.device)
        self.VMX = torch.zeros(self.n_envs, 1, device=self.device)
        self.KM0 = torch.zeros(self.n_envs, 1, device=self.device)
        self.M1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.M2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.M4 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KA1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KA2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.VI = torch.zeros(self.n_envs, 1, device=self.device)
        self.P2U = torch.zeros(self.n_envs, 1, device=self.device)
        self.IB = torch.zeros(self.n_envs, 1, device=self.device)
        self.KI = torch.zeros(self.n_envs, 1, device=self.device)
        self.M30 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KA1 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KA2 = torch.zeros(self.n_envs, 1, device=self.device)
        self.KD = torch.zeros(self.n_envs, 1, device=self.device)
        self.KSC = torch.zeros(self.n_envs, 1, device=self.device)
        self.VG = torch.zeros(self.n_envs, 1, device=self.device)

        self.SAMPLE_TIME = torch.ones(1, device=self.device) # min
        self.EAT_RATE = 5 * torch.ones(self.n_envs, 1, device=self.device)  # g/min CHO
        self.const_zero = torch.zeros(self.n_envs, 1, device=self.device)
        self.const_ones = torch.ones(self.n_envs, 1, device=self.device)

        self.lin_space = torch.linspace(0.0, 1.0, 500, device=self.device)#[1:-1]
        self.init_states()
        self.odeint = odeint

    def init_states(self):
        for p in range(0, self.n_envs):
            for s in range(0, 13):
                state_num = ' '+str(s+1) if (s+1) < 10 else str(s+1)
                self.states[p][s] = self.patients[p]['x0_'+state_num]

            self._last_Qsto[p] = self.patients[p]['x0_ 1'] + self.patients[p]['x0_ 2']
            self.BW[p] = self.patients[p]['BW']
            self.KMAX[p] = self.patients[p]['kmax']
            self.B[p] = self.patients[p]['b']
            self.D[p] = self.patients[p]['d']
            self.KMIN[p] = self.patients[p]['kmin']
            self.KABS[p] = self.patients[p]['kabs']
            self.F[p] = self.patients[p]['f']
            self.KP1[p] = self.patients[p]['kp1']
            self.KP2[p] = self.patients[p]['kp2']
            self.KP3[p] = self.patients[p]['kp3']
            self.FSNC[p] = self.patients[p]['Fsnc']
            self.KE1[p] = self.patients[p]['ke1']
            self.KE2[p] = self.patients[p]['ke2']
            self.K1[p] = self.patients[p]['k1']
            self.K2[p] = self.patients[p]['k2']
            self.VM0[p] = self.patients[p]['Vm0']
            self.VMX[p] = self.patients[p]['Vmx']
            self.KM0[p] = self.patients[p]['Km0']
            self.M1[p] = self.patients[p]['m1']
            self.M2[p] = self.patients[p]['m2']
            self.M4[p] = self.patients[p]['m4']
            self.KA1[p] = self.patients[p]['ka1']
            self.KA2[p] = self.patients[p]['ka2']
            self.VI[p] = self.patients[p]['Vi']
            self.P2U[p] = self.patients[p]['p2u']
            self.IB[p] = self.patients[p]['Ib']
            self.KI[p] = self.patients[p]['ki']
            self.M30[p] = self.patients[p]['m30']
            self.KA1[p] = self.patients[p]['ka1']
            self.KA2[p] = self.patients[p]['ka2']
            self.KD[p] = self.patients[p]['kd']
            self.KSC[p] = self.patients[p]['ksc']
            self.VG[p] = self.patients[p]['Vg']

        self.tuple_state = self.matrix_2_tuple(self.states)

    def forward(self, t, tuple_x):
        # x is a tuple - (patient1[1:13], patient2[1:13]... patientn[1:13])
        # break x into 2D array. once df_x calculated => return tuple().
        self.x_states = [torch.unsqueeze(self.tuple_2_matrix(tuple_x)[:, i], 1) for i in range(0, self.n_states)]

        self.f_insulin = self.action_ins * 6000 / self.BW  # U/min -> pmol/kg/min
        self.f_qsto = self.x_states[0] + self.x_states[1]
        self.f_Dbar = self._last_Qsto + self._last_foodtaken
        self.dxdt[0] = -self.KMAX * self.x_states[0] + self.action_CHO * 1000  # Stomach solid  #d = self.action_CHO * 1000  # g -> mg

        #temp_Dbar  = Dbar + 1e-7  # to avoid zero division. Note, self.B & self.D are never zero.
        self.f_aa = 5 / 2 / (self.const_ones - self.B) / (self.f_Dbar + 1e-7)
        self.f_cc = 5 / 2 / self.D / (self.f_Dbar + 1e-7)

        self.f_kgut_true = self.KMIN + (self.KMAX - self.KMIN) / 2 * (torch.tanh(self.f_aa * (self.f_qsto - self.B * self.f_Dbar)) - torch.tanh(self.f_cc * (self.f_qsto - self.D * self.f_Dbar)) + (2*self.const_ones))
        self.f_kgut = torch.where(self.f_Dbar > 0, self.f_kgut_true, self.KMAX)

        self.dxdt[1] = self.KMAX * self.x_states[0] - self.x_states[1] * self.f_kgut  # stomach liquid
        self.dxdt[2] = self.f_kgut * self.x_states[1] - self.KABS * self.x_states[2]  # intestine
        self.f_Rat = self.F * self.KABS * self.x_states[2] / self.BW  # Rate of appearance
        self.f_EGPt = self.KP1 - self.KP2 * self.x_states[3] - self.KP3 * self.x_states[8]  # Glucose Production
        self.f_Uiit = self.FSNC  # Glucose Utilization
        self.f_Et_true = self.KE1 * (self.x_states[3] - self.KE2)  # renal excretion
        self.f_Et = torch.where(self.x_states[3] > self.KE2, self.f_Et_true, self.const_zero)

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        self.dxdt[3] = torch.max(self.f_EGPt, self.const_zero) + self.f_Rat - self.f_Uiit - self.f_Et - self.K1 * self.x_states[3] + self.K2 * self.x_states[4]
        self.dxdt[3] = (self.x_states[3] >= self.const_zero) * self.dxdt[3]

        self.f_Vmt = self.VM0 + self.VMX * self.x_states[6]
        self.f_Kmt = self.KM0
        self.f_Uidt = self.f_Vmt * self.x_states[4] / (self.f_Kmt + self.x_states[4])
        self.dxdt[4] = -self.f_Uidt + self.K1 * self.x_states[3] - self.K2 * self.x_states[4]
        self.dxdt[4] = (self.x_states[4] >= self.const_zero) * self.dxdt[4]

        # insulin kinetics
        self.dxdt[5] = -(self.M2 + self.M4) * self.x_states[5] + self.M1 * self.x_states[9] + self.KA1 * self.x_states[10] + self.KA2 * self.x_states[11]  # plus insulin IV injection u[3] if needed
        self.f_It = self.x_states[5] / self.VI
        self.dxdt[5] = (self.x_states[5] >= self.const_zero) * self.dxdt[5]

        # insulin action on glucose utilization
        self.dxdt[6] = -self.P2U * self.x_states[6]  + self.P2U * (self.f_It  - self.IB)
        # insulin action on production
        self.dxdt[7] = -self.KI * (self.x_states[7] - self.f_It)
        self.dxdt[8] = -self.KI * (self.x_states[8] - self.x_states[7])
        # insulin in the liver (pmol/kg)
        self.dxdt[9] = -(self.M1 + self.M30) * self.x_states[9] + self.M2 * self.x_states[5]
        self.dxdt[9] = (self.x_states[9] >= self.const_zero) * self.dxdt[9]
        # subcutaneous insulin kinetics
        self.dxdt[10] = self.f_insulin - (self.KA1 + self.KD) * self.x_states[10]
        self.dxdt[10] = (self.x_states[10] >= self.const_zero) * self.dxdt[10]
        self.dxdt[11] = self.KD * self.x_states[10] - self.KA2 * self.x_states[11]
        self.dxdt[11] = (self.x_states[11] >= self.const_zero) * self.dxdt[11]
        # subcutaneous glcuose
        self.dxdt[12] = (-self.KSC * self.x_states[12] + self.KSC * self.x_states[3])
        self.dxdt[12] = (self.x_states[12] >= self.const_zero) * self.dxdt[12]

        self.final_dxdt = torch.cat(self.dxdt, 1)
        self.final_dxdt = self.matrix_2_tuple(self.final_dxdt)
        return self.final_dxdt

    def step(self, action):
        self.action_CHO = torch.unsqueeze(nn.utils.parameters_to_vector(action['CHO']), 1)
        self.action_ins = torch.unsqueeze(nn.utils.parameters_to_vector(action['insulin']), 1)
        self.action_CHO = self._announce_meal(self.action_CHO)

        # Detect eating or not and update last digestion amount
        # condition = ((self.action_CHO > 0) & (self._last_action_CHO <= 0))
        self.s_condition1 = torch.logical_and((self.action_CHO > 0), (self._last_action_CHO <= 0))
        self._last_Qsto = torch.where(self.s_condition1, (torch.unsqueeze(torch.sum(self.states[:, :2], 1), 1)), self._last_Qsto)
        self._last_foodtaken = torch.where(self.s_condition1, self.const_zero, self._last_foodtaken)
        self.is_eating = torch.where(self.s_condition1, self.true_bool_tensor, self.is_eating)

        self._last_foodtaken = torch.where(self.is_eating, (self._last_foodtaken+self.action_CHO), self._last_foodtaken)

        # Detect eating ended
        self.s_condition2 = torch.logical_and((self.action_CHO <= 0), self._last_action_CHO > 0)
        self.is_eating = torch.where(self.s_condition2, self.false_bool_tensor, self.is_eating)

        # update the action.
        self._last_action_ins = self.action_ins
        self._last_action_CHO = self.action_CHO

        solution = self.odeint(self, self.tuple_state, self.lin_space + self.t0, method='dopri5', atol=1e-8,
                               rtol=1e-8, options={'dtype': torch.float64})  # solve ODE
        x = torch.stack([s[-1] for s in solution], dim=1)
        y = x.detach().clone()  # without this, memory leaks!
        self.tuple_state = torch.tensor_split(torch.flatten(y), self.n_envs * self.n_states)
        self.states = self.tuple_2_matrix(self.tuple_state)  # update state as a matrix
        self.t0 = self.t0 + self.SAMPLE_TIME

    def _announce_meal(self, meal):
        # element-wise implementation
        self.planned_meal += meal
        self.a_adj_to_eat = torch.min(self.EAT_RATE, self.planned_meal)
        self.a_adj = self.planned_meal - self.a_adj_to_eat
        self.a_adj_plannedMeal = torch.max(self.const_zero, self.a_adj)
        self.a_false_to_eat = 0 * self.const_ones
        self.to_eat = torch.where(self.planned_meal > 0, self.a_adj_to_eat, self.a_false_to_eat)
        self.planned_meal = torch.where(self.planned_meal > 0, self.a_adj_plannedMeal, self.planned_meal)
        return self.to_eat
        # my original implementation.
        # for i in range(0, self.n_envs):
        #     self.planned_meal[i] += meal[i]
        #     if self.planned_meal[i] > 0:
        #         self.to_eat[i] = torch.min(self.EAT_RATE, self.planned_meal[i])
        #         self.planned_meal[i] -= self.to_eat[i]
        #         self.planned_meal[i] = torch.max(torch.zeros(1, device=self.device), self.planned_meal[i])
        #     else:
        #         self.to_eat[i] = torch.zeros(1, device=self.device)

    def reset(self, envs):
        for p in envs:
            for s in range(0, 13):
                state_num = ' '+str(s+1) if (s+1) < 10 else str(s+1)
                self.states[p][s] = self.patients[p]['x0_'+state_num]
            self._last_Qsto[p] = self.patients[p]['x0_ 1'] + self.patients[p]['x0_ 2']
        self.tuple_state = self.matrix_2_tuple(self.states)

    def full_reset(self):
        for p in range(0, self.n_envs):
            for s in range(0, 13):
                state_num = ' '+str(s+1) if (s+1) < 10 else str(s+1)
                self.states[p][s] = self.patients[p]['x0_'+state_num]
            self._last_Qsto[p] = self.patients[p]['x0_ 1'] + self.patients[p]['x0_ 2']
        self.tuple_state = self.matrix_2_tuple(self.states)
        self.t0 = torch.zeros(1, device=self.device)

    def matrix_2_tuple(self, matrix):
        return torch.tensor_split(torch.flatten(torch.reshape(matrix, (1, self.n_envs * self.n_states))), self.n_envs * self.n_states)
        # matrix = torch.flatten(torch.reshape(matrix, (1, self.n_envs * self.n_states)))
        # tuple = torch.tensor_split(matrix, self.n_envs * self.n_states)

    def tuple_2_matrix(self, input_tuple):
        return torch.reshape(torch.stack(list(input_tuple), dim=1), (self.n_envs, self.n_states))
        # matrix = torch.reshape(torch.stack(list(tuple), dim=1), (self.n_envs, self.n_states)) ==> This is implemented
        # matrix = torch.reshape(torch.tensor(tuple, device=self.device), (self.n_envs, self.n_states))  # inefficient

    def observation(self):
        self.o_GM = torch.unsqueeze(self.states[:, 12], 1)  # subcutaneous glucose (mg/kg)
        return torch.div(self.o_GM, self.VG)
        # Gsub = torch.div(self.o_GM, self.VG)

    def t(self):
        return self.t0


# Implementation for a single T1D patient, not used in the simulator
class T1DPatient(nn.Module):
    def __init__(self, device, params, init_state=None, random_init_bg=False, seed=None, t0=0.0):
        super().__init__()
        self.device = device
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = nn.Parameter(torch.tensor([seed]), requires_grad=False)
        self.t0 = torch.zeros(1, device=self.device) #nn.Parameter(torch.tensor([0.0]))
        self.name = self._params['Name']
        self.const_zero = torch.zeros(1, device=self.device)  # g/min CHO
        self.const_ones = torch.ones(1, device=self.device)  # g/min CHO
        self.reset()
        self.SAMPLE_TIME = nn.Parameter(torch.tensor([1.0]), requires_grad=False)  # min
        self.EAT_RATE = 5 * torch.ones(1, device=self.device)  # g/min CHO
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
            Et = self.const_ones * 0

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
        #print('initialising ########')
        if self._init_state is None:
            self.init_state = (self._params['x0_ 1'], self._params['x0_ 2'], self._params['x0_ 3'],
                               self._params['x0_ 4'], self._params['x0_ 5'], self._params['x0_ 6'],
                               self._params['x0_ 7'], self._params['x0_ 8'], self._params['x0_ 9'],
                               self._params['x0_10'], self._params['x0_11'], self._params['x0_12'],
                               self._params['x0_13'])
        else:
            self.init_state = self._init_state
        self.state = self.init_state  # chirath addition

        self.rand_gen = torch.Generator(device=self.device) #np.random.RandomState(self.seed)
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
        self._last_foodtaken = self.const_ones * 0
        self.planned_meal = self.const_ones * 0
        self.to_eat = self.const_ones * 0

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
            self.to_eat = self.const_ones * 0
        return self.to_eat

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action['CHO'])
        action.update({'CHO': nn.Parameter(to_eat, requires_grad=False)})  #action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action['CHO'] > 0 and self._last_action['CHO'] <= 0:
            #logger.info('t = {}, patient starts eating ...'.format(self.t))
            self._last_Qsto = self.state[0] + self.state[1]
            self._last_foodtaken = self.const_ones * 0
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
        # tt = torch.linspace(float(t0), float(event_t), int((float(event_t) - float(t0)) * 500), device=self.device)[1:-1]
        # tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
        # solution = odeint(self, self.state, tt, atol=1e-8, rtol=1e-8)
        #
        # self.state = tuple(s[-1] for s in solution)
        # self.t0 = event_t

    def update_state(self):
        # todo: appropriately update.
        return self.state

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
