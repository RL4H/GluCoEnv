import time
from glucoenv import T1DEnv
import gc
import argparse
from csv import writer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--n_env', type=int, default=3)
parser.add_argument('--steps', type=int, default=3)
args = parser.parse_args()

if __name__ == '__main__':
    device = args.device
    n_env = args.n_env
    steps = args.steps

    #names = ['adolescent#001', 'adolescent#002']
    names = ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
             'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010']

    for name in names:
        env = T1DEnv.make(env=name, n_env=n_env, scenario='moderate', device=device)
        SBB = T1DEnv.benchmark_controller(env=name, n_env=n_env, sample_time=env.sample_time,
                                          mode='perfect', device=device)
        action = SBB.init_action()
        start = time.time()
        for i in range(0, steps):
            obs, rew, done, info = env.step(action)
            action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
        end = time.time()
        execution_time = (end-start)
        del env
        del SBB
        gc.collect()

        FILE_NAME = 'data/'+str(n_env) + '_' + device + '_' + str(steps) +'.csv'
        with open(FILE_NAME, 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([execution_time])
            f_object.close()


