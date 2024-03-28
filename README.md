[comment]: <> (# )
<p align="center">
<img src="https://raw.githubusercontent.com/chirathyh/chirathyh.github.io/main/images/glucoenv.png" alt="GluCoEnv" width="477"/>
</p>

<div align="center">

---

**GluCoEnv - Glucose Control Environment** is a simulation environment which aims to facilitate the development of Reinforcement Learning based Artificial Pancreas Systems for Glucose Control in Type 1 Diabetes. 

![license](https://img.shields.io/badge/License-MIT-yellow.svg)
[![python_sup](https://img.shields.io/badge/python-3.8-black.svg?)](https://www.python.org/downloads/release/python-380/)
[![capsml](https://img.shields.io/badge/Web-CAPSML-red)](https://capsml.com/)
[![DOI](https://img.shields.io/badge/DOI-10.25911/CXAQ--3151-blue)](http://hdl.handle.net/1885/305591)
</div>

---
### About
This project implements in-silico Type 1 Diabetes (T1D) subjects for developing glucose control algorithms. The glucose control environment includes 30 subjects (10 children, adolescents, and adults each), which extends the work of [Simglucose](https://github.com/jxx123/simglucose) and UVA/Padova 2008 simulators by following an end-to-end GPU-based implmentation using the PyTorch framework. The project aim is to facilitate the development of Reinforcement Learning (RL) based control algorithms by providing a high-performance environment for experimentation. 

Research related to RL-based glucose control systems are relatively minimal compared to popular RL tasks (games, physics simulations etc). The task of glucose control requires ground up development where prolem formulations, state-action space representations, reward function formulations are not well established. Hence, researchers have to run significant amount of experiments to design, develop and tune hyperparameters. This groundup development requires significant compute and effort. 

The key highlights of **GluCoEnv** are the vectorized parallel environments designed to run on a GPU and the flexibility to develop RL-based algorithms for glucose control and benchmarking. The problem also lacks proper benchmarking scenarios and controllers, which have been implemented in this environment to provide some guidance on the task.

You can find more details and our RL-based glucose control algorithms by visiting the project [**CAPSML**](https://capsml.com/).

**This project is under active development, where additional glucose dynamics models, clinical metrics, RL algorithms, and visualisation tools will be introduced.**

### Installation & Dependencies
Create a Python 3.8.0 virtual environemnt and install PyTorch 1.12.0 with CUDA 11.3.
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```
The project can be installed using the source or pypi. To install using the source,
```
git clone https://github.com/chirathyh/GluCoEnv.git
cd GluCoEnv
pip install -e .
```
Install using pip
```

pip install git+https://github.com/chirathyh/GluCoEnv.git

```
### Quick Start
Sample python scripts are provided which highlights the creation of environments, running RL and benchmark algorithm and visualisaion of simulated glucose control.
<pre>
GluCoEnv
|- benchmark_example.py: Creating envrionments, runnign benchmarks.
|- ppo_example.py: Sample PPO Algorithm.
|- visualise_example.py: Visualise glucose control.
|- example_setings.yaaml: Sample config file
</pre>
### Creating an environment & running benchmarks
```python
from glucoenv import T1DEnv

env = T1DEnv.make(env='adolescent#001', n_env=32, scenario='moderate', device='cpu')
SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=32, sample_time=env.sample_time,
                                          mode='perfect', device='cpu')
action = SBB.init_action()
for _ in range(20):
    obs, rew, done, info = env.step(action)
    action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
```

### Visualising Glucose Control
Upon the conclusion of the simulation the glucose and insulin trajectories can be visualised.
```python
from glucoenv import T1DEnv
from glucoenv.env.memory import Memory
from glucoenv.visualiser.T1DVisu import render

# demonstrate visualisation and memory
env = T1DEnv.make(env='adolescent#001', n_env=4, scenario='moderate', device='cpu')
SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=4, sample_time=env.sample_time,
                                       mode='error_quadratic', device=cpu)
mem = Memory(device='cpu', env=env, n_steps=288)
action = SBB.init_action()
for i in range(0, 288):
    obs, rew, done, info = env.step(action)
    mem.update(step=i, glucose=info['BG'], CGM=obs, insulin=action,
                CHO=info['meal'], MA=info['meal_announcement'], t=info['time'])
    action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])
render(mem, env_ids=[0])  # provide the id's of the required graphs,
```

### Running RL Algorithms (e.g., PPO)
A basic PPO algorithm is implemented in the project. The parameters of the PPO algorithm can be provided through a yaml file, and results will be saved in the target folder under the "results" directory of the project.
```python
from glucoenv import T1DEnv
from glucoenv.agent.ppo.ppo import PPO
from glucoenv.utils.utils import load_args

ppo_args = load_args(FILE='glucoenv/agent/ppo/config.yaml', folder_id='test123')
env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='train', obs_type='past_history', scenario='moderate', device='cpu')
eval_env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='eval', obs_type='past_history', scenario='moderate', device='cpu')
model = PPO(args=ppo_args, env=env, eval_env=eval_env, device='cpu')
model.learn(total_timesteps=500000)
```

### Custom Configurations
Additional environment configurations are possible through the use of a yaml file, where the meal protocol, insulin pump, glucose sensor can be configured. These configurations are expected to be valuable for the research translation phase. 
```python
from glucoenv import T1DEnv

env = T1DEnv.make(settings='example_settings.yaml')
SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=env.n_env, sample_time=env.sample_time,
                                      mode='perfect', device=env.device, env_device='cpu')

```
A sample settings.yaml file is given below.
```yaml
device: 'cpu'
seed: 0
sensor: 'GuardianRT'
pump: 'Insulet'
env_type: 'train'
obs_type: 'current'
start_midnight: True 
use_start_time: False 
start_time: 0 
use_individual_meal_protocol: False  
meal_protocol:  
  probability: [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]  
  amount_mu: [45, 10, 70, 10, 80, 10]  
  amount_sigma: [10, 5, 10, 5, 10, 5]
  time_lower_bound: [5, 9, 10, 14, 16, 20]  
  time_upper_bound: [9, 10, 14, 16, 20, 23]
  time_mu: [7, 9.5, 12, 15, 18, 21.5]
  time_sigma: [60, 30, 60, 30, 60, 30] 
# list the patient names here:
patients:
  patient1:
    name: 'adolescent#001'
    n_env: 2
    meal_protocol:
      probability: [ 0.95, 0.3, 0.95, 0.3, 0.95, 0.3 ]
      amount_mu: [ 45, 10, 70, 10, 80, 10 ]
      amount_sigma: [ 10, 5, 10, 5, 10, 5 ]
      time_lower_bound: [ 5, 9, 10, 14, 16, 20 ]
      time_upper_bound: [ 9, 10, 14, 16, 20, 23 ]
      time_mu: [ 7, 9.5, 12, 15, 18, 21.5 ]
      time_sigma: [ 60, 30, 60, 30, 60, 30 ]
  patient2:
    name: 'adolescent#002'
    n_env: 1
    meal_protocol:
      probability: [ 0.95, 0.3, 0.95, 0.3, 0.95, 0.3 ]
      amount_mu: [ 45, 10, 70, 10, 80, 10 ]
      amount_sigma: [ 10, 5, 10, 5, 10, 5 ]
      time_lower_bound: [ 5, 9, 10, 14, 16, 20 ]
      time_upper_bound: [ 9, 10, 14, 16, 20, 23 ]
      time_mu: [ 7, 9.5, 12, 15, 18, 21.5 ]
      time_sigma: [ 60, 30, 60, 30, 60, 30 ]
```
### Documentation
[Documentation](https://glucoenv.readthedocs.io/en/latest/index.html)

### Benchmark
A benchmark controller based on Standard Basal Bolus (SBB) clinical treatment strategy is replicated along with pre-defines simulation protocols. A model which simulates the human error in carbohydrate estimation is also implemented.

### Learn More 
You can learn more about RL-based glucose control by visiting the project [website](https://capsml.com/).
RL-based systems, I have designed are hosted there, which provides you the ability to simulate for your own custom scenarios.


### Citing
```
@misc{hettiarachchi2022glucoenv,
     author={Hettiarachchi, Chirath},
     title={GluCoEnv v0.1.0(2022)},
     year = {2022},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/chirathyh/GluCoEnv}},
   }
```
```
@article{hettiarachchi2023reinforcement,
  title={Reinforcement Learning-based Artificial Pancreas Systems to Automate Treatment in Type 1 Diabetes},
  author={Hettiarachchi, Chirath},
  year={2023},
  publisher={The Australian National University}
}
```

### Reporting Issues, Feature Requests & Contributing to the project
Please contact Chirath Hettiarachchi (chirath.hettiarachchi@anu.edu.au) for reporting issues or contributing to the project. Your thoughts & comments are welcome, which will be valuable towards the research in this domain.

### Acknowledgement
The UVA/Padova model made it possible to design / develop control systems for glucose regulation in Type 1 Diabetes, in an in-silico manner. However, it was designed for the Matlab framework. The development of the [Simglucose](https://github.com/jxx123/simglucose) simulator by Jinyu Xie, has been a very valuable contribution to research for the development of Reinforcement Learning based control systems. I would like to thank everyone behind all these efforts.

The open-source community projects Python, PyTorch, [torchdiffeq](https://github.com/rtqichen/torchdiffeq), [torchcubicspline](https://github.com/patrick-kidger/torchcubicspline) was heavily used in this work. I thank all the developers behind these projects, the Stack Overflow, PyTorch communities who have been very helpful the development of this work.  

This project initiated as a hobby project of Chirath Hettiarachchi, who is currently a PhD student whose research is funded by the Australian National University and the Our Health in Our Hands initiative; and by the National Computational Infrastructure (NCI Australia), and NCRIS enabled capability supported by the Australian Government. 
