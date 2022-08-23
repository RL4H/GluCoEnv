.. _install:

Installation & Quick-Start
============

Installation
-------------

Create a Python 3.8.0 virtual environemnt and install PyTorch 1.12.0 with CUDA 11.3.

.. code-block:: bash

  python3 -m venv env
  source env/bin/activate
  pip install --upgrade pip

The project can be installed using the source or pypi. To install using the source,

.. code-block:: bash

  git clone https://github.com/chirathyh/GluCoEnv.git
  cd GluCoEnv
  pip install -e .

Install using pip

.. code-block:: bash

  pip install git+https://github.com/chirathyh/GluCoEnv.git


Quick-Start
-------------

Sample python scripts are provided which highlights the creation of environments, running RL and benchmark algorithm and visualisaion of simulated glucose control.

.. code-block:: bash 

  GluCoEnv
  |- benchmark_example.py: Creating envrionments, runnign benchmarks.
  |- ppo_example.py: Sample PPO Algorithm.
  |- visualise_example.py: Visualise glucose control.
  |- example_setings.yaaml: Sample config file

**Creating an environment & running benchmarks**

.. code-block:: python 

  from glucoenv import T1DEnv

  env = T1DEnv.make(env='adolescent#001', n_env=32, scenario='moderate', device='cpu')
  SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=32, sample_time=env.sample_time,
                                            mode='perfect', device='cpu')
  action = SBB.init_action()
  for _ in range(20):
      obs, rew, done, info = env.step(action)
      action = SBB.get_action(glucose=obs, done=done, meal=info['SBB_MA'], t=info['time'])


**Visualising Glucose Control**

Upon the conclusion of the simulation the glucose and insulin trajectories can be visualised.

.. code-block:: python 

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

**Running RL Algorithms (e.g., PPO)**
A basic PPO algorithm is implemented in the project. The parameters of the PPO algorithm can be provided through a yaml file, and results will be saved in the target folder under the "results" directory of the project.

.. code-block:: python 

  from glucoenv import T1DEnv
  from glucoenv.agent.ppo.ppo import PPO
  from glucoenv.utils.utils import load_args

  ppo_args = load_args(FILE='glucoenv/agent/ppo/config.yaml', folder_id='test123')
  env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='train', obs_type='past_history', scenario='moderate', device='cpu')
  eval_env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='eval', obs_type='past_history', scenario='moderate', device='cpu')
  model = PPO(args=ppo_args, env=env, eval_env=eval_env, device='cpu')
  model.learn(total_timesteps=500000)

