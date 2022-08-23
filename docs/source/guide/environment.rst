.. _env:

Environment
============

The environment represents the glucoregulatory system of a person with type 1 diabetes. In GluCoEnv, you parallelise environments, which can be used to run multiple workers for a single environment or multiple experiments with multiple workers. 

Environments can be created using provided API and through a custom config.yaml file where greater flexibility is possible. The standard API provide standardised settings to train RL algorithms and encouraged for benchmark purposes. The settings related to aspects such as the meal protocol for the simulation (carbohydrate content, time, probability of occurence), glucose sensor and insulin pump used, type 1 diabetes patient name, simulation start time etc..

Using standard API,

.. code-block:: python 

  from glucoenv import T1DEnv

  env = T1DEnv.make(env='adolescent#001', n_env=2, env_type='train', obs_type='past_history', scenario='moderate', device='cpu')


Using a custom config file,


.. code-block:: python 

  from glucoenv import T1DEnv

  env = T1DEnv.make(settings='example_settings.yaml')
  SBB = T1DEnv.benchmark_controller(env='adolescent#001', n_env=env.n_env, sample_time=env.sample_time,
                                        mode='perfect', device=env.device, env_device='cpu')


A sample custom yaml file;

.. code-block:: yaml 

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

Todo: Improve documentation by explainaing the parameters. 
