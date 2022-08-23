Welcome to GluCoEnv documentation!
===================================

<p align="center">
<img src="https://raw.githubusercontent.com/chirathyh/chirathyh.github.io/main/images/glucoenv.png" alt="GluCoEnv" width="477"/>
</p>

<div align="center">

---

**GluCoEnv - Glucose Control Environment** is a simulation environment which aims to facilitate the development of Reinforcement Learning based Artificial Pancreas Systems for Glucose Control in Type 1 Diabetes. 

### About
This project implements in-silico Type 1 Diabetes (T1D) subjects for developing glucose control algorithms. The glucose control environment includes 30 subjects (10 children, adolescents, and adults each), which extends the work of [Simglucose](https://github.com/jxx123/simglucose) and UVA/Padova 2008 simulators by following an end-to-end GPU-based implmentation using the PyTorch framework. The project aim is to facilitate the development of Reinforcement Learning (RL) based control algorithms by providing a high-performance environment for experimentation. 

Research related to RL-based glucose control systems are relatively minimal compared to popular RL tasks (games, physics simulations etc). The task of glucose control requires ground up development where prolem formulations, state-action space representations, reward function formulations are not well established. Hence, researchers have to run significant amount of experiments to design, develop and tune hyperparameters. This groundup development requires significant compute and effort. 

The key highlights of **GluCoEnv** are the vectorized parallel environments designed to run on a GPU and the flexibility to develop RL-based algorithms for glucose control and benchmarking. The problem also lacks proper benchmarking scenarios and controllers, which have been implemented in this environment to provide some guidance on the task.

You can find more details and our RL-based glucose control algorithms by visiting the project [**CAPSML**](https://github.com/jxx123/simglucose).

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
