.. _state:

Observation Space
============

The glucoregulatory system is a complex non-linear system. In the glucose control problem the true system state information is not available. Instead we are limited to sensor and actuator measurements. Hence, in we use an observation-space insteads of the true state-space. 

In GluCoEnv, two types of observation-spaces are provided. The default observation-space named  :guilabel:`current`, outputs the current glucose sensor measurement, while :guilabel:`past_history` provides an observation-space where glucose sensor and insulin pump measurements are augmented by a window of past historical values. The :guilabel:`past_history` is useful when formulating the problem as a Partially Observable Markov Decision Process (POMDP), while :guilabel:`current` is used in many open-loop clinical treatment algorithms and classical algorithms like PID and MPC.

The observation-space is a setting which should be provided when initialising the environment. 
