#!/bin/bash
#PBS -P ny83
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=scratch/ny83
#PBS -o out_ppo0.txt
#PBS -e err_ppo0.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/ny83/ch9972/ppo_example.py --env adolescent#001 --n_env 2 --d_env cpu --d_agent cpu
wait
