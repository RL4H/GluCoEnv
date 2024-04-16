#!/bin/bash
#PBS -P ny83
#PBS -q gpuvolta
#PBS -l walltime=24:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M chirath.hettiarachchi@anu.edu.au
#PBS -l storage=scratch/ny83
#PBS -o out_ppo000.txt
#PBS -e err_ppo000.txt
#PBS -l software=python

module load cuda/12.3.2
module load python3/3.12.1
source /scratch/ny83/ch9972/pytorch-env/bin/activate

python3 /scratch/ny83/ch9972/pytorch-env/GluCoEnv/analysis/ppo_example.py --env adolescent#001 --n_env 16 --d_env cuda:0 --d_agent cuda:0 --folder_id cudacuda0
wait



