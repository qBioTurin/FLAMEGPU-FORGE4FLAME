#!/bin/bash
#SBATCH --partition=cascadelake
#SBATCH -N 1
#SBATCH --gres gpu:v100:1
time bash /beegfs/home/scontald/gitProjects/FLAMEGPU2-F4F/run.sh -expdir NoCountermeasures -e ON -prun 8