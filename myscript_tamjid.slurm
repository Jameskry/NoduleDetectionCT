#!/bin/bash -l
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 10:30:00
#SBATCH -J nodule_tamjid
#SBATCH -L SCRATCH

module load python
source activate $HOME/cori-envs/tamjidcaffe

srun -n 1 -e final_output_t_a.err -o final_output_t_a.out python train.py --word_dim 311 --sentence_length 124 --class_size 5 > output_a.log



