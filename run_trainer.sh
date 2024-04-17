#! /bin/sh
#$ -N audemix
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=12:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=100G


. /etc/profile.d/modules.sh

export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
module load anaconda
source activate jax_gpu $1

python ./run.py
source deactivate