#! /bin/sh
#$ -N audemix
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=12:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=100G


bash run_trainer.sh $SGE_TASK_ID