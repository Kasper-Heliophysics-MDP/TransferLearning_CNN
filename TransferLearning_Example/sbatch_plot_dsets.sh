#!/bin/bash
#SBATCH --job-name=model_plot
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=minerva
#SBATCH --mem=64gb

module load python/venvs/tensorflow28-minerva

RUNPATH=/scratch2/pmurphy
cd $RUNPATH
# source $RUNPATH/venvs/ml/bin/activate
python3 -u /scratch2/pmurphy/scripts/plot_datasets.py
