#!/bin/bash
#SBATCH --job-name=model_plot
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow28-minerva

# RUNPATH=/scratch2/pmurphy
# cd $RUNPATH
# source $RUNPATH/venvs/ml/bin/activate
python3 -u /data/pmurphy/scripts/nenufar_ml/analyse_model.py
