#!/bin/bash
#SBATCH --job-name=model_plot
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow-2.12
python3 -u /data/pmurphy/scripts/nenufar_ml/spike_dataset_train.py