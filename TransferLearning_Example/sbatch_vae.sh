#!/bin/bash
#SBATCH --job-name=nenufar_VAE
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow28-minerva

python3 -u /data/pmurphy/scripts/nenufar_ml/nenufar_VAE.py