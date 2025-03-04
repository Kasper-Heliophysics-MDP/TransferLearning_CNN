#!/bin/bash
#SBATCH --job-name=unet_test
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow28-minerva
for i in $(ls -d /minerva/pmurphy/SUN_TRACKING*);
do
    python3 -u /data/pmurphy/scripts/nenufar_ml/output_prediction.py $i /data/pmurphy/Unet_resnet34_120epochs_bce_2classes_moreaugment_largercosinedecay_cleanerdset_kfold0
done