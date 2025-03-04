#!/bin/bash
#SBATCH --job-name=unet_test
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow28-minerva
# RUNPATH=/scratch/pmurphy
# VENVPATH=/data/pmurphy
# cd $RUNPATH
#echo "activating venv"
#source $VENVPATH/venvs/ml/bin/activate
# python3 -u /scratch2/pmurphy/scripts/test_model.py Unet -e 250 -b vgg16
python3 -u /data/pmurphy/scripts/nenufar_ml/data_gen_test.py Unet -e 120 -b resnet152