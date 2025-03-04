#!/bin/bash
#SBATCH --job-name=segnet_compare
#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minerva
#SBATCH --mem=32gb

module load python/venvs/tensorflow28-minerva
RUNPATH=/scratch2/pmurphy
VENVPATH=/data/pmurphy
cd $RUNPATH
#echo "activating venv"
#source $VENVPATH/venvs/ml/bin/activate
# python3 -u /scratch2/pmurphy/scripts/test_model.py Unet -e 250 -b vgg16
python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Unet -e 50 -b resnet18
python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Unet -e 50 -b resnet34
python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Unet -e 50 -b resnet50
python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Unet -e 50 -b resnet101
python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Unet -e 50 -b resnet152
# python3 -u /scratch2/pmurphy/scripts/data_gen_test.py Linknet -e 100 -b resnet34
# python3 -u /scratch2/pmurphy/scripts/data_gen_test.py FPN -e 100 -b resnet34
# python3 -u /scratch2/pmurphy/scripts/data_gen_test.py PSPNet -e 100 -b resnet34