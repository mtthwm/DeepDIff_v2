#!/bin/bash

#SBATCH --account=notchpeak-gpu

#SBATCH --partition=notchpeak-gpu

#SBATCH --gres=gpu:1

#SBATCH --nodes=1

##SBATCH --ntasks-per-node=1

##SBATCH --cpus-per-task=1

#SBATCH --mem=32G

#SBATCH --gres=gpu:1

#SBATCH --time=12:00:00

#SBATCH --job-name=bronch_lcd_11142025-1526_resume

#SBATCH --output=bronch_lcd_11142025-1526_resume.log

#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --mail-user=u1344001@umail.utah.edu



## Activate the conda environment

source activate /uufs/chpc.utah.edu/common/home/u1344001/BICEPS_HSI_2025/hsp_env



## Navigate to the project directory

cd /uufs/chpc.utah.edu/common/home/u1344001/BICEPS_HSI_2025/DeepDIff_v2



## Run the training script

python nll_train_test.py