#!/bin/bash

#SBATCH --account=notchpeak-gpu

#SBATCH --partition=notchpeak-gpu

#SBATCH --gres=gpu:1

#SBATCH --nodes=1

##SBATCH --ntasks-per-node=1

##SBATCH --cpus-per-task=1

#SBATCH --mem=32G

#SBATCH --gres=gpu:1

#SBATCH --time=2:00:00

#SBATCH --job-name=morales_macbeth_and_bronch_model_verif_11262025

#SBATCH --output=morales_macbeth_and_bronch_model_verif_11262025.log

#SBATCH --mail-type=FAIL,BEGIN,END

#SBATCH --mail-user=u1344001@umail.utah.edu

## Load Miniforge
module load miniforge3/24.9.0


## Activate the conda environment

source activate /uufs/chpc.utah.edu/common/home/u1344001/BICEPS_HSI_2025/hsp_env



## Navigate to the project directory

cd /uufs/chpc.utah.edu/common/home/u1344001/BICEPS_HSI_2025/DeepDIff_v2



## Run the test script

dataroot_0=/scratch/general/nfs1/u1344001/data/Exp3/physical_verification
name_0=bronch_lcd_pol-1
checkpoints_dir_0=/scratch/general/nfs1/u1344001/data/Exp3/checkpoints

python test.py \
    --dataroot "$dataroot_0" \
    --name "$name_0" \
    --checkpoints_dir "$checkpoints_dir_0" \
    --polarization "-1" \
    --model pix2pix_v2 \
    --input_nc 1 \
    --output_nc 212 \
    --netG unet_2048_to_512 \
    --netG_reps 2 \
    --netD_mult 0 \
    --norm_bitwise \
    --netD_mult 0 \
    --norm_bitwise \
    --use_nll \
    --lambda_l1 0 \
    --norm instance \
    --no_dropout \
    --eval

dataroot_1=/scratch/general/nfs1/u1344001/data/Exp7/physical_verification
name_1=macbeth_lcd_pol-1
checkpoints_dir_1=/scratch/general/nfs1/u1344001/data/Exp7/checkpoints

python test.py \
    --dataroot "$dataroot_1" \
    --name "$name_1" \
    --checkpoints_dir "$checkpoints_dir_1" \
    --polarization "-1" \
    --model pix2pix_v2 \
    --input_nc 1 \
    --output_nc 212 \
    --netG unet_2048_to_512 \
    --netG_reps 2 \
    --netD_mult 0 \
    --norm_bitwise \
    --netD_mult 0 \
    --norm_bitwise \
    --use_nll \
    --lambda_l1 0 \
    --norm instance \
    --no_dropout \
    --eval