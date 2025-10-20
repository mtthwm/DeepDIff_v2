import os
import subprocess
import shutil
import pandas as pd
import argparse

DATASETS = [
    # {
    #     'name': 'banknotes_augmented',
    #     'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/banknotes',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_banknotes_augmented_prob',
    # },
    # {
    #     'name': 'william_summer_augmented',
    #     'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/William_Summer',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_william_summer_augmented_prob',
    # },
    # {
    #     'name': 'produce_augmented',
    #     'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/produce',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/Diff_nll_modelsmodels/checkpoints_produce_augmented_prob',
    # },
    {
        'name': 'invertebrates_augmented',
        'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/invertebrates',
        'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_invertebrates_augmented_prob',
    },
    # {
    #     'name': 'rescharts_augmented',
    #     'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/rescharts',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_rescharts_augmented_prob',
    # },
    # {
    #     'name': 'cumulative_augmented',
    #     'dataroot': '/scratch/general/nfs1/u1528328/img_dir/mkrauss/produce',
    #     'checkpoints_dir': '/scratch/general/nfs1/u1528328/model_dir/nll_models/checkpoints_cumulative_augmented_prob',
    # },
]

POL_ANGLES = [0]
TRAIN_SCRIPT = 'train.py'
TEST_SCRIPT = 'test.py'
EVAL_SCRIPT = 'HSI_comparison_probabalistic.py'
PER_IMAGE_SCRIPT = 'HSI_comparison_probabalistic_per_image.py'
RESULTS_DIR = 'results'  # Directory where test images are saved
METRICS_DIR = '/uufs/chpc.utah.edu/common/home/u1528328/Probabalistic_UNET/metrics_prob_nll' 

# Fixed options for training and testing, matching banknotes_training.sh
TRAIN_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '212',
    '--n_epochs', '10',
    '--n_epochs_decay', '10',
    '--save_epoch_freq', '5',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise',
    '--use_nll',
    '--lambda_l1', '0',
    '--norm', 'instance',
    '--no_dropout'          # turn dropout off
]
TEST_OPTS = [
    '--model', 'pix2pix',
    '--input_nc', '1',
    '--output_nc', '212',
    '--netG', 'unet_1024',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise',
    '--use_nll',
    '--lambda_l1', '0',
    '--norm','instance',    # InstanceNorm
    '--no_dropout',
    '--eval'                # standard eval; with IN, outputs match train/eval
]

# Helper to run a command and print output
def run_cmd(cmd):
    print('Running:', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to process (from DATASETS)')
    args = parser.parse_args()

    selected_datasets = DATASETS
    if args.dataset:
        selected_datasets = [ds for ds in DATASETS if ds['name'] == args.dataset]
        if not selected_datasets:
            raise ValueError(f"Dataset {args.dataset} not found in DATASETS.")
        
    all_metrics = []         # rows from HSI_comparison_probabalistic.py (averages)
    all_metrics_per_image = []   # rows from HSI_comparison_prob_per_image.py (one row per image)
    for ds in selected_datasets:
        dataset_name = ds['name']
        for pol in POL_ANGLES:
            # Append polarization to model name and checkpoint dir
            model_name = f"{ds['name']}_pol{pol}"
            ckpt_dir = os.path.join(ds['checkpoints_dir'], f"pol{pol}")

            # 1. Train
            # train_cmd = [
            #     'python', TRAIN_SCRIPT,
            #     '--dataroot', ds['dataroot'],
            #     '--name', model_name,
            #     '--checkpoints_dir', ckpt_dir,
            #     '--polarization', str(pol),
            # ] + TRAIN_OPTS
            # run_cmd(train_cmd)

            # 2. Test
            test_cmd = [
                'python', TEST_SCRIPT,
                '--dataroot', ds['dataroot'],
                '--name', model_name,
                '--checkpoints_dir', ckpt_dir,
                '--polarization', str(pol),
            ] + TEST_OPTS
            run_cmd(test_cmd)

            # 3. Evaluate
            eval_img_dir = os.path.join(RESULTS_DIR, model_name, 'validation_latest', 'images')
            # Count number of images for --num_images
            num_images = 0
            if os.path.exists(eval_img_dir):
                num_images = len([f for f in os.listdir(eval_img_dir) if f.startswith('cb_raw_') and f.endswith('.tif')])
            # Write metrics CSV outside of results folder to avoid deletion
            metrics_csv = os.path.join(METRICS_DIR, f'metrics_prob_{model_name}.csv')
            print(f"Writing metrics to: {metrics_csv}")
            eval_cmd = [
                'python', EVAL_SCRIPT,
                '--results_dir', eval_img_dir,
                '--num_images', str(num_images if num_images > 0 else 50),
                '--metrics_csv', metrics_csv
            ]
            run_cmd(eval_cmd)

if __name__ == "__main__":
    main()