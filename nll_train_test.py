import os, sys
import subprocess
import shutil
import pandas as pd
import argparse

DATASETS = [
    {
        'name': 'bronch_lcd',
        'dataroot': '/scratch/general/nfs1/u1344001/data/Exp3/bronch_lcd',
        'checkpoints_dir': '/scratch/general/nfs1/u1344001/data/Exp3/checkpoints',
    }
]

POL_ANGLES = [-1]
TRAIN_SCRIPT = 'train.py'
TEST_SCRIPT = 'test.py'
EVAL_SCRIPT = 'HSI_comparison_probabalistic.py'
PER_IMAGE_SCRIPT = 'HSI_comparison_probabalistic_per_image.py'
RESULTS_DIR = 'results'  # Directory where test images are saved
METRICS_DIR = '/uufs/chpc.utah.edu/common/home/u1344001/Exp3/metrics_prob_nll' 

# Fixed options for training and testing, matching banknotes_training.sh
TRAIN_OPTS = [
    '--model', 'pix2pix_v2',
    '--input_nc', '1',
    '--output_nc', '212',
    '--n_epochs', '10',
    '--n_epochs_decay', '10',
    '--save_epoch_freq', '5',
    '--netG', 'unet_2048_to_512',
    '--netG_reps', '2',
    '--netD_mult', '0',
    '--norm_bitwise',
    '--use_nll',
    '--lambda_l1', '0',
    '--norm', 'instance',
    '--no_dropout'          # turn dropout off
]
TEST_OPTS = [
    '--model', 'pix2pix_v2',
    '--input_nc', '1',
    '--output_nc', '212',
    '--netG', 'unet_2048_to_512',
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
def run_cmd(cmd, label=None):
    """Run a command and stream its stdout/stderr in real time.
    - Forces unbuffered Python (-u) using the current interpreter (sys.executable).
    - Prefixes each printed line with [label] when provided.
    """
    # Normalize command to use the current Python interpreter unbuffered when it calls 'python'
    cmd = list(cmd)
    if len(cmd) > 0 and os.path.basename(cmd[0]).startswith('python'):
        cmd = [sys.executable, '-u'] + cmd[1:]
    elif len(cmd) > 0 and cmd[0] in ('python', 'python3'):
        cmd = [sys.executable, '-u'] + cmd[1:]    
        prefix = f"[{label}] " if label else ""
    print(f"{prefix}Running: {' '.join(str(x) for x in cmd)}")    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)    
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            print(f"{prefix}{line.rstrip()}")
    finally:
        if process.stdout:
            process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{prefix}Command failed (exit {return_code}): {' '.join(cmd)}")

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
            train_cmd = [
                'python', TRAIN_SCRIPT,
                '--dataroot', ds['dataroot'],
                '--name', model_name,
                '--checkpoints_dir', ckpt_dir,
                '--polarization', str(pol),
            ] + TRAIN_OPTS
            run_cmd(train_cmd)

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