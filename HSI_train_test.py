import os
import subprocess
import shutil
import pandas as pd

# -------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------
DATASETS = [
    {
        "name": "thorlabs_cubert_v2",
        "dataroot": "/scratch/general/nfs1/u1344001/data/Exp2/stretched",
        "checkpoints_dir": "/scratch/general/nfs1/u1528328/model_dir/checkpoints_thorlabs_cubert_v2",
    },
]

TRAIN_SCRIPT = "train.py"
TEST_SCRIPT = "test.py"
RESULTS_DIR = "results"
METRICS_DIR = "/uufs/chpc.utah.edu/common/home/u1528328/v2_model"

# -------------------------------------------------------------
# Training / Testing options
# -------------------------------------------------------------
TRAIN_OPTS = [
    "--model", "pix2pix_v2",
    "--dataset_mode", "aligned_v2",
    "--netG", "unet_1024_to_256",
    "--input_nc", "1",
    "--output_nc", "212",
    "--n_epochs", "10",
    "--n_epochs_decay", "10",
    "--save_epoch_freq", "5",
    "--use_nll",
    "--lambda_l1", "0",
    "--norm", "instance",
    "--no_dropout",
    "--polarization", "-1",
]

TEST_OPTS = [
    "--model", "pix2pix_v2",
    "--dataset_mode", "aligned_v2",
    "--netG", "unet_1024_to_256",
    "--input_nc", "1",
    "--output_nc", "212",
    "--use_nll",
    "--lambda_l1", "0",
    "--norm", "instance",
    "--no_dropout",
    "--eval",
    "--polarization", "-1",
]

# -------------------------------------------------------------
# Helper to execute shell commands
# -------------------------------------------------------------
def run_cmd(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

# -------------------------------------------------------------
# Main execution loop
# -------------------------------------------------------------
def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    for ds in DATASETS:
        dataset_name = ds["name"]
        ckpt_dir = ds["checkpoints_dir"]

        # ----------------------------
        # 1. Train model
        # ----------------------------
        # train_cmd = [
        #     "python", TRAIN_SCRIPT,
        #     "--dataroot", ds["dataroot"],
        #     "--name", dataset_name,
        #     "--checkpoints_dir", ckpt_dir,
        # ] + TRAIN_OPTS
        # run_cmd(train_cmd)

        # ----------------------------
        # 2. Test model
        # ----------------------------
        test_cmd = [
            "python", TEST_SCRIPT,
            "--dataroot", ds["dataroot"],
            "--name", dataset_name,
            "--checkpoints_dir", ckpt_dir,
        ] + TEST_OPTS
        run_cmd(test_cmd)

    print("All training + testing complete.")

if __name__ == "__main__":
    main()
