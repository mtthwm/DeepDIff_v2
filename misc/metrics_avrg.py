import pandas as pd
import numpy as np

# ---- Config ----
INPUT_CSV  = r"C:\Users\maxtk\Documents\Menon Lab\Initial Publication\Diffuser_encoded_snapshot_hyperspectral_polarimetry\Code\Probabalistic_UNET\metrics_prob_nll\master_metrics_cumulative_augmented.csv"                 # change if needed
OUT_DS     = "summary_per_dataset.csv"
OUT_DS_POL = "summary_per_dataset_polarization.csv"

# If your CSV has no header, set this to the exact column list and pass header=None below.
COLUMNS = [
    "avg_ssim_3d","avg_ssim_2d","avg_mse","avg_mae","avg_rase","avg_fidelity",
    "avg_RSE","avg_sre_l1","avg_sre_l2","avg_rsre_l1","avg_rsre_l2",
    "avg_sigma_mean","avg_sigma_std","avg_sigma_min","avg_sigma_max","avg_sigma_median",
    "num_images","dataset","polarization"
]

def weighted_avg(df, value_cols, weight_col="num_images"):
    w = df[weight_col].astype(float)
    wsum = w.sum()
    if wsum == 0:
        return df[value_cols].mean(numeric_only=True)
    return (df[value_cols].multiply(w, axis=0).sum(axis=0)) / wsum

def add_psnr(df):
    """Add PSNR column from avg_mse."""
    df["avg_psnr"] = 10 * np.log10(1.0 / df["avg_mse"].clip(lower=1e-12))
    return df

def main():
    # Read CSV safely
    try:
        df = pd.read_csv(INPUT_CSV)
        if set(COLUMNS) - set(df.columns):
            df = pd.read_csv(INPUT_CSV, header=None, names=COLUMNS)
    except Exception:
        df = pd.read_csv(INPUT_CSV, header=None, names=COLUMNS)

    # Ensure correct numeric conversion
    num_cols = [c for c in COLUMNS if c not in ("dataset","polarization")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    metric_cols = [c for c in num_cols if c != "num_images"]

    # ---- Per dataset ----
    rows = []
    for ds, g in df.groupby("dataset"):
        avgs = weighted_avg(g, metric_cols, "num_images")
        total_imgs = g["num_images"].sum()
        out = {"dataset": ds, "num_images": int(total_imgs)}
        out.update(avgs.to_dict())
        rows.append(out)
    ds_summary = pd.DataFrame(rows)
    ds_summary = add_psnr(ds_summary)
    ds_summary.to_csv(OUT_DS, index=False)

    # ---- Per dataset & polarization ----
    rows = []
    for (ds, pol), g in df.groupby(["dataset","polarization"]):
        avgs = weighted_avg(g, metric_cols, "num_images")
        total_imgs = g["num_images"].sum()
        out = {"dataset": ds, "polarization": pol, "num_images": int(total_imgs)}
        out.update(avgs.to_dict())
        rows.append(out)
    dsp_summary = pd.DataFrame(rows)
    dsp_summary = add_psnr(dsp_summary)
    dsp_summary.to_csv(OUT_DS_POL, index=False)

    print(f"Wrote {OUT_DS} and {OUT_DS_POL} with PSNR values.")

if __name__ == "__main__":
    main()