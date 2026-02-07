import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import yaml
import matplotlib

matplotlib.use("Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from paper.hcpa.fpr.analyze_feat_null_results import main as analyze_main  # noqa: E402
from paper.hcpa.fpr.run_fpr_pipeline import aggregate_results  # noqa: E402
from paper.hcpa.fpr.run_feat_null_firstlevel import main as feat_main  # noqa: E402


def _write_nifti(path: str, shape=(4, 4, 4, 8), fill=0.0):
    img = nib.Nifti1Image(np.full(shape, fill, dtype=np.float32), np.eye(4))
    img.to_filename(path)


def _make_config(tmpdir: str) -> str:
    cfg = {
        "paths": {
            "cleaned_dir": os.path.join(tmpdir, "cleaned"),
            "csmooth_dir": os.path.join(tmpdir, "cs"),
            "gaussian_dir": os.path.join(tmpdir, "gs"),
            "output_root": os.path.join(tmpdir, "out"),
            "mask_path": None,
            "subjects_glob": "sub-*",
        },
        "smoothing": {"fwhm_values": [3]},
        "designs": {
            "n_designs_per_run": 2,
            "min_duration": 1.0,
            "max_duration": 2.0,
            "seed": 1,
        },
        "glm": {
            "hrf_model": "glover",
            "drift_model": "cosine",
            "high_pass": None,
            "oversampling": 50,
        },
        "group": {
            "group_size": 5,
            "n_groups": 1,
            "cluster_forming_ps": [0.01, 0.001],
            "cluster_fwer": 0.05,
            "n_perm": 0,
            "two_sided": True,
            "seed": 123,
        },
        "feat": {
            "fsf_template": os.path.join(tmpdir, "template.fsf"),
            "container": None,
            "smoothing_fwhm": 0,
            "log_dir": os.path.join(tmpdir, "logs"),
        },
    }
    cfg_path = os.path.join(tmpdir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


def test_analyze_feat_null_results_detects_cluster(tmp_path, monkeypatch):
    cfg_path = _make_config(tmp_path)
    out_root = os.path.join(tmp_path, "out")
    feat_dir = os.path.join(out_root, "feat", "csmooth", "fwhm-3", "sub-01", "dir-AP_run-01", "design-000.feat", "stats")
    os.makedirs(feat_dir, exist_ok=True)
    z_path = os.path.join(feat_dir, "zstat1.nii.gz")
    data = np.zeros((5, 5, 5), dtype=np.float32)
    data[1:3, 1:3, 1:3] = 5.0
    nib.Nifti1Image(data, np.eye(4)).to_filename(z_path)

    argv = [
        "analyze",
        "--config",
        cfg_path,
        "--method",
        "csmooth",
        "--fwhm",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    analyze_main()

    summary_path = os.path.join(out_root, "feat", "csmooth", "fwhm-3", "feat_null_summary.csv")
    df = pd.read_csv(summary_path)
    assert not df.empty
    assert df["detected"].max() == 1


def test_aggregate_results_outputs_csv_and_plot(tmp_path):
    out_root = os.path.join(tmp_path, "out")
    os.makedirs(out_root, exist_ok=True)
    summary_dir = os.path.join(out_root, "feat", "csmooth", "fwhm-3")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "feat_null_summary.csv")
    pd.DataFrame({"method": ["csmooth", "csmooth"], "fwhm": [3, 3], "cluster_forming_p": [0.01, 0.001], "detected": [0, 1]}).to_csv(summary_path, index=False)

    out_csv = os.path.join(out_root, "feat", "fpr_summary.csv")
    out_png = os.path.join(out_root, "feat", "fpr_plot.png")
    aggregate_results(out_root, ["csmooth"], [3], out_csv, out_png, [0.01, 0.001])

    assert os.path.exists(out_csv)
    assert os.path.exists(out_png)
    agg = pd.read_csv(out_csv)
    assert set(["method", "fwhm", "cluster_forming_p", "fpr"]).issubset(set(agg.columns))
    assert agg.shape[0] == 2


def test_run_feat_null_firstlevel_writes_fsf(tmp_path, monkeypatch):
    cfg_path = _make_config(tmp_path)
    # create template
    template = "fmri_filename\noutput_directory\nsmoothing_fwhm\nevents_txt\n"
    with open(os.path.join(tmp_path, "template.fsf"), "w", encoding="utf-8") as f:
        f.write(template)

    # fmri and design inputs
    cs_dir = os.path.join(tmp_path, "cs", "sub-01", "func")
    os.makedirs(cs_dir, exist_ok=True)
    fmri_path = os.path.join(cs_dir, "sub-01_task-rest_dir-AP_run-01_space-MNI152NLin2009cAsym_desc-csmooth_fwhm-3_bold.nii.gz")
    _write_nifti(fmri_path)

    design_dir = os.path.join(tmp_path, "out", "designs", "sub-01", "dir-AP_run-01")
    os.makedirs(design_dir, exist_ok=True)
    design_path = os.path.join(design_dir, "design-000.tsv")
    pd.DataFrame({"onset": [0.0], "duration": [1.0], "trial_type": ["stim"], "amplitude": [1.0]}).to_csv(design_path, sep="\t", index=False)

    # capture subprocess without running feat
    calls = []
    monkeypatch.setattr("subprocess.run", lambda cmd, check: calls.append(cmd))
    argv = [
        "feat",
        "--config",
        cfg_path,
        "--method",
        "csmooth",
        "--fwhm",
        "3",
        "--design-id",
        "0",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    feat_main()

    fsf_path = os.path.join(tmp_path, "out", "feat", "csmooth", "fwhm-3", "sub-01", "dir-AP_run-01", "design-000.fsf")
    assert os.path.exists(fsf_path)
    with open(fsf_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert fmri_path in text
    assert design_path in text

