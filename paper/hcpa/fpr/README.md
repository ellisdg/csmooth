# Individual FPR experiment (FEAT, Eklund-style)

This folder adds a lightweight null-task pipeline to estimate individual-level false positive rates for constrained vs Gaussian smoothing using FEAT.

## Inputs
- Cleaned resting fMRI: `paths.cleaned_dir`
- Pre-smoothed data: `paths.csmooth_dir`, `paths.gaussian_dir` (FWHM 3/6/9/12)
- FEAT template with placeholders: `fmri_filename`, `output_directory`, `smoothing_fwhm`, `events_txt`
- Apptainer container for FSL: `feat.container`

## Quick start
1. Generate null designs (random events per run):
   ```bash
   python paper/hcpa/fpr/generate_null_designs.py --config paper/hcpa/fpr/fpr_config.yaml
   ```
2. Run FEAT null GLMs per smoothing/method (smoothing inside FEAT set to 0; uses the pre-smoothed data). Example for csmooth FWHM=6:
   ```bash
   python paper/hcpa/fpr/run_feat_null_firstlevel.py --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6 --dry-run
   # remove --dry-run to actually run feat inside the container
   ```
3. Summarize individual detections (cluster-wise, per design/run):
   ```bash
   python paper/hcpa/fpr/analyze_feat_null_results.py --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
   python paper/hcpa/fpr/analyze_feat_null_results.py --config paper/hcpa/fpr/fpr_config.yaml --method gaussian --fwhm 6
   ```

## Notes
- FEAT smoothing is disabled (`feat.smoothing_fwhm=0`) because images are already smoothed.
- Cluster detection is done post-hoc in Python using the zstat1 maps emitted by FEAT; thresholds come from `group.cluster_forming_ps` and `group.two_sided`.
- Update paths in `fpr_config.yaml` to match your environment (template, container, output root, mask).
- The scripts do not submit SLURM jobs automatically; wrap `run_feat_null_firstlevel.py` in your preferred sbatch runner if needed.

## One-shot pipeline
Use `run_fpr_pipeline.py` to generate designs, run FEAT (pre-smoothed csmooth/gaussian inputs), summarize, and plot FPR.

```bash
python paper/hcpa/fpr/run_fpr_pipeline.py --config paper/hcpa/fpr/fpr_config.yaml \
  --methods csmooth gaussian --fwhm 3 6 9 12
```

Flags:
- `--skip-designs` / `--skip-feat` / `--skip-analyze` to reuse existing outputs.
- `--feat-dry-run` to print FEAT commands without running.

Outputs:
- Per-run summaries: `paths.output_root/feat/<method>/fwhm-*/feat_null_summary.csv`
- Aggregated FPR: `paths.output_root/feat/fpr_summary.csv`
- Plot: `paths.output_root/feat/fpr_plot.png`

## Docker (optional)
A convenience Dockerfile is provided at `paper/hcpa/fpr/Dockerfile` that installs FSL and Python deps.

Build:
```bash
docker build -t csmooth-fpr -f paper/hcpa/fpr/Dockerfile .
```

Run (mount your data and config):
```bash
docker run --rm -v /data:/data -v /data2:/data2 -v $PWD:/app \
  csmooth-fpr --config /app/paper/hcpa/fpr/fpr_config.yaml --methods csmooth gaussian --fwhm 3 6 9 12
```

Note: If you prefer your Apptainer-based FSL instead of the image’s FSL install, set `feat.container` in the config and ensure Apptainer is available; `run_feat_null_firstlevel.py` will use the container when provided, otherwise it calls local `feat`.
