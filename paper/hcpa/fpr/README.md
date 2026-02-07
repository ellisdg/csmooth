# Individual FPR experiment (FEAT, Eklund-style)

- Null designs are now generated as alternating blocks: each block length is sampled uniformly between `designs.min_duration` and `designs.max_duration` (default 10–30s), starting with a stim block and alternating stim/rest until the scan ends.
- High-pass is off by default (`glm.high_pass: null`). The GLM settings in `glm` are only used by the optional nilearn path (`run_first_level_null_glm.py`); FEAT runs use your FSF template/container settings under `feat`.
- The folder remains self-contained: copy `paper/hcpa/fpr` anywhere and run with local relative paths defined in `fpr_config.yaml` (defaults point to `data/` inputs and `output/`).

## Inputs
- Cleaned resting fMRI: `paths.cleaned_dir` (default `data/cleaned`)
- Pre-smoothed data: `paths.csmooth_dir`, `paths.gaussian_dir` (default `data/csmooth`, `data/gaussian`)
- FEAT template placeholders: `fmri_filename`, `output_directory`, `smoothing_fwhm`, `events_txt` (default `config/template.fsf`)
- Optional Apptainer/Singularity container path: `feat.container`

## Quick start (per script)
```bash
python -m paper.hcpa.fpr.generate_null_designs --config paper/hcpa/fpr/fpr_config.yaml
python -m paper.hcpa.fpr.run_feat_null_firstlevel --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6 --dry-run
python -m paper.hcpa.fpr.analyze_feat_null_results --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
python -m paper.hcpa.fpr.run_group_null_glm --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
```
Remove `--dry-run` to actually call FEAT.

## Dispatcher
Use the lightweight dispatcher to mirror the above commands:
```bash
python -m paper.hcpa.fpr.cli generate-designs -- --config paper/hcpa/fpr/fpr_config.yaml
python -m paper.hcpa.fpr.cli feat-null-first -- --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
python -m paper.hcpa.fpr.cli analyze-null -- --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
python -m paper.hcpa.fpr.cli group-null -- --config paper/hcpa/fpr/fpr_config.yaml --method csmooth --fwhm 6
python -m paper.hcpa.fpr.cli pipeline -- --config paper/hcpa/fpr/fpr_config.yaml --methods csmooth gaussian --fwhm 3 6 9 12
```
(The `--` separates dispatcher args from subcommand args.)

## One-shot pipeline
`run_fpr_pipeline.py` generates designs, runs FEAT, summarizes, and plots.

```bash
python -m paper.hcpa.fpr.run_fpr_pipeline --config paper/hcpa/fpr/fpr_config.yaml \
  --methods csmooth gaussian --fwhm 3 6 9 12
```

Flags:
- `--skip-designs` / `--skip-feat` / `--skip-analyze` reuse existing outputs.
- `--feat-dry-run` prints FEAT commands only.

Outputs:
- Per-run summaries: `paths.output_root/feat/<method>/fwhm-*/feat_null_summary.csv`
- Aggregated FPR: `paths.output_root/feat/fpr_summary.csv`
- Plot: `paths.output_root/feat/fpr_plot.png`
