# Simulations

This folder contains runnable simulation scripts used for the paper.

## Constrained vs Gaussian

`constrained_vs_gauss.py` simulates a single active cortical region, adds Gaussian noise, then compares constrained graph smoothing against volumetric Gaussian smoothing across one or more FWHM values.

### Quick run

```bash
python /home/david/PycharmProjects/ConstrainedSmoothing/paper/simulations/constrained_vs_gauss.py --label-region 1030 --fwhm-list 4 6 8 --save-plots
```

Outputs are written under `paper/simulations/constrained_vs_gauss_outputs/` by default.

### Key options

- `--amplitude`: Amplitude inside the GT label region.
- `--threshold-quantile`: Quantile over brain-mask voxels used to define active voxels.
- `--no-save-volumes`: Skip saving NIfTI outputs and plots.
