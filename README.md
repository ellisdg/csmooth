# ConstrainedSmoothing

ConstrainedSmoothing (csmooth) performs anatomically informed smoothing of fMRIPrep preprocessed BOLD images using cortical surface + volume constraints.

## Quickstart

You only need Docker installed. The image `ellisdg/csmooth` will be pulled automatically if not present.

1. Create (or identify) two directories:
   - `/path/to/fmriprep` (fMRIPrep derivatives containing sub-*/ folders)
   - `/path/to/output` (will be created if it doesn't exist)
2. Specify a level of smoothing:
   - FWHM in mm: `--fwhm 6.0`
3. Run (all subjects):

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --fwhm 6.0
```

Single subject (e.g. sub-01):

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --fwhm 6.0
```

Outputs will appear under `/path/to/output/sub-01/func/` with filenames like: `sub-01_task-*_space-T1w_desc-csmooth_fwhm-6_bold.nii.gz` (format may include decimals if needed).

Need MNI output?

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --fwhm 6.0 \
           --output_to_mni
```

That's it for most users. See below for advanced options.

---
## Advanced Docker Usage

You can customize processing with the optional flags below.

Common additions:

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --tau 120 \
           --voxel_size 1.0 \
           --mask_dilation 3 \
           --multiproc 8 \
           --overwrite
```

Use custom BOLD files (must be T1w space and include `space-T1w` in name):

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/custom_bold:/data/custom \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --bold_files /data/custom/sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz \
           --fwhm 6.0
```

Low memory mode (process timepoints individually): add `--low_mem`.
Disable resampling (use native BOLD voxel grid): add `--no_resample` (then `--voxel_size` is ignored).

---
## Command Line Arguments (Summary)

Required positional:
- fMRIPrep derivatives directory
- Output directory

Required smoothing parameter:
- `--fwhm <float>`  Gaussian target FWHM in mm

Optional modifiers:
- `--subject <ID>` Only that subject (e.g. 01 for sub-01). If omitted, all `sub-*` directories processed
- `--bold_files <paths...>` Custom input BOLD(s) in T1w space (requires `--subject`)
- `--output_to_mni` Resample outputs to MNI152NLin2009cAsym using fMRIPrep transform
- `--mask_dilation <int>` Voxels to dilate mask (default 3)
- `--voxel_size <float>` Isotropic resample size prior to smoothing (default 1.0 mm)
- `--no_resample` Skip resampling entirely
- `--multiproc <int>` Parallel workers (default 4)
- `--overwrite` Replace existing output files
- `--low_mem` Lower memory usage (slower)
- `--debug` Verbose logging

Edge conditions handled:
- Existing outputs skipped unless `--overwrite`
- Duplicate output filenames will raise an error
- Missing required fMRIPrep files produce clear exceptions

---
## Local Installation (For Development or Custom Pipelines)

Install (editable):

```bash
pip install -r requirements.txt
pip install -e .
```

Run CLI exactly as Docker entrypoint does:

```bash
python -m csmooth.fmriprep /path/to/fmriprep /path/to/output --fwhm 6.0
```

Single subject, tau smoothing, MNI output:

```bash
python -m csmooth.fmriprep /path/to/fmriprep /path/to/output --subject 01 --tau 100 --output_to_mni
```

Custom BOLD files:

```bash
python -m csmooth.fmriprep /path/to/fmriprep /path/to/output --subject 01 \
    --bold_files /path/to/custom/sub-01_task-rest_space-T1w_desc-preproc_bold.nii.gz \
    --fwhm 6.0
```

Low memory mode:

```bash
python -m csmooth.fmriprep /path/to/fmriprep /path/to/output --fwhm 6.0 --low_mem
```

### Programmatic Use

```python
from csmooth.fmriprep import process_fmriprep_subject

params = {
    'fwhm': 6.0,
    'mask_dilation': 3,
    'voxel_size': 1.0,
    'multiproc': 4,
    'overwrite': False,
    'low_memory': False
}

process_fmriprep_subject(
    fmriprep_subject_dir='/path/to/fmriprep/sub-01',
    output_subject_dir='/path/to/output/sub-01',
    parameters=params
)
```

---
## Building the Image (Optional)

Only needed if you are modifying the source:

```bash
docker build -t ellisdg/csmooth .
```

---
## Tips & Troubleshooting

- Ensure fMRIPrep directory contains expected `sub-*` folders and anatomical + functional outputs.
- If you see missing file errors, verify you ran fMRIPrep with surface reconstruction (`--fs-license-file`).
- Reduce memory: use `--low_mem`, increase parallelism carefully (`--multiproc`) depending on CPU cores.
- For higher spatial fidelity, keep `--voxel_size 1.0`; larger values speed up processing but reduce graph detail.
- MNI output omits T1w-space images (only MNI saved).

---
## Citation

---
## License
