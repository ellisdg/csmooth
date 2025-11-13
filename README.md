# ConstrainedSmoothing

Detailed description of your project.

## Installation

### Docker (Recommended)

The easiest way to use csmooth is via Docker. Pull the pre-built image:

```bash
docker pull ellisdg/csmooth
```

### Local Installation

Install the package using pip:

```bash
pip install -e .
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Running with Docker (Recommended)

To process fMRIPrep outputs with Docker:

```bash
docker run -v /path/to/fmriprep:/data/fmriprep \
           -v /path/to/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --fwhm 6.0 \
           --subject <subject_id>
```

**Required Arguments:**
- First positional: Path to fMRIPrep derivatives directory
- Second positional: Path to output directory

**Smoothing Parameters (one required):**
- `--fwhm`: Target smoothing kernel FWHM in mm (mutually exclusive with `--tau`)
- `--tau`: Smoothing parameter in seconds (mutually exclusive with `--fwhm`)

**Optional Arguments:**
- `--subject`: Specific subject ID to process (e.g., `01` for `sub-01`). If not provided, processes all subjects
- `--bold_files`: List of custom BOLD files in T1w space (requires `--subject`)
- `--output_to_mni`: Resample outputs to MNI space
- `--mask_dilation`: Number of voxels to dilate the mask (default: 3)
- `--voxel_size`: Isotropic voxel size for resampling in mm (default: 1.0)
- `--no_resample`: Skip resampling step
- `--multiproc`: Number of parallel processes (default: 4)
- `--overwrite`: Overwrite existing output files
- `--low_mem`: Use low memory mode (process timepoints separately)
- `--debug`: Enable debug logging

### Running Locally

After installing the package, you can run the fMRIPrep processing script directly:

```bash
python -m csmooth.fmriprep \
    /path/to/fmriprep \
    /path/to/output \
    --fwhm 6.0 \
    --subject <subject_id>
```

Or use the same script directly:

```bash
python csmooth/fmriprep.py \
    /path/to/fmriprep \
    /path/to/output \
    --fwhm 6.0
```

You can also use it as a library in your own Python code:

```python
from csmooth.fmriprep import process_fmriprep_subject

# Define smoothing parameters
parameters = {
    'fwhm': 6.0,
    'mask_dilation': 3,
    'voxel_size': 1.0,
    'multiproc': 4,
    'overwrite': False,
    'low_memory': False
}

# Process a single subject
process_fmriprep_subject(
    fmriprep_subject_dir='/path/to/fmriprep/sub-01',
    output_subject_dir='/path/to/output/sub-01',
    parameters=parameters
)
```

## Examples

### Process all subjects with 6mm FWHM smoothing (Docker):

```bash
docker run -v /data/fmriprep:/data/fmriprep \
           -v /data/output:/data/output \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --fwhm 6.0
```

### Process single subject with custom parameters (Docker):

```bash
docker run -v /data:/data \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --fwhm 6.0 \
           --voxel_size 2.0 \
           --mask_dilation 5 \
           --multiproc 8
```

### Process with tau parameter and output to MNI space (Local):

```bash
python -m csmooth.fmriprep \
    /path/to/fmriprep \
    /path/to/output \
    --subject 01 \
    --tau 100 \
    --output_to_mni
```

### Process custom BOLD files (Docker):

```bash
docker run -v /data:/data \
           ellisdg/csmooth \
           /data/fmriprep \
           /data/output \
           --subject 01 \
           --bold_files /data/custom/sub-01_task-rest_space-T1w_bold.nii.gz \
           --fwhm 6.0
```

### Low memory mode for large datasets (Local):

```bash
python -m csmooth.fmriprep \
    /path/to/fmriprep \
    /path/to/output \
    --fwhm 6.0 \
    --low_mem \
    --overwrite
```
