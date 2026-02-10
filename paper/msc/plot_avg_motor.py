import os
import glob
import re
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from paper.plot_stat_maps import plot_mri_with_contours


# -----------------------------------------------------------------------------
# Helpers for locating files (mirrors logic from `compute_motor_stats.py`)
# -----------------------------------------------------------------------------


def _get_base_dirs() -> Dict[str, str]:
    """Return base directories for data and derivatives.

    Mirrors layout from `compute_motor_stats.py` and adds anatomical/surface dirs.
    """

    base_dir = "/media/conda2/public/MSC"
    return {
        "base": base_dir,
        "gaussian": f"{base_dir}/derivatives/fsl_gaussian",
        "constrained": f"{base_dir}/derivatives/fsl_constrained",
        "constrained_nr": f"{base_dir}/derivatives/fsl_constrained_nr",
        "stats": f"{base_dir}/derivatives/motor_stats",
        # fMRIPrep derivatives hold T1 and GIFTI surfaces in a common space
        "anat": f"{base_dir}/derivatives/fmriprep",
    }


def _find_subjects(gaussian_dir: str) -> List[str]:
    """Return list of subject directory paths under the gaussian dir."""

    subject_dirs = sorted(glob.glob(os.path.join(gaussian_dir, "sub-*")))
    return subject_dirs


def _extract_subject(subject_dir: str) -> str:
    """Extract MSC subject ID from subject directory path."""

    m = re.search(r"sub-(MSC\d+)", subject_dir)
    if not m:
        raise ValueError(f"Could not parse subject ID from {subject_dir}")
    return m.group(1)


def _load_avg_zstat_image(stats_dir: str, subject: str, task: str, event: str) -> nib.Nifti1Image:
    """Load the per-subject, no-smoothing average zstat image.

    This uses the filename convention from `compute_motor_stats.py`:
    `sub-{subject}_task-{task}_event-{event}_zstat.nii.gz`
    """

    avg_fname = os.path.join(
        stats_dir,
        os.path.dirname(f"sub-{subject}"),
        "func",
        f"sub-{subject}_task-{task}_event-{event}_zstat.nii.gz",
    )
    if not os.path.exists(avg_fname):
        raise FileNotFoundError(
            f"Average zstat image not found for subject {subject}, event {event}: {avg_fname}"
        )
    return nib.load(avg_fname)


# -----------------------------------------------------------------------------
# Slice selection logic
# -----------------------------------------------------------------------------


def _find_best_axial_block(data: np.ndarray, block_len: int = 6) -> Tuple[int, int]:
    """Find consecutive axial slice block with the greatest activation.

    Parameters
    ----------
    data : 3D numpy array
        Z-statistic volume in voxel space.
    block_len : int
        Number of consecutive slices to include in the block.

    Returns
    -------
    start, stop : int
        Start (inclusive) and stop (exclusive) slice indices along z-axis.
    """

    if data.ndim != 3:
        raise ValueError("Expected 3D data array for slice selection")

    # We define "activation" as the sum of positive z values per slice.
    z_axis = 2  # axial
    slice_scores = []
    for z in range(data.shape[z_axis]):
        sl = data[:, :, z]
        positive = np.clip(sl, 0, None)
        slice_scores.append(positive.sum())
    slice_scores = np.asarray(slice_scores)

    if block_len >= data.shape[z_axis]:
        return 0, data.shape[z_axis]

    window_sums = np.convolve(slice_scores, np.ones(block_len, dtype=float), mode="valid")
    start = int(window_sums.argmax())
    stop = start + block_len
    return start, stop


def _find_best_axial_block_from_file(stat_fname: str, block_len: int = 6) -> List[int]:
    """Compute best axial slice block in the *zstat* volume.

    We work purely in the stat map's index (voxel) space here, to respect
    its native geometry when deciding where activation is strongest.
    Later we will map these indices onto the nearest T1 slices using the
    image affines.
    """

    img = nib.load(stat_fname)
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D stat map, got shape {data.shape}")

    z_axis = 2
    slice_scores = []
    for z in range(data.shape[z_axis]):
        sl = data[:, :, z]
        positive = np.clip(sl, 0, None)
        slice_scores.append(positive.sum())
    slice_scores = np.asarray(slice_scores)

    if block_len >= data.shape[z_axis]:
        return list(range(data.shape[z_axis]))

    window_sums = np.convolve(slice_scores, np.ones(block_len, dtype=float), mode="valid")
    start = int(window_sums.argmax())
    stop = start + block_len
    return list(range(start, stop))


def _map_zstat_slices_to_t1(z_indices: List[int], stat_fname: str, t1_fname: str) -> List[int]:
    """Map z-stat axial slice indices onto nearest T1 slices using affines.

    Parameters
    ----------
    z_indices : list of int
        Axial slice indices in the zstat (stat map) volume.
    stat_fname : str
        Path to the zstat NIfTI.
    t1_fname : str
        Path to the T1 NIfTI.

    Returns
    -------
    t1_slices : list of int
        Nearest axial slice indices in the T1 volume.
    """

    stat_img = nib.load(stat_fname)
    t1_img = nib.load(t1_fname)

    stat_aff = stat_img.affine
    t1_aff = t1_img.affine
    t1_aff_inv = np.linalg.inv(t1_aff)

    t1_n_slices = t1_img.shape[2]
    t1_slices = []

    for z in z_indices:
        # Choose the centre of the slice in stat voxel coordinates
        ijk_stat = np.array([stat_img.shape[0] / 2.0, stat_img.shape[1] / 2.0, z, 1.0])
        xyz = stat_aff @ ijk_stat
        ijk_t1 = t1_aff_inv @ xyz
        z_t1 = int(round(ijk_t1[2]))
        # Clamp to valid T1 slice range
        z_t1 = max(0, min(t1_n_slices - 1, z_t1))
        t1_slices.append(z_t1)

    # Deduplicate and sort to satisfy plot_mri_with_contours requirements
    t1_slices = sorted(set(t1_slices))
    return t1_slices


# -----------------------------------------------------------------------------
# Plotting logic
# -----------------------------------------------------------------------------


def _average_subject_from_sessions(session_files: List[str], output_fname: str) -> str:
    """Average session-level zstat images into a per-subject image.

    Parameters
    ----------
    session_files : list of str
        Paths to NIfTI images to average.
    output_fname : str
        Destination filename for the averaged image.

    Returns
    -------
    output_fname : str
        The path to the written average image.
    """

    if not session_files:
        raise ValueError("No session files provided for averaging")

    print("Averaging session-level zstat images:")
    for f in session_files:
        print(f"  {f}")

    img0 = nib.load(session_files[0])
    data_stack = np.stack([np.asarray(nib.load(f).dataobj) for f in session_files], axis=0)
    avg_data = data_stack.mean(axis=0)
    avg_img = img0.__class__(avg_data, img0.affine)

    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    print(f"Writing subject-average stat map to {output_fname}")
    avg_img.to_filename(output_fname)
    return output_fname


def _get_session_zstats_for_subject(stats_dir: str, subject: str, task: str, event: str) -> List[str]:
    """Collect all session-level zstat images for a subject/event.

    These are the per-session *no-smoothing* averages written by
    `compute_motor_stats.py`, with filenames of the form

    `sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_event-{event}_zstat.nii.gz`.
    """

    pattern = os.path.join(
        stats_dir,
        f"sub-{subject}",
        "ses-*",
        "func",
        f"sub-{subject}_ses-*_task-{task}_event-{event}_zstat.nii.gz",
    )
    files = sorted(glob.glob(pattern))
    return files


def _get_t1_and_surfaces(base_dirs: Dict[str, str], subject: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Return T1w file and GIFTI surface paths for a subject.

    Uses fMRIPrep-style derivatives where both the T1 and surfaces live
    in a consistent space expected by `plot_mri_with_contours`.
    """

    anat_dir = base_dirs["anat"]

    # T1w in subject's fMRIPrep anat directory (T1w in T1w space)
    t1_fname = os.path.join(
        anat_dir,
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_desc-preproc_T1w.nii.gz",
    )
    if not os.path.exists(t1_fname):
        # Fall back to non-preproc T1 if needed
        alt_t1 = os.path.join(
            anat_dir,
            f"sub-{subject}",
            "anat",
            f"sub-{subject}_T1w.nii.gz",
        )
        if os.path.exists(alt_t1):
            t1_fname = alt_t1
        else:
            raise FileNotFoundError(f"Could not find T1w for subject {subject} in {anat_dir}")

    # Freesurfer-derived GIFTI surfaces produced by fMRIPrep
    lh_pial = os.path.join(
        anat_dir,
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_hemi-L_pial.surf.gii",
    )
    rh_pial = os.path.join(
        anat_dir,
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_hemi-R_pial.surf.gii",
    )

    lh_white = os.path.join(
        anat_dir,
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_hemi-L_white.surf.gii",
    )
    rh_white = os.path.join(
        anat_dir,
        f"sub-{subject}",
        "anat",
        f"sub-{subject}_hemi-R_white.surf.gii",
    )

    surfaces = []
    # Use white and pial surfaces with different colors for context
    if os.path.exists(lh_white):
        surfaces.append((lh_white, "b"))
    if os.path.exists(rh_white):
        surfaces.append((rh_white, "b"))
    if os.path.exists(lh_pial):
        surfaces.append((lh_pial, "r"))
    if os.path.exists(rh_pial):
        surfaces.append((rh_pial, "r"))

    if not surfaces:
        raise FileNotFoundError(f"Could not find any GIFTI surfaces for subject {subject} in {anat_dir}")

    return t1_fname, surfaces


def _plot_subject_event(
    subject: str,
    event: str,
    stat_map_fname: str,
    t1_fname: str,
    surfaces: List[Tuple[str, str]],
    output_dir: str,
    block_len: int = 6,
) -> str:
    """Create an axial montage using `plot_mri_with_contours` for one subject/event.

    We:
    - determine the best axial slice block *in zstat space*;
    - map those slices to the nearest T1 axial indices via the affines;
    - call `plot_mri_with_contours` with the subject's T1 and surfaces;
    - save the figure as a PNG.
    """

    # Step 1: best slices in zstat index space
    zstat_slices = _find_best_axial_block_from_file(stat_map_fname, block_len=block_len)

    # Step 2: map those to nearest T1 slices
    t1_slices = _map_zstat_slices_to_t1(zstat_slices, stat_map_fname, t1_fname)

    # Derive color scaling from the subject-average zstat map
    stat_img = nib.load(stat_map_fname)
    stat_data = np.asarray(stat_img.dataobj)
    pos_vals = stat_data[np.isfinite(stat_data) & (stat_data > 0)]
    if pos_vals.size > 0:
        stat_vmin = float(np.percentile(pos_vals, 50))  # median positive
        stat_vmax = float(np.percentile(pos_vals, 99))  # near upper tail
        # Ensure vmin < vmax; if not, fall back to simple range
        if not np.isfinite(stat_vmin) or not np.isfinite(stat_vmax) or stat_vmin >= stat_vmax:
            stat_vmin = float(pos_vals.min())
            stat_vmax = float(pos_vals.max())
    else:
        # Fallback: use full finite range
        finite_vals = stat_data[np.isfinite(stat_data)]
        if finite_vals.size == 0:
            stat_vmin, stat_vmax = 0.0, 1.0
        else:
            stat_vmin = float(finite_vals.min())
            stat_vmax = float(finite_vals.max())

    fig = plot_mri_with_contours(
        mri_fname=t1_fname,
        surfaces=surfaces,
        orientation="axial",
        slices=t1_slices,
        show=False,
        slices_as_subplots=True,
        stat_map_fname=stat_map_fname,
        stat_map_cmap="hot",
        stat_map_alpha=0.9,
        stat_map_threshold=2.0,
        stat_map_vmin=stat_vmin,
        stat_map_vmax=stat_vmax,
        surface_thickness=0.25,
        colorbar=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    png_fname = os.path.join(output_dir, f"sub-{subject}_event-{event}_avg_zstat_mri_contours.png")
    print(f"Writing MRI+stat map montage to {png_fname}")
    fig.savefig(png_fname, dpi=300)
    plt.close(fig)
    return png_fname


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def main():
    """Create average zstat MRI+surface plots per subject for each motor event.

    Steps per subject & event:
    - gather all session-level no-smoothing averages from `motor_stats`;
    - compute a subject-level average zstat volume (if needed);
    - choose consecutive axial slices of maximal activation;
    - use `plot_mri_with_contours` with subject T1 and pial surfaces to
      generate an overlay plot.
    """

    base_dirs = _get_base_dirs()
    stats_dir = base_dirs["stats"]

    # Identify subjects from the stats directory (those with session-level zstats)
    subject_dirs = sorted(glob.glob(os.path.join(stats_dir, "sub-*")))
    subjects = [re.search(r"sub-(MSC\d+)", d).group(1) for d in subject_dirs]

    task = "motor"
    events = [
        "leftfoot",
        "lefthand",
        "rightfoot",
        "righthand",
        "tongue",
    ]

    output_root = os.path.join(stats_dir, "plots", "avg_motor")

    for subject in subjects:
        print(f"Processing subject {subject}")
        # Get subject T1 and surfaces
        try:
            t1_fname, surfaces = _get_t1_and_surfaces(base_dirs, subject)
        except FileNotFoundError as exc:
            print(str(exc))
            continue

        for event_name in events:
            # Collect session-level stat maps for this subject/event
            session_files = _get_session_zstats_for_subject(stats_dir, subject, task, event_name)
            if not session_files:
                print(f"No session-level zstats found for subject {subject}, event {event_name}")
                continue

            # Subject-average filename (derived from compute_motor_stats convention)
            subj_avg_fname = os.path.join(
                stats_dir,
                os.path.dirname(f"sub-{subject}"),
                "func",
                f"sub-{subject}_task-{task}_event-{event_name}_zstat.nii.gz",
            )

            if not os.path.exists(subj_avg_fname):
                _average_subject_from_sessions(session_files, subj_avg_fname)

            # Now plot using T1, surfaces, and subject-average stat map
            _plot_subject_event(
                subject=subject,
                event=event_name,
                stat_map_fname=subj_avg_fname,
                t1_fname=t1_fname,
                surfaces=surfaces,
                output_dir=output_root,
                block_len=3,
            )


if __name__ == "__main__":
    main()
