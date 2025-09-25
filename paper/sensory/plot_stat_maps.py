"""Functions to make simple plots with M/EEG data."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import io
import os.path as op

import nilearn.image
import numpy as np
import nibabel as nib


from mne._freesurfer import _mri_orientation



from mne.utils import (
    _check_option,


)
from mne.viz.utils import (
    _figure_agg,
    _prepare_trellis,
    _validate_type,
    plt_show,
)

def plot_mri_with_contours(
    *,
    mri_fname,
    surfaces,
    orientation="axial",
    slices=None,
    show=True,
    show_indices=False,
    show_orientation=False,
    width=512,
    slices_as_subplots=True,
    stat_map_fname=None,
    stat_map_cmap="hot",
    stat_map_alpha=0.9,
    stat_map_threshold=3.1,  # <-- Add threshold argument
    mri_alpha=0.9,  # <-- Add alpha for MRI display
    surface_alpha=0.9,  # <-- Add alpha for surfaces
    stat_map_interpolation="nearest",
    stat_map_display_interpolation="nearest",  # <-- Add interpolation argument
    surface_thickness=0.75,  # <-- Add surface thickness argument
    ax=None,  # <-- new argument
    crop_percentile=50,  # <-- new argument for cropping
    stat_map_vmin=None,  # <-- new argument
    stat_map_vmax=None,  # <-- new argument
    colorbar=False,      # <-- new argument
):
    """Plot BEM contours on anatomical MRI slices, with optional statistical map overlay.

    Parameters
    ----------
    stat_map_fname : str | None
        Path to a NIfTI file containing a statistical map to overlay.
    stat_map_cmap : str
        Colormap for the statistical map.
    stat_map_alpha : float
        Alpha transparency for the statistical map overlay.
    stat_map_threshold : float
        Threshold for displaying statistical map values (default: 3.1).
    mri_alpha : float
        Alpha transparency for the anatomical MRI display (default: 0.8).
    surface_alpha : float
        Alpha transparency for the surface contours (default: 0.9).
    stat_map_interpolation : str
        Interpolation method for displaying the statistical map (default: "nearest").
    surface_thickness : float
        Line thickness for the surface contours (default: 0.75).
    slices_as_subplots : bool
        Whether to add all slices as subplots to a single figure, or to
        create a new figure for each slice. If ``False``, return NumPy
        arrays instead of Matplotlib figures.

    Returns
    -------
    matplotlib.figure.Figure | list of array
        The plotted slices.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    # For ease of plotting, we will do everything in voxel coordinates.
    _validate_type(show_orientation, (bool, str), "show_orientation")
    if isinstance(show_orientation, str):
        _check_option(
            "show_orientation", show_orientation, ("always",), extra="when str"
        )
    _check_option("orientation", orientation, ("coronal", "axial", "sagittal"))

    # Load the T1 data
    nim = nib.load(mri_fname)
    data = nim.get_fdata()

    # Load statistical map if provided
    if stat_map_fname is not None:
        stat_nim = nib.load(stat_map_fname)
        stat_data = stat_nim.get_fdata()
        # Reorient stat map to match MRI if needed
        if not np.allclose(stat_nim.affine, nim.affine):
            stat_nim = nilearn.image.resample_to_img(stat_nim, nim,
                                                     interpolation=stat_map_interpolation,
                                                     force_resample=True,
                                                     copy_header=True)
            stat_data = stat_nim.get_fdata()
    else:
        stat_data = None

    axis, x, y = _mri_orientation(orientation)

    n_slices = data.shape[axis]

    # if no slices were specified, pick some equally-spaced ones automatically
    if slices is None:
        slices = np.round(np.linspace(start=0, stop=n_slices - 1, num=14)).astype(int)

        # omit first and last one (not much brain visible there anyway…)
        slices = slices[1:-1]

    slices = np.atleast_1d(slices).copy()
    slices[slices < 0] += n_slices  # allow negative indexing
    if (
        not np.array_equal(np.sort(slices), slices)
        or slices.ndim != 1
        or slices.size < 1
        or slices[0] < 0
        or slices[-1] >= n_slices
        or slices.dtype.kind not in "iu"
    ):
        raise ValueError(
            "slices must be a sorted 1D array of int with unique "
            "elements, at least one element, and no elements "
            f"greater than {n_slices - 1:d}, got {slices}"
        )

    # create of list of surfaces
    surfs = list()
    for file_name, color in surfaces:
        surf = dict()
        surface_image = nib.load(file_name)
        verts, faces = surface_image.darrays[0].data, surface_image.darrays[1].data
        # transform the vertices to voxel coordinates
        voxel_verts = np.linalg.solve(nim.affine, np.hstack((verts, np.ones((verts.shape[0], 1)))).T).T
        surf["rr"], surf["tris"] = voxel_verts, faces
        # move surface to voxel coordinate system
        surfs.append((surf, color))

    sources = list()


    # get the figure dimensions right
    if slices_as_subplots:
        n_col = 4
        fig, axs, _, _ = _prepare_trellis(len(slices), n_col)
        fig.set_facecolor("k")
        dpi = fig.get_dpi()
        n_axes = len(axs)
    else:
        n_col = n_axes = 1
        dpi = 96
        # 2x standard MRI resolution is probably good enough for the
        # traces
        w = width / dpi
        figsize = (w, w / data.shape[x] * data.shape[y])

    bounds = np.concatenate(
        [[-np.inf], slices[:-1] + np.diff(slices) / 2.0, [np.inf]]
    )  # float
    slicer = [slice(None)] * 3
    ori_labels = dict(R="LR", A="PA", S="IS")
    xlabels, ylabels = ori_labels["RAS"[x]], ori_labels["RAS"[y]]
    path_effects = [patheffects.withStroke(linewidth=4, foreground="k", alpha=0.75)]
    figs = []
    # If ax is provided, plot only one slice into that axis
    if ax is not None:
        # Only plot the first slice in slices
        sl = slices[0]
        lower, upper = bounds[0], bounds[1]
        slicer[axis] = sl
        dat = data[tuple(slicer)].T

        # --- Cropping logic ---
        bg_thresh = np.percentile(dat, crop_percentile)
        mask = dat > bg_thresh
        if np.any(mask):
            coords = np.argwhere(mask)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            dat_cropped = dat[min_y:max_y+1, min_x:max_x+1]
        else:
            min_y, min_x, max_y, max_x = 0, 0, dat.shape[0]-1, dat.shape[1]-1
            dat_cropped = dat

        # Plot MRI
        ax.imshow(dat_cropped, cmap=plt.cm.gray, origin="lower", alpha=mri_alpha)

        # Plot stat map overlay
        if stat_data is not None:
            stat_slice = stat_data[tuple(slicer)].T
            stat_slice_cropped = stat_slice[min_y:max_y+1, min_x:max_x+1]
            stat_mask = np.isfinite(stat_slice_cropped) & (stat_slice_cropped >= stat_map_threshold)
            if np.any(stat_mask):
                im = ax.imshow(
                    np.ma.masked_where(~stat_mask, stat_slice_cropped),
                    cmap=stat_map_cmap,
                    alpha=stat_map_alpha,
                    origin="lower",
                    interpolation=stat_map_display_interpolation,
                    vmin=stat_map_vmin,
                    vmax=stat_map_vmax,
                )
                if colorbar:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(cbar.ax.get_yticklabels(), color='white')
                    cbar.outline.set_edgecolor('white')
                    # Set colorbar label values to be visible
                    cbar.ax.tick_params(axis='y', colors='white')

        ax.set_autoscale_on(False)
        ax.axis("off")
        ax.set_aspect("equal")

        # Plot contours
        for surf, color in surfs:
            surf_x = surf["rr"][:, x]
            surf_y = surf["rr"][:, y]
            surf_axis = surf["rr"][:, axis]
            # Only plot contours within the cropped region
            mask_surf = (
                (surf_x >= min_x) & (surf_x <= max_x) &
                (surf_y >= min_y) & (surf_y <= max_y)
            )
            if not np.any(mask_surf):
                continue
            ax.tricontour(
                surf_x[mask_surf] - min_x,
                surf_y[mask_surf] - min_y,
                surf["tris"],
                surf_axis[mask_surf],
                levels=[sl],
                colors=color,
                linewidths=surface_thickness,
                alpha=surface_alpha,
                zorder=1,
            )

        if len(sources):
            in_slice = (sources[:, axis] >= lower) & (sources[:, axis] < upper)
            ax.scatter(
                sources[in_slice, x] - min_x,
                sources[in_slice, y] - min_y,
                marker=".",
                color="#FF00FF",
                s=1,
                zorder=2,
            )
        if show_indices:
            ax.text(
                dat_cropped.shape[1] // 8 + 0.5,
                0.5,
                str(sl),
                color="w",
                fontsize="x-small",
                va="bottom",
                ha="left",
            )
        # label the axes
        kwargs = dict(
            color="#66CCEE",
            fontsize="medium",
            path_effects=path_effects,
            family="monospace",
            clip_on=False,
            zorder=5,
            weight="bold",
        )
        always = show_orientation == "always"
        if show_orientation:
            if ai % n_col == 0 or always:  # left
                ax.text(
                    0, dat_cropped.shape[0] / 2.0, xlabels[0], va="center", ha="left", **kwargs
                )
            if ai % n_col == n_col - 1 or ai == n_axes - 1 or always:  # right
                ax.text(
                    dat_cropped.shape[1] - 1,
                    dat_cropped.shape[0] / 2.0,
                    xlabels[1],
                    va="center",
                    ha="right",
                    **kwargs,
                )
            if ai >= n_axes - n_col or always:  # bottom
                ax.text(
                    dat_cropped.shape[1] / 2.0,
                    0,
                    ylabels[0],
                    ha="center",
                    va="bottom",
                    **kwargs,
                )
            if ai < n_col or n_col == 1 or always:  # top
                ax.text(
                    dat_cropped.shape[1] / 2.0,
                    dat_cropped.shape[0] - 1,
                    ylabels[1],
                    ha="center",
                    va="top",
                    **kwargs,
                )

        return  # nothing to return
    # Otherwise, original behavior (for single stat map plotting)
    bounds = np.concatenate(
        [[-np.inf], slices[:-1] + np.diff(slices) / 2.0, [np.inf]]
    )  # float
    slicer = [slice(None)] * 3
    ori_labels = dict(R="LR", A="PA", S="IS")
    xlabels, ylabels = ori_labels["RAS"[x]], ori_labels["RAS"[y]]
    path_effects = [patheffects.withStroke(linewidth=4, foreground="k", alpha=0.75)]
    figs = []
    # If ax is provided, plot only one slice into that axis
    for ai, (sl, lower, upper) in enumerate(zip(slices, bounds[:-1], bounds[1:])):
        if ax is not None and ai == 0:
            plot_ax = ax
        elif slices_as_subplots:
            plot_ax = axs[ai]
        else:
            fig = _figure_agg(figsize=figsize, dpi=dpi, facecolor="k")
            plot_ax = fig.add_axes([0, 0, 1, 1], frame_on=False, facecolor="k")

        slicer[axis] = sl
        dat = data[tuple(slicer)].T

        # --- Cropping logic ---
        bg_thresh = np.percentile(dat, crop_percentile)
        mask = dat > bg_thresh
        if np.any(mask):
            coords = np.argwhere(mask)
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            dat_cropped = dat[min_y:max_y+1, min_x:max_x+1]
        else:
            min_y, min_x, max_y, max_x = 0, 0, dat.shape[0]-1, dat.shape[1]-1
            dat_cropped = dat

        # Plot MRI
        plot_ax.imshow(dat_cropped, cmap=plt.cm.gray, origin="lower", alpha=mri_alpha)

        # Plot stat map overlay
        if stat_data is not None:
            stat_slice = stat_data[tuple(slicer)].T
            stat_slice_cropped = stat_slice[min_y:max_y+1, min_x:max_x+1]
            stat_mask = np.isfinite(stat_slice_cropped) & (stat_slice_cropped >= stat_map_threshold)
            if np.any(stat_mask):
                im = plot_ax.imshow(
                    np.ma.masked_where(~stat_mask, stat_slice_cropped),
                    cmap=stat_map_cmap,
                    alpha=stat_map_alpha,
                    origin="lower",
                    interpolation=stat_map_interpolation,
                    vmin=stat_map_vmin,
                    vmax=stat_map_vmax,
                )
                if colorbar:
                    cbar = plt.colorbar(im, ax=plot_ax, fraction=0.046, pad=0.04)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(cbar.ax.get_yticklabels(), color='white')
                    cbar.outline.set_edgecolor('white')
                    # Set colorbar label values to be visible
                    cbar.ax.tick_params(axis='y', colors='white')

        plot_ax.set_autoscale_on(False)
        plot_ax.axis("off")
        plot_ax.set_aspect("equal")

        # Plot contours
        for surf, color in surfs:
            surf_x = surf["rr"][:, x]
            surf_y = surf["rr"][:, y]
            surf_axis = surf["rr"][:, axis]
            # Only plot contours within the cropped region
            mask_surf = (
                (surf_x >= min_x) & (surf_x <= max_x) &
                (surf_y >= min_y) & (surf_y <= max_y)
            )
            if not np.any(mask_surf):
                continue
            plot_ax.tricontour(
                surf_x[mask_surf] - min_x,
                surf_y[mask_surf] - min_y,
                surf["tris"],
                surf_axis[mask_surf],
                levels=[sl],
                colors=color,
                linewidths=surface_thickness,
                alpha=surface_alpha,
                zorder=1,
            )

        if len(sources):
            in_slice = (sources[:, axis] >= lower) & (sources[:, axis] < upper)
            plot_ax.scatter(
                sources[in_slice, x] - min_x,
                sources[in_slice, y] - min_y,
                marker=".",
                color="#FF00FF",
                s=1,
                zorder=2,
            )
        if show_indices:
            plot_ax.text(
                dat_cropped.shape[1] // 8 + 0.5,
                0.5,
                str(sl),
                color="w",
                fontsize="x-small",
                va="bottom",
                ha="left",
            )
        # label the axes
        kwargs = dict(
            color="#66CCEE",
            fontsize="medium",
            path_effects=path_effects,
            family="monospace",
            clip_on=False,
            zorder=5,
            weight="bold",
        )
        always = show_orientation == "always"
        if show_orientation:
            if ai % n_col == 0 or always:  # left
                plot_ax.text(
                    0, dat_cropped.shape[0] / 2.0, xlabels[0], va="center", ha="left", **kwargs
                )
            if ai % n_col == n_col - 1 or ai == n_axes - 1 or always:  # right
                plot_ax.text(
                    dat_cropped.shape[1] - 1,
                    dat_cropped.shape[0] / 2.0,
                    xlabels[1],
                    va="center",
                    ha="right",
                    **kwargs,
                )
            if ai >= n_axes - n_col or always:  # bottom
                plot_ax.text(
                    dat_cropped.shape[1] / 2.0,
                    0,
                    ylabels[0],
                    ha="center",
                    va="bottom",
                    **kwargs,
                )
            if ai < n_col or n_col == 1 or always:  # top
                plot_ax.text(
                    dat_cropped.shape[1] / 2.0,
                    dat_cropped.shape[0] - 1,
                    ylabels[1],
                    ha="center",
                    va="top",
                    **kwargs,
                )

        if ax is not None:
            break  # only plot one slice if ax is provided

        if not slices_as_subplots:
            # convert to NumPy array
            with io.BytesIO() as buff:
                fig.savefig(
                    buff, format="raw", bbox_inches="tight", pad_inches=0, dpi=dpi
                )
                w_, h_ = fig.canvas.get_width_height()
                plt.close(fig)
                buff.seek(0)
                fig_array = np.frombuffer(buff.getvalue(), dtype=np.uint8)

            fig = fig_array.reshape((int(h_), int(w_), -1))
            figs.append(fig)

    if slices_as_subplots:
        plt_show(show, fig=fig)
        return fig
    else:
        return figs




def plot_multiple_stat_maps(
    *,
    mri_fname,
    surfaces,
    stat_map_fnames,
    slices=None,
    orientation="axial",
    width=512,
    slices_as_subplots=True,
    stat_map_cmap="hot",
    stat_map_alpha=0.9,
    stat_map_threshold=3.1,
    mri_alpha=0.9,
    surface_alpha=0.9,
    stat_map_interpolation="nearest",
    surface_thickness=0.75,
    show=True,
    stat_map_vmin=None,  # <-- new argument
    stat_map_vmax=None,  # <-- new argument
    colorbar=False,      # <-- new argument
):
    """
    Plot multiple statistical maps on MRI contours and combine into a large figure.

    Parameters
    ----------
    stat_map_fnames : list of str
        List of NIfTI filenames for statistical maps.
    All other arguments are passed to _plot_mri_contours.

    Returns
    -------
    matplotlib.figure.Figure
        Combined figure with all stat maps (rows) and slices (columns).
    """
    import matplotlib.pyplot as plt

    n_maps = len(stat_map_fnames)
    n_slices = len(slices) if slices is not None else 1
    fig_width = width / 96 * n_slices
    fig_height = width / 96 * n_maps
    fig, axs = plt.subplots(
        n_maps, n_slices, figsize=(fig_width, fig_height),
        squeeze=False
    )
    fig.patch.set_facecolor("black")
    for i, stat_map_fname in enumerate(stat_map_fnames):
        for j, sl in enumerate(slices):
            plot_mri_with_contours(
                mri_fname=mri_fname,
                surfaces=surfaces,
                stat_map_fname=stat_map_fname,
                slices=[sl],
                orientation=orientation,
                width=width,
                slices_as_subplots=False,
                stat_map_cmap=stat_map_cmap,
                stat_map_alpha=stat_map_alpha,
                stat_map_threshold=stat_map_threshold,
                mri_alpha=mri_alpha,
                surface_alpha=surface_alpha,
                stat_map_interpolation=stat_map_interpolation,
                surface_thickness=surface_thickness,
                show=False,
                ax=axs[i, j],
                stat_map_vmin=stat_map_vmin,
                stat_map_vmax=stat_map_vmax,
                colorbar=(colorbar and j == n_slices - 1),  # show colorbar on last column only
            )
            axs[i, j].set_axis_off()
            axs[i, j].set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    base_dir = "/media/conda2/public/sensory"
    for subject in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        fmriprep_dir = op.join(base_dir, "derivatives", "fmriprep")
        subjects_dir = op.join(fmriprep_dir, "sourcedata", "freesurfer")
        mri_fname = op.join(fmriprep_dir, f"sub-{subject}", "anat", f"sub-{subject}_desc-preproc_T1w.nii.gz")
        fwhm = 0
        stat_fnames = list()
        for fwhm in (0, 3, 6, 9, 12):
            stat_map_fname = f"{base_dir}/derivatives/fsl_gaussian/sub-{subject}/func/sub-{subject}_task-lefthand_run-1_space-T1w_desc-preproc_bold_fwhm-{fwhm}.feat/stats/zstat1.nii.gz"
            stat_fnames.append(stat_map_fname)
        surfaces = [
            (op.join(fmriprep_dir, f"sub-{subject}", "anat",
                     f"sub-{subject}_hemi-L_pial.surf.gii"), "g"),
            (op.join(fmriprep_dir, f"sub-{subject}", "anat",
                     f"sub-{subject}_hemi-R_pial.surf.gii"), "g"),
            (op.join(fmriprep_dir, f"sub-{subject}", "anat",
                     f"sub-{subject}_hemi-L_white.surf.gii"), "b"),
            (op.join(fmriprep_dir, f"sub-{subject}", "anat",
                     f"sub-{subject}_hemi-R_white.surf.gii"), "b"),

        ]

        kwargs = dict(
            mri_fname=mri_fname,
            surfaces=surfaces,
            slices=[203, 204, 205, 206, 207],
            orientation="axial",
            width=512,
            slices_as_subplots=True,
            stat_map_cmap="hot",
            stat_map_alpha=0.9,
            stat_map_threshold=3.1,
            mri_alpha=0.9,
            surface_alpha=0.9,
            stat_map_interpolation="nearest",
            surface_thickness=0.75,
            show=True,
            stat_map_vmin=3.1,
            stat_map_vmax=12.0,
            colorbar=True
        )

        slices_str = "_".join(map(str, kwargs["slices"]))

        fig = plot_multiple_stat_maps(
            stat_map_fnames=stat_fnames,
            **kwargs
        )
        fig.savefig(f"sub-{subject}_sensory_stat_maps_gaussian_{slices_str}.png",
                    dpi=300,
                    bbox_inches="tight")

        stat_fnames = list()
        for fwhm in (3, 6, 9, 12):
            stat_map_fname = f"{base_dir}/derivatives/fsl_constrained/sub-{subject}/func/sub-{subject}_task-lefthand_run-1_space-T1w_desc-csmooth_fwhm-{fwhm}_bold.feat/stats/zstat1.nii.gz"
            stat_fnames.append(stat_map_fname)

        fig = plot_multiple_stat_maps(
            stat_map_fnames=stat_fnames,
            **kwargs
        )
        fig.savefig(f"sub-{subject}_sensory_stat_maps_constrained_{slices_str}.png",
                    dpi=300,
                    bbox_inches="tight")
