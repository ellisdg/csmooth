from os import path as op
import matplotlib.pyplot as plt
from paper.plot_stat_maps import plot_multiple_stat_maps


if __name__ == "__main__":
    base_dir = "/media/conda2/public/sensory"
    slices = [211, 212, 213]
    gaussian_fwhms = (0, 3, 6, 9, 12)
    constrained_fwhms = (3, 6, 9, 12)
    for subject in ["02"]:
        fmriprep_dir = op.join(base_dir, "derivatives", "fmriprep")
        subjects_dir = op.join(fmriprep_dir, "sourcedata", "freesurfer")
        mri_fname = op.join(fmriprep_dir, f"sub-{subject}", "anat", f"sub-{subject}_desc-preproc_T1w.nii.gz")
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
            show=False,
            stat_map_vmin=3.1,
            stat_map_vmax=12.0,
            colorbar=True,
            crop_enabled=False,
        )

        for fwhm in gaussian_fwhms:
            stat_map_fname = f"{base_dir}/derivatives/fsl_gaussian/sub-{subject}/func/sub-{subject}_task-lefthand_run-1_space-T1w_desc-preproc_bold_fwhm-{fwhm}.feat/stats/zstat1.nii.gz"
            for sl in slices:
                fig = plot_multiple_stat_maps(
                    stat_map_fnames=[stat_map_fname],
                    slices=[sl],
                    **kwargs
                )
                fig.savefig(
                    f"paper/sensory/gaussian_plots/sub-{subject}_slice-{sl}_fwhm-{fwhm}_gaussian.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

        for fwhm in constrained_fwhms:
            stat_map_fname = f"{base_dir}/derivatives/fsl_constrained_nr/sub-{subject}/func/sub-{subject}_task-lefthand_run-1_space-T1w_desc-csmooth_fwhm-{fwhm}_bold.feat/stats/zstat1.nii.gz"
            for sl in slices:
                fig = plot_multiple_stat_maps(
                    stat_map_fnames=[stat_map_fname],
                    slices=[sl],
                    **kwargs
                )
                fig.savefig(
                    f"paper/sensory/no_resample_plots/sub-{subject}_slice-{sl}_fwhm-{fwhm}_constrained.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
