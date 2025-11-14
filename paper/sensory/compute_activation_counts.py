import nibabel as nib
import nilearn
import nilearn.image
import numpy as np
import glob
import pandas as pd
import re
from tqdm import tqdm


def main():
    base_dir = "/media/conda2/public/sensory"
    task = "lefthand"
    data = list()
    gaussian_fsl_dirs = sorted(glob.glob(f"{base_dir}/derivatives/fsl_gaussian/**/sub-*task-{task}*space-T1w*.feat", recursive=True))
    constrained_fsl_dirs = sorted(glob.glob(f"{base_dir}/derivatives/fsl_constrained/**/sub-*task-{task}*space-T1w*.feat", recursive=True))
    constrained_nr_fsl_dirs = sorted(
        glob.glob(f"{base_dir}/derivatives/fsl_constrained_nr/**/sub-*task-{task}*space-T1w*.feat", recursive=True))
    fsl_dirs = gaussian_fsl_dirs + constrained_fsl_dirs + constrained_nr_fsl_dirs
    for fsl_dir in tqdm(fsl_dirs):
        print(fsl_dir)
        subject = re.search(r"sub-(\d+)", fsl_dir).group(1)
        run = re.search(r"run-(\d+)", fsl_dir)
        if run is not None:
            run = run.group(1)
        else:
            # If no run is found this indicates that the folder is the average of two runs
            # We are only interested in the results of the individual runs
            continue
        fwhm = re.search(r"fwhm-(\d+)", fsl_dir).group(1)
        if "constrained_nr" in fsl_dir:
            smoothing_method = "Constrained_NR"
        elif "constrained" in fsl_dir:
            smoothing_method = "Constrained"
        elif "gaussian" in fsl_dir:
            smoothing_method = "Gaussian"
        else:
            raise ValueError(f"Unknown smoothing method in {fsl_dir}")
        aparc_aseg_filename = f"{base_dir}/derivatives/fmriprep/sourcedata/freesurfer/sub-{subject}/mri/aparc+aseg.mgz"
        aseg_filename = f"{base_dir}/derivatives/fmriprep/sourcedata/freesurfer/sub-{subject}/mri/aseg.mgz"
        # dseg_filename = f"{base_dir}/derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_dseg.nii.gz"
        aparc_aseg_image = nib.load(aparc_aseg_filename)
        aseg_image = nib.load(aseg_filename)
        zstat_image = nib.load(f"{fsl_dir}/stats/zstat1.nii.gz")
        zstat_data = np.asarray(zstat_image.dataobj)
        func_image = nib.load(f"{fsl_dir}/filtered_func_data.nii.gz")
        func_data = np.asarray(func_image.dataobj)

        # resample the aseg and dseg images to the zstat image space
        resampled_aparc_aseg = nilearn.image.resample_to_img(aparc_aseg_image, zstat_image, interpolation="nearest",
                                                             force_resample=True, copy_header=True)
        resampled_aseg = nilearn.image.resample_to_img(aseg_image, zstat_image, interpolation="nearest",
                                                         force_resample=True, copy_header=True)
        aparc_aseg_data = np.asarray(resampled_aparc_aseg.dataobj)
        rh_precentral_mask = aparc_aseg_data == 2024
        rh_postcentral_mask = aparc_aseg_data == 2022
        aseg_data = np.asarray(resampled_aseg.dataobj)
        wm_mask = np.isin(aseg_data, (2, 41))
        gm_mask = np.isin(aseg_data, (3, 42))
        background_mask = aseg_data == 0

        for mask, name in zip((rh_precentral_mask, rh_postcentral_mask, wm_mask, gm_mask, background_mask),
                              ("rh_precentral", "rh_postcentral", "wm", "gm", "background")):
            row = [subject, name, run,
                   np.mean(zstat_data[mask]),  # mean zstat
                   np.std(zstat_data[mask]),  # std zstat
                   np.sum(zstat_data[mask] > 3.1),  # n active voxels
                   np.mean(func_data[mask]),  # mean functional data
                   np.std(func_data[mask]),  # std functional data
                   np.mean(func_data[mask])/np.std(func_data[mask]),  # tSNR
                   fwhm,
                   smoothing_method]
            data.append(row)

    df = pd.DataFrame(data, columns=["subject", "region", "run",
                                     "mean_zstat", "std_zstat", "n_active",
                                     "mean_func", "std_func", "tSNR",
                                     "fwhm", "smoothing_method"])
    df.to_csv(f"{base_dir}/derivatives/stats/fsl_stats_task-{task}.csv")



if __name__ == "__main__":
    main()
