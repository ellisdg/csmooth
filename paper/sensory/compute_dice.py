import os
import re
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import tqdm


def compute_dice(image_mask1, image_mask2):
    """
    Compute the Dice coefficient between two binary masks.
    """
    intersection = (image_mask1 & image_mask2).sum()
    return 2.0 * intersection / (image_mask1.sum() + image_mask2.sum())


def main():
    base_dir = "/media/conda2/public/sensory"
    task = "lefthand"

    df_filename = f"{base_dir}/derivatives/stats/dice_scores_task-{task}.csv"

    data = list()
    for method in tqdm.tqdm(("constrained", "constrained_nr", "gaussian"), desc="Smoothing Methods"):
        for fsl_dir_run1 in tqdm.tqdm(sorted(glob.glob(f"{base_dir}/derivatives/fsl_{method}/**/sub-*task-{task}*run-1*space-T1w*.feat",
                                             recursive=True)), desc="FSL Directories"):
            subject = re.search(r"sub-(\d+)", fsl_dir_run1).group(1)
            print(fsl_dir_run1)
            fsl_dir_run2 = fsl_dir_run1.replace("run-1_", "run-2_")
            fwhm = re.search(r"fwhm-(\d+)", fsl_dir_run1).group(1)

            subject_data = list()
            for fsl_dir in (fsl_dir_run1, fsl_dir_run2):
                input_filename = os.path.join(fsl_dir, "stats", "zstat1.nii.gz")
                image = nib.load(input_filename)
                image_data = np.asarray(image.dataobj)
                subject_data.append(image_data)

            data.append([subject, fwhm, method, compute_dice(image_mask1=subject_data[0] > 3.1, image_mask2=subject_data[1] > 3.1)])
    df = pd.DataFrame(data, columns=["Subject", "FWHM", "SmoothingMethod", "Dice"])
    df.to_csv(df_filename, index=False)


if __name__ == "__main__":
    main()