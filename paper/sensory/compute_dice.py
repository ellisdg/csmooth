import os
import re
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



def compute_dice(image_mask1, image_mask2):
    """
    Compute the Dice coefficient between two binary masks.
    """
    intersection = (image_mask1 & image_mask2).sum()
    return 2.0 * intersection / (image_mask1.sum() + image_mask2.sum())


def main():
    base_dir = "/media/conda2/public/sensory"
    task = "lefthand"

    df_filename = "/home/david/PycharmProjects/torchfMRI/sensory/dice.csv"

    overwrite = False
    if overwrite or not os.path.exists(df_filename):
        data = list()
        for method in ("constrained", "gaussian"):
            for fsl_dir_run1 in sorted(glob.glob(f"{base_dir}/derivatives/fsl_{method}/**/sub-*task-{task}*run-1*space-T1w*.feat",
                                                 recursive=True)):
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
    else:
        df = pd.read_csv(df_filename)

    no_smoothing_mask = df["FWHM"] == "0"
    df.loc[no_smoothing_mask, "SmoothingMethod"] = "None"

    seaborn.set_style("darkgrid")
    seaborn.set_palette("muted")
    fig, ax = plt.subplots()
    seaborn.boxplot(df, x="FWHM", y="Dice", ax=ax, hue="SmoothingMethod")
    ax.set_xlabel("FWHM (mm)")
    ax.set_ylabel("Dice Score Coefficient")

    fig.savefig("/home/david/PycharmProjects/torchfMRI/sensory/dice_boxplot.png", dpi=300, bbox_inches="tight")

    print("Running mixed effects model ")
    model = smf.mixedlm("Dice ~ FWHM + C(SmoothingMethod, Treatment(reference='gaussian')):FWHM", data=df, groups="Subject")
    result = model.fit()
    print(result.summary())
    result.pvalues.to_csv("/home/david/PycharmProjects/torchfMRI/sensory/dice_mixed_effects_pvalues.csv")
    result.fe_params.to_csv("/home/david/PycharmProjects/torchfMRI/sensory/dice_mixed_effects_fe_params.csv")
    result.conf_int().to_csv("/home/david/PycharmProjects/torchfMRI/sensory/dice_mixed_effects_conf_int.csv")




if __name__ == "__main__":
    main()