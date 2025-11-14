from paper.sensory.archive.compute_averages import transform_image
import glob
import re
import os
import nibabel as nib
import numpy as np
from torchfmri.utils.stats import batch_icc
import seaborn
from matplotlib import pyplot as plt
import pandas as pd
import torch
import templateflow.api as tpf
import nilearn.image


def main():
    base_dir = "/media/conda2/public/sensory"
    task = "lefthand"
    reference_filename = ("/media/conda2/public/sensory/derivatives/fmriprep/sub-01/func/"
                          "sub-01_task-lefthand_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz")
    reference_mask_filename = tpf.get("MNI152NLin2009cAsym",
                                      atlas="HOCPAL",
                                      desc="th25",
                                      resolution=2)
    print("Reference mask filename:", reference_mask_filename)
    tau_df_filename = "/home/david/PycharmProjects/torchfMRI/sensory/tau_icc.csv"
    fwhm_df_filename = "/home/david/PycharmProjects/torchfMRI/sensory/fwhm_icc.csv"
    overwrite = False
    if overwrite or not os.path.exists(tau_df_filename) or not os.path.exists(fwhm_df_filename):
        data = dict()
        for fsl_dir_run1 in sorted(glob.glob(f"{base_dir}/derivatives/fsl*/**/sub-*task-{task}*run-1*space-T1w*.feat",
                                             recursive=True)):
            print(fsl_dir_run1)
            fsl_dir_run2 = fsl_dir_run1.replace("run-1_", "run-2_")
            ext = fsl_dir_run1.split("_")[-1].replace(".feat", "")

            subject = re.search("sub-(\d+)", fsl_dir_run1).group(1)
            transform_file = f"{base_dir}/derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

            subject_data = list()
            for fsl_dir in (fsl_dir_run1, fsl_dir_run2):
                input_filename = os.path.join(fsl_dir, "stats", "zstat1.nii.gz")

                output_mni_filename = input_filename.replace("-T1w_", "-MNI152NLin2009cAsym_")
                if not os.path.exists(output_mni_filename):
                    print("Transforming image:", input_filename)
                    print("Output:", output_mni_filename)
                    transform_image(input_filename, transform_file, output_mni_filename, reference_filename)
                else:
                    print("Using existing file:", output_mni_filename)

                image = nib.load(output_mni_filename)
                image_data = np.asarray(image.dataobj)
                subject_data.append(image_data)
            if ext not in data:
                data[ext] = list()
            data[ext].append(subject_data)

        reference_mask_image = nib.load(reference_mask_filename)
        reference_mask_image = nilearn.image.resample_to_img(reference_mask_image,
                                                             nib.load(reference_filename),
                                                             interpolation="nearest")
        reference_mask = np.asarray(reference_mask_image.dataobj) == 34  # right postcentral gyrus
        tau_list = list()
        fwhm_list = list()
        for ext in data:
            ext_data = np.moveaxis(np.array(data[ext])[..., reference_mask], -1, 0)  # Move voxel dimension to the front
            print("Data shape:", ext_data.shape)
            icc = batch_icc(torch.from_numpy(ext_data)).numpy()
            df = pd.DataFrame(icc, columns=["ICC"])
            df["parameter"] = int(float(ext.split("-")[-1]))
            if "tau" in ext:
                tau_list.append(df)
            else:
                fwhm_list.append(df)

        tau_df = pd.concat(tau_list, ignore_index=True)
        fwhm_df = pd.concat(fwhm_list, ignore_index=True)
        tau_df.to_csv(tau_df_filename, index=False)
        fwhm_df.to_csv(fwhm_df_filename, index=False)
    else:
        tau_df = pd.read_csv(tau_df_filename)
        fwhm_df = pd.read_csv(fwhm_df_filename)

    # add the rows from fwhm to tau df where the parameter is 0
    fwhm_df_p0 = fwhm_df[fwhm_df["parameter"] == 0].copy()
    tau_df = pd.concat([tau_df, fwhm_df_p0], ignore_index=True)

    seaborn.set_style("darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    seaborn.boxplot(fwhm_df, x="parameter", y="ICC", ax=axes[0])
    axes[0].set_xlabel("FWHM (mm)")
    seaborn.boxplot(tau_df, x="parameter", y="ICC", ax=axes[1])
    axes[1].set_xlabel("tau")

    fig.savefig("/home/david/PycharmProjects/torchfMRI/sensory/icc_boxplot.png", dpi=300, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for df, name, ax in zip((tau_df, fwhm_df), ("tau", "FWHM (mm)"), axes):
        icc_50 = list()
        for parameter in sorted(df["parameter"].unique()):
            subset = df[df["parameter"] == parameter]
            print("Name:", name, "Parameter:", parameter, "Mean:", subset["ICC"].mean())
            icc_50.append(np.sum(subset["ICC"] > 0.5)/ len(subset))
        icc_50 = pd.DataFrame(np.array(icc_50), columns=["ICC"])
        icc_50[name] = df["parameter"].unique()
        seaborn.barplot(icc_50, x=name, y="ICC", ax=ax)
    fig.savefig("/home/david/PycharmProjects/torchfMRI/sensory/icc50_count_barplot.png", dpi=300)


if __name__ == "__main__":
    main()