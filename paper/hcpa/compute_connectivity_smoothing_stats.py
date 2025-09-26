import json
import numpy as np
import os
import glob
import templateflow.api as tf
import nibabel as nib
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def find_subjects(cleaned_fmri_dir, csmooth_dir, remaining_volumes_threshold=0.8, n_subjects=np.inf):
    included_subjects = list()
    excluded_subjects = list()
    all_fmri_files = dict()
    for subject_dir in sorted(glob.glob(os.path.join(cleaned_fmri_dir, "sub-*"))):
        subject_id = os.path.basename(subject_dir)
        fmri_rest_files = sorted(glob.glob(os.path.join(
            subject_dir, "func",
            f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")))
        csmooth_rest_files = sorted(glob.glob(os.path.join(
            csmooth_dir, subject_id, "func",
            f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_desc-csmooth*_bold.nii.gz")))
        if len(fmri_rest_files) < 4:
            print(f"Excluding {subject_id} because only found {len(fmri_rest_files)} rest runs")
            excluded_subjects.append(subject_id)
        elif len(fmri_rest_files)*4 != len(csmooth_rest_files):
            print(f"Excluding {subject_id} because number of cleaned rest runs "
                  f"({len(fmri_rest_files)}) does not match number of csmooth rest runs ({len(csmooth_rest_files)})")
            excluded_subjects.append(subject_id)
        else:
            nvols = 0
            total_vols = 0
            for fmri_rest_file in fmri_rest_files:
                sidecar_file = fmri_rest_file.replace(".nii.gz", ".json")
                metadata = read_json(sidecar_file)
                nvols += metadata["NumberOfVolumesAfterScrubbing"]
                total_vols += metadata["NumberOfVolumesBeforeScrubbing"]
            proportion_remaining = nvols / total_vols
            if proportion_remaining < remaining_volumes_threshold:
                print(f"Excluding {subject_id} because only {proportion_remaining:.2f} of volumes remain after scrubbing")
                excluded_subjects.append(subject_id)
            else:
                print(f"Including {subject_id} with {proportion_remaining:.2f} of volumes remaining after scrubbing")
                included_subjects.append(subject_id)
                all_fmri_files[(subject_id, "no_smoothing", 0)] = fmri_rest_files
                for csmooth_rest_file in csmooth_rest_files:
                    fwhm = int(re.search(r"fwhm-(\d+)", csmooth_rest_file).group(1))
                    key = (subject_id, "csmooth", fwhm)
                    if key not in all_fmri_files:
                        all_fmri_files[key] = list()
                    all_fmri_files[key].append(csmooth_rest_file)

        if len(included_subjects) >= n_subjects:
            break
    return included_subjects, all_fmri_files, excluded_subjects


def fetch_atlas_image(template="MNI152NLin2009cAsym", atlas="Schaefer2018", description="1000Parcels7Networks", resolution=1):
    # Fetch the atlas image from templateflow
    atlas_file = tf.get(atlas=atlas, template=template, resolution=resolution, desc=description)
    atlas_img = nib.load(atlas_file)
    atlas_labels_df = fetch_atlas_tsv_file(template=template, atlas=atlas, description=description)
    return atlas_img, atlas_labels_df


def fetch_atlas_tsv_file(template="MNI152NLin2009cAsym", atlas="Schaefer2018", description="1000Parcels7Networks"):
    atlas_labels_file = tf.get(atlas=atlas, template=template, desc=description, extension=".tsv")
    atlas_labels_df = pd.read_csv(atlas_labels_file, sep="\t", index_col=0)
    return atlas_labels_df


def compute_roi_to_roi_distances(atlas_img):
    atlas_data = atlas_img.get_fdata()
    roi_indices = np.unique(atlas_data)
    roi_indices = roi_indices[roi_indices != 0]  # Exclude background
    roi_centroids = list()
    for roi in tqdm(roi_indices, desc="Computing ROI centroids", unit="ROI"):
        coords = np.column_stack(np.where(atlas_data == roi))
        coords_mm = nib.affines.apply_affine(atlas_img.affine, coords)
        centroid = coords_mm.mean(axis=0)
        roi_centroids.append(centroid)
    roi_centroids = np.array(roi_centroids)
    distance_matrix = np.sqrt(np.sum((roi_centroids[:, np.newaxis, :] - roi_centroids[np.newaxis, :, :]) ** 2,
                                     axis=-1))
    return distance_matrix, roi_indices


def compute_fmri_connectivity(fmri_files, atlas_img):
    # Initialize the masker with the atlas image
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize="zscore", detrend=False, background_label=0,
                               resampling_target="data", strategy="mean")
    time_series = list()
    for fmri_file in tqdm(fmri_files, desc="Extracting time series from FMRI files", unit="file"):
        fmri_img = nib.load(fmri_file)
        ts = masker.fit_transform(fmri_img)
        time_series.append(ts)
    time_series = np.vstack(time_series)

    # Compute the connectivity matrix using Pearson correlation
    connectivity_measure = ConnectivityMeasure(kind='correlation')
    connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]
    # fisher z-transform
    connectivity_matrix = np.arctanh(connectivity_matrix)
    return connectivity_matrix


def compute_connectivity_for_key(args):
    key, fmri_files, atlas_img = args
    connectivity_matrix = compute_fmri_connectivity(fmri_files, atlas_img)
    return key, connectivity_matrix


def compute_fmri_connectivity_matrices(all_fmri_files, atlas_img, n_procs=1):
    from multiprocessing import Pool

    # Prepare arguments for multiprocessing (include atlas_img)
    args_list = [(key, all_fmri_files[key], atlas_img) for key in all_fmri_files]

    connectivity_matrices = dict()
    with Pool(processes=n_procs) as pool:
        results = list(tqdm(pool.imap(compute_connectivity_for_key, args_list),
                           total=len(args_list),
                           desc="Computing FMRI connectivity matrices",
                           unit="subject/method"))

    for key, connectivity_matrix in results:
        connectivity_matrices[key] = connectivity_matrix

    return connectivity_matrices



if __name__ == "__main__":
    n_procs = 2
    cleaned_fmri_dir = "/media/conda2/public/HCPA/myderivatives/cleaned"
    csmooth_dir = "/media/conda2/public/HCPA/myderivatives/csmooth"
    included_subjects, all_fmri_files, excluded_subjects = find_subjects(cleaned_fmri_dir,
                                                                         csmooth_dir,
                                                                         remaining_volumes_threshold=0.8,
                                                                         n_subjects=1)

    print("Included subjects:", included_subjects)
    print("Excluded subjects:", excluded_subjects)
    print("Number of included subjects:", len(included_subjects))
    print("Number of excluded subjects:", len(excluded_subjects))
    atlas_image, atlas_labels_df = fetch_atlas_image()

    distance_matrix, roi_indices = compute_roi_to_roi_distances(atlas_image)

    connectivity_matrices = compute_fmri_connectivity_matrices(all_fmri_files, atlas_image, n_procs=n_procs)

    print("Distance matrix shape:", distance_matrix.shape)

    # extract upper triangle of the distance matrix
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distance_values = distance_matrix[triu_indices]
    connectivity_values = dict()
    for key in connectivity_matrices:
        subject_id, method, fwhm = key
        connectivity_matrix = connectivity_matrices[key]
        new_key = (method, fwhm)
        if new_key not in connectivity_values:
            connectivity_values[new_key] = list()
        connectivity_values[new_key].append(connectivity_matrix[triu_indices])

    mean_connectivity  = dict()
    for key in connectivity_values:
        mean_connectivity[key] = np.mean(connectivity_values[key], axis=0)
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(10, 6))
    # plt.scatter(distance_values, mean_connectivity, alpha=0.01)
    # plt.xlabel("ROI-to-ROI Distance (mm)")
    # plt.ylabel("Mean Fisher Z-transformed Connectivity")
    # plt.title("Mean ROI-to-ROI Connectivity vs. Distance")
    # plt.axhline(0, color='k', linestyle='--')
    # plt.tight_layout()
    # plt.savefig("connectivity_vs_distance.png", dpi=300)
    print("Saving connectivity and distance data...")
    # save the data to a csv file
    data = list()
    for key in mean_connectivity:
        method, fwhm = key
        _mean_connectivity = mean_connectivity[key]
        for dist, i, j, conn in zip(distance_values, triu_indices[0], triu_indices[1], _mean_connectivity):
            roi1_name = atlas_labels_df["name"][roi_indices[i]]
            roi1_hemi = roi1_name.split("_")[1]
            assert roi1_hemi in ("LH", "RH"), f"Unexpected hemisphere in ROI name: {roi1_name}"
            roi2_name = atlas_labels_df["name"][roi_indices[j]]
            roi2_hemi = roi2_name.split("_")[1]
            assert roi2_hemi in ("LH", "RH"), f"Unexpected hemisphere in ROI name: {roi2_name}"
            intrahemispheric = roi1_hemi == roi2_hemi
            data.append((roi_indices[i], atlas_labels_df["name"][roi_indices[i]],
                         roi_indices[j], atlas_labels_df["name"][roi_indices[j]],
                         intrahemispheric, method, fwhm,
                         dist, conn))
    df = pd.DataFrame(data, columns=["ROI1_index", "ROI1_name",
                                     "ROI2_index", "ROI2_name",
                                     "Intrahemispheric",
                                     "Method", "FWHM",
                                     "Distance_mm", "FisherZ_Connectivity"])
    df.to_csv("mean_connectivity_distance_data.csv", index=False)
    # data = list()
    # for _connectivity_matrix, subject in zip(connectivity_matrices, included_subjects):
    #     for dist, i, j in zip(distance_values, triu_indices[0], triu_indices[1]):
    #         conn = _connectivity_matrix[i, j]
    #         data.append((subject, roi_indices[i], roi_indices[j], dist, conn))
    # df = pd.DataFrame(data, columns=["Subject", "ROI1", "ROI2", "Distance_mm", "FisherZ_Connectivity"])
    # df.to_csv("connectivity_distance_data.csv", index=False)
    print("Done.")

