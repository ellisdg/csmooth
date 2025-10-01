import numpy as np
import os
import glob
import templateflow.api as tf
import nibabel as nib
from tqdm import tqdm
import pandas as pd


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


def append_data(data, subject, method, fwhm, distance_values, connectivity_values):
    for dist, conn in zip(distance_values, connectivity_values):
        data.append((subject, method, fwhm, dist, conn))
    return data

if __name__ == "__main__":
    atlas_image, atlas_labels_df = fetch_atlas_image()

    distance_matrix, roi_indices = compute_roi_to_roi_distances(atlas_image)

    print("Distance matrix shape:", distance_matrix.shape)

    # extract upper triangle of the distance matrix
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distance_values = distance_matrix[triu_indices].astype(np.float32)

    connectivity_dir = "/media/conda2/public/HCPA/myderivatives/connectivity"
    subject_dirs = sorted(glob.glob(os.path.join(connectivity_dir, "sub-*")))[:50]
    data = list()
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects", unit="subject"):
        subject_name = os.path.basename(subject_dir)
        ns_filename = os.path.join(subject_dir, "func",
                                   f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_no_smoothing_fwhm-0_connectivity.npy")
        ns_values = np.load(ns_filename)[triu_indices].astype(np.float32)
        # append no smoothing values twice, once for each method
        data = append_data(data, subject_name, "gaussian", 0, distance_values, ns_values)
        data = append_data(data, subject_name, "constrained", 0, distance_values, ns_values)

        del ns_values
        for fwhm in (3, 6, 9, 12):
            cs_filename = os.path.join(subject_dir, "func",
                                       f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_csmooth_fwhm-{fwhm}_connectivity.npy")
            cs_values = np.load(cs_filename)[triu_indices].astype(np.float32)
            data = append_data(data, subject_name, "constrained", fwhm, distance_values, cs_values)
            del cs_values
            gs_filename = os.path.join(subject_dir, "func",
                                       f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_gaussian_fwhm-{fwhm}_connectivity.npy")
            gs_values = np.load(gs_filename)[triu_indices].astype(np.float32)
            data = append_data(data, subject_name, "gaussian", fwhm, distance_values, gs_values)
            del gs_values
    df = pd.DataFrame(data, columns=["Subject", "Method", "FWHM", "Distance_mm", "FisherZ_Connectivity"])
    print("Total data points:", len(df))
    df.to_parquet("connectivity_distance_data.parquet", index=False)
    print("Done.")

