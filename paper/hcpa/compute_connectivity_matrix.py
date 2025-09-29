import argparse
import os
import templateflow.api as tf
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input_files", nargs="+")
    argument_parser.add_argument("--output_file", type=str)
    argument_parser.add_argument("--template", type=str, default="MNI152NLin2009cAsym")
    argument_parser.add_argument("--resolution", type=int, default=1)
    argument_parser.add_argument("--atlas", type=str, default="Schaefer2018")
    argument_parser.add_argument("--description", type=str, default="1000Parcels7Networks")
    return argument_parser.parse_args()


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


def main():
    args = parse_args()
    atlas_img, atlas_labels_df = fetch_atlas_image(template=args.template,
                                                   atlas=args.atlas,
                                                   description=args.description,
                                                   resolution=args.resolution)
    connectivity_matrix = compute_fmri_connectivity(args.input_files, atlas_img)
    np.save(args.output_file, connectivity_matrix)
    print(f"Saved connectivity matrix to {args.output_file}")


if __name__ == "__main__":
    main()
