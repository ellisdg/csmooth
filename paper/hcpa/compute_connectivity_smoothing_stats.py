import numpy as np
import os
import glob
import templateflow.api as tf
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import networkx as nx
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


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


def compute_adjacency_matrix(connectivity_matrix, graph_density=0.2):
    # Make the connectivity matrix symmetric
    sym_matrix = (connectivity_matrix + connectivity_matrix.T) / 2
    # Zero out the diagonal
    np.fill_diagonal(sym_matrix, 0)
    # Determine the threshold for the top X% of connections
    threshold = np.percentile(sym_matrix, 100 * (1 - graph_density))
    # Create the adjacency matrix
    adjacency_matrix = (sym_matrix >= threshold).astype(int)
    return adjacency_matrix


def create_network_communities(atlas_labels_df, roi_indices):
    """Create community assignments based on network labels from the atlas."""
    # Extract network information from the label names
    # Schaefer atlas labels follow format: "7Networks_LH_Vis_1" or "7Networks_RH_Default_1"
    communities = []
    network_to_community = {}
    community_id = 0
    
    for roi_idx in roi_indices:
        # roi_indices are 1-based, atlas_labels_df index is also 1-based
        label_name = atlas_labels_df.loc[int(roi_idx), 'name']
        
        # Extract network name (e.g., "Vis", "Default", etc.)
        parts = label_name.split('_')
        if len(parts) >= 3:
            network = parts[2]  # Network name is the 3rd part
        else:
            network = 'Unknown'
        
        # Assign community ID for this network
        if network not in network_to_community:
            network_to_community[network] = community_id
            community_id += 1
        
        communities.append(network_to_community[network])
    
    return communities


def compute_graph_metrics(adjacency_matrix, communities=None):
    G = nx.from_numpy_array(adjacency_matrix)
    local_eff = nx.local_efficiency(G)
    global_eff = nx.global_efficiency(G)
    clustering_coeff = np.mean(list(nx.clustering(G).values()))
    if communities is None:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))
        modularity = nx.algorithms.community.quality.modularity(G, communities)
    else:
        # Convert community assignments to community sets for NetworkX
        community_sets = []
        unique_communities = set(communities)
        for comm_id in unique_communities:
            community_sets.append(set([i for i, c in enumerate(communities) if c == comm_id]))
        modularity = nx.algorithms.community.quality.modularity(G, community_sets)
    return local_eff, global_eff, clustering_coeff, modularity


def append_data(data, subject, method, fwhm, distance_values, connectivity_values):
    for dist, conn in zip(distance_values, connectivity_values):
        data.append((subject, method, fwhm, dist, conn))
    return data


def process_subject_connectivity_distance(subject_dir, distance_values):
    """Process a single subject and return connectivity-distance data."""
    subject_name = os.path.basename(subject_dir)
    subject_data = []

    # Process no smoothing
    ns_filename = os.path.join(subject_dir, "func",
                               f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_no_smoothing_fwhm-0_connectivity.npy")
    ns_matrix = np.load(ns_filename)
    triu_indices = np.triu_indices_from(ns_matrix, k=1)
    ns_values = ns_matrix.astype(np.float32)[triu_indices]

    # append no smoothing values
    subject_data = append_data(subject_data, subject_name, "NoSmoothing", 0, distance_values, ns_values)
    del ns_values, ns_matrix

    # Process different FWHM values
    for fwhm in (3, 6, 9, 12):
        cs_filename = os.path.join(subject_dir, "func",
                                   f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_csmooth_fwhm-{fwhm}_connectivity.npy")
        cs_matrix = np.load(cs_filename)
        cs_values = cs_matrix[triu_indices].astype(np.float32)
        subject_data = append_data(subject_data, subject_name, "Constrained", fwhm, distance_values, cs_values)
        del cs_values, cs_matrix

        gs_filename = os.path.join(subject_dir, "func",
                                   f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_gaussian_fwhm-{fwhm}_connectivity.npy")
        gs_matrix = np.load(gs_filename)
        gs_values = gs_matrix[triu_indices].astype(np.float32)
        subject_data = append_data(subject_data, subject_name, "Gaussian", fwhm, distance_values, gs_values)
        del gs_values, gs_matrix

    return subject_data


def process_subject_metrics(subject_dir, graph_density, network_communities):
    """Process a single subject and return graph metrics."""
    subject_name = os.path.basename(subject_dir)
    subject_metrics = []

    # Process no smoothing
    ns_filename = os.path.join(subject_dir, "func",
                               f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_no_smoothing_fwhm-0_connectivity.npy")
    ns_matrix = np.load(ns_filename)
    ns_graph_metrics = compute_graph_metrics(compute_adjacency_matrix(ns_matrix, graph_density=graph_density), communities=network_communities)
    subject_metrics.append((subject_name, "NoSmoothing", 0, graph_density, *ns_graph_metrics))
    del ns_matrix

    # Process different FWHM values
    for fwhm in (3, 6, 9, 12):
        cs_filename = os.path.join(subject_dir, "func",
                                   f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_csmooth_fwhm-{fwhm}_connectivity.npy")
        cs_matrix = np.load(cs_filename)
        subject_metrics.append((subject_name, "Constrained", fwhm, graph_density,
                        *compute_graph_metrics(compute_adjacency_matrix(cs_matrix, graph_density=graph_density), communities=network_communities)))
        del cs_matrix

        gs_filename = os.path.join(subject_dir, "func",
                                   f"{subject_name}_space-MNI152NLin2009cAsym_desc-Schaefer20181000Parcels7Networks_gaussian_fwhm-{fwhm}_connectivity.npy")
        gs_matrix = np.load(gs_filename)
        subject_metrics.append((subject_name, "Gaussian", fwhm, graph_density,
                        *compute_graph_metrics(compute_adjacency_matrix(gs_matrix, graph_density=graph_density), communities=network_communities)))
        del gs_matrix

    return subject_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compute connectivity smoothing statistics.")
    parser.add_argument("--connectivity_dir", type=str, required=True,
                        help="Directory containing subject connectivity data.")
    parser.add_argument("--output_metrics", type=str, default="connectivity_graph_metrics.csv",
                        help="Output CSV file for graph metrics.")
    parser.add_argument("--output_data", type=str, default="connectivity_distance_data.parquet",
                        help="Output Parquet file for connectivity-distance data.")
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="Number of subjects to process.")
    parser.add_argument("--graph_density", type=float, default=0.2,
                        help="Graph density for adjacency matrix computation.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs if set.")
    return parser.parse_args()


def main(connectivity_dir, output_metrics, output_data, n_subjects=None, graph_density=0.2, force=False):

    print("Fetching atlas image and labels...")
    atlas_image, atlas_labels_df = fetch_atlas_image()
    distance_matrix, roi_indices = compute_roi_to_roi_distances(atlas_image)
    print("Distance matrix shape:", distance_matrix.shape)
    
    # Create network communities based on atlas labels
    network_communities = create_network_communities(atlas_labels_df, roi_indices)
    print(f"Created {len(set(network_communities))} network communities")
    
    # extract upper triangle of the distance matrix
    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    distance_values = distance_matrix[triu_indices].astype(np.float32)

    subject_dirs = sorted(glob.glob(os.path.join(connectivity_dir, "sub-*")))
    if n_subjects is not None:
        subject_dirs = subject_dirs[:n_subjects]

    if not os.path.exists(output_data) or force:
        if force and os.path.exists(output_data):
            print(f"Force enabled: overwriting existing {output_data}")
        print(f"Processing {len(subject_dirs)} subjects for connectivity-distance data using {cpu_count()} CPU cores...")
        process_func = partial(process_subject_connectivity_distance, distance_values=distance_values)
        data = []
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_func, subject_dirs),
                               total=len(subject_dirs),
                               desc="Processing subjects",
                               unit="subject"))
        for subject_data in results:
            data.extend(subject_data)
        conn_distance_df = pd.DataFrame(data, columns=["Subject", "Method", "FWHM", "Distance_mm", "FisherZ_Connectivity"])
        print("Total data points:", len(conn_distance_df))
        conn_distance_df.to_parquet(output_data,
                                    index=False)
        print(f"Saved connectivity-distance data to {output_data}")
        del conn_distance_df, data
    else:
        print(f"Output data file {output_data} already exists. Skipping connectivity-distance data processing.")

    if not os.path.exists(output_metrics) or force:
        if force and os.path.exists(output_metrics):
            print(f"Force enabled: overwriting existing {output_metrics}")
        print(f"Processing {len(subject_dirs)} subjects for graph metrics using {cpu_count()} CPU cores...")
        process_func = partial(process_subject_metrics, graph_density=graph_density, network_communities=network_communities)
        metrics = []
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_func, subject_dirs),
                               total=len(subject_dirs),
                               desc="Processing subjects",
                               unit="subject"))
        for subject_metrics in results:
            metrics.extend(subject_metrics)
        metrics_df = pd.DataFrame(metrics,
                                  columns=["Subject", "Method", "FWHM", "GraphDensity",
                                           "LocalEfficiency", "GlobalEfficiency", "ClusteringCoefficient", "Modularity"])
        print("Graph metrics data points:", len(metrics_df))
        metrics_df.to_csv(output_metrics,
                          index=False)
        print(f"Saved graph metrics to {output_metrics}")
        del metrics_df, metrics
    else:
        print(f"Output metrics file {output_metrics} already exists. Skipping graph metrics processing.")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(connectivity_dir=args.connectivity_dir,
         output_metrics=args.output_metrics,
         output_data=args.output_data,
         n_subjects=args.n_subjects,
         graph_density=args.graph_density,
         force=args.force)
