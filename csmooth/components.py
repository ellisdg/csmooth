import time
from collections import Counter
import os

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components
import nilearn.image
import nibabel as nib


from csmooth.matrix import create_adjacency_matrix
from csmooth.graph import select_nodes
from csmooth.utils import logger


def identify_connected_components(edge_src, edge_dst, edge_distances):
    """
    Identify connected components in a graph defined by edges and distances.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :return: labels: numpy array of labels for each node in the graph,
             sorted_labels: numpy array of sorted labels by size of components.
             unique_nodes: numpy array of unique nodes in the graph.
    """

    adjacency_matrix, unique_nodes = create_adjacency_matrix(edge_src, edge_dst, weights=edge_distances)
    n_components, labels = connected_components(csgraph=adjacency_matrix.tocsr(), directed=False, return_labels=True)
    sorted_labels = np.argsort([(labels == l).sum() for l in np.unique(labels)])[::-1]

    return labels, sorted_labels, unique_nodes


def find_common_edges(edge_src, edge_dst, edge_distances, nodes, resampled_image_data, node_labels,
                      node_counts, component_i,
                      sampling_fraction, min_samples,
                      node_counts_image_data=None):
    # create an undirected networkx graph
    G = nx.Graph()
    G.add_weighted_edges_from(zip(edge_src, edge_dst, edge_distances))
    if G.is_directed():
        logger.warning("Graph is directed when it should be undirected. Converting to undirected graph.")
        G = G.to_undirected()

    # randomly sample a fraction of the nodes and use those to estimate edges that when removed
    # would disconnect the graph into multiple components.
    # sample nodes from the largest dseg label group and take the same number of nodes from the second largest group
    _sampling_fraction = max(sampling_fraction, min_samples / len(G.nodes))

    logger.debug(f"Sampling fraction for component: {_sampling_fraction:.4f}")

    # sample nodes from the graph
    max_dseg_label, second_max_dseg_label = node_labels[np.argsort(node_counts)[-2:]]
    logger.debug(f"Max dseg label: {max_dseg_label}, second max dseg label: {second_max_dseg_label}")

    max_dseg_nodes = nodes[resampled_image_data[nodes] == max_dseg_label]
    second_max_dseg_nodes = nodes[resampled_image_data[nodes] == second_max_dseg_label]

    if len(second_max_dseg_nodes) < min_samples:
        logger.warning(f"Not enough nodes in the second largest dseg label {second_max_dseg_label} "
                        f"to sample {min_samples}. Exiting component {component_i} check.")
        return None

    sampled_max_nodes = np.random.choice(max_dseg_nodes, size=int(len(max_dseg_nodes) * _sampling_fraction),
                                         replace=False)
    sampled_second_max_nodes = np.random.choice(second_max_dseg_nodes, size=len(sampled_max_nodes), replace=False)
    edge_counter = Counter()
    for u, v in zip(sampled_max_nodes, sampled_second_max_nodes):
        # convert the sampled nodes to their indices in the graph
        idx_u = np.where(nodes == u)[0][0]
        idx_v = np.where(nodes == v)[0][0]
        shortest_path = nx.shortest_path(G, idx_u, idx_v)
        for k in range(len(shortest_path) - 1):
            if shortest_path[k] < shortest_path[k + 1]:
                edge = (shortest_path[k], shortest_path[k + 1])
            else:
                edge = (shortest_path[k + 1], shortest_path[k])
            edge_counter[edge] += 1
    logger.debug(f"Identified {len(edge_counter)} edges in the sampled paths for component {component_i}")

    if node_counts_image_data is not None:
        logger.info(f"Updating node counts image data for component {component_i}")
        for edge, count in edge_counter.items():
            node_counts_image_data[nodes[edge[0]]] += count
            node_counts_image_data[nodes[edge[1]]] += count

    return edge_counter


def find_outliers(edge_counter, outlier_threshold=5):
    """
    Find outliers in the edge counter that have higher counts than expected.
    :param edge_counter: Counter object with edges as keys and their counts as values.
    :param outlier_threshold: number of standard deviations above the mean to consider an edge as an outlier.
    (default is 5).
    :return: list of outlier edges.
    """
    counts = np.array(list(edge_counter.values()))
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    threshold = mean_count + outlier_threshold * std_count
    logger.debug(f"Mean count: {mean_count:.2f}, Std count: {std_count:.2f}, Threshold for outliers: {threshold:.2f}")
    outliers = [edge for edge, count in edge_counter.items() if count > threshold]
    logger.debug(f"Found {len(outliers)} outlier edges with counts above the threshold of {threshold:.2f}")
    return outliers


def check_components(edge_src, edge_dst, edge_distances, labels, unique_nodes, sorted_labels,
                     dseg_file, reference_image,
                     n_components=5,
                     sampling_fraction=0.0001, min_samples=100,
                     qc_threshold=0.60,
                     max_removal_attempts=500,
                     output_removed_edges_filename=None):
    """
    Check for bottlenecks in the graph defined by edges and distances.
    :param edge_src: numpy array of source nodes
    :param edge_dst: numpy array of destination nodes
    :param edge_distances: numpy array of distances between nodes
    :param labels: numpy array of labels for each node
    :param unique_nodes: numpy array of unique nodes in the graph
    :param sorted_labels: numpy array of sorted labels by size of components
    :param dseg_file: path to the dseg file for the subject
    :param reference_image: path to the reference for the components
    :param n_components: number of largest components to check (should always be 5)
    :param sampling_fraction: fraction of nodes from a component to sample for estimating bottlenecks.
    :param min_samples: minimum number of nodes from a component to sample for estimating bottlenecks.
    :param qc_threshold: threshold for quality control, default is 0.9, which would indicate that at least 90% of the nodes
    in the component should come from the same dseg label.
    :param max_removal_attempts: maximum number of attempts to remove edges to disconnect the component.
    :param output_removed_edges_filename: if provided, will save an image of the nodes of removed edges for the component.
    :return: edge_src_bottleneck, edge_dst_bottleneck, edge_distances_bottleneck:
             numpy arrays of edges and distances that are bottlenecks.
    """
    logger.info('Checking components')
    start_time = time.time()

    # resample dseg_file to the reference_file space
    image = nib.load(dseg_file)
    resampled_image = nilearn.image.resample_to_img(
        image, reference_image, interpolation='nearest',
                                                    force_resample=True, copy_header=True)
    resampled_image_data = np.asarray(resampled_image.dataobj).flatten()
    # swap label 3 with label 0
    # 3 refers to CSF and 0 refers to background
    # so we will treat CSF as background
    # resampled_image_data[resampled_image_data == 3] = 0

    for i, label in enumerate(sorted_labels[:n_components]):
        # select nodes that belong to the specified component
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src, edge_dst, edge_distances, labels, label, unique_nodes)

        # count the number of nodes belonging to each dseg label
        node_labels, node_counts = np.unique(resampled_image_data[_nodes], return_counts=True)
        # ignore counts from CSF as ventricles are included in the white matter component
        if 3 in node_labels:
            csf_index = np.where(node_labels == 3)[0][0]
            node_labels = np.delete(node_labels, csf_index)
            node_counts = np.delete(node_counts, csf_index)
        node_counts_normalized = node_counts / np.sum(node_counts)
        logger.debug(f"Component {i}: {node_labels}, counts: {node_counts}, normalized: {node_counts_normalized}")

        # check if the component is large enough
        if np.max(node_counts_normalized) > qc_threshold:
            logger.info(f"Component {i} is homogenous enough with "
                         f"{np.max(node_counts_normalized) * 100:.1f}% "
                         f"of nodes belonging to the same dseg label")
        else:
            logger.warning(f"Component {i} is too heterogenous with only "
                            f"{np.max(node_counts_normalized) * 100:.1f}% "
                            f"of nodes belonging to the same dseg label")
            logger.info(f"Attempting to identify errant edges in component {i} ")

            component_labels, sorted_component_labels, component_nodes = identify_connected_components(
                    edge_src=_edge_src,
                                                edge_dst=_edge_dst,
                                                edge_distances=_edge_distances)
            n_components = len(sorted_component_labels)
            logger.debug(f"Number of components before removing edges: {n_components}")
            edges_removed = list()
            while n_components < 2:
                edge_counter = find_common_edges(
                    edge_src=_edge_src,
                    edge_dst=_edge_dst,
                    edge_distances=_edge_distances,
                    nodes=_nodes,
                    resampled_image_data=resampled_image_data,
                    node_labels=node_labels,
                    node_counts=node_counts,
                    component_i=i,
                    sampling_fraction=sampling_fraction,
                    min_samples=min_samples
                )
                if edge_counter is None or len(edge_counter) == 0:
                    logger.warning(f"No edges found in component {i}. Skipping edge removal.")
                    continue

                outliers = find_outliers(edge_counter)

                for edge_to_remove in outliers:
                    logger.debug(f"Removing edge {edge_to_remove} with count")

                    _edge_mask = ~((np.isin(_edge_src, edge_to_remove) & np.isin(_edge_dst, edge_to_remove)) |
                            (np.isin(_edge_dst, edge_to_remove) & np.isin(_edge_src, edge_to_remove)))
                    logger.debug(f"Component edges before removing edge {edge_to_remove}: {len(_edge_src)} edges")
                    _edge_src = _edge_src[_edge_mask]
                    _edge_dst = _edge_dst[_edge_mask]
                    _edge_distances = _edge_distances[_edge_mask]
                    logger.debug(f"Component edges after removing edge {edge_to_remove}: {len(_edge_src)} edges remaining")


                    # remove edges from the original edge lists as well
                    # convert edge_to_remove to indices in the original edge lists
                    edge_to_remove = (_nodes[edge_to_remove[0]], _nodes[edge_to_remove[1]])
                    edge_mask = ~((np.isin(edge_src, edge_to_remove) & np.isin(edge_dst, edge_to_remove)) |
                                  (np.isin(edge_dst, edge_to_remove) & np.isin(edge_src, edge_to_remove)))
                    logger.debug(f"Total number of edges before removing edge {edge_to_remove}: {len(edge_src)} edges")
                    edge_src = edge_src[edge_mask]
                    edge_dst = edge_dst[edge_mask]
                    removed_edge_distance = edge_distances[~edge_mask]
                    if len(removed_edge_distance) > 1:
                        if np.all(np.isclose(removed_edge_distance, removed_edge_distance[0])):
                            removed_edge_distance = removed_edge_distance[0]
                        else:
                            raise ValueError(f"Removed edge distances are not equal: {removed_edge_distance}")
                    else:
                        removed_edge_distance = np.squeeze(removed_edge_distance)
                    edge_distances = edge_distances[edge_mask]
                    logger.debug(f"Total number of edges after removing edge {edge_to_remove}: {len(edge_src)} edges remaining")

                    edges_removed.append((edge_to_remove, removed_edge_distance))

                logger.debug(f"Checking connected components after removing edges")
                component_labels, sorted_component_labels, component_nodes = identify_connected_components(
                    edge_src=_edge_src,
                    edge_dst=_edge_dst,
                    edge_distances=_edge_distances)
                n_components = len(sorted_component_labels)
                logger.debug(f"Number of components after removing edges: {n_components}")

                if len(edges_removed) >= max_removal_attempts:
                    logger.warning(f"Maximum number of removal attempts ({max_removal_attempts}) reached. "
                                    f"Exiting component {i} check.")
                    break
                else:
                    logger.info(f"Removed {len(edges_removed)} edges so far...")

            logger.info(f"Removed {len(edges_removed)} edges to disconnect the component {i} into multiple components.")
            if n_components < 2:
                logger.warning(f"Component {i} could not be disconnected into multiple components.")
            else:
                # add back edges thar were removed but belong to the same final component
                logger.info(f"Component {i} was successfully disconnected into {n_components} components.")
                logger.info(f"Adding back edges that begin and terminate in the same final component {i}")
                final_edges_removed = list()
                for edge, removed_edge_distance in edges_removed:
                    node1 = np.squeeze(np.where(component_nodes == edge[0]))
                    node2 = np.squeeze(np.where(component_nodes == edge[1]))
                    label1 = component_labels[node1]
                    label2 = component_labels[node2]
                    if label1.size == 1 and label2.size == 1 and np.all(label1 == label2):
                        logger.debug(f"Adding back edge {edge} with distance {removed_edge_distance:.2f}")
                        edge_src = np.append(edge_src, edge[0])
                        edge_dst = np.append(edge_dst, edge[1])
                        edge_distances = np.append(edge_distances, removed_edge_distance)
                    else:
                        final_edges_removed.append(edge)
                logger.info(f"Total edges removed after adding back in edges belonging to the same component: {final_edges_removed}")

                if output_removed_edges_filename is not None:
                    component_output_removed_edges_filename = output_removed_edges_filename.replace('.nii.gz', f'_component_{i}.nii.gz')
                    logger.info(f"Saving image of nodes of removed edges for component {i} to "
                                 f"{component_output_removed_edges_filename}")
                    # create an image of the node counts
                    node_counts_image_data = np.zeros(resampled_image_data.shape, dtype=np.int32)
                    for edge in final_edges_removed:
                        node_counts_image_data[edge[0]] += 1
                        node_counts_image_data[edge[1]] += 1
                    node_counts_image = nib.Nifti1Image(node_counts_image_data.reshape(resampled_image.shape),
                                                        resampled_image.affine)
                    os.makedirs(os.path.dirname(component_output_removed_edges_filename), exist_ok=True)
                    node_counts_image.to_filename(component_output_removed_edges_filename)



    end_time = time.time()
    logger.info(f"Time taken to check components: {(end_time - start_time) / 60:.2f} minutes")
    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst, edge_distances)
    return edge_src, edge_dst, edge_distances, labels, sorted_labels, unique_nodes
