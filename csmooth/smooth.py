import os
import time
import warnings

import networkx as nx
import nibabel as nib
import nilearn.image
import numpy as np
import scipy
from tqdm import tqdm

from csmooth.affine import adjust_affine_spacing, resample_data_to_affine
from csmooth.components import identify_connected_components
from csmooth.gaussian import gaussian_smoothing, compute_gaussian_kernels, apply_gaussian_smoothing
from csmooth.graph import create_graph, select_nodes
from csmooth.heat import heat_kernel_smoothing
from csmooth.optimization import find_optimal_tau
from csmooth.utils import logger
from csmooth.resampling import resample_image


def _smooth_component(edge_src, edge_dst, edge_distances, signal_data, tau=None,
                      fwhm=None, n_jobs=4):
    """
    Smooth a single component of the graph signal.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :param signal_data:
    :param tau:
    :param fwhm:
    :param n_jobs: number of parallel processes to use for smoothing. Only used for Gaussian smoothing. (default=4)
    :return:
    """
    if tau is None and fwhm is None:
        raise ValueError("Must provide either tau or fwhm")
    if tau is not None and fwhm is not None:
        raise ValueError("Must provide either tau or fwhm, not both")
    if fwhm is not None:
        smoothed_data = gaussian_smoothing(data=signal_data,
                                           edge_src=edge_src,
                                           edge_dst=edge_dst,
                                           edge_distances=edge_distances,
                                           fwhm=fwhm)
    elif tau is not None:
        smoothed_data = heat_kernel_smoothing(edge_src=edge_src,
                                              edge_dst=edge_dst,
                                              edge_distances=edge_distances,
                                              signal_data=signal_data,
                                              tau=tau)
    else:
        raise ValueError("Must provide either tau or fwhm")
    return smoothed_data


def smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                     tau=None, fwhm=None, n_jobs=4, low_memory=False):
    start_time = time.time()
    _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(edge_src=edge_src,
                                                                 edge_dst=edge_dst,
                                                                 edge_distances=edge_distances,
                                                                 labels=labels,
                                                                 label=label,
                                                                 unique_nodes=unique_nodes)
    if low_memory:
        # if low memory mode is enabled, smooth each timepoint separately
        for t in range(signal_data.shape[1]):
            signal_data[_nodes, t] = _smooth_component(edge_src=_edge_src,
                                                     edge_dst=_edge_dst,
                                                     edge_distances=_edge_distances,
                                                     signal_data=signal_data[_nodes, t],
                                                     tau=tau,
                                                     fwhm=fwhm,
                                                     n_jobs=n_jobs)
    else:
        signal_data[_nodes, :] = _smooth_component(edge_src=_edge_src,
                                           edge_dst=_edge_dst,
                                           edge_distances=_edge_distances,
                                           signal_data=signal_data[_nodes, :],
                                           tau=tau,
                                           fwhm=fwhm,
                                           n_jobs=n_jobs)
    elapsed_time = time.time() - start_time
    logger.debug(f"Smoothing component {label} with {len(_nodes)} nodes took {elapsed_time:.2f} seconds.")


def process_mask(mask_file, reference_image, mask_dilation):
    if mask_file is not None:
        mask_image = nib.load(mask_file)
    else:
        mask_image = nib.Nifti1Image(np.ones(reference_image.shape[:3], dtype=np.uint8), reference_image.affine)

    mask_image = nilearn.image.resample_to_img(mask_image, reference_image,
                                               interpolation="nearest",
                                               force_resample=True,
                                               copy_header=True)
    mask_array = mask_image.get_fdata() > 0.5

    if mask_dilation is not None and mask_file is not None:
        mask_array = scipy.ndimage.binary_dilation(mask_array, iterations=mask_dilation)

    return mask_array


def save_labelmap(output_labelmap, shape, affine, labels, sorted_labels, unique_nodes):
    os.makedirs(os.path.dirname(output_labelmap), exist_ok=True)
    labelmap = np.zeros(shape, dtype=np.uint32).flatten()

    for i, label in enumerate(sorted_labels):
        labelmap[unique_nodes[labels == label]] = i + 1

    labelmap_image = nib.Nifti1Image(labelmap.reshape(shape), affine)
    labelmap_image.to_filename(output_labelmap)


def smooth_image(in_file, out_file, surface_files, **kwargs):
    """
    Smooth an image using graph signal smoothing. This function simply calls smooth_images with a single image.
    :param in_file: Path to a Nifti file to be smoothed.
    :param out_file: Output filename to save the smoothed image.
    :param surface_files: List of surface filenames to use for edge pruning.
    :param kwargs: Additional parameters to pass to smooth_images.
    :return:
    """
    smooth_images([in_file], [out_file], surface_files, **kwargs)


def precompute_guassian_kernels(edge_src, edge_dst, edge_distances, labels, sorted_labels, unique_nodes,
                                out_kernel_basename, fwhm):
    """
    Precompute Gaussian kernels for each component and save to file.
    :param edge_src:
    :param edge_dst:
    :param edge_distances:
    :param labels:
    :param sorted_labels:
    :param unique_nodes:
    :param out_kernel_basename:
    :param fwhm:
    :return:
    """
    # for each component, compute the gaussian kernels and save to file
    kernel_filenames = []
    os.makedirs(os.path.dirname(out_kernel_basename), exist_ok=True)
    for label in tqdm(sorted_labels, desc="Computing smoothing kernels", unit="component"):
        out_kernel_filename = out_kernel_basename + f"_{label}.npz"
        if not os.path.exists(out_kernel_filename):
            _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(edge_src=edge_src,
                                                                         edge_dst=edge_dst,
                                                                         edge_distances=edge_distances,
                                                                         labels=labels,
                                                                         label=label,
                                                                         unique_nodes=unique_nodes)
            _edge_src, _edge_dst, _edge_weights = compute_gaussian_kernels(edge_src=_edge_src,
                                                                           edge_dst=_edge_dst,
                                                                           edge_distances=_edge_distances,
                                                                           fwhm=fwhm)
            np.savez_compressed(out_kernel_filename,
                                src=_edge_src,
                                dst=_edge_dst,
                                weights=_edge_weights,
                                nodes=_nodes)
        kernel_filenames.append(out_kernel_filename)
    return kernel_filenames


def load_image(in_file, reference_image=None):
    """
    Load an image and resample it to the reference image if provided.
    :param in_file: Path to the input image file.
    :param reference_image: Optional reference image to resample to.
    :return: Loaded image as a nibabel Nifti1Image object.
    """
    source_image = nib.load(in_file)
    if reference_image is not None:
        resampled_image = nilearn.image.resample_to_img(source_img=source_image,
                                                        target_img=reference_image,
                                                        interpolation="linear",
                                                        force_resample=True,
                                                        copy_header=True)
        return resampled_image, source_image
    else:
        return source_image, None


def load_reference_image(reference_file, resample_resolution=None):
    """
    :param reference_file:
    :param resample_resolution:
    :return:  If any resampling is done, the refererence image and the resampled reference image are the same.
    Otherwise, the reference image is the first image and the resampled reference is None.

    """
    first_image = nib.load(reference_file)
    if resample_resolution is not None:
        logger.debug(f"Resampling reference image from "
                      f"resolution {first_image.header.get_zooms()[:3]} to resolution {resample_resolution}")
        _affine = adjust_affine_spacing(first_image.affine, np.asarray(resample_resolution))
        reference_data = resample_data_to_affine(first_image.get_fdata()[..., 0],
                                                 target_affine=_affine,
                                                 original_affine=first_image.affine)
        _shape = reference_data.shape[:3]
        reference_image = nib.Nifti1Image(reference_data, _affine)
        resampled_reference = reference_image
    else:
        reference_image = first_image
        resampled_reference = None

    return reference_image, resampled_reference


def write_image(image, out_file, target_image=None, output_transform=None):
    """
    Write the smoothed image to a file, optionally resampling it to match a target image.
    :param image:
    :param out_file:
    :param target_image:
    :param output_transform:
    :return:
    """
    if output_transform is not None:
        if target_image is not None:
            image = resample_image(input_image=image,
                                   local_reference_image=target_image,
                                   transform_file=output_transform)
        else:
            image = resample_image(input_image=image,
                                   local_reference_image=image,
                                   transform_file=output_transform)
    elif target_image is not None:
        logger.debug("Resampling smoothed image from shape %s to %s",
                      image.shape, target_image.shape)
        image = nilearn.image.resample_to_img(
            source_img=image,
            target_img=target_image,
            interpolation="linear",
            force_resample=True,
            copy_header=True)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    logger.info(f"Saving smoothed image to {out_file}")
    image.to_filename(out_file)


def apply_estimated_gaussian_smoothing(in_files, out_files, edge_src, edge_dst, edge_distances, labels, sorted_labels,
                                       unique_nodes, fwhm, resampled_reference=None, low_memory=False,
                                       output_transform=None):
    """
    Apply estimated Gaussian smoothing to a list of images based on the provided graph structure.
    Finds the optimal tau for each of the five largest components in the graph (background, wm left/right, gm left/right).
    The rest of the components are smoothed with the mean tau of the five largest components.
    :param in_files: List of input image files to be smoothed.
    :param out_files: List of output image files to save the smoothed images.
    :param edge_src: Source nodes of the graph edges.
    :param edge_dst: Destination nodes of the graph edges.
    :param edge_distances: Distances of the graph edges.
    :param labels: Labels for each node in the graph.
    :param sorted_labels: Sorted list of labels by size of components.
    :param unique_nodes: Unique nodes in the graph.
    :param fwhm: Target full width at half maximum in mm for smoothing.
    :param resampled_reference: Optional reference image to resample the smoothed images to.
    :param low_memory: If True, use low memory mode. This will reduce memory usage but may increase runtime.
    :param output_transform: Optional transformation to apply to the smoothed images before writing them to file.
    :return: None
    """
    # Estimate optimal tau for each component to achieve the target fwhm
    main_labels = sorted_labels[:5]
    taus = list()
    initial_tau = 2 * fwhm
    for label in tqdm(main_labels, desc="Finding optimal taus", unit="component"):
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            labels=labels,
            label=label,
            unique_nodes=unique_nodes)
        _tau = find_optimal_tau(fwhm=fwhm,
                                edge_src=_edge_src,
                                edge_dst=_edge_dst,
                                edge_distances=_edge_distances,
                                shape=(len(_nodes),),
                                initial_tau=initial_tau)
        taus.append(_tau)
        # update initial tau to the last estimated tau
        initial_tau = _tau

    for i in tqdm(range(len(in_files)), desc="Smoothing images", unit="image"):
        in_file = in_files[i]
        out_file = out_files[i]

        signal_image, orig_image = load_image(in_file, reference_image=resampled_reference)
        signal_data = signal_image.get_fdata()

        if signal_data.ndim == 3:
            signal_data = signal_data[..., None]

        _shape = signal_data.shape

        signal_data = signal_data.reshape(-1, signal_data.shape[-1])
        # smooth the background data with the mean estimated tau
        smooth_component(
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_distances=edge_distances,
            signal_data=signal_data,
            labels=labels,
            label=sorted_labels[5:],  # smooth the rest of the components with the mean tau
            unique_nodes=unique_nodes,
            tau=np.mean(taus),  # Use the mean tau for all components
            low_memory=low_memory
        )
        # smooth each main component with the estimated tau
        for label, _tau in zip(main_labels, taus):
            smooth_component(edge_src=edge_src,
                             edge_dst=edge_dst,
                             edge_distances=edge_distances,
                             signal_data=signal_data,
                             labels=labels,
                             label=label,
                             unique_nodes=unique_nodes,
                             tau=_tau,
                             low_memory=low_memory)

        smoothed_image = nib.Nifti1Image(signal_data.reshape(_shape),
                                         affine=signal_image.affine)

        write_image(image=smoothed_image,
                    out_file=out_file,
                    target_image=orig_image,
                    output_transform=output_transform)


def apply_precomputed_kernels(in_files, out_files, kernel_filenames, resampled_reference=None):
    """
    Apply precomputed Gaussian kernels to smooth images.
    :param in_files:
    :param out_files:
    :param kernel_filenames:
    :param resampled_reference:
    :return:
    """
    for in_file, out_file in zip(in_files, out_files):

        signal_image, orig_image = load_image(in_file, reference_image=resampled_reference)
        signal_data = signal_image.get_fdata()

        if signal_data.ndim == 3:
            signal_data = signal_data[..., None]

        _shape = signal_data.shape

        signal_data = signal_data.reshape(-1, signal_data.shape[-1])
        smoothed_signal_data = signal_data.copy()

        for kernel_filename in kernel_filenames:
            kernel_data = np.load(kernel_filename)
            _edge_src = kernel_data['src']
            _edge_dst = kernel_data['dst']
            _edge_weights = kernel_data['weights']
            nodes = kernel_data['nodes']

            smoothed_signal_data[nodes, :] = apply_gaussian_smoothing(signal_data[nodes, :],
                                                                      _edge_src,
                                                                      _edge_dst,
                                                                      _edge_weights)

        smoothed_image = nib.Nifti1Image(smoothed_signal_data.reshape(_shape),
                                         affine=signal_image.affine)
        write_image(image=smoothed_image,
                    out_file=out_file,
                    target_image=orig_image)


def smooth_images(in_files, out_files, surface_files, out_kernel_basename=None, tau=None, fwhm=None,
                  output_labelmap=None,
                  resample_resolution=None, mask_file=None, mask_dilation=3,
                  estimate=True, low_memory=False,
                  t1w_to_mni_transform=None):
    """
    Smooth an image using graph signal smoothing.
    :param in_files: Path to a Nifti files to be smoothed.
    :param out_files: Output filenames to save the smoothed image.
    :param out_kernel_basename: filepath to save smoothing kernel files. The label number and '.npz' will be
    appended to the basename for each component. Required if fwhm is provided and estimate is False.
    :param surface_files: List of surface filenames to use for edge pruning.
    :param tau: Value of tau to use for graph signal smoothing. Either tau or fwhm must be provided.
    :param fwhm: Value of FWHM to use for Gaussian smoothing. Either tau or fwhm must be provided.
    :param output_labelmap: Optional output labelmap filename to save the individual components that were smoothed. To disable, set to None.
    :param resample_resolution: Optional (x, y, z) resolution to resample the image to. If None, no resampling is done.
    Otherwise, the image is resampled to the specified resolution prior to formation of the graph and smoothing. After
    smoothing, the image is resampled back to the original resolution.
    :param mask_file: Optional filename of a mask to use for smoothing. This can speed up processing and reduce
     computational requirements, If None, no mask is used.
    :param mask_dilation: Optional number of voxels to dilate the mask by. This can help to include more voxels in the
        smoothing process. If None, no dilation is done. Mask dilation is done in the resampled image space. If no
        signal image resampling is done, the mask is dilated in the signal image space (not the mask image space).
        If no mask filename is provided, this parameter is ignored.
    :param estimate: If True, estimate the optimal tau for each component to achieve the target fwhm. This is
    much faster than computing the Gaussian kernels but may result in slightly different smoothing. If False,
    precompute the Gaussian kernels for each component and save to file which could take a very long time.
    :param low_memory: If True, use low memory mode. This will reduce memory usage but may increase runtime.
    Memory usage is reduced by smoothing each timepoint separately instead of all at once. This is useful for large
    images with many timepoints. If False, all timepoints are smoothed at once which is faster but requires more memory.
    :param t1w_to_mni_transform: Optional transformation matrix to apply to the smoothed image after processing
    to align it to MNI space before writing it to file.
    :raises ValueError: If both tau and fwhm are None or if both are provided.
    :return:
    """
    # TODO: check that all the bold images are aligned

    reference_image, resampled_reference = load_reference_image(in_files[0], resample_resolution)
    # reference image and resampled reference are the same if resampling is done, otherwise the resampled reference is None.

    mask_array = process_mask(mask_file, reference_image, mask_dilation)
    edge_src, edge_dst, edge_distances = create_graph(mask_array, reference_image.affine, surface_files)

    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst)

    if output_labelmap is not None:
        save_labelmap(output_labelmap, reference_image.shape, reference_image.affine, labels, sorted_labels,
                      unique_nodes)

    if tau is not None:
        raise NotImplementedError("Smoothing with tau is not yet supported for multiple images.")
    elif fwhm is not None:
        if estimate:
            apply_estimated_gaussian_smoothing(
                in_files=in_files,
                out_files=out_files,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_distances=edge_distances,
                labels=labels,
                sorted_labels=sorted_labels,
                unique_nodes=unique_nodes,
                fwhm=fwhm,
                resampled_reference=resampled_reference,
                low_memory=low_memory,
                output_transform=t1w_to_mni_transform)
        else:
            warnings.warn("Computing Gaussian kernels for each component. This may take a very long time for "
                          "large Guassian FWHM values and/or high resolution images.")
            if out_kernel_basename is None:
                raise ValueError("out_kernel_basename must be provided if fwhm is provided and estimate is False.")

            kernel_filenames = precompute_guassian_kernels(
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_distances=edge_distances,
                labels=labels,
                sorted_labels=sorted_labels,
                unique_nodes=unique_nodes,
                out_kernel_basename=out_kernel_basename,
                fwhm=fwhm)

            apply_precomputed_kernels(
                in_files=in_files,
                out_files=out_files,
                kernel_filenames=kernel_filenames,
                resampled_reference=resampled_reference)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Smooth fMRI images using constrained smoothing.")
    args = parser.parse_args()

    check_parameters(args, parser)

    return args


def add_file_args(parser):
    parser.add_argument("in_file", type=str,
                        help="Input image to be smoothed. In most cases this should be a preprocessed fMRI image. "
                             "Input image must be aligned in the same space as the surface files.")
    parser.add_argument("out_file", type=str,
                        help="Output smoothed image filename.")
    parser.add_argument("--surface_files", type=str, nargs='+', required=True,
                        help="List of surface files to use for edge pruning. Must be in GIFTI format.")
    parser.add_argument("--mask_file", type=str, default=None,
                        help="Optional mask file to use for smoothing. "
                             "Must be in the same space as the input image but may be a different resolution. "
                             "Will be resampled to match input image resolution. "
                             "If not provided, the whole image is used which increases computational requirements "
                             "and runtime.")
    # TODO: add option to use a labelmap instead of mask_file
    parser.add_argument("--output_labelmap", type=str,
                        help="Optional output labelmap filename to save the individual components that were smoothed. "
                             "By default, this is saved to the '{output_basename}_components.nii.gz'. "
                             "To disable, set to None. ")
    return parser


def add_parameter_args(parser):
    parser.add_argument("--tau", type=float,
                        help="Tau value for heat kernel smoothing. Either --tau or --fwhm must be provided.")
    parser.add_argument("--fwhm", type=float,
                        help="FWHM value for Gaussian smoothing. Either --tau or --fwhm must be provided.")
    parser.add_argument("--mask_dilation", type=int, default=3,
                        help="Number of voxels to dilate the mask by. "
                             "This can help make sure no parts of the brain are being erroneously excluded due to any "
                             "masking errors. "
                             "If None, no dilation is done. Default is 3.")
    parser.add_argument("--multiproc", type=int, default=4,
                        help="Number of parallel processes to use for smoothing.")
    parser.add_argument("--overwrite", action='store_true',
                        help="If set, overwrite existing output files. Default is to not overwrite.")
    parser.add_argument("--voxel_size", type=float, default=1.0,
                        help="Isotropic voxel size for resampling the image and mask prior to smoothing. "
                             "Smaller voxel sizes allow for a more continuous graph but increase computational "
                             "requirements and runtime. Default is 1.0 mm.")
    parser.add_argument("--low_mem", action='store_true',
                        help="If set, use low memory mode. This will reduce memory usage but may increase runtime. "
                             "Memory usage is reduced by smoothing each timepoint separately instead of all at once. "
                             "This is useful for very large images or when running on machines with limited memory. "
                             "Default is to smooth all timepoints at once.")
    parser.add_argument("--debug", action='store_true',
                        help="If set, enable debug logging. Default is to use info level logging.")
    return parser


def check_parameters(args, parser):
    # Validation to ensure either tau or fwhm is provided
    if args.tau is None and args.fwhm is None:
        parser.error("Either --tau or --fwhm must be provided.")
    if args.tau is not None and args.fwhm is not None:
        parser.error("Only one of --tau or --fwhm can be provided, not both.")
    if args.debug:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

def main():
    args = parse_args()

    # Enable NetworkX parallel configuration
    nx.config.backends.parallel.active = True
    nx.config.backends.parallel.n_jobs = args.multiproc

    output_labelmap = args.output_labelmap
    if output_labelmap.lower() == "none":
        output_labelmap = None
    elif output_labelmap is None:
        output_labelmap = os.path.splitext(args.out_file)[0] + "_components.nii.gz"

    if os.path.exists(args.out_file):
        if args.overwrite:
            warnings.warn(f"Overwriting existing file: {args.out_file}.")
        else:
            warnings.warn(f"Output file {args.out_file} already exists.")
            warnings.warn("Exiting. Use --overwrite to overwrite existing files.")
            return

    smooth_image(in_file=args.in_file,
                 out_file=args.out_file,
                 surface_files=args.surface_files,
                 tau=args.tau,
                 fwhm=args.fwhm,
                 mask_file=args.mask_file,
                 mask_dilation=args.mask_dilation,
                 output_labelmap=output_labelmap,
                 resample_resolution=(args.voxel_size, args.voxel_size, args.voxel_size),
                 low_memory=args.low_mem)

    logger.info("Smoothing complete.")


if __name__ == "__main__":
    main()
