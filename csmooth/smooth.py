import logging
import warnings
import nilearn.image
import numpy as np
import scipy
import nibabel as nib
import os
from tqdm import tqdm

from csmooth.gaussian import gaussian_smoothing, compute_gaussian_kernels, apply_gaussian_smoothing
from csmooth.graph import create_graph, identify_connected_components, select_nodes
from csmooth.heat import heat_kernel_smoothing
from csmooth.affine import adjust_affine_spacing, resample_data_to_shape, resample_data_to_affine


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
                                           fwhm=fwhm,
                                           n_jobs=n_jobs)
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
                     smoothed_signal_data, tau=None, fwhm=None, n_jobs=4):
    _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(edge_src=edge_src,
                                                                 edge_dst=edge_dst,
                                                                 edge_distances=edge_distances,
                                                                 labels=labels,
                                                                 label=label,
                                                                 unique_nodes=unique_nodes)
    _smoothed_data = _smooth_component(edge_src=_edge_src,
                                       edge_dst=_edge_dst,
                                       edge_distances=_edge_distances,
                                       signal_data=signal_data[_nodes, :],
                                       tau=tau,
                                       fwhm=fwhm,
                                       n_jobs=n_jobs)
    smoothed_signal_data[_nodes, :] = _smoothed_data


def load_and_resample_image(in_file, resample_resolution):
    reference_image = nib.load(in_file)
    original_shape = reference_image.shape[:3]
    original_affine = reference_image.affine

    if resample_resolution is not None:
        affine = adjust_affine_spacing(reference_image.affine, np.asarray(resample_resolution)).numpy()
        reference_data = resample_data_to_affine(reference_image.get_fdata()[..., 0],
                                                 target_affine=affine,
                                                 original_affine=reference_image.affine)
        shape = reference_data.shape[:3]
        reference_image = nib.Nifti1Image(reference_data, affine)
    else:
        affine = original_affine
        shape = original_shape

    return reference_image, affine, shape, original_shape, original_affine


def process_mask(mask_file, reference_image, mask_dilation):
    if mask_file is not None:
        mask_image = nib.load(mask_file)
    else:
        mask_image = nib.Nifti1Image(np.ones(reference_image.shape[:3], dtype=np.uint8), reference_image.affine)

    mask_image = nilearn.image.resample_to_img(mask_image, reference_image,
                                               interpolation="nearest",
                                               force_resample=True)
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


def smooth_components(edge_src, edge_dst, edge_distances, signal_data, labels, sorted_labels, unique_nodes, tau, fwhm,
                      n_jobs=4):
    smoothed_signal_data = signal_data.copy()
    for label in tqdm(sorted_labels, desc="Smoothing components", unit="component"):
        smooth_component(edge_src, edge_dst, edge_distances, signal_data, labels, label, unique_nodes,
                         smoothed_signal_data, tau=tau, fwhm=fwhm, n_jobs=n_jobs)
    return smoothed_signal_data


def smooth_image(in_file, out_file, surface_files, tau=None, fwhm=None, output_labelmap=None,
                 resample_resolution=None, mask_file=None, mask_dilation=3, multiproc=4):
    """
    Smooth an image using graph signal smoothing.
    :param in_file: Path to a Nifti file to be smoothed.
    :param out_file: Output filename to save the smoothed image.
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
    :return:
    """

    reference_image, affine, shape, original_shape, original_affine = load_and_resample_image(in_file,
                                                                                              resample_resolution)
    mask_array = process_mask(mask_file, reference_image, mask_dilation)
    edge_src, edge_dst, edge_distances = create_graph(mask_array, affine, surface_files)

    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst, edge_distances)

    if output_labelmap is not None:
        save_labelmap(output_labelmap, shape, affine, labels, sorted_labels, unique_nodes)

    signal_image = nib.load(in_file)
    signal_data = signal_image.get_fdata()
    if signal_data.ndim == 3:
        signal_data = signal_data[..., None]

    if resample_resolution is not None:
        signal_data = resample_data_to_shape(signal_data, shape)

    signal_data = signal_data.reshape(-1, signal_data.shape[-1])
    smoothed_signal_data = smooth_components(edge_src, edge_dst, edge_distances, signal_data, labels, sorted_labels,
                                             unique_nodes, tau, fwhm, n_jobs=multiproc)

    if resample_resolution is not None:
        smoothed_signal_data = resample_data_to_shape(smoothed_signal_data, original_shape)
    smoothed_image = nib.Nifti1Image(smoothed_signal_data.reshape(original_shape + (-1,)), signal_image.affine)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    smoothed_image.to_filename(out_file)


def smooth_images(in_files, out_files, surface_files, out_kernel_basename, tau=None, fwhm=None, output_labelmap=None,
                  resample_resolution=None, mask_file=None, mask_dilation=3, multiproc=4):
    """
    Smooth an image using graph signal smoothing.
    :param in_files: Path to a Nifti file to be smoothed.
    :param out_files: Output filename to save the smoothed image.
    :param out_kernel_basename: filepath to save smoothing kernel files. The label number and '.npz' will be
    appended to the basename for each component.
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
    :return:
    """

    if tau is not None:
        raise NotImplementedError("Smoothing with tau is not yet supported for multiple images.")

    # TODO: check that all the bold images are aligned
    reference_image, affine, shape, original_shape, original_affine = load_and_resample_image(in_files[0],
                                                                                              resample_resolution)
    mask_array = process_mask(mask_file, reference_image, mask_dilation)
    edge_src, edge_dst, edge_distances = create_graph(mask_array, affine, surface_files)

    labels, sorted_labels, unique_nodes = identify_connected_components(edge_src, edge_dst, edge_distances)

    if output_labelmap is not None:
        save_labelmap(output_labelmap, shape, affine, labels, sorted_labels, unique_nodes)

    # for each component, compute the gaussian kernels and save to file
    kernel_filenames = []
    for label in tqdm(sorted_labels, desc="Computing smoothing kernels", unit="component"):
        out_kernel_filename = out_kernel_basename + f"_{label}.npz"
        _edge_src, _edge_dst, _edge_distances, _nodes = select_nodes(edge_src=edge_src,
                                                                     edge_dst=edge_dst,
                                                                     edge_distances=edge_distances,
                                                                     labels=labels,
                                                                     label=label,
                                                                     unique_nodes=unique_nodes)
        _edge_src, _edge_dst, _edge_weights = compute_gaussian_kernels(edge_src=_edge_src,
                                                                       edge_dst=_edge_dst,
                                                                       edge_distances=_edge_distances,
                                                                       fwhm=fwhm,
                                                                       n_jobs=multiproc)
        np.savez_compressed(out_kernel_filename,
                            src=_edge_src,
                            dst=_edge_dst,
                            weights=_edge_weights,
                            nodes=_nodes)
        kernel_filenames.append(out_kernel_filename)


    for in_file, out_file in zip(in_files, out_files):

        signal_image = nib.load(in_file)
        signal_data = signal_image.get_fdata()
        if signal_data.ndim == 3:
            signal_data = signal_data[..., None]

        if resample_resolution is not None:
            signal_data = resample_data_to_shape(signal_data, shape)

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

            if resample_resolution is not None:
                smoothed_signal_data = resample_data_to_shape(smoothed_signal_data, original_shape)
            smoothed_image = nib.Nifti1Image(smoothed_signal_data.reshape(original_shape + (-1,)), signal_image.affine)

            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            smoothed_image.to_filename(out_file)


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
    #TODO: add option to use a labelmap instead of mask_file
    parser.add_argument("--output_labelmap", type=str,
                        help="Optional output labelmap filename to save the individual components that were smoothed. "
                             "By default, this is saved to the '{output_basename}_components.nii.gz'. "
                             "To disable, set to None. ")
    parser.add_argument("--surface_affine", type=str,
                        help="Optional affine affine matrix to apply to the surface coordinates to align them to the "
                             "image space. Typically, labeled 'from-fsnative_to-T1w' in the fmriprep output.")
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
    parser.add_argument("--no_overwrite", action='store_true',
                        help="If set, do not overwrite existing output files. Default is to overwrite.")
    parser.add_argument("--voxel_size", type=float, default=2.0,
                        help="Isotropic voxel size for resampling the image and mask prior to smoothing. "
                             "Smaller voxel sizes allow for a more continuous graph but increase computational "
                             "requirements and runtime. Default is 2.0 mm.")
    return parser


def check_parameters(args, parser):
    # Validation to ensure either tau or fwhm is provided
    if args.tau is None and args.fwhm is None:
        parser.error("Either --tau or --fwhm must be provided.")
    if args.tau is not None and args.fwhm is not None:
        parser.error("Only one of --tau or --fwhm can be provided, not both.")



def main():
    args = parse_args()

    output_labelmap = args.output_labelmap
    if output_labelmap.lower() == "none":
        output_labelmap = None
    elif output_labelmap is None:
        output_labelmap = os.path.splitext(args.out_file)[0] + "_components.nii.gz"

    if os.path.exists(args.out_file):
        if args.no_overwrite:
            warnings.warn(f"Output file {args.out_file} already exists.")
            warnings.warn("Exiting. Use --no_overwrite to overwrite existing files.")
        else:
            warnings.warn(f"Overwriting existing file: {args.out_file}.")

    smooth_image(in_file=args.in_file,
                 out_file=args.out_file,
                 surface_files=args.surface_files,
                 tau=args.tau,
                 fwhm=args.fwhm,
                 mask_file=args.mask_file,
                 mask_dilation=args.mask_dilation,
                 output_labelmap=output_labelmap,
                 resample_resolution=(args.voxel_size, args.voxel_size, args.voxel_size),)

    logging.log("Smoothing complete.")


if __name__ == "__main__":
    main()
