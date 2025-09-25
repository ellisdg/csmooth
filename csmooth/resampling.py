import nitransforms as nt
import templateflow.api as tf
import re
import nibabel as nib
from csmooth.utils import logger
from csmooth.affine import adjust_affine_spacing
import nilearn
import numpy as np
from tqdm import tqdm


def resample_image(input_image, local_reference_image, transform_file, resolution=1, suffix='T1w'):
    """
    Resample an image to a local reference space or MNI space using a specified transform file.
    :param input_image: nibabel image to resample.
    :param local_reference_image: nibabel image defining the local reference space.
    :param transform_file: transform file to apply for resampling.
    :param resolution: resolution of the MNI template to use, default is 1mm.
    :param suffix: suffix of the MNI template to use, default is 'T1
    :return:
    """
    logger.info(f"Resampling image to local reference space using transform file: {transform_file}")
    # use regex to get the specific MNI template from the transform file
    # the MNI space name should be prefaced by "space-" and then end with "_"
    mni_space_match = re.search(r'to-(\w+)_', transform_file)
    if mni_space_match:
        mni_space = mni_space_match.group(1)
        logger.debug(f"Using template space: {mni_space} from transform file: {transform_file}. "
                     f"If this is not the intended template space, please check the transform file name.")
        logger.debug("Fetching MNI reference image from TemplateFlow.")
        mni_reference_file = tf.get(mni_space, resolution=resolution, suffix=suffix)
        if type(mni_reference_file) is list:
            logger.debug(f"Multiple MNI reference images found for space {mni_space} with resolution {resolution}mm. "
                         f"Using the first one: {mni_reference_file[0]}")
            mni_reference_file = mni_reference_file[0]
    else:
        raise ValueError("Transform file does not contain a valid template space identifier. "
                         "Please ensure the transform filename contains 'to-<template_name>_' format.")

    # Load the MNI reference image
    logger.debug(f"Loading MNI reference image from: {mni_reference_file}")
    mni_reference_image = nib.load(mni_reference_file)
    logger.debug(f"Adjusting MNI reference image affine to match local reference image spacing: "
                 f"{local_reference_image.header.get_zooms()[:3]}")
    mni_reference_affine = adjust_affine_spacing(mni_reference_image.affine,
                                                 local_reference_image.header.get_zooms()[:3])
    mni_reference_image =  nilearn.image.resample_img(mni_reference_image,
                                                      target_affine=mni_reference_affine,
                                                      force_resample=True,
                                                      copy_header=True)

    logger.debug(f"Loading transform file: {transform_file}")
    transform = nt.manip.load(transform_file)

    return resample_4d_image(input_image, mni_reference_image, transform)


def resample_4d_image(input_image, reference_image, transform):
    # Get data and affine from input and reference
    input_data = input_image.get_fdata()
    n_vols = input_data.shape[-1]
    resampled_vols = []
    for t in tqdm(range(n_vols), desc="Resampling volumes"):
        vol_img = nib.Nifti1Image(input_data[..., t], input_image.affine, input_image.header)
        resampled_vol = nt.resampling.apply(transform=transform,
                                            spatialimage=vol_img,
                                            reference=reference_image)
        resampled_vols.append(resampled_vol.get_fdata())
    resampled_data = np.stack(resampled_vols, axis=-1)
    # Create new 4D image with reference affine/header
    resampled_img = nib.Nifti1Image(resampled_data, reference_image.affine)
    return resampled_img
