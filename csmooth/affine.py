import numpy as np


def get_spacing_from_affine(affine):
    """
    Get the spacing from the affine matrix.
    :param affine: affine matrix
    :return: spacing
    """
    RZS = affine[:3, :3]
    return np.sqrt(np.sum(np.multiply(RZS, RZS), axis=0))

def scale(x, y, z, dtype=np.float32):
    """
    Generate a transformation matrix to scale an image
    :param x: scaling in the x-direction
    :param y: scaling in the y-direction
    :param z: scaling in the z-direction
    :param dtype: the data type for the resulting affine matrix
    :return: affine matrix to scale the image
    """
    scale = np.array([[x, 0, 0, (x - 1) / 2],
                      [0, y, 0, (y - 1) / 2],
                      [0, 0, z, (z - 1) / 2],
                      [0, 0, 0, 1]], dtype=dtype)
    return scale


def adjust_affine_spacing(affine, new_spacing):
    affine = np.array(affine, dtype=np.float32)
    new_spacing = np.array(new_spacing, dtype=np.float32)
    spacing = get_spacing_from_affine(affine)
    spacing_scale = np.divide(new_spacing, spacing)
    scale_affine = scale(spacing_scale[0], spacing_scale[1], spacing_scale[2])
    return np.matmul(affine, scale_affine)


def resample_data_to_shape(data, target_shape):
    """
    Resample data to a target shape.
    :param data: numpy array containing the data to resample.
    :param target_shape: target shape of the data.
    :return: resampled data as a numpy array.
    """
    from scipy.ndimage import zoom
    data = np.asarray(data)
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, zoom_factors, order=1)

def resample_data_to_affine(data, target_affine, original_affine, interpolation="continuous"):
    """
    Resample data to a target affine.
    :param data: numpy array containing the data to resample.
    :param target_affine: target affine matrix.
    :param original_affine: original affine matrix of the data.
    :return: resampled data as a numpy array.
    """
    import nilearn
    import nibabel as nib

    original_image = nib.Nifti1Image(data, original_affine)
    resampled_image = nilearn.image.resample_img(original_image,
                                                 interpolation=interpolation,
                                                 target_affine=target_affine,
                                                 force_resample=True,
                                                 copy_header=True)
    return resampled_image.get_fdata()



