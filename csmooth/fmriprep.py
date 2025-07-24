import glob
import os


import networkx as nx

from csmooth.smooth import add_parameter_args, check_parameters, smooth_images
from csmooth.utils import logger


def find_surface_files(fmriprep_subject_dir):
    """
    Find surface files in the fmriprep subject directory.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :return: A list of surface files
    """

    pial_files = glob.glob(os.path.join(fmriprep_subject_dir, "**", "anat", "*_pial.surf.gii"), recursive=True)
    whitematter_files = glob.glob(os.path.join(fmriprep_subject_dir, "**", "anat", "*_white.surf.gii"), recursive=True)
    surface_files = pial_files + whitematter_files
    if not len(pial_files) == 2:
        raise ValueError(f"Expected exactly 2 pial surface files, "
                         f"found {len(pial_files)} in {fmriprep_subject_dir}/anat")
    elif not len(whitematter_files) == 2:
        raise ValueError(f"Expected exactly 2 white matter surface files, "
                         f"found {len(whitematter_files)} in {fmriprep_subject_dir}/anat")
    return surface_files


def find_bold_files(fmriprep_subject_dir):
    """
    Find BOLD files in the fmriprep subject directory.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :return: A list of BOLD files
    """

    bold_files = sorted(glob.glob(os.path.join(fmriprep_subject_dir, "**", "func", "*_space-T1w_desc-preproc_bold.nii.gz"),
                           recursive=True))
    if not bold_files:
        raise FileNotFoundError(f"No BOLD files found in {fmriprep_subject_dir}/func")

    logger.info(f"Found {len(bold_files)} BOLD files in {fmriprep_subject_dir}/func: {bold_files}")

    return bold_files


def find_mask_file(fmriprep_subject_dir):
    """
    Find brain mask file in the fmriprep subject directory.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :return: Path to the brain mask file
    """

    mask_file = os.path.join(fmriprep_subject_dir, "**", "anat", "*_desc-brain_mask.nii.gz")
    mask_files = [f for f in glob.glob(mask_file, recursive=True) if "space-MNI" not in f]

    if not mask_files:
        raise FileNotFoundError(f"No brain mask file found in {fmriprep_subject_dir}/func")

    if len(mask_files) > 1:
        raise ValueError(f"Expected exactly one brain mask file, found {len(mask_files)}: {mask_files}")

    return mask_files[0]  # Return the single mask file found


def find_fmriprep_files(fmriprep_subject_dir):
    """
    Find all necessary fMRIPrep files for a subject.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :return: A dictionary with surface files, BOLD files, and mask file
    """
    if not os.path.exists(fmriprep_subject_dir):
        raise FileNotFoundError(f"fMRIPrep subject directory does not exist: {fmriprep_subject_dir}")

    surface_files = find_surface_files(fmriprep_subject_dir)
    bold_files = find_bold_files(fmriprep_subject_dir)
    mask_file = find_mask_file(fmriprep_subject_dir)

    return {
        "surface_files": surface_files,
        "bold_files": bold_files,
        "mask_file": mask_file
    }


def derive_output_filenames(output_subject_dir, input_filenames, tau=None, fwhm=None):
    """
    Based on the input parameters and found files, derive output filenames for smoothed images.
    :param output_subject_dir: Output directory for smoothed images
    :param input_filenames: a list of file paths to the input BOLD images
    :param tau: Smoothing parameter in seconds (optional)
    :param fwhm: Smoothing parameter in mm (optional)
    :return: output_filenames: a list of output filenames for smoothed images
    """
    output_filenames = []
    for input_filename in input_filenames:
        base_name = os.path.basename(input_filename)
        output_base_name = base_name.replace("_desc-preproc_bold", "_desc-csmooth_{}_bold")

        # Add smoothing parameters to the output filename
        if tau is None and fwhm is None:
            raise ValueError(f"Must specify either tau or fwhm")
        elif fwhm is not None:
            fwhm_str = f"{fwhm:.1f}" if fwhm % 1 > 1e-6 else str(int(fwhm))
            output_base_name = output_base_name.format(f"fwhm-{fwhm_str}")
        elif tau is not None:
            tau_str = f"{tau:.1f}" if tau % 1 > 1e-6 else str(int(tau))
            output_base_name = output_base_name.format(f"tau-{tau_str}")
        else:
            raise ValueError("Invalid smoothing parameters provided.")

        output_filenames.append(os.path.join(output_subject_dir, "func", output_base_name))

    return output_filenames



def process_fmriprep_subject(fmriprep_subject_dir, output_subject_dir, parameters):
    """
    Process a single fMRIPrep subject directory to prepare for constrained smoothing.
    :param fmriprep_subject_dir: Directory containing fMRIPrep outputs for a subject
    :param output_subject_dir: Directory to save the processed outputs
    :param parameters: Dictionary of smoothing parameters
    """
    files = find_fmriprep_files(fmriprep_subject_dir)

    logger.info(f"Processing fMRIPrep subject directory: {fmriprep_subject_dir}")

    # TODO: add subject id to the output labelmap filename
    output_labelmap_file = os.path.join(output_subject_dir, "anat", "components_labelmap.nii.gz")
    output_filenames = derive_output_filenames(output_subject_dir, files["bold_files"],
                                               tau=parameters.get("tau", None),
                                               fwhm=parameters.get("fwhm", None))
    for input_filename, output_filename in zip(list(files["bold_files"]), list(output_filenames)):
        if not parameters["overwrite"] and os.path.exists(output_filename):
            logger.warning(f"Output file already exists, skipping: {output_filename}")
            files["bold_files"].remove(input_filename)
            output_filenames.remove(output_filename)
    if not files["bold_files"]:
        logger.info(f"All output files already exist in {output_subject_dir}, skipping processing.")
        return
    kernel_basename = os.path.join(output_subject_dir,
                                   "cache",
                                   "csmooth_kernel_fwhm-{}mm_voxel-{}mm".format(
        parameters.get("fwhm", None), parameters.get("voxel_size", None)))
    resample_resolution = (parameters.get("voxel_size"), parameters.get("voxel_size"), parameters.get("voxel_size"))

    smooth_images(in_files=files["bold_files"],
                  surface_files=files["surface_files"],
                  mask_file=files["mask_file"],
                  out_kernel_basename=kernel_basename,
                  out_files=output_filenames,
                  output_labelmap=output_labelmap_file,
                  tau=parameters.get("tau", None),
                  fwhm=parameters.get("fwhm", None),
                  mask_dilation=parameters.get("mask_dilation", None),
                  resample_resolution=resample_resolution)

    logger.info(f"Processed subject in {fmriprep_subject_dir}, outputs saved to {output_subject_dir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run constrained smoothing on fMRIPrep outputs")
    parser.add_argument("fmriprep_dir", type=str,
                        help="Path to the fMRIPrep derivatives directory")
    parser.add_argument("output_dir", type=str,
                        help="Path to the output directory to save the smoothed images")
    parser.add_argument("--subject", type=str,
                        help="Specify a single subject to process (default: all subjects in fmriprep_dir)",)
    add_parameter_args(parser)
    args = parser.parse_args()
    check_parameters(args, parser)
    return args


def main():
    args = parse_args()
    kwargs = vars(args)
    fmriprep_dir = kwargs.pop("fmriprep_dir")
    output_dir = kwargs.pop("output_dir")
    subject_id = kwargs.pop("subject")

    # Enable NetworkX parallel configuration
    nx.config.backends.parallel.active = True
    nx.config.backends.parallel.n_jobs = kwargs["multiproc"]

    if subject_id is None:
        subject_dirs = sorted(glob.glob(os.path.join(fmriprep_dir, "sub-*")))
        for fmriprep_subject_dir in subject_dirs:
            if not os.path.isdir(fmriprep_subject_dir):
                continue
            logger.info(f"Processing subject directory: {fmriprep_subject_dir}")
            output_subject_dir = os.path.join(output_dir, os.path.basename(fmriprep_subject_dir))
            os.makedirs(output_subject_dir, exist_ok=True)
            process_fmriprep_subject(fmriprep_subject_dir, output_subject_dir, kwargs)
    else:
        fmriprep_subject_dir = os.path.join(fmriprep_dir, f"sub-{subject_id}")
        if not os.path.exists(fmriprep_subject_dir):
            raise FileNotFoundError(f"Subject directory does not exist: {fmriprep_subject_dir}")
        output_subject_dir = os.path.join(output_dir, f"sub-{subject_id}")
        os.makedirs(output_subject_dir, exist_ok=True)
        process_fmriprep_subject(fmriprep_subject_dir, output_subject_dir, kwargs)




if __name__ == "__main__":
    main()
