import glob
import os

import networkx as nx

from csmooth.smooth import check_parameters, smooth_images
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


def find_fmriprep_bold_files(fmriprep_subject_dir):
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

def find_t1w_to_mni_transform(fmriprep_subject_dir):
    """
    Find the T1w to MNI transformation file in the fmriprep subject directory.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :return: Path to the T1w to MNI transformation file
    """
    # TODO: allow for use of different MNI spaces, e.g. MNI152NLin6Asym
    transform_file = os.path.join(fmriprep_subject_dir, "**", "anat", "*_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")
    transform_files = glob.glob(transform_file, recursive=True)

    if not transform_files:
        raise FileNotFoundError(f"No T1w to MNI transformation file found in {fmriprep_subject_dir}/anat")

    if len(transform_files) > 1:
        raise ValueError(f"Expected exactly one T1w to MNI transformation file, found {len(transform_files)}: {transform_files}")

    return transform_files[0]  # Return the single transformation file found


def find_fmriprep_files(fmriprep_subject_dir, find_bold_files=True, find_t1w_to_mni=False):
    """
    Find all necessary fMRIPrep files for a subject.
    :param fmriprep_subject_dir: Directory containing fmriprep outputs for a subject
    :param find_bold_files: If True, also find the BOLD files in T1w space. (default: True)
    This can be set to False if you want to use custom BOLD files.
    :param find_t1w_to_mni: If True, also find the T1w to MNI transformation file
    :return: A dictionary with surface files, BOLD files, and mask file
    """
    if not os.path.exists(fmriprep_subject_dir):
        raise FileNotFoundError(f"fMRIPrep subject directory does not exist: {fmriprep_subject_dir}")

    surface_files = find_surface_files(fmriprep_subject_dir)

    if find_bold_files:
        bold_files = find_fmriprep_bold_files(fmriprep_subject_dir)
    else:
        bold_files = []


    mask_file = find_mask_file(fmriprep_subject_dir)
    if find_t1w_to_mni:
        t1w_to_mni_transform = find_t1w_to_mni_transform(fmriprep_subject_dir)
        logger.info(f"Found T1w to MNI transformation file: {t1w_to_mni_transform}")
    else:
        t1w_to_mni_transform = None

    return {
        "surface_files": surface_files,
        "bold_files": bold_files,
        "mask_file": mask_file,
        "t1w_to_mni_transform": t1w_to_mni_transform
    }


def derive_output_filenames(output_subject_dir, input_filenames, tau=None, fwhm=None, output_to_mni=False):
    """
    Based on the input parameters and found files, derive output filenames for smoothed images.
    :param output_subject_dir: Output directory for smoothed images
    :param input_filenames: a list of file paths to the input BOLD images
    :param tau: Smoothing parameter in seconds (optional)
    :param fwhm: Smoothing parameter in mm (optional)
    :param output_to_mni: If True, output filenames will be modified to indicate MNI space.
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

        if output_to_mni:
            if "space-T1w" in output_base_name:
                output_base_name = output_base_name.replace("space-T1w", "space-MNI152NLin2009cAsym")
            else:
                raise ValueError("Output to MNI space requested, but input filenames do not contain 'space-T1w'. "
                                 "Please ensure input files have space-T1w in their file names.")

        output_filename = os.path.join(output_subject_dir, "func", output_base_name)
        if output_filename in output_filenames:
            raise ValueError(f"Output filename already exists: {output_filename}. "
                             f"Please check that your input file basenames are unique.")
        output_filenames.append(output_filename)

    return output_filenames



def process_fmriprep_subject(fmriprep_subject_dir, output_subject_dir, parameters, bold_files=None,
                             output_to_mni=False):
    """
    Process a single fMRIPrep subject directory to prepare for constrained smoothing.
    :param fmriprep_subject_dir: Directory containing fMRIPrep outputs for a subject
    :param output_subject_dir: Directory to save the processed outputs
    :param parameters: Dictionary of smoothing parameters
    :param bold_files: Optional list of input BOLD files to process instead of finding them in fMRIPrep directory
    :param output_to_mni: If True, outputs will be resampled to MNI space
    :return: None
    """
    files = find_fmriprep_files(fmriprep_subject_dir, find_bold_files=bold_files is None,
                                find_t1w_to_mni=output_to_mni)
    if bold_files is not None:
        files["bold_files"] = bold_files

    logger.info(f"Processing fMRIPrep subject directory: {fmriprep_subject_dir}")

    # TODO: add subject id to the output labelmap filename
    output_labelmap_file = os.path.join(output_subject_dir, "anat", "components_labelmap.nii.gz")
    output_filenames = derive_output_filenames(output_subject_dir, files["bold_files"],
                                               tau=parameters.get("tau", None),
                                               fwhm=parameters.get("fwhm", None),
                                               output_to_mni=output_to_mni)
    logger.info(f"Overwriting output filenames: {parameters['overwrite']}")
    for input_filename, output_filename in zip(list(files["bold_files"]), list(output_filenames)):
        if not parameters["overwrite"] and os.path.exists(output_filename):
            logger.warning(f"Output file already exists, skipping: {output_filename}")
            files["bold_files"].remove(input_filename)
            output_filenames.remove(output_filename)
        else:
            logger.info(f"Processing {input_filename} to {output_filename}")
    if not files["bold_files"]:
        logger.info(f"All output files already exist in {output_subject_dir}; skipping processing.")
        return
    kernel_basename = os.path.join(output_subject_dir,
                                   "cache",
                                   "csmooth_kernel_fwhm-{}mm_voxel-{}mm".format(
        parameters.get("fwhm", None), parameters.get("voxel_size", None)))
    if parameters.get("no_resample", False):
        logger.warning("Ignoring --voxel_size parameter and not resampling the image prior to smoothing.")
        resample_resolution = None
    else:
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
                  resample_resolution=resample_resolution,
                  low_memory=parameters.get("low_memory", False),
                  t1w_to_mni_transform=files.get("t1w_to_mni_transform", None))

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
    parser.add_argument("--bold_files", type=str, nargs="+",
                        help="List of input BOLD files in T1w space to process "
                             "(optional, requires --subject to be specified, will ignore BOLD files in fmriprep_dir). "
                             "Use this option to specify custom input files for constrained smoothing, but still use "
                             "the surface and mask files from the fMRIPrep directory. "
                             "Input files should be in T1w space and have 'space-T1w' in their filenames. "
                             "If not specified, the script will find BOLD files in the fMRIPrep directory.")
    parser.add_argument("--output_to_mni", action="store_true",
                        help="If set, outputs will be resampled to MNI space. "
                             "This is accomplished by finding the T1w to MNI transformation file in the fMRIPrep "
                             "directory and applying it to the smoothed images. "
                             "If True, T1w space files will not be saved to the output directory")
    add_parameter_args(parser)
    args = parser.parse_args()
    check_parameters(args, parser)
    return args


def main():
    kwargs = vars(parse_args())
    fmriprep_dir = kwargs.pop("fmriprep_dir")
    output_dir = kwargs.pop("output_dir")
    subject_id = kwargs.pop("subject")
    bold_files = kwargs.pop("bold_files", None)
    output_to_mni = kwargs.pop("output_to_mni")

    # Enable NetworkX parallel configuration
    nx.config.backends.parallel.active = True
    nx.config.backends.parallel.n_jobs = kwargs["multiproc"]

    if subject_id is None:
        if bold_files is not None:
            raise ValueError("Cannot specify --in_files without --subject. "
                             "Please specify a subject to process or remove the --in_files option.")
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
        process_fmriprep_subject(fmriprep_subject_dir, output_subject_dir, kwargs, bold_files=bold_files,
                                 output_to_mni=output_to_mni)




if __name__ == "__main__":
    main()


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
    parser.add_argument("--no_resample", action='store_true',
                        help="If set, do not resample the image prior to smoothing. "
                             "The graph will be formed using the resolution of the BOLD images. "
                             "This overrides the --voxel_size parameter. "
                             "Default is to resample the image to the specified voxel size.")
    parser.add_argument("--low_mem", action='store_true',
                        help="If set, use low memory mode. This will reduce memory usage but may increase runtime. "
                             "Memory usage is reduced by smoothing each timepoint separately instead of all at once. "
                             "This is useful for very large images or when running on machines with limited memory. "
                             "Default is to smooth all timepoints at once.")
    parser.add_argument("--debug", action='store_true',
                        help="If set, enable debug logging. Default is to use info level logging.")
    return parser
