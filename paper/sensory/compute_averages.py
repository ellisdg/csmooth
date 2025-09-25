import glob
import re
import os
import sys
from nipype.interfaces.ants import ApplyTransforms

def average_image(filenames, output_filename):
    import nibabel as nib
    import numpy as np
    import os

    image1 = nib.load(filenames[0])
    image = image1.__class__(np.stack([np.asarray(nib.load(f).dataobj) for f in filenames]).mean(axis=0),
                             image1.affine)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print("Writing averaged image to {}".format(output_filename))
    image.to_filename(output_filename)
    return output_filename


def transform_image(input_filename, transform, output_filename, reference_image):

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    ants_apply = ApplyTransforms()
    ants_apply.inputs.input_image = input_filename
    ants_apply.inputs.transforms = transform
    ants_apply.inputs.reference_image = reference_image
    ants_apply.inputs.output_image = output_filename
    ants_apply.run()
    return output_filename

def main():
    sys.path.append(os.environ["ANTSPATH"])
    base_dir = "/media/conda2/public/sensory"
    task = "lefthand"
    data = dict()
    reference_filename = ("/media/conda2/public/sensory/derivatives/fmriprep/sub-01/func/"
                          "sub-01_task-lefthand_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz")
    for fsl_dir_run1 in sorted(glob.glob(f"{base_dir}/derivatives/fsl*/**/sub-*task-{task}*run-1*space-T1w*.feat",
                                         recursive=True)):
        print(fsl_dir_run1)
        fsl_dir_run2 = fsl_dir_run1.replace("run-1_", "run-2_")
        input_filenames = list()
        ext = fsl_dir_run1.split("_")[-1].replace(".feat", "")
        task = re.search("task-(\w+)_", fsl_dir_run1).group(1)
        key = base_dir + "/derivatives/fsl/" + "_".join((task, ext)) + ".nii.gz"
        print(key)
        for fsl_dir in (fsl_dir_run1, fsl_dir_run2):
            input_filenames.append(os.path.join(fsl_dir, "stats", "zstat1.nii.gz"))

        output_filename = input_filenames[0].replace("run-1_", "")
        if not os.path.exists(output_filename):
            average_image(input_filenames, output_filename=output_filename)

        subject = re.search("sub-(\d+)", fsl_dir_run1).group(1)
        transform_file = f"{base_dir}/derivatives/fmriprep/sub-{subject}/anat/sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

        output_mni_filename = output_filename.replace("-T1w_", "-MNI152NLin2009cAsym_")
        if not os.path.exists(output_mni_filename):
            transform_image(output_filename, transform_file, output_mni_filename, reference_filename)

        if key not in data:
            data[key] = list()

        data[key].append(output_mni_filename)

    for key in data:
        output_filename = key
        if not os.path.exists(output_filename):
            average_image(data[key], output_filename=output_filename)

if __name__ == "__main__":
    main()
