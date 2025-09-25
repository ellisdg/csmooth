import glob
import subprocess
import os
import shutil
import re


def modify_feat_file(filename, input_filename, output_directory, smoothing, ev_filename):
    with open(filename, "r") as op:
        s = op.read()

    s = s.replace("fmri_filename", input_filename)
    s = s.replace("output_directory", output_directory)
    s = s.replace("smoothing_fwhm", str(smoothing))
    s = s.replace("events_txt", ev_filename)

    with open(filename, "w") as op:
        op.write(s)


def submit_feat_file(filename):
    """
    Submit a feat file to the cluster via slurm
    :param filename:
    :return:
    """
    slurm_str = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --tasks=1
#SBATCH --nodelist=conda-gpu-02.unmc.edu
#SBATCH --output=/data2/david.ellis/public/sensory/cache/fsl/logs/%j.out
#SBATCH --error=/data2/david.ellis/public/sensory/cache/fsl/logs/%j.err
#SBATCH --job-name=fsl-feat
echo "Running feat {filename}"
echo "Starting at $(date)"
apptainer exec -B /data -B /data2 /data2/david.ellis/public/HCPA/code/tractography/fslantsnipype.sif feat {filename}
echo "Finished at $(date)"
""".format(filename=filename)
    os.makedirs("/data2/david.ellis/public/sensory/cache/fsl/logs", exist_ok=True)
    slurm_filename = filename.replace(".fsf", ".slurm")
    with open(slurm_filename, "w") as op:
        op.write(slurm_str)
    subprocess.call(["sbatch", slurm_filename])


def main():
    base_directory = "/data2/david.ellis/public/sensory"
    fmriprep_dir = f"{base_directory}/derivatives/fmriprep"
    fsl_template_filename = f"{base_directory}/code/fsl/template.fsf"
    fsl_directory = f"{base_directory}/derivatives/fsl"
    # search for files in the fmriprep directory
    for task in ("lefthand", "righthand"):
        wildcard = os.path.join(fmriprep_dir, f"sub-*/func/sub-*_task-{task}_*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        filenames = sorted(glob.glob(wildcard))
        ev_filename = f"{base_directory}/code/fsl/task-{task}_events.txt"
        assert os.path.exists(ev_filename)
        for filename in filenames:
            subject = re.search(r"sub-(\d+)", filename).group(1)
            for smoothing in (0, 3, 6, 9, 12):
                ouptut_directory = os.path.join(fsl_directory, f"sub-{subject}/func",
                                                os.path.basename(filename).replace(".nii.gz", f"_fwhm-{smoothing}"))
                os.makedirs(os.path.dirname(ouptut_directory), exist_ok=True)
                # copy the template file to the output directory
                subject_feat_filename = ouptut_directory + ".fsf"
                shutil.copy(fsl_template_filename, subject_feat_filename)
                modify_feat_file(subject_feat_filename, filename, ouptut_directory, smoothing, ev_filename)
                submit_feat_file(subject_feat_filename)





if __name__ == "__main__":
    main()