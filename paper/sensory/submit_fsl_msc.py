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
#SBATCH --output=/data2/david.ellis/public/MSC/cache/fsl/logs/%j.out
#SBATCH --error=/data2/david.ellis/public/MSC/cache/fsl/logs/%j.err
#SBATCH --job-name=fsl-feat
echo "Running feat {filename}"
echo "Starting at $(date)"
apptainer exec -B /data -B /data2 /data2/david.ellis/public/HCPA/code/tractography/fslantsnipype.sif feat {filename}
echo "Finished at $(date)"
""".format(filename=filename)
    os.makedirs("/data2/david.ellis/public/MSC/cache/fsl/logs", exist_ok=True)
    slurm_filename = filename.replace(".fsf", ".slurm")
    with open(slurm_filename, "w") as op:
        op.write(slurm_str)
    subprocess.call(["sbatch", slurm_filename])

def convert_events_tsv_to_txt(events_filename_tsv, txt_filename):
    """
    Convert a TSV file to a text file with the first two columns
    :param events_filename_tsv: Path to the TSV file
    :param txt_filename: Path to the output text file
    :return: Path to the converted text file
    """
    with open(events_filename_tsv, "r") as tsv_file, open(txt_filename, "w") as txt_file:
        for line in tsv_file:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                txt_file.write(f"{parts[0]}\t{parts[1]}\t1\n")
    return txt_filename


def main():
    base_directory = "/data2/david.ellis/public/MSC"
    assert os.path.exists(base_directory), f"Base directory {base_directory} does not exist"

    fmriprep_dir = f"{base_directory}/derivatives/fmriprep"
    assert os.path.exists(fmriprep_dir), f"fmriprep directory {fmriprep_dir} does not exist"

    bids_dir = f"{base_directory}/bids"
    assert os.path.exists(bids_dir), f"BIDS directory {bids_dir} does not exist"

    fsl_template_filename = f"{base_directory}/code/fsl/template.fsf"
    assert os.path.exists(fsl_template_filename), f"FSL template file {fsl_template_filename} does not exist"

    os.makedirs(f"{base_directory}/cache/fsl/logs", exist_ok=True)

    fsl_directory = f"{base_directory}/derivatives/fsl"
    # search for files in the fmriprep directory
    for task in ("motor",):
        fmriprep_wildcard = os.path.join(fmriprep_dir, f"sub-*/func/sub-*_task-{task}_*space-T1w_desc-preproc_bold.nii.gz")
        fmriprep_filenames = sorted(glob.glob(fmriprep_wildcard))
        for fmriprep_filename in fmriprep_filenames:

            subject = re.search(r"sub-(\d+)", fmriprep_filename).group(1)
            events_filename_tsv = fmriprep_filename.replace(
                "space-T1w_desc-preproc_bold.nii.gz", "_events.tsv").replace(
                fmriprep_dir, bids_dir)
            assert os.path.exists(events_filename_tsv), \
                f"Events file {events_filename_tsv} does not exist for {fmriprep_filename}"

            ev_filename = os.path.join(fsl_directory,
                                       f"sub-{subject}",
                                       "func",
                                       os.path.basename(events_filename_tsv).replace(".tsv", ".txt"))
            if not os.path.exists(ev_filename):
                os.makedirs(os.path.dirname(ev_filename), exist_ok=True)
                ev_filename = convert_events_tsv_to_txt(events_filename_tsv, ev_filename)

            for smoothing in (0, 3, 6, 9, 12):
                output_directory = os.path.join(fsl_directory, f"sub-{subject}/func",
                                                os.path.basename(fmriprep_filename).replace(".nii.gz", f"_fwhm-{smoothing}"))
                os.makedirs(os.path.dirname(output_directory), exist_ok=True)
                # copy the template file to the output directory
                subject_feat_filename = output_directory + ".fsf"
                shutil.copy(fsl_template_filename, subject_feat_filename)
                modify_feat_file(subject_feat_filename, fmriprep_filename, output_directory, smoothing, ev_filename)
                submit_feat_file(subject_feat_filename)





if __name__ == "__main__":
    main()