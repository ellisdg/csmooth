import os
import glob
import subprocess
import json
import re


def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def submit_files_dict(files_dict):
    for (subject_id, method, fwhm), fmri_files in files_dict.items():
        output_file = derive_output_filename(subject_id, method, fwhm)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping submission.")
        else:
            submit_compute_connectivity_matrix(fmri_files, output_file)

def submit_compute_connectivity_matrix(fmri_files, output_file, verbose=False):
    fmri_files_str = " ".join(fmri_files)
    command = ["sbatch",
               "/data2/david.ellis/public/HCPA/code/connectivity/submit_compute_matrix.sbatch",
               f'"{fmri_files_str}"',
               output_file]
    if verbose:
        print("Submitting command:", " ".join(command))
    subprocess.run(" ".join(command), shell=True, check=True)


def derive_output_filename(subject_id, method, fwhm, template="MNI152NLin2009cAsym", atlas="Schaefer2018",
                           description="1000Parcels7Networks"):
    return (f"/data2/david.ellis/public/HCPA/myderivatives/connectivity/{subject_id}/func/"
            f"{subject_id}_space-{template}_desc-{atlas}{description}_"
            f"{method}_fwhm-{fwhm}_connectivity.npy")


def find_subjects(cleaned_fmri_dir, csmooth_dir, gaussian_dir, remaining_volumes_threshold=0.8):
    submitted_subjects = []
    for subject_dir in sorted(glob.glob(os.path.join(cleaned_fmri_dir, "sub-*"))):
        subject_id = os.path.basename(subject_dir)
        fmri_rest_files = sorted(glob.glob(os.path.join(
            subject_dir, "func",
            f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")))
        csmooth_rest_files = sorted(glob.glob(os.path.join(
            csmooth_dir, subject_id, "func",
            f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_desc-csmooth*_bold.nii.gz")))
        gaussian_rest_files = sorted(glob.glob(os.path.join(
            gaussian_dir, subject_id, "func",
            f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_desc-gaussian*_bold.nii.gz"
        )))
        if len(fmri_rest_files) < 4:
            print(f"Excluding {subject_id} because only found {len(fmri_rest_files)} rest runs")
        elif len(fmri_rest_files)*4 != len(csmooth_rest_files):
            print(f"Excluding {subject_id} because number of cleaned rest runs "
                  f"({len(fmri_rest_files)}) does not match number of csmooth rest runs ({len(csmooth_rest_files)})")
        elif len(fmri_rest_files)*4 != len(gaussian_rest_files):
            print(f"Excluding {subject_id} because number of cleaned rest runs "
                  f"({len(fmri_rest_files)}) does not match number of Gaussian rest runs ({len(gaussian_rest_files)})")
        else:
            nvols = 0
            total_vols = 0
            for fmri_rest_file in fmri_rest_files:
                sidecar_file = fmri_rest_file.replace(".nii.gz", ".json")
                metadata = read_json(sidecar_file)
                nvols += metadata["NumberOfVolumesAfterScrubbing"]
                total_vols += metadata["NumberOfVolumesBeforeScrubbing"]
            proportion_remaining = nvols / total_vols
            if proportion_remaining < remaining_volumes_threshold:
                print(f"Excluding {subject_id} because only {proportion_remaining:.2f} of volumes remain after scrubbing")
            else:
                print(f"Including {subject_id} with {proportion_remaining:.2f} of volumes remaining after scrubbing")
                tmp_fmri_files = dict()
                tmp_fmri_files[(subject_id, "no_smoothing", 0)] = fmri_rest_files
                for csmooth_rest_file in csmooth_rest_files:
                    fwhm = int(re.search(r"fwhm-(\d+)", csmooth_rest_file).group(1))
                    key = (subject_id, "csmooth", fwhm)
                    if key not in tmp_fmri_files:
                        tmp_fmri_files[key] = list()
                    tmp_fmri_files[key].append(csmooth_rest_file)
                for gaussian_rest_file in gaussian_rest_files:
                    fwhm = int(re.search(r"fwhm-(\d+)", gaussian_rest_file).group(1))
                    key = (subject_id, "gaussian", fwhm)
                    if key not in tmp_fmri_files:
                        tmp_fmri_files[key] = list()
                    tmp_fmri_files[key].append(gaussian_rest_file)
                submit_files_dict(tmp_fmri_files)
                submitted_subjects.append(subject_id)
    print(f"Number of submitted subjects: {len(submitted_subjects)}")
    return submitted_subjects


def main():
    clean_dir = "/data2/david.ellis/public/HCPA/myderivatives/cleaned"
    csmooth_dir = "/data2/david.ellis/public/HCPA/myderivatives/csmooth"
    gaussian_dir = "/data2/david.ellis/public/HCPA/myderivatives/gaussian"
    find_subjects(cleaned_fmri_dir=clean_dir, csmooth_dir=csmooth_dir, gaussian_dir=gaussian_dir,
                  remaining_volumes_threshold=0.8)

if __name__ == "__main__":
    main()