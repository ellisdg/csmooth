import os
import glob
import nibabel as nib
import re
import numpy as np
import pandas as pd
import scipy.stats


def average_image(filenames, output_filename):
    image1 = nib.load(filenames[0])
    image = image1.__class__(np.stack([np.asarray(nib.load(f).dataobj) for f in filenames]).mean(axis=0),
                             image1.affine)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print("Writing averaged image to {}".format(output_filename))
    image.to_filename(output_filename)
    return output_filename


def compute_dice(array1, array2, threshold=3.1):
    """
    Compute the Dice coefficient between two binary arrays.
    :param array1: first binary array
    :param array2: second binary array
    :param threshold: threshold to apply to the arrays before computing the Dice coefficient
    :return: Dice coefficient
    """
    bin1 = array1 > threshold
    bin2 = array2 > threshold
    intersection = np.logical_and(bin1, bin2).sum()
    size1 = bin1.sum()
    size2 = bin2.sum()
    if size1 + size2 == 0:
        return 1.0
    dice = 2.0 * intersection / (size1 + size2)
    return dice


def compute_stats(_filename, _method, avg_img, average_filename, subject, session, task, run, zstat_name, fwhm,
                  output_dir, stats, exist_ok=True):
    print(f"Processing {_method} file {os.path.basename(_filename)}")
    print(f"Comparing {average_filename} to {_filename}")

    _img = nib.load(_filename)
    if avg_img.shape != _img.shape:
        raise ValueError(f"Image shapes do not match: {avg_img.shape} vs {_img.shape}")
    diff = np.asarray(avg_img.dataobj) - np.asarray(_img.dataobj)
    diff_filename = os.path.join(output_dir, f"sub-{subject}",
                                 f"ses-{session}",
                                 "func",
                                 f"sub-{subject}_ses-{session}_task-{task}_run-{run}_event-{zstat_name}_fwhm-{fwhm}_{_method}_diff.nii.gz")
    if os.path.exists(diff_filename) and not exist_ok:
        raise FileExistsError(f"Diff file {diff_filename} already exists and exist_ok is False")
    os.makedirs(os.path.dirname(diff_filename), exist_ok=True)
    print(f"Writing difference image to {diff_filename}")
    diff_img = avg_img.__class__(diff, avg_img.affine)
    diff_img.to_filename(diff_filename)
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    dice = compute_dice(np.asarray(avg_img.dataobj), np.asarray(_img.dataobj))
    pearson_r = scipy.stats.pearsonr(np.asarray(avg_img.dataobj).ravel(), np.asarray(_img.dataobj).ravel())[0]
    stats.append({
        "subject": subject,
        "session": session,
        "task": task,
        "event": zstat_name,
        "fwhm": fwhm,
        "method": _method,
        "mse": mse,
        "mae": mae,
        "dice": dice,
        "pearson_r": pearson_r,
        "n_voxels": np.prod(avg_img.shape)
    })


def main():
    # find all the motor task zstat images and average them per subject with no smoothing
    # those will be the ground truth for the motor task
    #base_dir = "/data2/david.ellis/public/MSC"
    base_dir = "/media/conda2/public/MSC"
    gaussian_dir = f"{base_dir}/derivatives/fsl_gaussian"
    constrained_dir = f"{base_dir}/derivatives/fsl_constrained"
    output_dir = f"{base_dir}/derivatives/motor_stats"
    overwrite = False
    exist_ok = True  # will raise an error if the diff image already exists
    os.makedirs(output_dir, exist_ok=True)
    task = "motor"
    stats = list()
    for zstat_num, zstat_name in ((1, "leftfoot"), (2, "lefthand"), (3, "rightfoot"), (4, "righthand"), (5, "tongue")):
        print(f"Processing zstat {zstat_num} ({zstat_name})")
        subject_dirs = sorted(glob.glob(os.path.join(gaussian_dir, "sub-*")))
        for subject_dir in subject_dirs:
            subject = re.search(r"sub-(MSC\d+)", subject_dir).group(1)
            print(f"Processing subject dir {subject_dir}, subject {subject}")
            # find all the gaussian smoothed images with fwhm=0
            wildcard = os.path.join(subject_dir, f"func/*task-{task}*_fwhm-0*.feat/stats/zstat{zstat_num}.nii.gz")
            no_smoothing_filenames = sorted(glob.glob(wildcard))
            print(f"Found {len(no_smoothing_filenames)} no smoothing files")
            assert len(no_smoothing_filenames) == 20, f"Expected 20 no smoothing files, found {len(no_smoothing_filenames)}"
            average_filename = os.path.join(output_dir, os.path.dirname(subject_dir), "func",
                                            f"sub-{subject}_task-{task}_event-{zstat_name}_zstat.nii.gz")
            if overwrite or not os.path.exists(average_filename):
                average_image(no_smoothing_filenames, average_filename)
            else:
                print(f"Average file {average_filename} exists, skipping")
            avg_img = nib.load(average_filename)


            for no_smoothing_session_filename in no_smoothing_filenames:
                run = re.search(r"run-(\d+)", no_smoothing_session_filename).group(1)
                session = re.search(r"ses-(func\d+)", no_smoothing_session_filename).group(1)
                compute_stats(
                    _filename=no_smoothing_session_filename,
                    _method="no_smoothing",
                    avg_img=avg_img,
                    average_filename=average_filename,
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    zstat_name=zstat_name,
                    fwhm=0,
                    output_dir=output_dir,
                    stats=stats,
                    exist_ok=exist_ok
                )

            for fwhm in (3, 6, 9, 12):
                print(f"Processing fwhm {fwhm}")
                constrained_wildcard = os.path.join(constrained_dir, f"sub-{subject}/func/*task-{task}*_fwhm-{fwhm}*.feat/stats/zstat{zstat_num}.nii.gz")
                constrained_filenames = sorted(glob.glob(constrained_wildcard))
                print(f"Found {len(constrained_filenames)} constrained files")
                if len(constrained_filenames) != 20:
                    # figure out which files are missing
                    # there should be 10 sessions with 2 runs each
                    expected_sessions = {f"func{n:02d}" for n in range(1, 11)}
                    expected_runs = {f"run-{n:02d}" for n in range(1, 3)}
                    for expected_session in expected_sessions:
                        for expected_run in expected_runs:
                            expected_filename = os.path.join(constrained_dir, f"sub-{subject}",
                                                             "func",
                                                             f"sub-{subject}_ses-{expected_session}_task-{task}_{expected_run}_space-T1w_desc-csmooth_fwhm-{fwhm}_bold.feat",
                                                             "stats",
                                                             f"zstat{zstat_num}.nii.gz")
                            if expected_filename not in constrained_filenames:
                                print(f"Missing constrained file: {expected_filename}")

                    raise ValueError(f"Expected 20 constrained files for subject {subject}, fwhm {fwhm}, found {len(constrained_filenames)}")

                for constrained_filename in constrained_filenames:
                    gaussian_filename = constrained_filename.replace(
                        constrained_dir, gaussian_dir).replace(
                        "desc-csmooth", "desc-preproc").replace(
                        f"_fwhm-{fwhm}_bold", f"_bold_fwhm-{fwhm}"
                    )

                    for _filename, _method in ((constrained_filename, "constrained"),
                                               (gaussian_filename, "gaussian")):
                        session = re.search(r"ses-(func\d+)", _filename).group(1)
                        run = re.search(r"run-(\d+)", _filename).group(1)
                        compute_stats(_filename, _method, avg_img, average_filename, subject, session, task,
                                      run=run,
                                      zstat_name=zstat_name,
                                      fwhm=fwhm,
                                      output_dir=output_dir,
                                      stats=stats,
                                      exist_ok=exist_ok)



    stats_df = pd.DataFrame(stats)
    stats_filename = os.path.join(output_dir, "motor_stats_single_run.csv")
    print(f"Writing stats to {stats_filename}")
    stats_df.to_csv(stats_filename, index=False)





if __name__ == "__main__":
    main()
