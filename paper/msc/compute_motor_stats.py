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


def compute_stats(_filename, _method, avg_img, average_filename, subject, session, task, zstat_name, fwhm, output_dir,
                  stats, exist_ok=True):
    """Compute voxelwise difference image and scalar stats versus the per-subject average.

    Parameters
    ----------
    _filename : str
        Path to the session-level zstat image for a given smoothing method.
    _method : str
        Smoothing method label (e.g., 'no_smoothing', 'constrained', 'gaussian', 'constrained_nr').
    avg_img : nibabel image
        Subject-level reference average (fwhm=0) for this event.
    average_filename : str
        Path to the subject-level average image, used only for logging.
    exist_ok : bool
        If False, raise if the *stats row* already exists. Currently this flag only
        controls whether we short-circuit when the diff file exists; by default we
        allow overwriting diff images so the script can be re-run safely.
    """
    print(f"Processing {_method} file {os.path.basename(_filename)}")
    print(f"Comparing {average_filename} to {_filename}")

    _img = nib.load(_filename)
    if avg_img.shape != _img.shape:
        raise ValueError(f"Image shapes do not match: {avg_img.shape} vs {_img.shape}")
    diff = np.asarray(avg_img.dataobj) - np.asarray(_img.dataobj)
    diff_filename = os.path.join(output_dir, f"sub-{subject}",
                                 f"ses-{session}",
                                 "func",
                                 f"sub-{subject}_ses-{session}_task-{task}_event-{zstat_name}_fwhm-{fwhm}_{_method}_diff.nii.gz")
    # For robustness, always overwrite the diff image if it exists. This makes the
    # script idempotent and avoids crashes when re-running on partially processed data.
    os.makedirs(os.path.dirname(diff_filename), exist_ok=True)
    print(f"Writing difference image to {diff_filename}")
    diff_img = avg_img.__class__(diff, avg_img.affine)
    diff_img.to_filename(diff_filename)
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    pearson_r = scipy.stats.pearsonr(np.asarray(avg_img.dataobj).ravel(), np.asarray(_img.dataobj).ravel())[0]
    dice = compute_dice(np.asarray(avg_img.dataobj), np.asarray(_img.dataobj))
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
    # then compute stats for each smoothing method and fwhm
    #base_dir = "/data2/david.ellis/public/MSC"
    base_dir = "/media/conda2/public/MSC"
    gaussian_dir = f"{base_dir}/derivatives/fsl_gaussian"
    constrained_dir = f"{base_dir}/derivatives/fsl_constrained"
    constrained_nr_dir = f"{base_dir}/derivatives/fsl_constrained_nr"
    output_dir = f"{base_dir}/derivatives/motor_stats"
    # If True, recompute all intermediate averages and overwrite *.nii.gz outputs
    overwrite = False
    # We now always overwrite diff images inside compute_stats, so exist_ok mainly
    # governs behaviour for future extensions; keep it True to allow re-runs.
    exist_ok = True
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

            # compute stats for no smoothing
            no_smoothing_sessions = list()
            for filename in no_smoothing_filenames:
                session = re.search(r"ses-(func\d+)", filename).group(1)
                no_smoothing_sessions.append(session)
            print(f"No smoothing sessions: {set(no_smoothing_sessions)}")
            for session in set(no_smoothing_sessions):
                no_smoothing_session_filenames = [f for f in no_smoothing_filenames if re.search(r"ses-(func\d+)", f).group(1) == session]
                assert len(no_smoothing_session_filenames) == 2, f"Expected 2 no smoothing files for session {session}, found {len(no_smoothing_session_filenames)}"
                no_smoothing_session_filename = os.path.join(output_dir, f"sub-{subject}",
                                                            f"ses-{session}",
                                                            "func",
                                                            f"sub-{subject}_ses-{session}_task-{task}_event-{zstat_name}_zstat.nii.gz")
                if overwrite or not os.path.exists(no_smoothing_session_filename):
                    average_image(no_smoothing_session_filenames, no_smoothing_session_filename)
                else:
                    print(f"No smoothing average file {no_smoothing_session_filename} exists, skipping")
                compute_stats(no_smoothing_session_filename, "no_smoothing", avg_img, average_filename, subject, session, task, zstat_name, 0, output_dir, stats, exist_ok=exist_ok)

            for fwhm in (3, 6, 9, 12):
                print(f"Processing fwhm {fwhm}")
                # Constrained smoothing
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

                constrained_sessions = list()
                for filename in constrained_filenames:
                    session = re.search(r"ses-(func\d+)", filename).group(1)
                    constrained_sessions.append(session)
                print(f"Constrained sessions: {set(constrained_sessions)}")

                for session in set(constrained_sessions):
                    constrained_session_filenames = [f for f in constrained_filenames if re.search(r"ses-(func\d+)", f).group(1) == session]
                    assert len(constrained_session_filenames) == 2, f"Expected 2 constrained files for session {session}, found {len(constrained_session_filenames)}"
                    constrained_session_filename = os.path.join(output_dir, f"sub-{subject}",
                                                              f"ses-{session}",
                                                              "func",
                                                              f"sub-{subject}_ses-{session}_task-{task}_event-{zstat_name}_fwhm-{fwhm}_constrained_zstat.nii.gz")
                    if overwrite or not os.path.exists(constrained_session_filename):
                        average_image(constrained_session_filenames, constrained_session_filename)
                    else:
                        print(f"Constrained average file {constrained_session_filename} exists, skipping")

                    gaussian_session_filenames = [f.replace(
                        constrained_dir, gaussian_dir).replace(
                        "desc-csmooth", "desc-preproc").replace(
                        f"_fwhm-{fwhm}_bold", f"_bold_fwhm-{fwhm}"
                    ) for f in constrained_session_filenames]
                    for g_filename in gaussian_session_filenames:
                        assert os.path.exists(g_filename), f"Gaussian file {g_filename} does not exist"
                    gaussian_session_filename = os.path.join(output_dir, f"sub-{subject}",
                                                            f"ses-{session}",
                                                            "func",
                                                            f"sub-{subject}_ses-{session}_task-{task}_event-{zstat_name}_fwhm-{fwhm}_gaussian_zstat.nii.gz")
                    if overwrite or not os.path.exists(gaussian_session_filename):
                        average_image(gaussian_session_filenames, gaussian_session_filename)
                    else:
                        print(f"Gaussian average file {gaussian_session_filename} exists, skipping")

                    for _filename, _method in ((constrained_session_filename, "constrained"),
                                               (gaussian_session_filename, "gaussian")):
                        compute_stats(_filename, _method, avg_img, average_filename, subject, session, task,
                                      zstat_name, fwhm, output_dir, stats, exist_ok=exist_ok)

                # Constrained_nr smoothing (non-regularized)
                constrained_nr_wildcard = os.path.join(constrained_nr_dir, f"sub-{subject}/func/*task-{task}*_fwhm-{fwhm}*.feat/stats/zstat{zstat_num}.nii.gz")
                constrained_nr_filenames = sorted(glob.glob(constrained_nr_wildcard))
                print(f"Found {len(constrained_nr_filenames)} constrained_nr files")
                if len(constrained_nr_filenames) != 20:
                    # figure out which files are missing
                    expected_sessions = {f"func{n:02d}" for n in range(1, 11)}
                    expected_runs = {f"run-{n:02d}" for n in range(1, 3)}
                    for expected_session in expected_sessions:
                        for expected_run in expected_runs:
                            expected_filename = os.path.join(constrained_nr_dir, f"sub-{subject}",
                                                             "func",
                                                             f"sub-{subject}_ses-{expected_session}_task-{task}_{expected_run}_space-T1w_desc-csmooth_fwhm-{fwhm}_bold.feat",
                                                             "stats",
                                                             f"zstat{zstat_num}.nii.gz")
                            if expected_filename not in constrained_nr_filenames:
                                print(f"Missing constrained_nr file: {expected_filename}")

                    raise ValueError(f"Expected 20 constrained_nr files for subject {subject}, fwhm {fwhm}, found {len(constrained_nr_filenames)}")

                constrained_nr_sessions = list()
                for filename in constrained_nr_filenames:
                    session = re.search(r"ses-(func\d+)", filename).group(1)
                    constrained_nr_sessions.append(session)
                print(f"Constrained_nr sessions: {set(constrained_nr_sessions)}")

                for session in set(constrained_nr_sessions):
                    constrained_nr_session_filenames = [f for f in constrained_nr_filenames if re.search(r"ses-(func\d+)", f).group(1) == session]
                    assert len(constrained_nr_session_filenames) == 2, f"Expected 2 constrained_nr files for session {session}, found {len(constrained_nr_session_filenames)}"
                    constrained_nr_session_filename = os.path.join(output_dir, f"sub-{subject}",
                                                                   f"ses-{session}",
                                                                   "func",
                                                                   f"sub-{subject}_ses-{session}_task-{task}_event-{zstat_name}_fwhm-{fwhm}_constrained_nr_zstat.nii.gz")
                    if overwrite or not os.path.exists(constrained_nr_session_filename):
                        average_image(constrained_nr_session_filenames, constrained_nr_session_filename)
                    else:
                        print(f"Constrained_nr average file {constrained_nr_session_filename} exists, skipping")

                    for _filename, _method in ((constrained_nr_session_filename, "constrained_nr"),):
                        compute_stats(_filename, _method, avg_img, average_filename, subject, session, task,
                                      zstat_name, fwhm, output_dir, stats, exist_ok=exist_ok)

    stats_df = pd.DataFrame(stats)
    stats_filename = os.path.join(output_dir, "motor_stats.csv")
    print(f"Writing stats to {stats_filename}")
    stats_df.to_csv(stats_filename, index=False)




if __name__ == "__main__":
    main()
