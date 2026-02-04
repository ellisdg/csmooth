import pytest

import paper.sensory.archive.plot_stat_maps as psm


def test_plot_multiple_stat_maps_uses_per_map_thresholds(monkeypatch):
    """Ensure per-map thresholds from `stat_map_thresholds` are forwarded to
    `plot_mri_with_contours` for each map."""
    called = []

    def fake_plot_mri_with_contours(*args, **kwargs):
        # capture the stat_map_threshold passed for each map
        called.append(kwargs.get("stat_map_threshold"))
        return None

    monkeypatch.setattr(psm, "plot_mri_with_contours", fake_plot_mri_with_contours)

    stat_map_fnames = ["m1.nii", "m2.nii", "m3.nii"]
    thresholds = [1.1, 2.2, 3.3]

    # Call the multi-map plot with explicit per-map thresholds
    fig = psm.plot_multiple_stat_maps(
        mri_fname="t1.nii",
        surfaces=[],
        stat_map_fnames=stat_map_fnames,
        slices=[0],
        stat_map_cmap="hot",
        stat_map_alpha=0.9,
        stat_map_threshold=0.0,
        stat_map_thresholds=thresholds,
        stat_map_vmin=0.0,
        stat_map_vmax=5.0,
        colorbar=False,
        crop_map_fname=None,
        show=False,
    )

    # The fake function should have been called once per map
    assert called == thresholds


def test_plot_multiple_stat_maps_fallback_single_threshold(monkeypatch):
    """When stat_map_thresholds is None the single stat_map_threshold should be
    used for every map."""
    called = []

    def fake_plot_mri_with_contours(*args, **kwargs):
        called.append(kwargs.get("stat_map_threshold"))
        return None

    monkeypatch.setattr(psm, "plot_mri_with_contours", fake_plot_mri_with_contours)

    stat_map_fnames = ["a.nii", "b.nii"]
    base_thr = 4.5

    fig = psm.plot_multiple_stat_maps(
        mri_fname="t1.nii",
        surfaces=[],
        stat_map_fnames=stat_map_fnames,
        slices=[0],
        stat_map_threshold=base_thr,
        stat_map_thresholds=None,
        show=False,
    )

    assert called == [base_thr, base_thr]


def test_integration_constrained_vs_gauss_plot_thresholds(monkeypatch, tmp_path):
    """Run a lightweight integration-style call to the single-region experiment and
    assert the combined multi-map plot receives the per-map thresholds that match
    the metrics written to the results row.

    We monkeypatch smoothing and plotting to avoid heavy computation and file I/O.
    """
    import numpy as np
    import nibabel as nib
    import paper.simulations.constrained_vs_gauss as csg

    # Create tiny synthetic images
    shape = (8, 8, 8)
    affine = np.eye(4)
    aparc_arr = np.zeros(shape, dtype=np.int32)
    # small cubic label region
    aparc_arr[2:5, 2:5, 2:5] = 2000
    aparc_img = nib.Nifti1Image(aparc_arr, affine)

    brain_mask_arr = np.ones(shape, dtype=np.uint8)
    brain_mask_image = nib.Nifti1Image(brain_mask_arr, affine)

    t1_img = nib.Nifti1Image(np.zeros(shape, dtype=float), affine)

    mask_array = brain_mask_arr.astype(bool)
    gm_mask = np.zeros(shape, dtype=bool)
    gm_mask[aparc_arr == 2000] = True
    wm_mask = np.zeros(shape, dtype=bool)
    other_mask = mask_array & ~gm_mask & ~wm_mask

    # Minimal dummy graph/connectivity inputs (not used because we monkeypatch smoothing)
    edge_src = np.array([], dtype=int)
    edge_dst = np.array([], dtype=int)
    edge_distances = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    sorted_labels = np.array([], dtype=int)
    unique_nodes = np.array([], dtype=int)

    # Monkeypatch computationally-heavy routines to be no-ops or identity.
    # Return the provided signal_data kwarg if present, otherwise first positional arg.
    monkeypatch.setattr(
        csg,
        "apply_constrained_smoothing",
        lambda *args, **kwargs: kwargs.get("signal_data") if "signal_data" in kwargs else (args[0] if args else None),
    )
    monkeypatch.setattr(csg, "smooth_img", lambda img, fwhm: nib.Nifti1Image(np.asarray(img.get_fdata()), img.affine))

    captured_multi = []

    class _FakeFig:
        def savefig(self, *a, **k):
            return None

    def _fake_multi(*args, **kwargs):
        # Capture relevant kwargs (thresholds and vmin/vmax)
        captured_multi.append({
            "stat_map_thresholds": kwargs.get("stat_map_thresholds"),
            "stat_map_vmin": kwargs.get("stat_map_vmin"),
            "stat_map_vmax": kwargs.get("stat_map_vmax"),
        })
        return _FakeFig()

    # Monkeypatch both the combined-plot and single-map plotting used later
    monkeypatch.setattr(csg, "plot_multiple_stat_maps", _fake_multi)
    monkeypatch.setattr(csg, "plot_mri_with_contours", lambda *a, **kw: _FakeFig())

    # plt.close is called in the code; make it a no-op so our fake figure
    # objects don't cause a TypeError when closed.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "close", lambda fig: None)

    outdir = str(tmp_path / "out")
    rows, _ = csg.run_constrained_vs_gauss_experiment_single_region(
        aparc_img=aparc_img,
        brain_mask_image=brain_mask_image,
        t1_img=t1_img,
        surface_files=["", "", "", ""],
        label=2000,
        center_ijk=(3, 3, 3),
        fwhm_list=[6.0],
        amplitude=1.0,
        noise_std=0.1,
        output_dir=outdir,
        mask_array=mask_array,
        gm_mask=gm_mask,
        wm_mask=wm_mask,
        other_mask=other_mask,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_distances=edge_distances,
        labels=labels,
        sorted_labels=sorted_labels,
        unique_nodes=unique_nodes,
        optimal_taus_by_fwhm=None,
        random_seed=0,
        pred_threshold_quantile=None,
        overwrite_volumes=True,
        save_images=False,
        plot_outputs=True,
    )

    assert rows, "Expected at least one result row"
    assert len(captured_multi) == 1, "Expected combined multi-map plot to be called once"

    multi_kwargs = captured_multi[0]
    sm_thrs = multi_kwargs.get("stat_map_thresholds")
    assert sm_thrs is not None and len(sm_thrs) == 4

    row = rows[0]
    # indices 1..3 correspond to raw, constrained, gaussian thresholds
    assert pytest.approx(sm_thrs[1]) == row["raw_pred_threshold"]
    assert pytest.approx(sm_thrs[2]) == row["constrained_pred_threshold"]
    assert pytest.approx(sm_thrs[3]) == row["gaussian_pred_threshold"]


def test_stat_map_vmin_vmax_passthrough(monkeypatch):
    """Ensure that `stat_map_vmin` and `stat_map_vmax` passed into
    `plot_multiple_stat_maps` are forwarded unchanged to the underlying
    `plot_mri_with_contours` calls (i.e., shared colormap behaviour).
    """
    import paper.sensory.archive.plot_stat_maps as psm

    captured = []

    class _FakeFig:
        def savefig(self, *a, **k):
            return None

    def _fake_plot_mri_with_contours(*args, **kwargs):
        captured.append((kwargs.get("stat_map_vmin"), kwargs.get("stat_map_vmax")))
        return _FakeFig()

    monkeypatch.setattr(psm, "plot_mri_with_contours", _fake_plot_mri_with_contours)

    # Call with explicit vmin/vmax; two maps should result in two calls
    psm.plot_multiple_stat_maps(
        mri_fname="t1.nii",
        surfaces=[],
        stat_map_fnames=["a.nii", "b.nii"],
        slices=[0],
        stat_map_vmin=0.5,
        stat_map_vmax=3.5,
        show=False,
    )

    assert captured == [(0.5, 3.5), (0.5, 3.5)]
