# tests/test_grid_resolution_effect_smoke.py
import numpy as np
import nibabel as nib

import paper.simulations.grid_resolution_effect as gre


def test_run_grid_resolution_experiment_smoke(tmp_path, monkeypatch):
    # Fake graph/labelmap/meta
    class DummyGraph:
        pass

    def fake_create_graph(*args, **kwargs):
        # Create a small 5x5x5 grid and return simple linear-chain edges between
        # successive flattened indices. This provides a connected graph for
        # identify_connected_components to operate on in tests.
        shape = (5, 5, 5)
        nvox = int(np.prod(shape))
        # edges between i and i+1 for i in [0, nvox-2]
        edge_src = np.arange(0, nvox - 1, dtype=int)
        edge_dst = np.arange(1, nvox, dtype=int)
        edge_distances = np.ones(edge_src.shape, dtype=float)
        return edge_src, edge_dst, edge_distances

    def fake_smooth_images(*args, **kwargs):
        # Identity smoothing: return input image as-is
        if "images" in kwargs:
            return kwargs["images"]
        return args[1]

    def fake_save_labelmap(*args, **kwargs):
        return None

    # Monkeypatch nib.load to return a small fake NIfTI-like object
    class FakeHeader:
        def get_zooms(self):
            return (1.0, 1.0, 1.0)

    class FakeImg:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.affine = np.eye(4)
            self.header = FakeHeader()

        def get_fdata(self):
            return self._data

        @property
        def shape(self):
            return self._data.shape

    def fake_load(fname, **kwargs):
        # Return a 5x5x5 volume with label 1035 at [2,2,2]
        data = np.zeros((5, 5, 5), dtype=np.int32)
        data[2, 2, 2] = 1035
        return FakeImg(data)

    monkeypatch.setattr(gre, "create_graph", fake_create_graph)
    monkeypatch.setattr(gre, "smooth_images", fake_smooth_images)
    monkeypatch.setattr(gre, "save_labelmap", fake_save_labelmap)
    monkeypatch.setattr(nib, "load", fake_load)
    # Avoid calling nilearn.resample_img in tests; use identity resample helper
    def fake_resample_data_to_affine(data, target_affine, original_affine, interpolation="continuous"):
        return np.asarray(data)
    monkeypatch.setattr(gre, "resample_data_to_affine", fake_resample_data_to_affine)
    # Avoid calling csmooth.smooth.process_mask which uses nilearn.resample_to_img
    def fake_process_mask(mask_file, reference_image, mask_dilation=0):
        return np.ones(reference_image.shape[:3], dtype=bool)
    monkeypatch.setattr(gre, "process_mask", fake_process_mask)
    # Avoid running heavy smoothing implementation in tests
    def fake_apply_estimated_gaussian_smoothing(signal_data, *args, **kwargs):
        # return flattened signal (the code expects a flattened array later)
        arr = np.asarray(signal_data)
        return arr.ravel()
    monkeypatch.setattr(gre, "apply_estimated_gaussian_smoothing", fake_apply_estimated_gaussian_smoothing)

    rows = gre.run_grid_resolution_experiment(
        aparc_file="aparc.nii.gz",
        brain_mask_file="mask.nii.gz",
        pial_l_file="pialL.surf.gii",
        pial_r_file="pialR.surf.gii",
        white_l_file="whiteL.surf.gii",
        white_r_file="whiteR.surf.gii",
        ground_truth_parcellation_label=1035,
        fwhm=6.0,
        gt_center_ijk=(2, 2, 2),
        timepoints=20,
        signal_amplitude=5.0,
        noise_std=1.0,
        output_dir=str(tmp_path),
        voxel_sizes=[1.0, 2.0],
        random_seed=0,
    )

    assert len(rows) == 2
    for r in rows:
        assert "dice" in r
        # Dice may be NaN in some degenerate monkeypatched paths; if finite it should be in [0,1]
        if np.isfinite(r["dice"]):
            assert 0.0 <= r["dice"] <= 1.0
        assert r["n_voxels"] == 125
