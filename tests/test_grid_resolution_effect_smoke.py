# tests/test_grid_resolution_effect_smoke.py
import numpy as np

import paper.simulations.grid_resolution_effect as gre


def test_run_grid_resolution_experiment_smoke(tmp_path, monkeypatch):
    # Fake graph/labelmap/meta
    class DummyGraph:
        pass

    def fake_create_graph(**kwargs):
        # Create a small "grid" with a known active region label 1035
        labelmap = np.zeros((5, 5, 5), dtype=np.int32)
        labelmap[2, 2, 2] = 1035
        meta = {"affine": np.eye(4), "shape": labelmap.shape}
        return DummyGraph(), labelmap, meta

    def fake_smooth_images(*args, **kwargs):
        # Identity smoothing: return input image as-is
        if "images" in kwargs:
            return kwargs["images"]
        return args[1]

    def fake_save_labelmap(*args, **kwargs):
        return None

    monkeypatch.setattr(gre, "create_graph", fake_create_graph)
    monkeypatch.setattr(gre, "smooth_images", fake_smooth_images)
    monkeypatch.setattr(gre, "save_labelmap", fake_save_labelmap)

    rows = gre.run_grid_resolution_experiment(
        aparc_file="aparc.nii.gz",
        brain_mask_file="mask.nii.gz",
        pial_l_file="pialL.surf.gii",
        pial_r_file="pialR.surf.gii",
        white_l_file="whiteL.surf.gii",
        white_r_file="whiteR.surf.gii",
        ground_truth_parcellation_label=1035,
        fwhm=6.0,
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
        assert 0.0 <= r["dice"] <= 1.0
        assert r["n_voxels"] == 125
