import numpy as np
import pytest

from robo_harness.regions import Regions, geometry, project
from test_tracked_points import scene


def test_region_geometry_uses_only_selected_depth():
    camera = scene()["vision"]["cam"]
    mask = np.zeros((100, 100), bool)
    mask[40:60, 20:80] = True
    stats, xyz, _ = geometry(camera, mask)
    assert stats["plane_rmse_m"] < 1e-8
    assert stats["surface_normal_toward_camera_world"][2] == -1
    assert stats["major_axis_ratio"] > 2
    uv, depth, inside = project(camera, xyz)
    assert inside.all() and np.allclose(depth, 1)
    assert np.all((uv[:, 0] >= 20) & (uv[:, 0] < 80))


def test_missing_depth_is_not_fabricated():
    camera = scene()["vision"]["cam"]
    camera["depth"][:] = np.nan
    with pytest.raises(ValueError, match="Insufficient"):
        geometry(camera, np.ones((100, 100), bool))


def test_mask_remains_visible_when_geometry_unknown(monkeypatch):
    import robo_harness.regions as module

    mask = np.zeros((100, 100), bool)
    mask[25:75, 25:75] = True
    monkeypatch.setattr(module, "segment", lambda *args: (mask, 0.8))
    obs = scene()
    obs["arms"] = {}
    obs["vision"]["cam"]["depth"][:] = np.nan
    manager = Regions(backend="lk", device="cpu")
    result = manager.inspect(
        {
            "camera": "cam",
            "box": {"left": 20, "top": 20, "right": 80, "bottom": 80},
            "coordinate_space": "pixels",
            "label": "patch",
        },
        obs,
    )
    assert result["status"] == "mask_visible_geometry_unknown"
    assert "surface_reference_world_m" not in result
    manager.overlay(obs["vision"]["cam"]["color"], "cam", obs)


def test_regions_refresh_and_loss(monkeypatch):
    import robo_harness.regions as module

    mask = np.zeros((100, 100), bool)
    mask[25:75, 25:75] = True
    monkeypatch.setattr(module, "segment", lambda *args: (mask, 0.8))
    obs = scene()
    obs["arms"] = {"arm": {"xyz_world_m": [0, 0, 0], "axes_world": np.eye(3).tolist()}}
    manager = Regions(backend="lk", device="cpu")
    result = manager.inspect(
        {"camera": "cam", "roi": [20, 20, 80, 80], "coordinate_space": "pixels", "label": "patch"},
        obs,
    )
    assert result["relative_to_robot"]["arm"]["distance_m"] > 0.9
    raw = obs["vision"]["cam"]["color"].copy()
    manager.overlay(raw, "cam", obs)
    np.testing.assert_equal(raw, obs["vision"]["cam"]["color"])
    # Stale geometry cannot be queried until refreshed.
    obs["frame_id"] = 1
    assert manager.summary("S1", obs)["status"] == "unknown_reinspect"
    manager.update(obs, refresh=True)
    assert manager.summary("S1", obs)["status"] == "current_visible_surface_estimate"
    obs["frame_id"] = 20
    manager.update(obs, refresh=True)
    assert manager.summary("S1", obs)["status"] == "unknown_reinspect"


def test_cross_camera_depth_disagreement_is_not_match(monkeypatch):
    import robo_harness.regions as module

    mask = np.zeros((100, 100), bool)
    mask[25:75, 25:75] = True
    monkeypatch.setattr(module, "segment", lambda *args: (mask, 0.8))
    obs = scene()
    obs["arms"] = {}
    obs["vision"]["other"] = scene()["vision"]["cam"]
    obs["vision"]["other"]["depth"] *= 0.5
    manager = Regions(backend="lk", device="cpu")
    manager.inspect(
        {"camera": "cam", "roi": [20, 20, 80, 80], "coordinate_space": "pixels", "label": "patch"},
        obs,
    )
    result = manager.cross_view("S1", "other", obs)
    assert result["projected_in_image_samples"] > 0
    assert result["depth_consistent_samples"] == 0
    obs["vision"]["other"]["depth"] *= 2
    manager.refresh_cross_views(obs)
    assert len(manager.entries["S1"]["cross"]["pixels"]) > 0
    assert manager.entries["S1"]["cross"]["frame"] == obs["frame_id"]
