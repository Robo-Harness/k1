import numpy as np
import pytest
import robo_harness.regions as module
from test_tracked_points import scene


def setup(monkeypatch):
    obs = scene()
    obs["arms"] = {}
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True
    monkeypatch.setattr(
        module,
        "segment_text",
        lambda rgb, query, dev: [(mask, 0.9)] * (4 if query == "bowl" else 1),
    )
    monkeypatch.setattr(module, "segment", lambda *args: (mask, 0.9))
    return module.Regions(device="cpu", backend="lk"), obs


def test_bowl_then_plate_preserves_every_returned_id(monkeypatch):
    regions, obs = setup(monkeypatch)
    regions.find({"camera": "cam", "query": "bowl"}, obs)
    regions.find({"camera": "cam", "query": "plate"}, obs)
    assert set(regions.entries) == {"S1", "S2", "S3", "S4", "S5"}
    assert regions.summary("S1", obs)["status"] == "current_visible_surface_estimate"
    for i in range(16):
        regions.find({"camera": "cam", "query": f"new{i}"}, obs)
    assert len(regions.entries) == 21
    assert regions.summary("S1", obs)["label_proposed_by_agent"] == "bowl"


def test_stale_memory_is_not_current_geometry(monkeypatch):
    regions, obs = setup(monkeypatch)
    regions.find({"camera": "cam", "query": "plate"}, obs)
    obs["frame_id"] += 1
    old = regions.summary("S1", obs)
    assert old["status"] == "unknown_reinspect"
    assert "surface_reference_world_m" not in old
    assert old["historical_measurement"]["usable_as_current_geometry"] is False
    assert "S1" in regions.entries
    with pytest.raises(ValueError, match="was not created"):
        regions.summary("S999", obs)


def test_box_label_is_not_semantic_confirmation_and_reinspection_retained(monkeypatch):
    regions, obs = setup(monkeypatch)
    first = regions.find({"camera": "cam", "query": "plate"}, obs)["candidates"][0]
    assert first["provenance"]["method"] == "text_prompt"
    args = {
        "camera": "cam",
        "roi": [20, 20, 80, 80],
        "coordinate_space": "pixels",
        "label": "bowl",
        "region_id": "S1",
    }
    new = regions.inspect(args, obs)
    assert new["revision"] == 1
    assert new["provenance"]["label_used_for_segmentation"] is False
    assert new["identity_verified"] is False
    assert regions.entries["S1"]["reinspections"][0]["label_proposed_by_agent"] == "plate"


def test_point_capacity_does_not_delete_memory():
    from robo_harness.tracked_points import TrackedPoints
    from robo_harness.geometry import unproject

    tracks = TrackedPoints(capacity=1, backend="lk", device="cpu")
    obs = scene()
    for i in range(3):
        tracks.start(
            {**unproject(obs["vision"]["cam"], [50, 50]), "id": f"P{i}", "camera": "cam"}, obs
        )
    assert len(tracks.entries) == 3
    tracks.invalidate(["P0"], "occluded", obs["frame_id"])
    old = tracks.context()[0]
    assert old["status"] == "lost_remeasure"
    assert "historical_xyz_world_m_not_current" in old
    assert "xyz_world_m" not in old


def test_region_id_drawn_even_without_reliable_major_axis(monkeypatch):
    regions, obs = setup(monkeypatch)
    regions.find({"camera": "cam", "query": "plate"}, obs)
    regions.entries["S1"]["stats"]["major_axis_reliable"] = False
    labels = []
    original = module.cv2.putText

    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(module.cv2, "putText", capture)
    regions.overlay(obs["vision"]["cam"]["color"], "cam", obs)
    assert "S1" in labels
