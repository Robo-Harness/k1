import json
import shutil

import cv2
import numpy as np
import pytest

from robo_harness.model_video import make_model_video


def test_empty_video(tmp_path):
    assert make_model_video(tmp_path) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_video_includes_auxiliary_and_unlogged_inputs(tmp_path):
    inputs = tmp_path / "model_inputs"
    inputs.mkdir()
    for call, views in [(1, [0, 1, 2]), (2, [0, 1])]:
        for view in views:
            cv2.imwrite(
                str(inputs / f"call_{call:03d}_view_{view}.jpg"),
                np.full((96, 128, 3), 50 + view * 70, dtype=np.uint8),
            )
    (tmp_path / "agent.jsonl").write_text(
        json.dumps({"call_id": 1, "frame_id": 12, "results": []}) + '\n{"incomplete":'
    )
    path = make_model_video(tmp_path)
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    capture.release()
    assert not list(tmp_path.glob("*.avi"))
    assert not (tmp_path / "rollout.mp4").exists()
    assert len(list(inputs.glob("*.jpg"))) == 5
    with pytest.raises(FileExistsError):
        make_model_video(tmp_path)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_single_current_camera_does_not_misclassify_historical_image(tmp_path):
    inputs = tmp_path / "model_inputs"
    inputs.mkdir()
    for view in range(2):
        cv2.imwrite(
            str(inputs / f"call_001_view_{view}.jpg"),
            np.full((96, 128, 3), 50 + view * 100, dtype=np.uint8),
        )
    (tmp_path / "agent.jsonl").write_text(
        json.dumps(
            {
                "call_id": 1,
                "frame_id": 4,
                "results": [],
                "current_camera_count": 1,
                "image_manifest": [
                    {"index": 0, "caption": "CURRENT camera frame 4"},
                    {"index": 1, "caption": "HISTORICAL camera call 0 frame 0"},
                ],
            }
        )
        + "\n"
    )
    capture = cv2.VideoCapture(str(make_model_video(tmp_path)))
    # Separate current panel and historical panel, not a false two-current-camera panel.
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 16
    capture.release()
