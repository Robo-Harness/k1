"""Encode archived model-input images directly to MP4, without an AVI copy."""

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


def make_model_video(output):
    output = Path(output)
    inputs = output / "model_inputs"
    groups = {}
    for path in sorted(inputs.glob("call_*_view_*.jpg")):
        match = re.fullmatch(r"call_(\d+)_view_(\d+)\.jpg", path.name)
        if match:
            call, view = map(int, match.groups())
            groups.setdefault(call, {})[view] = path
    if not groups:
        return None
    rows = {}
    log = output / "agent.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["call_id"]] = row
    destination = output / "model_view.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    command = [
        "ffmpeg",
        "-n",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "768x424",
        "-r",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def fit(image, width, height):
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        )
        canvas = np.zeros((height, width, 3), np.uint8)
        y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return canvas

    try:
        for call, views in sorted(groups.items()):
            row = rows.get(call, {})
            title = f"call {call} frame {row.get('frame_id', '?')}: " + ",".join(
                r["name"] for r in row.get("results", [])
            )
            camera_count = row.get("current_camera_count", 2)
            captions = {item["index"]: item["caption"] for item in row.get("image_manifest", [])}
            primary = [cv2.imread(str(views[v])) for v in range(camera_count) if v in views]
            primary = [im for im in primary if im is not None]
            panels = []
            if primary:
                panels.append(
                    (
                        "CURRENT CAMERAS" if "current_camera_count" in row else "MAIN + WRIST",
                        np.concatenate(
                            [fit(im, 768 // len(primary), 384) for im in primary], axis=1
                        ),
                    )
                )
            for view, path in sorted(views.items()):
                if view >= camera_count:
                    im = cv2.imread(str(path))
                    if im is not None:
                        panels.append(
                            (
                                f"INPUT {view}: "
                                + captions.get(view, "auxiliary / historical")[:105],
                                fit(im, 768, 384),
                            )
                        )
            for label, panel in panels:
                frame = np.zeros((424, 768, 3), np.uint8)
                frame[40:] = panel
                cv2.putText(frame, title[:110], (5, 16), 0, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, label, (5, 33), 0, 0.4, (100, 255, 255), 1)
                for _ in range(8):
                    process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"model_view encoding failed: {code}")
    return destination
