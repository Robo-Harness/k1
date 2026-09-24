"""Pose schematics on original views. No new camera or synthesized geometry."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .regions import project


def candidate_panel(result, obs, arm, camera_name):
    camera = obs["vision"][camera_name]
    state = obs["arms"][arm]
    pads = np.asarray(state["geometry"]["finger_pad_centers_local_m"])
    approach_local = np.asarray(state["geometry"]["axes_local"]["approach"])
    panels = []
    for candidate in result["candidates"]:
        rgb = camera["color"].copy()
        rotation = Rotation.from_quat(candidate["target_quaternion_xyzw"]).as_matrix()
        tcp = np.asarray(candidate["candidate_tcp_world_m"])
        tips = pads @ rotation.T + tcp
        backs = tips - (rotation @ approach_local) * 0.03
        points = np.vstack([tips, backs, tcp, np.asarray(candidate["pregrasp_tcp_world_m"])])
        uv, _, inside = project(camera, points)
        pixels = np.rint(uv).astype(int)
        for i, j in [(0, 2), (1, 3), (2, 3)]:
            if inside[i] and inside[j]:
                cv2.line(rgb, tuple(pixels[i]), tuple(pixels[j]), (255, 70, 190), 2)
        for i in (0, 1):
            if inside[i]:
                cv2.circle(rgb, tuple(pixels[i]), 3, (255, 230, 0), -1)
        if inside[4] and inside[5]:
            cv2.arrowedLine(rgb, tuple(pixels[5]), tuple(pixels[4]), (80, 210, 255), 1)
        cv2.putText(
            rgb, candidate["candidate_id"] + " HYPOTHESIS", (5, 18), 0, 0.45, (255, 255, 0), 1
        )
        panels.append(rgb)
    return np.concatenate(panels, axis=1) if panels else None
