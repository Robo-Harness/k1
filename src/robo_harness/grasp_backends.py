"""Optional point-cloud grasp backend. No task, simulator, or action dependencies.

Not activated by importing this module. Caller explicitly supplies endpoint and
robot/provider frame calibration. Returned poses are hypotheses, not certificates.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def validate_predictions(poses, scores):
    poses, scores = np.asarray(poses, float), np.asarray(scores, float).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
        raise ValueError("Expected matching (N,4,4) poses and N scores")
    if len(poses) > 256 or not np.isfinite(poses).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite or excessive grasp predictions")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous transform")
    matrices = poses[:, :3, :3]
    if not np.allclose(
        matrices.transpose(0, 2, 1) @ matrices, np.eye(3), atol=1e-3
    ) or not np.allclose(np.linalg.det(matrices), 1, atol=1e-3):
        raise ValueError("Grasp rotations must be proper orthonormal matrices")
    return poses, scores


def request_graspgen(points, endpoint, timeout_ms=30000, num_grasps=100, topk=24):
    """Only the observed metric object point cloud leaves this process."""
    import msgpack
    import msgpack_numpy
    import zmq

    xyz = np.asarray(points, dtype=np.float32)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not 30 <= len(xyz) <= 65536
        or not np.isfinite(xyz).all()
    ):
        raise ValueError("Need 30..65536 finite visible metric XYZ points")
    if not 1 <= topk <= 256 or not topk <= num_grasps <= 512:
        raise ValueError("Invalid grasp sampling budget")
    if not endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("Use an explicitly configured local grasp service")
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(endpoint)
        payload = {
            "action": "infer",
            "point_cloud": xyz,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk,
            "min_grasps": min(10, topk),
            "max_tries": 2,
            "grasp_threshold": -1.0,
            "remove_outliers": True,
        }
        socket.send(msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True))
        response = msgpack.unpackb(socket.recv(), object_hook=msgpack_numpy.decode, raw=False)
        if not isinstance(response, dict) or response.get("error"):
            raise RuntimeError(
                "Grasp provider failed: "
                + str(response.get("error") if isinstance(response, dict) else "invalid response")
            )
        return validate_predictions(response.get("grasps"), response.get("confidences"))
    except zmq.ZMQError as exc:
        raise RuntimeError("Local grasp service transport failure: " + type(exc).__name__) from exc
    finally:
        socket.close()


def contact_aligned_pose(provider_pose, robot_state, provider_axes, contact_reference_local_m):
    """Map provider semantic axes and nominal contact reference to robot pad midpoint.

    This is an explicit contact-alignment hypothesis, NOT exact hand-mesh
    registration. Contact offset must come from the provider's documented model,
    not from a task-specific position or a simulator target oracle.
    """
    pose, _ = validate_predictions(np.asarray(provider_pose)[None], [0])
    pose = pose[0]
    geometry = robot_state["geometry"]
    names = ("approach", "closing", "lateral")
    local = np.column_stack([geometry["axes_local"][n] for n in names])
    provider = np.column_stack([provider_axes[n] for n in names])
    for basis in (local, provider):
        if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-3):
            raise ValueError("Need calibrated orthonormal semantic axes")
    target = pose[:3, :3] @ provider @ local.T
    if np.linalg.det(target) < 0.99:
        raise ValueError("Robot/provider semantic conventions have incompatible handedness")
    flipped = pose[:3, :3] @ provider @ np.diag([1, -1, -1]) @ local.T
    current = np.asarray(robot_state["axes_world"])
    if (
        Rotation.from_matrix(flipped @ current.T).magnitude()
        < Rotation.from_matrix(target @ current.T).magnitude()
    ):
        target = flipped
    reference = np.asarray(contact_reference_local_m, float)
    pads = np.asarray(geometry["finger_pad_centers_local_m"], float)
    if (
        reference.shape != (3,)
        or pads.shape != (2, 3)
        or not np.isfinite(reference).all()
        or not np.isfinite(pads).all()
    ):
        raise ValueError("Need finite provider contact reference and robot pad centers")
    contact = pose[:3, 3] + pose[:3, :3] @ reference
    tcp = contact - target @ pads.mean(axis=0)
    return {
        "target_quaternion_xyzw": Rotation.from_matrix(target).as_quat().tolist(),
        "candidate_tcp_world_m": tcp.tolist(),
        "contact_reference_world_m": contact.tolist(),
        "approach_world": (target @ local[:, 0]).tolist(),
        "closing_world_unsigned": (target @ local[:, 1]).tolist(),
        "conversion": "nominal_provider_contact_aligned_to_robot_pad_midpoint",
        "collision_certified": False,
        "attachment_verified": False,
    }


def learned_candidates(
    manager, region_id, arm, obs, endpoint, provider_axes, contact_reference_local_m
):
    """Optional provider adapted to measured robot geometry; does not execute."""
    from .grasp_geometry import visibility_score

    summary = manager.summary(region_id, obs)
    if summary["status"] != "current_visible_surface_estimate":
        raise ValueError("Need a current measured region")
    state = obs["arms"][arm]
    if state.get("gripper_opening_m", 0) < 0.025:
        raise ValueError("Open gripper before evaluating open-jaw candidates")
    xyz = np.asarray(manager.entries[region_id]["xyz"])
    poses, confidence = request_graspgen(xyz, endpoint)
    probes = np.asarray(state["geometry"]["sweep_probes_local_m"])
    proposals = []
    for pose, probability in zip(poses, confidence):
        item = contact_aligned_pose(pose, state, provider_axes, contact_reference_local_m)
        rotation = Rotation.from_quat(item["target_quaternion_xyzw"]).as_matrix()
        target = np.asarray(item["candidate_tcp_world_m"])
        pre = target - 0.07 * np.asarray(item["approach_world"])
        samples = np.concatenate(
            [probes @ rotation.T + pre + f * (target - pre) for f in (0, 0.25, 0.5, 0.75, 1)]
        )
        geometry = visibility_score(samples, obs, arm)
        rank = (
            geometry["observed_free_probe_fraction"]
            - 2 * geometry["near_surface_probe_fraction"]
            - 0.25 * geometry["unknown_probe_fraction"]
        )
        item.update(
            pregrasp_tcp_world_m=pre.tolist(),
            visibility_score=rank,
            learned_confidence_not_success_probability=float(probability),
            rotation_from_current_deg=float(
                np.degrees(
                    Rotation.from_matrix(rotation @ np.asarray(state["axes_world"]).T).magnitude()
                )
            ),
            jaw_fit_verified=False,
            **geometry,
        )
        proposals.append(item)
    # Keep independent scores interpretable: safety visibility first, learned
    # confidence only as tie-breaker. No task-specific positional filters.
    proposals.sort(
        key=lambda p: (p["visibility_score"], p["learned_confidence_not_success_probability"]),
        reverse=True,
    )
    selected = []
    for item in proposals:
        q = Rotation.from_quat(item["target_quaternion_xyzw"])
        if any(
            np.linalg.norm(np.asarray(item["candidate_tcp_world_m"]) - x["candidate_tcp_world_m"])
            < 0.015
            and np.degrees((q * Rotation.from_quat(x["target_quaternion_xyzw"]).inv()).magnitude())
            < 20
            for x in selected
        ):
            continue
        item["candidate_id"] = f"{region_id}-G{len(selected) + 1}"
        selected.append(item)
        if len(selected) == 3:
            break
    return {
        "region_id": region_id,
        "frame_id": obs["frame_id"],
        "provider": "graspgen_visible_point_cloud",
        "candidates": selected,
        "meaning": "Learned POSE HYPOTHESES from visible target RGB-D points only. Model confidence is not calibrated success probability. Provider nominal contact reference is aligned to robot finger-pad midpoint; hand meshes and jaw opening are NOT exactly registered. Inspect candidate contacts and jaw fit in original views. Sparse visibility is not collision checking, IK or path planning. Execute bounded primitives with clearance, settle closure, and verify target co-motion before transport. No whole-object width rejection: a local rim grasp need not enclose the whole object.",
    }
