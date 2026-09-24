"""Evaluator-only mutable MuJoCo model state, never exposed to policy tools."""

import numpy as np

FIELDS = (
    "body_pos",
    "body_quat",
    "geom_pos",
    "geom_quat",
    "geom_rgba",
    "site_pos",
    "site_quat",
    "site_rgba",
    "cam_pos",
    "cam_quat",
    "geom_friction",
)


def capture(sim):
    return {key: np.asarray(getattr(sim.model, key)).copy() for key in FIELDS}


def restore(sim, snapshot):
    for key, array in snapshot.items():
        target = getattr(sim.model, key)
        if target.shape != array.shape:
            raise ValueError(f"Model snapshot shape mismatch: {key}")
        target[:] = array
    sim.forward()
