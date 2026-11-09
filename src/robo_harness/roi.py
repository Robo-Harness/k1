"""Explicit original-image rectangles; legacy xyxy remains readable."""

import numpy as np

KEYS = ("x_min", "y_min", "x_max", "y_max")


def schema():
    return dict(
        type="object",
        additionalProperties=False,
        description="ORIGINAL image pixels: x horizontal/right, y vertical/down. Not yxyx, normalized or crop coordinates.",
        properties={
            k: dict(type="number", description=k + " in original-image pixels") for k in KEYS
        },
        required=list(KEYS),
    )


def rectangle(value, shape):
    if isinstance(value, dict):
        if set(value) != set(KEYS):
            raise ValueError("ROI requires x_min,y_min,x_max,y_max")
        value = [value[k] for k in KEYS]
    a = np.asarray(value, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("Expected finite original-image ROI [x_min,y_min,x_max,y_max]")
    x0, y0, x1, y1 = a.astype(int)
    h, w = shape[:2]
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError(
            "ROI outside original image: x horizontal, y vertical. No automatic swapping."
        )
    return [int(x0), int(y0), int(x1), int(y1)]


def receipt(value):
    return dict(zip(KEYS, map(int, value)))
