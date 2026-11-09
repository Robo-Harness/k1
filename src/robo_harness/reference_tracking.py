"""Image-only registration audit, not an identity detector or auto-reacquisition.

Kept opt-in until real replay qualification. A failed reference match is UNKNOWN,
not proof of lost tracking: viewpoint/appearance changes can explain it.
"""

import cv2
import numpy as np


class ReferenceTrackingAudit:
    def __init__(self, rgb, pixels):
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.points = np.asarray(pixels, np.float32).reshape(-1, 1, 2).copy()

    def compare(self, rgb, streamed_pixels):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        streamed = np.asarray(streamed_pixels, float).reshape(-1, 2)
        if gray.shape != self.gray.shape or len(streamed) != len(self.points):
            raise ValueError("Registration must use the same original image layout and point set")
        unknown = [dict(status="reference_unobservable", identity_verified=False) for _ in streamed]
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21), maxLevel=3
        )
        if nxt is None:
            return unknown
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, nxt, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            return unknown
        for i, (old, new, estimate) in enumerate(zip(self.points[:, 0], nxt[:, 0], streamed)):
            if not ok[i, 0] or not back_ok[i, 0] or not np.isfinite([*new, *estimate]).all():
                continue
            fb = float(np.linalg.norm(back[i, 0] - old))
            if fb > 0.8:
                continue
            # Do not accept padded boundary patches or textureless areas as a match.
            h, w = gray.shape
            if not all(6 <= u < w - 6 and 6 <= v < h - 6 for u, v in (old, new)):
                continue
            p = cv2.getRectSubPix(self.gray, (11, 11), tuple(old)).astype(float)
            q = cv2.getRectSubPix(gray, (11, 11), tuple(new)).astype(float)
            if min(p.std(), q.std()) < 8:
                continue
            corr = float(np.corrcoef(p.ravel(), q.ravel())[0, 1])
            if not np.isfinite(corr) or corr < 0.8:
                continue
            disagreement = float(np.linalg.norm(new - estimate))
            unknown[i] = dict(
                status="reference_disagreement" if disagreement > 3 else "reference_consistent",
                reference_pixel=new.tolist(),
                stream_reference_distance_px=disagreement,
                reference_forward_backward_error_px=fb,
                patch_correlation=corr,
                identity_verified=False,
            )
        return unknown
