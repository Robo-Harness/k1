"""Minimal plugin contract. Observations contain sensors, not hidden task state."""

from typing import Protocol


class EnvironmentAdapter(Protocol):
    frame: int
    horizon: int
    delta_axis: (
        float  # Configured by MotionController; transport envelope radius sqrt(3)*delta_axis.
    )

    def reset(self, initial_state_id: int = 0) -> dict: ...
    def observe(self) -> dict: ...
    def move(
        self, arm: str, delta: list, rotation_deg: list, gripper: float | None, steps: int
    ) -> dict: ...
    # Optional capability. Required only to expose the long-range move_toward tool.
    def move_to_position(
        self,
        arm: str,
        target_xyz_world_m: list,
        target_quaternion_xyzw: list,
        gripper: float | None,
        steps: int = 1,
    ) -> dict: ...
    def success(self) -> bool: ...  # evaluator only, never placed in actor observation
    def close(self) -> None: ...
