"""Explicit optional dependency paths; no machine-specific defaults or secrets."""

import os
from pathlib import Path


class DeploymentError(RuntimeError):
    """A required external dependency has not been configured."""


def path(name, required=True):
    names = {
        "sam3_repo": "SAM3_REPO",
        "sam3_checkpoint": "SAM3_CHECKPOINT",
        "tapnext_repo": "TAPNEXT_REPO",
        "tapnext_checkpoint": "TAPNEXT_CHECKPOINT",
        "libero_native": "LIBERO_ROOT",
    }
    variable = names.get(name, "ROBO_HARNESS_" + name.upper())
    value = os.environ.get(variable) or os.environ.get("ROBO_HARNESS_" + name.upper())
    if not value:
        if required:
            raise DeploymentError("Configure " + variable + " before enabling this dependency")
        return None
    return Path(value).expanduser().resolve()
