"""Do not convert infrastructure failures into tracking loss or policy errors."""


class UncertainExecution(RuntimeError):
    """An action may have executed without a valid ACK. Never auto-retry it."""


def raise_if_infrastructure_error(exc):
    from .deployment_config import DeploymentError

    message = str(exc).lower()
    resource_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cuda error: an illegal memory access",
        "cuda error: device-side assert",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "local grasp service transport failure",
    )
    if (
        isinstance(exc, (MemoryError, UncertainExecution, DeploymentError))
        or type(exc).__name__ == "OutOfMemoryError"
        or any(m in message for m in resource_markers)
    ):
        # Preserve original exception and traceback. Each episode runner writes
        # infrastructure_error.json; no accepted policy score is emitted.
        raise exc
