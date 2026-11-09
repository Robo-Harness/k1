"""Execution-failure boundary for simulators hosted in the actor process.

Parameter validation belongs before native_step. Once native execution starts,
an exception may follow a partial physical action even without a frame ACK.
Never turn that into a recoverable tool error or automatically replay it.
"""

from functools import wraps

from .resource_errors import UncertainExecution


def require_open(adapter):
    if getattr(adapter, "_closed", False):
        raise UncertainExecution("Simulator is closed; construct a new episode explicitly")
    if getattr(adapter, "native_error", False):
        raise UncertainExecution("Previous native execution failed; do not retry in this episode")


def require_usable(adapter):
    require_open(adapter)
    if getattr(adapter, "_native_busy", False):
        raise UncertainExecution("Nested motion from a native callback is not supported")


def motion_entry(function):
    @wraps(function)
    def checked(adapter, *args, **kwargs):
        require_usable(adapter)
        attempts = getattr(adapter, "_execution_attempts", 0)
        try:
            return function(adapter, *args, **kwargs)
        except BaseException as exc:
            # A later solve, observation or receipt can fail between native
            # ticks. Earlier ACKs still mean this whole tool call is not safe
            # to repeat, even when the failing statement was not native_step.
            if getattr(adapter, "_execution_attempts", 0) != attempts:
                adapter.native_error = True
                adapter._cached = None
                if isinstance(exc, Exception) and not isinstance(exc, UncertainExecution):
                    raise UncertainExecution(
                        "Motion failed after native execution began; "
                        "stop the unscored episode, do not resend the tool call."
                    ) from exc
            raise

    return checked


def native_step(function):
    @wraps(function)
    def guarded(adapter, *args, **kwargs):
        require_usable(adapter)
        adapter._native_busy = True
        adapter._execution_attempts = getattr(adapter, "_execution_attempts", 0) + 1
        try:
            return function(adapter, *args, **kwargs)
        except BaseException as exc:
            adapter.native_error = True
            adapter._cached = None
            if not isinstance(exc, Exception) or isinstance(exc, UncertainExecution):
                raise
            raise UncertainExecution(
                "Native action or its observation/recording callback failed; execution may "
                "be partial. Stop this unscored episode; do not resend the action."
            ) from exc
        finally:
            adapter._native_busy = False

    return guarded


def close(adapter):
    """Release this adapter's environment once; a failed close may be retried."""
    if getattr(adapter, "_closed", False):
        return
    adapter.on_step = None
    env = getattr(adapter, "env", None)
    if env is not None:
        env.close()
    adapter._closed = True
    adapter._cached = None
