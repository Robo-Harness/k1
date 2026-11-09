"""Bounded RPC over inherited pipes to our own trusted simulator child only.

No listener and no untrusted pickle inputs. A response timeout is NOT evidence
that an action stopped. Poison the channel, terminate the owned child on close,
and never retransmit a request whose execution may have begun.
"""

from copy import deepcopy
import io
import os
import pickle
import select
import struct
import subprocess
import time

from .resource_errors import UncertainExecution

MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _wait(pipe, deadline, writing=False):
    remaining = deadline - time.monotonic()
    ready = select.select(
        [] if writing else [pipe], [pipe] if writing else [], [], max(0, remaining)
    )
    if remaining <= 0 or not ready[1 if writing else 0]:
        raise TimeoutError("Private RPC message deadline exceeded")


def send(pipe, value, timeout=600):
    data = pickle.dumps(value, protocol=4)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Oversized private RPC")
    pending = memoryview(struct.pack("!Q", len(data)) + data)
    if isinstance(pipe, io.BytesIO):  # deterministic in-memory tests only
        pipe.write(pending)
        return
    deadline = time.monotonic() + timeout
    fd = pipe.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        while pending:
            _wait(pipe, deadline, writing=True)
            try:
                count = os.write(fd, pending[: 256 * 1024])
            except BlockingIOError:
                continue
            if not count:
                raise EOFError("Private RPC pipe write failed")
            pending = pending[count:]
    finally:
        os.set_blocking(fd, blocking)


def receive(pipe, timeout=600):
    deadline = time.monotonic() + timeout

    def exact(size):
        parts = []
        while size:
            if isinstance(pipe, io.BytesIO):
                part = pipe.read(size)
            else:
                _wait(pipe, deadline)
                part = os.read(pipe.fileno(), size)
            if not part:
                raise EOFError("Private RPC pipe closed")
            parts.append(part)
            size -= len(part)
        return b"".join(parts)

    size = struct.unpack("!Q", exact(8))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Oversized private RPC")
    return pickle.loads(exact(size))


def _frame(value):
    if type(value) is not int or value < 0:
        raise ValueError("Malformed RPC frame counter")
    return value


def call(adapter, method, args, timeout=600):
    if getattr(adapter, "_closed", False) and method != "close":
        raise UncertainExecution("Native RPC adapter is closed; no further requests")
    if getattr(adapter, "_broken", False):
        raise UncertainExecution("Native RPC channel unusable; no request retransmission")
    if getattr(adapter, "_rpc_in_flight", False):
        raise UncertainExecution(
            "Concurrent or callback-nested RPC is not supported; do not resend"
        )
    request_id = getattr(adapter, "_request_counter", 0) + 1
    adapter._request_counter = request_id
    start = adapter.frame
    events = 0
    deadline = time.monotonic() + timeout
    recoverable_error = None
    adapter._rpc_in_flight = True
    try:
        send(adapter.child.stdin, dict(request_id=request_id, method=method, args=args), timeout)
        while True:
            response = receive(adapter.child.stdout, deadline - time.monotonic())
            if (
                not isinstance(response, dict)
                or type(response.get("request_id")) is not int
                or response["request_id"] != request_id
            ):
                raise ValueError("Malformed or mismatched RPC response ID")
            if response.get("event") != "step":
                if "event" in response:
                    raise ValueError("Unknown RPC event")
                break
            frame = _frame(response["observation"]["frame_id"])
            if frame <= adapter.frame:
                raise ValueError("RPC step frame did not advance")
            adapter.frame = frame
            events += 1
            if adapter.on_step is not None:
                adapter.on_step(response["observation"], response["action"])
        frame = _frame(response.get("frame"))
        if frame < adapter.frame:
            raise ValueError("RPC final frame moved backwards")
        adapter.frame = frame
        if response.get("ok") is not True:
            if (
                response.get("ok") is False
                and response.get("before_execution") is True
                and response.get("type") in ("ValueError", "KeyError")
                and frame == start
                and events == 0
            ):
                recoverable_error = str(response.get("error", "Invalid native arguments"))
            else:
                raise UncertainExecution(
                    "Native command failed after dispatch: " + str(response.get("error"))
                )
        else:
            if "ended" in response:
                if type(response["ended"]) is not bool:
                    raise ValueError("Malformed native termination flag")
                adapter._ended = response["ended"]
            return response["value"]
    except BaseException as exc:
        adapter._broken = True
        adapter._cached = None
        if isinstance(exc, (KeyboardInterrupt, SystemExit, UncertainExecution)):
            raise
        raise UncertainExecution(
            "Native response unavailable, malformed or callback failed; do not retry motion"
        ) from exc
    finally:
        adapter._rpc_in_flight = False
    # Only a verified pre-dispatch rejection is a recoverable model/tool error.
    raise ValueError(recoverable_error)


def close_child(adapter, grace=15):
    """Idempotent cleanup of this adapter's child, never an action retry."""
    if getattr(adapter, "_closed", False):
        return
    adapter._closed = True
    adapter._cached = None
    try:
        if adapter.child.poll() is None and not adapter._broken:
            adapter._rpc("close", {})
    finally:
        # An uncertain command may still be running. Don't send another message
        # or wait a long normal-close grace before stopping the owned child.
        if adapter._broken and adapter.child.poll() is None:
            adapter.child.terminate()
        try:
            adapter.child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            adapter.child.terminate()
            try:
                adapter.child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                adapter.child.kill()
                adapter.child.wait()
        finally:
            for stream in (adapter.child.stdin, adapter.child.stdout, adapter.log):
                if stream is not None:
                    stream.close()


def require_observable(adapter):
    """Cached observations do not stay current after closure or channel failure."""
    if getattr(adapter, "_closed", False) or getattr(adapter, "_broken", False):
        raise UncertainExecution(
            "Native RPC adapter closed or uncertain; cached observation is not current"
        )


def execution_marker(adapter, native_frame=None, physics_steps=None):
    """Private server evidence, never a policy observation or task-success check.

    Attempt counter advances BEFORE entering the native actuator. It catches
    failures before a native frame/physics counter has had time to advance.
    """
    if adapter is None or not hasattr(adapter, "_execution_attempts"):
        return None
    if native_frame is None:
        native_frame = getattr(getattr(adapter, "env", None), "take_action_cnt", adapter.frame)
    if physics_steps is None:
        physics_steps = getattr(adapter, "physics_steps", None)
    return dict(
        attempts=int(adapter._execution_attempts),
        frame=int(adapter.frame),
        native_frame=int(native_frame),
        physics_steps=physics_steps,
        grippers=deepcopy(adapter.grippers),
        uncertain=bool(getattr(adapter, "_uncertain", False)),
    )


def rejected_before_execution(error, before, after):
    return (
        isinstance(error, (ValueError, KeyError))
        and before is not None
        and before == after
        and not before["uncertain"]
    )
