"""Explicit evaluation and local benchmark configuration entry points."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .model_video import make_model_video
from .multi_effector import MultiEffectorRegistry
from .runtime import run_agent
from .transport import APIClient, api_settings


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) - {"environment", "harness"}:
        raise ValueError("Configuration must contain environment and optional harness mappings")
    environment = config.get("environment", {})
    if not isinstance(environment, dict) or environment.get("name") not in (
        "libero-pro",
        "robosuite",
    ):
        raise ValueError("environment.name must be libero-pro or robosuite")
    allowed = (
        {"name", "suite", "task_id", "initial_state", "resolution", "horizon", "instruction_source"}
        if environment["name"] == "libero-pro"
        else {"name", "task", "robot", "seed", "horizon"}
    )
    if set(environment) - allowed:
        raise ValueError("Unknown environment options: " + str(sorted(set(environment) - allowed)))
    options = config.get("harness", {})
    allowed = {
        "max_calls",
        "history_rounds",
        "history_image_rounds",
        "delta_axis",
        "motion_tools",
        "arrival_guard",
        "tracking_backend",
        "tracking_device",
        "region_tools",
        "grasp_provider",
        "interaction_feedback",
        "completion_feedback",
        "move_toward_max_steps",
        "move_toward_stall_steps",
        "move_toward_tolerance_m",
    }
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("Unknown harness options")
    return config


def make_adapter(environment):
    args = dict(environment)
    name = args.pop("name")
    if name == "libero-pro":
        from .libero_adapter import LiberoAdapter

        state = args.pop("initial_state", 0)
        args.setdefault("instruction_source", "environment")
        env = LiberoAdapter(**args)
        env.reset(state)
    else:
        from .robosuite_adapter import RobosuiteAdapter

        env = RobosuiteAdapter(**args)
    return env


def configure_libero(args):
    repo, data, output = args.repo.resolve(), args.data.resolve(), args.output.resolve()
    paths = {
        "benchmark_root": repo / "libero/libero",
        "bddl_files": data / "bddl_files",
        "init_states": data / "init_files",
        "datasets": data,
        "assets": repo / "libero/libero/assets",
    }
    if any(not p.is_dir() for p in paths.values()):
        raise ValueError(
            "Expected LIBERO-Pro checkout/assets and data with bddl_files/ and init_files/"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("x") as stream:
        yaml.safe_dump({k: str(v) for k, v in paths.items()}, stream)
    print(
        "Local configuration created. Set LIBERO_ROOT and LIBERO_CONFIG_PATH in your environment."
    )


def run(args):
    config = read_config(args.config)
    # Check credentials before allocating a simulator. Smoke mode never uses the API.
    settings = None if args.smoke else api_settings()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps({"algorithm": "robo-harness k1", **config}, indent=2)
    )
    env = None
    try:
        env = make_adapter(config["environment"])
        with (output / "steps.jsonl").open("x") as stream:

            def record(obs, action):
                stream.write(
                    json.dumps(
                        {
                            "frame_id": obs["frame_id"],
                            "arms": obs["arms"],
                            "command": np.asarray(action).tolist(),
                        }
                    )
                    + "\n"
                )
                stream.flush()

            env.on_step = record
            initial = env.observe()
            record(initial, [])
            # Evaluator archive only; it is not sent to the policy.
            np.savez_compressed(
                output / "initial_state.npz", state=env.env.sim.get_state().flatten()
            )
            if args.smoke:
                arm = next(iter(initial["arms"]))
                env.move(arm, [0, 0, 0], steps=1)
                result = {
                    "smoke_only": True,
                    "native_steps": env.frame,
                    "cameras": list(initial["vision"]),
                    "arms": list(initial["arms"]),
                }
                (output / "smoke.json").write_text(json.dumps(result, indent=2))
            else:
                base, key, model, options = settings
                result = run_agent(
                    env,
                    output,
                    base,
                    key,
                    model=model,
                    registry_class=MultiEffectorRegistry
                    if config["environment"]["name"] == "robosuite"
                    else None,
                    client_factory=APIClient,
                    request_options=options,
                    **config.get("harness", {}),
                )
                make_model_video(output)
        print(json.dumps(result))
    except Exception as error:
        # Do not serialize exception text, request headers, credentials or arbitrary URLs.
        (output / "failure.json").write_text(
            json.dumps({"status": "incomplete", "error_type": type(error).__name__})
        )
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description="robo-harness k1")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluation = sub.add_parser("run", help="Run one isolated episode")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument(
        "--smoke", action="store_true", help="One native hold action; no model call"
    )
    setup = sub.add_parser("configure-libero", help="Write a new user-local LIBERO-Pro path config")
    setup.add_argument("--repo", type=Path, required=True)
    setup.add_argument("--data", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else configure_libero)(args)


if __name__ == "__main__":
    main()
