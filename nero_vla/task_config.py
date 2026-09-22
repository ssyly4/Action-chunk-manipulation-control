"""Load a validated policy task preset for the shell launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import tomllib


ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODES = {"dual", "right"}
WARMUPS = {"bimanual", "right"}


def load_task(path: Path) -> dict:
    """Read and validate one trusted repository task preset."""
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    task = config.get("task")
    if not isinstance(task, dict):
        raise ValueError("task config must contain a [task] table")
    for key in ("name", "mode", "runtime", "warmup"):
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError(f"task.{key} must be a non-empty string")
    if task["mode"] not in MODES:
        raise ValueError(f"task.mode must be one of {sorted(MODES)}")
    if task["warmup"] not in WARMUPS:
        raise ValueError(f"task.warmup must be one of {sorted(WARMUPS)}")

    arguments = config.get("arguments", {}).get("values")
    if not isinstance(arguments, list) or not arguments:
        raise ValueError("arguments.values must be a non-empty array")
    if not all(isinstance(value, (str, int, float)) for value in arguments):
        raise ValueError("arguments.values may contain only strings and numbers")

    for table_name in ("environment", "defaults"):
        table = config.get(table_name, {})
        if not isinstance(table, dict):
            raise ValueError(f"{table_name} must be a table")
        for name, value in table.items():
            if not ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"{table_name}.{name} must be scalar")
    return config


def _shell_value(value: object) -> str:
    if isinstance(value, bool):
        value = "1" if value else "0"
    return shlex.quote(str(value))


def emit_shell(config: dict) -> str:
    """Return shell declarations without evaluating arbitrary TOML content."""
    task = config["task"]
    lines = [
        f"export NERO_TASK_NAME={_shell_value(task['name'])}",
        f"export NERO_TASK_MODE={_shell_value(task['mode'])}",
        f"export NERO_TASK_RUNTIME={_shell_value(task['runtime'])}",
        f"export NERO_TASK_WARMUP={_shell_value(task['warmup'])}",
    ]
    for name, value in config.get("environment", {}).items():
        lines.append(f"export {name}={_shell_value(value)}")
    for name, value in config.get("defaults", {}).items():
        if name not in os.environ:
            lines.append(f"export {name}={_shell_value(value)}")
    arguments = " ".join(
        _shell_value(value) for value in config["arguments"]["values"]
    )
    lines.append(f"declare -a NERO_TASK_ARGS=({arguments})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(emit_shell(load_task(args.config)))


if __name__ == "__main__":
    main()
