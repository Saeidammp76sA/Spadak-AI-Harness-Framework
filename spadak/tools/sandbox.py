"""Execution sandbox: wall-clock timeouts + policy knobs."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SandboxPolicy:
    max_timeout_s: float = 60.0        # global cap on any single tool call
    allow_network: bool = True
    allow_filesystem: bool = True
    fs_root: str | None = None         # if set, file tools jailed under this dir


class ToolTimeout(Exception):
    pass


def run_with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    """Run `fn` in a worker thread; raise ToolTimeout after `seconds`.

    Daemon thread => a hung tool cannot wedge the harness; the orphan
    thread dies with the process. For true isolation swap this for
    subprocess/exec-based execution implementing the same interface.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise ToolTimeout(f"tool exceeded {seconds:.1f}s timeout")
    if "error" in box:
        raise box["error"]
    return box.get("value")
