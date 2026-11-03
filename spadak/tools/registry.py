"""ToolRegistry — the tool manager injected into agents.

Responsibilities:
  * register/lookup tools, generate provider-agnostic specs,
  * enforce per-agent allow-lists,
  * apply sandbox policy (timeout / caching) around execution,
  * validate arguments against the declared JSON schema (required keys).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from spadak.core.messages import ToolCall
from spadak.tools.sandbox import SandboxPolicy, run_with_timeout
from spadak.tools.tool import Tool


class ToolRegistry:
    def __init__(self, sandbox: SandboxPolicy | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache: dict[tuple[str, str], Any] = {}
        self.sandbox = sandbox or SandboxPolicy()

    # ---------- registration ----------
    def register(self, obj: Tool | Callable[..., Any]) -> "ToolRegistry":
        t = getattr(obj, "spadak_tool", None)
        if t is None:
            if isinstance(obj, Tool):
                t = obj
            else:
                raise TypeError(f"{obj!r} is not a @tool-decorated function")
        self._tools[t.name] = t
        return self

    def register_all(self, tools: Iterable[Tool | Callable[..., Any]]) -> "ToolRegistry":
        for x in tools:
            self.register(x)
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    # ---------- specs ----------
    def specs(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for name, t in self._tools.items():
            if allowed is not None and name not in allowed:
                continue
            out.append(t.spec())
        return out

    # ---------- execution ----------
    def execute(self, call: ToolCall) -> Any:
        t = self._tools.get(call.name)
        if t is None:
            raise KeyError(f"unknown tool: {call.name!r}")
        _validate_args(t, call.arguments)

        cache_key: tuple[str, str] | None = None
        if t.cache_result:
            cache_key = (t.name, json.dumps(call.arguments, sort_keys=True, default=str))
            if cache_key in self._cache:
                return {"cached": True, **({"value": self._cache[cache_key]}
                        if not isinstance(self._cache[cache_key], dict)
                        else self._cache[cache_key])}

        result = run_with_timeout(lambda: t.call(call.arguments),
                                  seconds=min(t.timeout_s, self.sandbox.max_timeout_s))
        if cache_key is not None:
            self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()


def _validate_args(t: Tool, args: dict[str, Any]) -> None:
    required = t.parameters.get("required", [])
    missing = [k for k in required if k not in args]
    if missing:
        raise ValueError(f"tool {t.name!r} missing required args: {missing}")
    props = t.parameters.get("properties", {})
    unknown = [k for k in args if k not in props]
    if unknown:
        # be forgiving: LLMs add junk fields; drop them rather than fail hard
        for k in unknown:
            args.pop(k)
