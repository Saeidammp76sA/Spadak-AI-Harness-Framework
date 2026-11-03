"""@tool decorator: derive JSON-schema from signatures & docstrings."""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

_JSON_TYPE = {
    str: "string", int: "integer", float: "number", bool: "boolean",
    list: "array", dict: "object",
    typing.Optional[str]: "string",
}


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)   # JSON schema
    dangerous: bool = False        # guardrails can require confirmation for these
    read_only: bool = False
    timeout_s: float = 30.0
    cache_result: bool = False     # memoize per-run by (name,args)

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}

    def call(self, args: dict[str, Any]) -> Any:
        sig = inspect.signature(self.func)
        coerced = _coerce(args, sig)
        return self.func(**coerced)


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None,
         description: str | None = None, dangerous: bool = False,
         read_only: bool = False, timeout_s: float = 30.0,
         cache_result: bool = False) -> Any:
    """Usable bare (@tool) or configured (@tool(dangerous=True))."""
    def wrap(fn: Callable[..., Any]) -> Tool:
        t = Tool(
            name=name or fn.__name__,
            description=description or _docstring_first_line(fn),
            func=fn,
            parameters=_schema_from_signature(fn),
            dangerous=dangerous, read_only=read_only,
            timeout_s=timeout_s, cache_result=cache_result,
        )
        fn.spadak_tool = t  # type: ignore[attr-defined]
        return t
    if func is not None:
        return wrap(func)
    return wrap


def _docstring_first_line(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn)
    return doc.split("\n")[0].strip() if doc else fn.__name__


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 — forward refs etc.
        hints = {}
    props: dict[str, Any] = {}
    required: list[str] = []
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        hint = hints.get(pname, str)
        base = typing.get_origin(hint) or hint
        jtype = _JSON_TYPE.get(base, "string")
        prop: dict[str, Any] = {"type": jtype}
        if base in (list, tuple) and typing.get_args(hint):
            item = typing.get_args(hint)[0]
            prop["items"] = {"type": _JSON_TYPE.get(item, "string")}
        if p.default is not inspect.Parameter.empty:
            prop["default"] = p.default
        else:
            required.append(pname)
        props[pname] = prop
    return {"type": "object", "properties": props,
            "required": required, "additionalProperties": False}


def _coerce(args: dict[str, Any], sig: inspect.Signature) -> dict[str, Any]:
    """Best-effort cast of LLM-supplied values to annotated types."""
    out: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name not in args:
            continue
        val = args[name]
        ann = p.annotation
        try:
            if ann is int and isinstance(val, str):
                val = int(float(val))
            elif ann is float and isinstance(val, str):
                val = float(val)
            elif ann is bool and isinstance(val, str):
                val = val.lower() in ("true", "1", "yes")
        except (ValueError, TypeError):
            pass
        out[name] = val
    return out
