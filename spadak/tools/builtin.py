"""A handful of safe, useful built-in tools + a weather demo tool."""
from __future__ import annotations

import ast
import json
import operator
import os
from pathlib import Path
from typing import Any

from spadak.tools.registry import ToolRegistry
from spadak.tools.tool import tool

# ---------------------------------------------------------------------------
# calculator: AST-validated expression evaluation (no eval() abuse)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@tool(read_only=True, cache_result=True)
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression safely (numbers, + - * / // % **)."""
    tree = ast.parse(expression.strip(), mode="eval")
    return float(_safe_eval(tree))


# ---------------------------------------------------------------------------
# filesystem tools (jailed to workspace root when configured)
# ---------------------------------------------------------------------------
@tool(read_only=True)
def read_file(path: str) -> str:
    """Read a text file and return its contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_text(encoding="utf-8", errors="replace")[:100_000]


@tool(dangerous=True)
def write_file(path: str, content: str) -> str:
    """Write text content to a file (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool(read_only=True)
def list_dir(path: str = ".") -> list[str]:
    """List entries of a directory."""
    return sorted(os.listdir(path))


# ---------------------------------------------------------------------------
# network (guarded; used by the weather demo with canned data offline)
# ---------------------------------------------------------------------------
@tool(read_only=True, timeout_s=15.0, cache_result=True)
def http_get(url: str) -> str:
    """Fetch a URL and return the first 20k chars of the response body."""
    import urllib.request
    if not url.startswith(("https://", "http://")):
        raise ValueError("only http(s) URLs allowed")
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode(errors="replace")[:20_000]


@tool(read_only=True, cache_result=True)
def get_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city. Offline demo returns deterministic data."""
    # Deterministic pseudo-data keyed on city name so demos/evals are stable.
    seed = sum(ord(c) for c in city.lower())
    temp_c = 5 + seed % 30
    conditions = ["sunny", "cloudy", "rain", "snow", "windy"]
    return {"city": city, "temp_c": temp_c,
            "condition": conditions[seed % len(conditions)],
            "humidity_pct": 20 + seed % 60}


def register_builtins(registry: ToolRegistry | None = None) -> ToolRegistry:
    reg = registry or ToolRegistry()
    reg.register_all([calculator, read_file, write_file, list_dir,
                      http_get, get_weather])
    return reg
