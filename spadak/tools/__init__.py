"""spadak.tools — typed tool registry sub-framework.

Tools are declared with plain Python functions + type hints; JSON schema
is generated automatically (no hand-written schemas). Includes execution
sandboxing: timeouts, retries and result caching.
"""
from spadak.tools.tool import Tool, tool
from spadak.tools.registry import ToolRegistry
from spadak.tools.builtin import register_builtins, calculator, read_file, write_file, http_get
from spadak.tools.sandbox import SandboxPolicy, run_with_timeout

__all__ = [
    "Tool", "tool", "ToolRegistry",
    "register_builtins", "calculator", "read_file", "write_file", "http_get",
    "SandboxPolicy", "run_with_timeout",
]
