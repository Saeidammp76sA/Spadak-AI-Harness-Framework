"""Ready-made guardrails. Compose them per-agent via the Harness builder."""
from __future__ import annotations

import re
from typing import Any, Callable

from spadak.safety.guardrail import Action, Guardrail, Verdict

# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(the\s+)?(system\s+)?prompt",
    r"you\s+are\s+now\s+(dan|jailbroken|in\s+developer\s+mode)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"<\s*/?\s*(system|assistant)\s*>",
]


class PromptInjectionGuard(Guardrail):
    """Blocks obvious prompt-injection phrasing in user input & tool outputs."""
    name = "prompt_injection"

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self.patterns = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
        self.patterns += [re.compile(p, re.IGNORECASE) for p in (extra_patterns or [])]

    def check_input(self, agent: Any, user_input: str) -> Verdict:
        for pat in self.patterns:
            if pat.search(user_input):
                return Verdict(Action.BLOCK, reason=f"injection pattern: {pat.pattern!r}")
        return Verdict.allow()

    def check_output(self, agent: Any, state: Any, message: Any) -> Verdict:
        # tool results smuggled into outputs are a classic injection vector
        for res in message.tool_results:
            text = str(res.output)
            for pat in self.patterns:
                if pat.search(text):
                    return Verdict(Action.REWRITE,
                                   reason="sanitized injected tool output",
                                   replaced_text=message.text +
                                   "\n[warning: tool output contained injection-like text and was withheld]")
        return Verdict.allow()


class ToolAllowListGuard(Guardrail):
    """Hard-enforces the agent's tool allow-list even if specs leaked."""
    name = "tool_allowlist"

    def check_tool_call(self, agent: Any, state: Any, call: Any) -> Verdict:
        allowed = agent.allowed_tools()
        if allowed is not None and call.name not in allowed:
            return Verdict(Action.BLOCK,
                           reason=f"tool {call.name!r} not in allow-list {allowed}")
        return Verdict.allow()


class DangerousToolConfirmGuard(Guardrail):
    """Tools marked `dangerous` require an explicit confirmation callback."""
    name = "dangerous_confirm"

    def __init__(self, confirmer: Callable[[str, dict], bool] | None = None) -> None:
        # default: deny-all; production code passes an approver (UI/webhook)
        self.confirmer = confirmer or (lambda name, args: False)

    def check_tool_call(self, agent: Any, state: Any, call: Any) -> Verdict:
        registry = agent.tools
        tool_obj = registry.get(call.name) if registry else None
        if tool_obj is not None and tool_obj.dangerous:
            if not self.confirmer(call.name, call.arguments):
                return Verdict(Action.BLOCK,
                               reason=f"dangerous tool {call.name!r} not confirmed")
        return Verdict.allow()


# ---------------------------------------------------------------------------
_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_ir": re.compile(r"(?<!\d)0\d{9}(?!\d)|(?<!\d)\+98\s?9\d{9}(?!\d)"),
    "card": re.compile(r"(?<!\d)\d{16}(?!\d)"),
    "national_id_ir": re.compile(r"(?<!\d)\d{10}(?!\d)"),
}


class PIIRedactionGuard(Guardrail):
    """Redacts PII from model output before it reaches the user."""
    name = "pii_redaction"

    def check_output(self, agent: Any, state: Any, message: Any) -> Verdict:
        text = message.text
        if not text:
            return Verdict.allow()
        redacted = text
        for label, pat in _PII_PATTERNS.items():
            redacted = pat.sub(f"[REDACTED:{label}]", redacted)
        if redacted != text:
            return Verdict(Action.REWRITE, reason="PII redacted",
                           replaced_text=redacted)
        return Verdict.allow()


class MaxToolCallsGuard(Guardrail):
    """Runaway-agent protection: caps total tool calls per run."""
    name = "max_tool_calls"

    def __init__(self, limit: int = 25) -> None:
        self.limit = limit

    def check_tool_call(self, agent: Any, state: Any, call: Any) -> Verdict:
        used = sum(len(m.tool_calls) for m in state.messages
                   if m.role.value == "assistant")
        if used > self.limit:
            return Verdict(Action.BLOCK,
                           reason=f"exceeded {self.limit} tool calls this run")
        return Verdict.allow()


class OutputLengthGuard(Guardrail):
    """Truncate absurdly long outputs (protects downstream token budgets)."""
    name = "output_length"

    def __init__(self, max_chars: int = 20_000) -> None:
        self.max_chars = max_chars

    def check_output(self, agent: Any, state: Any, message: Any) -> Verdict:
        text = message.text
        if text and len(text) > self.max_chars:
            return Verdict(Action.REWRITE, reason="output truncated",
                           replaced_text=text[: self.max_chars] + "\n…[truncated]")
        return Verdict.allow()
