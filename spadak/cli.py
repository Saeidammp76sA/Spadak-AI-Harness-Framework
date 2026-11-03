"""spadak CLI — inspect the harness, scaffold agents, run demos & evals.

Stdlib-only argparse; `spadak init <dir>` scaffolds a ready-to-run project.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spadak",
                                     description="Spadak agent harness CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="show framework components & versions")

    demo = sub.add_parser("demo", help="run the offline weather/calculator demo")

    ev = sub.add_parser("eval", help="run the bundled evaluation suite")

    initp = sub.add_parser("init", help="scaffold a new Spadak project")
    initp.add_argument("directory", nargs="?", default="my-spadak-app")

    args = parser.parse_args(argv)

    if args.cmd == "info":
        return _info()
    if args.cmd == "demo":
        return _demo()
    if args.cmd == "eval":
        return _eval()
    if args.cmd == "init":
        return _init(Path(args.directory))
    parser.print_help()
    return 0


def _info() -> int:
    import spadak
    print(f"spadak-harness v{spadak.__version__}")
    print("sub-frameworks:")
    for mod in ["core", "llm", "memory", "tools", "orchestration", "evaluation", "safety"]:
        print(f"  - spadak.{mod}")
    return 0


def _demo() -> int:
    from examples.weather_demo import run as demo_run
    return 0 if demo_run(verbose=True) else 1


def _eval() -> int:
    from examples.eval_suite_demo import run as eval_run
    report = eval_run(verbose=True)
    print(json.dumps(report.summary(), indent=2))
    return 0 if report.pass_rate >= 0.99 else 1


_TEMPLATE_MAIN = '''"""Scaffolded by `spadak init`."""
from spadak import AgentConfig, HarnessBuilder
from spadak.llm import MockLLM, LLMResponse
from spadak.tools.builtin import register_builtins
from spadak.safety import PIIRedactionGuard, PromptInjectionGuard
from spadak.memory import LongTermMemory


def build_harness():
    # Swap MockLLM for OpenAIProvider()/AnthropicProvider() when going live.
    llm = MockLLM(script=[LLMResponse(text="Hello! I am your scaffolded agent.")])
    tools = register_builtins()
    return (HarnessBuilder()
            .llm(llm)
            .memory(LongTermMemory())
            .guardrails(PromptInjectionGuard(), PIIRedactionGuard())
            .agent(AgentConfig(name="assistant",
                               system_prompt="You are a helpful assistant."))
            .build())


if __name__ == "__main__":
    out = build_harness().run("hi")
    print(out.answer)
'''


def _init(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "main.py").write_text(_TEMPLATE_MAIN, encoding="utf-8")
    (directory / "README.md").write_text(
        f"# {directory.name}\n\nRun: `python main.py`\n", encoding="utf-8")
    print(f"scaffolded Spadak project at {directory.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
