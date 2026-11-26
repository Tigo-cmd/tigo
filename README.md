# Tigo (`.tg`)

A minimal language and durable runtime for safe, typed AI agents—focused on **Developer & DevOps automation** (PR reviews, bug triage, repo tasks, CLI workflows). Zero external dependencies (Python 3 standard library + SQLite).

## Features

- **Typed Prompts & Refinement Constraints (`type`, `where`, `ask`, `"""..."""`)**: Multiline prompts return schema-validated objects (including field-level `where` expressions like `risk_score: float where risk_score >= 0.0 && risk_score <= 1.0`) and auto-repair invalid JSON.
- **Multi-Provider Support & Real Token Cost**: Built-in zero-dep HTTP adapters for **OpenAI** (`OPENAI_API_KEY`), **Anthropic** (`ANTHROPIC_API_KEY`), **Google Gemini** (`GEMINI_API_KEY`), and **local Ollama** (`ollama:<model>`, `$0` cost) with automatic `fallback` failover.
- **Compile-Time Verification (`tigo check`)**: Catches undefined `${variables}` inside prompts, missing agent `budget`/`max_steps`, unknown types, and unguarded I/O before spending a single token.
- **Capability Sandboxing (`allow`)**: File, network, and shell I/O (`fs.read_text`, `fs.write_text`, `http.get`, `sh.run`) are blocked outside `tool` definitions and restricted to explicit glob rules (`allow fs.read("examples/**")`, `allow sh("git diff*")`).
- **Durable Execution & TTY-Aware Approval (`require approval`)**: Every step is journaled to `.tigo/state.db`. Prompts interactively (`[y/N]`) in a terminal, or pauses durably in CI/`--non-interactive` mode so you can resume via `tigo replay <run_id> --approve` without re-paying for earlier LLM calls.
- **Concurrent Fan-Out (`parallel for`)**: Run multiple `ask` or tool calls concurrently with deterministic result ordering and thread-safe SQLite checkpoint replay (see [examples/multi_audit.tg](examples/multi_audit.tg)).
- **Built-in `.tg` Test Runner (`tigo test`) & Multi-file `import`**: Write `test name() { assert(...) }` directly inside `.tg` files.

## Install & Quick Start

```bash
# Optional: install the 'tigo' CLI command (or run './tigo.py' / 'python3 -m tigo' directly)
pip install -e .

# Static check (0 tokens spent)
python3 -m tigo check examples/pr_review.tg

# Run embedded .tg tests
python3 -m tigo test examples/pr_review.tg
python3 -m tigo test examples/triage.tg

# Run a DevOps PR review workflow (offline synthesis by default, or set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)
python3 -m tigo run examples/pr_review.tg review_pr base_ref=HEAD

# Inspect step trace & token cost
python3 -m tigo trace <run_id>

# Resume a paused non-interactive run after approval
python3 -m tigo replay <run_id> --approve

# Start interactive REPL
python3 -m tigo repl

# Run Python unit test suite
python3 test_tigo.py -v
```

## Editor Support
VS Code syntax highlighting for `.tg` files is included in `editors/vscode/`.
