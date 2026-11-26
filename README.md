# Tigo (`.tg`)

A minimal language and durable runtime for safe, typed AI agents. Zero external dependencies (Python 3 standard library + SQLite).

## Features

- **Typed Prompts (`type` + `ask`)**: LLM outputs are automatically constrained to your declared schema and self-repaired on validation failure.
- **Compile-Time Verification (`tigo.py check`)**: Catches undefined `${variables}` inside prompt strings, missing agent budgets/step limits, unknown types, and unguarded I/O before spending a single token.
- **Capability Sandboxing (`allow`)**: File and network I/O is blocked outside `tool` definitions and restricted to explicit glob rules (`allow fs.read("examples/**")`, `allow net("api.github.com")`).
- **Durable Execution & Human Approval (`require approval`)**: Every tool and LLM step is journaled to `.tigo/state.db`. Runs can pause for human approval and resume via `tigo.py replay <run_id> --approve` without re-paying for completed LLM calls.

## Example (`examples/triage.tg`)

```text
type BugReport {
    severity: "low" | "medium" | "critical"
    summary: string
    affected_files: list<string>
    confidence: float
}

tool read_source(path: string) -> string
    allow fs.read("examples/**")
{
    return fs.read_text(path)
}

agent TriageBot {
    model: "gpt-4o-mini" fallback "gpt-4o"
    tools: [read_source]
    budget: $0.05
    max_steps: 5
    system: "You are a senior engineer triaging incoming bug reports."
}

workflow triage_issue(issue_text: string) -> BugReport {
    let snippet: string = read_source("examples/triage.tg")
    let report: BugReport = ask TriageBot("Analyze issue: ${issue_text} using context: ${snippet}")

    if report.severity == "critical" {
        require approval("Critical bug detected (${report.summary}). Page on-call engineer?")
    }

    return report
}
```

## Quick Start

```bash
# Static check
./tigo.py check examples/triage.tg

# Run workflow (uses offline synthesis by default, or live OpenAI-compatible API when OPENAI_API_KEY is set)
./tigo.py run examples/triage.tg triage_issue issue_text="Memory leak on startup"

# Inspect run trace & cost
./tigo.py trace <run_id>

# Resume a paused run after approval
./tigo.py replay <run_id> --approve

# Run test suite
python3 test_tigo.py -v
```
