#!/usr/bin/env python3
"""Self-check test suite for Tigo compiler, checker, sandbox, and durable runtime."""
import tempfile
from pathlib import Path
import unittest
from tigo import ApprovalPaused, DurableStore, Runtime, TigoError, compile_source


class TigoTests(unittest.TestCase):
    def test_static_checker_catches_bad_interpolation_and_missing_guardrails(self):
        # 1. Undefined variable inside prompt interpolation
        with self.assertRaisesRegex(TigoError, "undefined variable 'missing_var'"):
            compile_source("""
            agent Bot { model: "m", budget: $0.01, max_steps: 2 }
            workflow wf(x: string) -> string {
                return ask Bot("Hello ${missing_var}")
            }
            """)

        # 2. Missing budget guardrail on agent
        with self.assertRaisesRegex(TigoError, "missing required guardrail 'budget'"):
            compile_source("""
            agent Bot { model: "m", max_steps: 2 }
            """)

        # 3. Direct fs.read_text inside workflow (must be inside a tool with allow)
        with self.assertRaisesRegex(TigoError, "only allowed inside a tool"):
            compile_source("""
            workflow wf(p: string) -> string {
                return fs.read_text(p)
            }
            """)

    def test_capability_sandbox_blocks_unauthorized_paths(self):
        src = """
        tool read_safe(p: string) -> string allow fs.read("examples/**") {
            return fs.read_text(p)
        }
        workflow wf(p: string) -> string {
            return read_safe(p)
        }
        """
        prog = compile_source(src)
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableStore(f"{tmp}/state.db")
            rt = Runtime(prog, store, "run1")
            # Allowed path works
            res = rt.run_workflow("wf", {"p": "examples/triage.tg"})
            self.assertIn("TriageBot", res)
            # Unauthorized path in workspace is blocked
            with self.assertRaisesRegex(TigoError, "sandbox denied fs.read"):
                rt.run_workflow("wf", {"p": "tigo.py"})
            # Path traversal outside workspace is blocked
            with self.assertRaisesRegex(TigoError, "sandbox blocked path traversal"):
                rt.run_workflow("wf", {"p": "../outside.txt"})

    def test_auto_repair_tool_use_and_durable_approval_replay(self):
        src = Path("examples/triage.tg").read_text(encoding="utf-8")
        prog = compile_source(src)

        calls = []

        def fake_llm(agent_name, history, schema, step):
            calls.append(step)
            if step == 1:
                # Return invalid enum value on step 1 to test auto-repair
                return {"severity": "INVALID_ENUM", "summary": "bad", "affected_files": [], "confidence": 0.5}
            # Step 2: valid critical report
            return {
                "severity": "critical",
                "summary": "Null pointer in parser",
                "affected_files": ["tigo.py"],
                "confidence": 0.95,
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = DurableStore(f"{tmp}/state.db")
            store.start_run("run_pause", "examples/triage.tg", "triage_issue", {"issue_text": "crash"})

            # First run: auto-repairs step 1 -> step 2, then pauses on require approval(...)
            rt1 = Runtime(prog, store, "run_pause", approved=False, mock_llm=fake_llm)
            with self.assertRaises(ApprovalPaused):
                rt1.run_workflow("triage_issue", {"issue_text": "crash"})
            self.assertEqual(len(calls), 2)  # Step 1 (invalid) + Step 2 (repaired)

            # Replay with approval: must NOT re-call the LLM (cached in SQLite)
            rt2 = Runtime(prog, store, "run_pause", approved=True, mock_llm=fake_llm)
            out = rt2.run_workflow("triage_issue", {"issue_text": "crash"})
            self.assertEqual(len(calls), 2)  # Still 2! Zero extra LLM calls on replay
            self.assertEqual(out["severity"], "critical")

    def test_budget_enforcement(self):
        src = """
        type Out { ok: bool }
        agent CheapBot { model: "m", budget: $0.003, max_steps: 5 }
        workflow wf() -> Out {
            let r: Out = ask CheapBot("loop")
            return r
        }
        """
        prog = compile_source(src)
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableStore(f"{tmp}/state.db")
            # Always return invalid schema so it keeps retrying until budget ($0.003 < 2 * $0.002) is exceeded
            rt = Runtime(prog, store, "run_budget", mock_llm=lambda *_: {"ok": "not_a_bool"})
            with self.assertRaisesRegex(TigoError, "exceeded budget"):
                rt.run_workflow("wf", {})

    def test_where_constraint_auto_repair_and_shell_sandbox(self):
        src = """
        type Score {
            val: float where val >= 0.0 && val <= 1.0
        }
        tool git_check(cmd: string) -> string allow sh("echo *") {
            return sh.run(cmd)
        }
        agent Scorer { model: "m", budget: $0.02, max_steps: 3 }
        workflow wf(cmd: string) -> Score {
            let out: string = git_check(cmd)
            let s: Score = ask Scorer("Score: ${out}")
            return s
        }
        """
        prog = compile_source(src)
        with tempfile.TemporaryDirectory() as tmp:
            store = DurableStore(f"{tmp}/state.db")
            # Step 1 violates 'where' (1.5 > 1.0), Step 2 succeeds (0.8)
            rt = Runtime(
                prog,
                store,
                "run_where",
                mock_llm=lambda _a, _h, _s, step: {"val": 1.5 if step == 1 else 0.8},
            )
            res = rt.run_workflow("wf", {"cmd": "echo hello"})
            self.assertEqual(res["val"], 0.8)

            # Unauthorized shell command is blocked
            with self.assertRaisesRegex(TigoError, "sandbox denied sh"):
                rt.run_workflow("wf", {"cmd": "rm -rf /"})


if __name__ == "__main__":
    unittest.main()
