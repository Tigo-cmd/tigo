#!/usr/bin/env python3
"""Self-check test suite for Tigo compiler, checker, providers, sandbox, and durable runtime."""
import os, tempfile, time, unittest
from pathlib import Path
from unittest import mock

from tigo import ApprovalPaused, ApprovalRejected, DurableStore, Runtime, TigoError, compile_file, compile_source
from tigo.cli import run_repl
from tigo.providers import call_provider
from tigo.syntax import AgentDecl, TypeRef, format_error


class TigoTests(unittest.TestCase):
    def test_static_checker_catches_bad_interpolation_and_missing_guardrails(self):
        with self.assertRaisesRegex(TigoError, "undefined variable 'missing_var'"):
            compile_source("""
            agent Bot { model: "m", budget: $0.01, max_steps: 2 }
            workflow wf(x: string) -> string {
                return ask Bot("Hello ${missing_var}")
            }
            """)

        with self.assertRaisesRegex(TigoError, "missing required guardrail 'budget'"):
            compile_source("""
            agent Bot { model: "m", max_steps: 2 }
            """)

        with self.assertRaisesRegex(TigoError, "only allowed inside a tool"):
            compile_source("""
            workflow wf(p: string) -> string {
                return fs.read_text(p)
            }
            """)

    def test_multiline_triple_quoted_prompt_and_interpolation_check(self):
        prog = compile_source('''
        agent Bot {
            model: "gpt-4o-mini"
            budget: $0.02
            max_steps: 2
            system: """
You are a DevOps assistant.
Be concise.
"""
        }
        workflow wf(repo: string) -> string {
            return ask Bot("""
Check repository:
${repo}
""")
        }
        ''')
        self.assertIn("DevOps assistant.", prog.agents["Bot"].config["system"])

        with self.assertRaisesRegex(TigoError, "undefined variable 'bad_ref'"):
            compile_source('''
            agent Bot { model: "m", budget: $0.01, max_steps: 2 }
            workflow wf(repo: string) -> string {
                return ask Bot("""
Repo: ${bad_ref}
""")
            }
            ''')

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
        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            rt = Runtime(prog, store, "run1")
            res = rt.run_workflow("wf", {"p": "examples/triage.tg"})
            self.assertIn("TriageBot", res)
            with self.assertRaisesRegex(TigoError, "sandbox denied fs.read"):
                rt.run_workflow("wf", {"p": "tigo.py"})
            with self.assertRaisesRegex(TigoError, "sandbox blocked path traversal"):
                rt.run_workflow("wf", {"p": "../outside.txt"})

    def test_auto_repair_and_durable_approval_replay(self):
        prog = compile_file("examples/triage.tg")
        calls = []

        def fake_llm(agent_name, history, schema, step):
            calls.append(step)
            if step == 1:
                return {"severity": "INVALID_ENUM", "summary": "bad", "affected_files": [], "confidence": 0.5}
            return {"severity": "critical", "summary": "Null pointer", "affected_files": ["tigo.py"], "confidence": 0.95}

        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            store.start_run("run_pause", "examples/triage.tg", "triage_issue", {"issue_text": "crash"})

            rt1 = Runtime(prog, store, "run_pause", approved=False, interactive=False, mock_llm=fake_llm)
            with self.assertRaises(ApprovalPaused):
                rt1.run_workflow("triage_issue", {"issue_text": "crash"})
            self.assertEqual(len(calls), 2)

            rt2 = Runtime(prog, store, "run_pause", approved=True, mock_llm=fake_llm)
            out = rt2.run_workflow("triage_issue", {"issue_text": "crash"})
            self.assertEqual(len(calls), 2)
            self.assertEqual(out["severity"], "critical")

    def test_interactive_tty_approval_accept_and_reject(self):
        src = """
        workflow wf() -> string {
            require approval("Deploy to prod?")
            return "deployed"
        }
        """
        prog = compile_source(src)
        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            store.start_run("r_yes", "mem", "wf", {})
            rt_yes = Runtime(prog, store, "r_yes", interactive=True, input_fn=lambda _: "y")
            self.assertEqual(rt_yes.run_workflow("wf", {}), "deployed")

            store.start_run("r_no", "mem", "wf", {})
            rt_no = Runtime(prog, store, "r_no", interactive=True, input_fn=lambda _: "n")
            with self.assertRaises(ApprovalRejected):
                rt_no.run_workflow("wf", {})
            self.assertEqual(store.get_run("r_no")[4], "rejected")

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
        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            rt = Runtime(prog, store, "run_budget", mock_llm=lambda *_: {"ok": "not_a_bool"})
            with self.assertRaisesRegex(TigoError, "exceeded budget"):
                rt.run_workflow("wf", {})

    def test_where_constraint_auto_repair_and_shell_sandbox(self):
        src = """
        type Score {
            val: float where val >= 0.0 && val <= 1.0
        }
        tool run_cmd(cmd: string) -> string allow sh("echo*") {
            return sh.run(cmd)
        }
        tool safe_echo(msg: string) -> string allow sh("echo") {
            return sh.run("echo", [msg])
        }
        agent Scorer { model: "m", budget: $0.02, max_steps: 3 }
        workflow wf(cmd: string) -> Score {
            let out: string = run_cmd(cmd)
            let q: string = safe_echo("quoted \\"msg\\"")
            assert(q == "quoted \\"msg\\"", "safe argv preserves quotes")
            let s: Score = ask Scorer("Score: ${out}")
            return s
        }
        """
        prog = compile_source(src)
        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            rt = Runtime(
                prog,
                store,
                "run_where",
                mock_llm=lambda _a, _h, _s, step: {"val": 1.5 if step == 1 else 0.8},
            )
            res = rt.run_workflow("wf", {"cmd": "echo hello"})
            self.assertEqual(res["val"], 0.8)
            with self.assertRaisesRegex(TigoError, "sandbox denied sh"):
                rt.run_workflow("wf", {"cmd": "rm -rf /"})

    def test_multi_provider_routing_cost_and_fallback(self):
        env_keys = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "ant-test",
            "GEMINI_API_KEY": "gem-test",
        }
        with mock.patch.dict(os.environ, env_keys, clear=False):
            # 1. Anthropic primary fails -> OpenAI fallback succeeds with exact token cost
            seen_urls = []

            def fake_post(url, payload, headers):
                seen_urls.append(url)
                if "anthropic.com" in url:
                    raise RuntimeError("503 overloaded")
                return {
                    "choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                }

            ag = AgentDecl("A", {"model": "anthropic:claude-3-5-sonnet", "fallback": "openai:gpt-4o-mini"}, 1)
            val, cost = call_provider(ag, [{"role": "user", "content": "hi"}], {"type": "object"}, TypeRef("Out"), post_fn=fake_post)
            self.assertEqual(val, {"ok": True})
            self.assertEqual(len(seen_urls), 2)
            self.assertAlmostEqual(cost, 0.00045, places=6)

            # 2. Gemini response parsing + token cost
            ag_gem = AgentDecl("G", {"model": "gemini-2.0-flash"}, 1)
            val_g, cost_g = call_provider(
                ag_gem,
                [{"role": "user", "content": "hi"}],
                {"type": "string"},
                TypeRef("string"),
                post_fn=lambda u, p, h: {
                    "candidates": [{"content": {"parts": [{"text": "hello gemini"}]}}],
                    "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 1000},
                },
            )
            self.assertEqual(val_g, "hello gemini")
            self.assertAlmostEqual(cost_g, 0.0005, places=6)

            # 3. Ollama local model ($0 cost)
            ag_ol = AgentDecl("O", {"model": "ollama:llama3.2"}, 1)
            val_o, cost_o = call_provider(
                ag_ol,
                [{"role": "user", "content": "hi"}],
                {"type": "string"},
                TypeRef("string"),
                post_fn=lambda u, p, h: {"message": {"content": "hello local"}},
            )
            self.assertEqual(val_o, "hello local")
            self.assertEqual(cost_o, 0.0)

    def test_parallel_for_concurrency_ordering_and_replay(self):
        src = """
        type ItemOut { idx: int, label: string }
        agent Worker { model: "m", budget: $0.05, max_steps: 2 }
        workflow fan_out(items: list<int>) -> list<ItemOut> {
            let out: list<ItemOut> = parallel for x in items {
                return ask Worker("item:${x}")
            }
            return out
        }
        """
        prog = compile_source(src)
        calls = []

        def slow_llm(agent_name, history, schema, step):
            idx = int(history[0]["content"].split(":")[1])
            calls.append(idx)
            # Branch 0 sleeps longest (0.06s), branch 2 shortest (0.04s) to test deterministic output order
            time.sleep(0.06 - idx * 0.01)
            return {"idx": idx, "label": f"done-{idx}"}

        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            store.start_run("par1", "mem", "fan_out", {"items": [0, 1, 2]})
            rt1 = Runtime(prog, store, "par1", mock_llm=slow_llm)

            t0 = time.monotonic()
            res1 = rt1.run_workflow("fan_out", {"items": [0, 1, 2]})
            elapsed = time.monotonic() - t0

            # Sequential would be 0.06 + 0.05 + 0.04 = 0.15s; parallel should finish well under 0.13s
            self.assertLess(elapsed, 0.13)
            self.assertEqual([r["idx"] for r in res1], [0, 1, 2])
            self.assertEqual(len(calls), 3)

            # Replay must hit SQLite checkpoints with zero extra LLM calls
            rt2 = Runtime(prog, store, "par1", mock_llm=slow_llm)
            res2 = rt2.run_workflow("fan_out", {"items": [0, 1, 2]})
            self.assertEqual(res2, res1)
            self.assertEqual(len(calls), 3)

    def test_error_diagnostics_and_repl(self):
        src = 'workflow f() -> string {\n    return "hi ${nope}"\n}'
        try:
            compile_source(src)
            self.fail("Expected TigoError")
        except TigoError as err:
            formatted = format_error(err, src, "demo.tg")
            self.assertIn("demo.tg:2: TigoError:", formatted)
            self.assertIn('return "hi ${nope}"', formatted)
            self.assertIn("^^^", formatted)

        lines = iter([
            'let x = 2 + 3',
            'x * 10',
            'agent Bot { model: "m", budget: $0.05, max_steps: 2 }',
            'ask Bot("ping")',
            'quit',
        ])
        outputs = []
        with tempfile.TemporaryDirectory() as tmp, DurableStore(f"{tmp}/state.db") as store:
            rc = run_repl(
                store,
                input_fn=lambda _: next(lines),
                output_fn=outputs.append,
                mock_llm=lambda *_: "pong",
            )
        self.assertEqual(rc, 0)
        self.assertIn("50", outputs)
        self.assertIn("ok", outputs)
        self.assertIn("pong", outputs)


if __name__ == "__main__":
    unittest.main()
