"""Command-line interface for Tigo (check, test, run, replay, trace, repl)."""
import hashlib, json, os, sys, time
from pathlib import Path
from typing import Callable
from .checker import check_program, check_stmts
from .runtime import ApprovalPaused, ApprovalRejected, DurableStore, Runtime
from .syntax import Parser, Program, TigoError, compile_file, format_error, lex


def run_repl(
    store: DurableStore,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    mock_llm: object = None,
) -> int:
    prog = Program()
    env: dict = {}
    scope: dict = {}
    run_id = f"repl_{int(time.time())}"
    store.start_run(run_id, "<repl>", "repl", {})
    rt = Runtime(prog, store, run_id, approved=True, interactive=False, mock_llm=mock_llm)
    output_fn("Tigo REPL (type 'exit' or 'quit' to leave)")

    buf: list[str] = []
    while True:
        try:
            line = input_fn("... " if buf else "tigo> ")
        except EOFError:
            break
        if not buf and line.strip() in ("exit", "quit"):
            break
        if not buf and not line.strip():
            continue
        buf.append(line)
        src = "\n".join(buf)
        if src.count("{") > src.count("}"):
            continue
        buf.clear()

        try:
            toks = lex(src)
            first = toks[0].val
            if first in ("type", "tool", "agent", "workflow", "fn", "test"):
                sub = Parser(toks).parse()
                merged = Program(
                    types={**prog.types, **sub.types},
                    tools={**prog.tools, **sub.tools},
                    agents={**prog.agents, **sub.agents},
                    workflows={**prog.workflows, **sub.workflows},
                    tests={**prog.tests, **sub.tests},
                )
                check_program(merged)
                prog.types, prog.tools, prog.agents, prog.workflows, prog.tests = (
                    merged.types, merged.tools, merged.agents, merged.workflows, merged.tests
                )
                output_fn("ok")
            else:
                stmt = Parser(toks).parse_stmt()
                tmp_scope = dict(scope)
                check_stmts(prog, [stmt], tmp_scope, is_tool=False)
                if stmt[0] == "expr":
                    val = rt.eval_expr(stmt[1], env)
                    if val is not None:
                        output_fn(json.dumps(val) if isinstance(val, (dict, list)) else str(val))
                else:
                    rt.exec_block([stmt], env)
                scope.update(tmp_scope)
        except Exception as err:
            output_fn(format_error(err, src, "<repl>"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  tigo check <file.tg>\n"
            "  tigo test <file.tg>\n"
            "  tigo run <file.tg> [workflow] [key=val ...] [--approve] [--non-interactive]\n"
            "  tigo replay <run_id> [--approve] [--non-interactive]\n"
            "  tigo trace <run_id>\n"
            "  tigo repl"
        )
        return 0

    cmd = args[0]
    active_file: str | None = None

    with DurableStore(os.environ.get("TIGO_DB", ".tigo/state.db")) as store:
        try:
            if cmd == "repl":
                return run_repl(store)

            if cmd == "check":
                active_file = args[1]
                prog = compile_file(active_file)
                print(
                    f"OK: {active_file} ({len(prog.types)} types, {len(prog.tools)} tools, "
                    f"{len(prog.agents)} agents, {len(prog.workflows)} workflows, {len(prog.tests)} tests)"
                )
                return 0

            if cmd == "test":
                active_file = args[1]
                prog = compile_file(active_file)
                for tname, tdecl in prog.tests.items():
                    run_id = f"test_{tname}_{int(time.time()*1000)}"
                    store.start_run(run_id, active_file, tname, {})
                    rt = Runtime(prog, store, run_id, approved=True)
                    rt.exec_block(tdecl.body, {})
                    store.set_status(run_id, "completed")
                    print(f"PASS: {tname}")
                print(f"All {len(prog.tests)} test(s) in {active_file} passed.")
                return 0

            if cmd == "run":
                active_file = args[1]
                prog = compile_file(active_file)
                flags = {a for a in args[2:] if a.startswith("--")}
                rest = [a for a in args[2:] if not a.startswith("--")]
                approved = "--approve" in flags
                interactive = sys.stdin.isatty() and ("--non-interactive" not in flags)
                wf_name = rest[0] if rest and "=" not in rest[0] else next(iter(prog.workflows))
                kv_args = rest[1:] if (rest and "=" not in rest[0]) else rest
                wf_inputs = {}
                for kv in kv_args:
                    k, v = kv.split("=", 1)
                    try:
                        wf_inputs[k] = json.loads(v)
                    except json.JSONDecodeError:
                        wf_inputs[k] = v
                run_id = hashlib.sha1(
                    f"{active_file}:{wf_name}:{json.dumps(wf_inputs, sort_keys=True)}:{time.time()}".encode()
                ).hexdigest()[:10]
                store.start_run(run_id, active_file, wf_name, wf_inputs)
                rt = Runtime(prog, store, run_id, approved=approved, interactive=interactive)
                out = rt.run_workflow(wf_name, wf_inputs)
                print(json.dumps({"run_id": run_id, "status": "completed", "result": out}, indent=2))
                return 0

            if cmd == "replay":
                run_id = args[1]
                flags = {a for a in args[2:] if a.startswith("--")}
                approved = "--approve" in flags
                interactive = sys.stdin.isatty() and ("--non-interactive" not in flags)
                row = store.get_run(run_id)
                if not row:
                    raise TigoError(f"Unknown run_id {run_id!r}")
                _, active_file, wf_name, args_json, _, _ = row
                prog = compile_file(active_file)
                store.set_status(run_id, "running")
                rt = Runtime(prog, store, run_id, approved=approved, interactive=interactive)
                out = rt.run_workflow(wf_name, json.loads(args_json))
                print(json.dumps({"run_id": run_id, "status": "completed", "result": out}, indent=2))
                return 0

            if cmd == "trace":
                run_id = args[1]
                row = store.get_run(run_id)
                if not row:
                    raise TigoError(f"Unknown run_id {run_id!r}")
                print(f"Run {row[0]} | workflow={row[2]} | status={row[4]} | total_cost=${row[5]:.4f}")
                for skey, kind, detail, res_json, cost in store.trace(run_id):
                    print(f"  [{skey}] {kind} (${cost:.4f}): {detail} -> {res_json}")
                return 0

            raise TigoError(f"Unknown command {cmd!r}")
        except ApprovalPaused as ap:
            print(json.dumps({"run_id": ap.run_id, "status": "waiting_approval", "message": ap.message}, indent=2))
            return 2
        except ApprovalRejected as ar:
            print(json.dumps({"run_id": ar.run_id, "status": "rejected", "message": ar.message}, indent=2))
            return 3
        except TigoError as te:
            if active_file and Path(active_file).exists():
                print(format_error(te, Path(active_file).read_text(encoding="utf-8"), active_file), file=sys.stderr)
            else:
                print(f"TigoError: {te}", file=sys.stderr)
            return 1
