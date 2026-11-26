"""Command-line interface for Tigo (check, test, run, replay, trace)."""
import hashlib, json, os, sys, time
from .runtime import ApprovalPaused, ApprovalRejected, DurableStore, Runtime
from .syntax import TigoError, compile_file


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  tigo check <file.tg>\n"
            "  tigo test <file.tg>\n"
            "  tigo run <file.tg> [workflow] [key=val ...] [--approve] [--non-interactive]\n"
            "  tigo replay <run_id> [--approve] [--non-interactive]\n"
            "  tigo trace <run_id>"
        )
        return 0

    cmd = args[0]
    store = DurableStore(os.environ.get("TIGO_DB", ".tigo/state.db"))

    try:
        if cmd == "check":
            path = args[1]
            prog = compile_file(path)
            print(f"OK: {path} ({len(prog.types)} types, {len(prog.tools)} tools, {len(prog.agents)} agents, {len(prog.workflows)} workflows, {len(prog.tests)} tests)")
            return 0

        if cmd == "test":
            path = args[1]
            prog = compile_file(path)
            for tname, tdecl in prog.tests.items():
                run_id = f"test_{tname}_{int(time.time()*1000)}"
                store.start_run(run_id, path, tname, {})
                rt = Runtime(prog, store, run_id, approved=True)
                rt.exec_block(tdecl.body, {})
                store.set_status(run_id, "completed")
                print(f"PASS: {tname}")
            print(f"All {len(prog.tests)} test(s) in {path} passed.")
            return 0

        if cmd == "run":
            path = args[1]
            prog = compile_file(path)
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
            run_id = hashlib.sha1(f"{path}:{wf_name}:{json.dumps(wf_inputs, sort_keys=True)}:{time.time()}".encode()).hexdigest()[:10]
            store.start_run(run_id, path, wf_name, wf_inputs)
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
            _, path, wf_name, args_json, _, _ = row
            prog = compile_file(path)
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
        print(f"TigoError: {te}", file=sys.stderr)
        return 1
