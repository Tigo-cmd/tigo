"""Durable SQLite store, capability sandbox, schema validation, and interpreter runtime."""
from concurrent.futures import ThreadPoolExecutor
import fnmatch, json, re, shlex, sqlite3, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from typing import Any, Callable
from .providers import DEFAULT_STEP_COST, call_provider
from .syntax import INTERP_RE, Program, TigoError, TypeRef


def eval_pure_expr(e: tuple, env: dict[str, Any]) -> Any:
    kind = e[0]
    if kind == "lit":
        return e[1]
    if kind == "var":
        return env[e[1]]
    if kind == "binop":
        op, l, r = e[1], eval_pure_expr(e[2], env), eval_pure_expr(e[3], env)
        if op == "+": return l + r
        if op == "-": return l - r
        if op == "*": return l * r
        if op == "/": return l / r
        if op == "==": return l == r
        if op == "!=": return l != r
        if op == "<": return l < r
        if op == "<=": return l <= r
        if op == ">": return l > r
        if op == ">=": return l >= r
        if op == "&&": return bool(l and r)
        if op == "||": return bool(l or r)
    raise ValueError(f"Unsupported expression in where constraint: {kind}")


def validate_type(prog: Program, tref: TypeRef | None, val: Any, path: str = "root") -> Any:
    if tref is None or tref.base in ("any", "void"):
        return val
    if tref.base == "string":
        if not isinstance(val, str):
            raise ValueError(f"{path}: expected string, got {type(val).__name__}")
        return val
    if tref.base == "int":
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(f"{path}: expected int, got {type(val).__name__}")
        return val
    if tref.base == "float":
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"{path}: expected float, got {type(val).__name__}")
        return float(val)
    if tref.base == "bool":
        if not isinstance(val, bool):
            raise ValueError(f"{path}: expected bool, got {type(val).__name__}")
        return val
    if tref.base == "union":
        if val not in tref.arg:
            raise ValueError(f"{path}: expected one of {tref.arg}, got {val!r}")
        return val
    if tref.base == "list":
        if not isinstance(val, list):
            raise ValueError(f"{path}: expected list, got {type(val).__name__}")
        return [validate_type(prog, tref.arg, x, f"{path}[{i}]") for i, x in enumerate(val)]
    if tref.base in prog.types:
        if not isinstance(val, dict):
            raise ValueError(f"{path}: expected object {tref.base}, got {type(val).__name__}")
        td = prog.types[tref.base]
        out = {}
        for k, ft in td.fields.items():
            if k not in val:
                raise ValueError(f"{path}: missing required field {k!r} for {tref.base}")
            out[k] = validate_type(prog, ft, val[k], f"{path}.{k}")
            if k in td.constraints and not eval_pure_expr(td.constraints[k], {k: out[k]}):
                raise ValueError(f"{path}.{k}: value {out[k]!r} failed 'where' constraint")
        return out
    return val


def type_to_schema_hint(prog: Program, tref: TypeRef | None) -> dict[str, Any]:
    if not tref or tref.base in ("any", "string", "int", "float", "bool"):
        return {"type": tref.base if tref else "string"}
    if tref.base == "union":
        return {"enum": tref.arg}
    if tref.base == "list":
        return {"type": "array", "items": type_to_schema_hint(prog, tref.arg)}
    if tref.base in prog.types:
        td = prog.types[tref.base]
        return {
            "type": "object",
            "properties": {k: type_to_schema_hint(prog, ft) for k, ft in td.fields.items()},
            "required": list(td.fields.keys()),
        }
    return {"type": "string"}


class DurableStore:
    def __init__(self, db_path: str = ".tigo/state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock, self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, file TEXT, workflow TEXT, args_json TEXT,
                    status TEXT, cost_usd REAL DEFAULT 0.0, created_at REAL
                )"""
            )
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS steps (
                    run_id TEXT, step_key TEXT, kind TEXT, detail TEXT,
                    result_json TEXT, cost_usd REAL DEFAULT 0.0, ts REAL,
                    PRIMARY KEY (run_id, step_key)
                )"""
            )

    def start_run(self, run_id: str, file: str, workflow: str, args: dict) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, 'running', 0.0, ?)",
                (run_id, file, workflow, json.dumps(args), time.time()),
            )
            self.conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (run_id,))

    def get_run(self, run_id: str) -> tuple | None:
        with self._lock:
            return self.conn.execute(
                "SELECT run_id, file, workflow, args_json, status, cost_usd FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    def get_step(self, run_id: str, step_key: str) -> Any | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT result_json FROM steps WHERE run_id=? AND step_key=?", (run_id, step_key)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_step(self, run_id: str, step_key: str, kind: str, detail: str, result: Any, cost: float = 0.0) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, step_key, kind, detail, json.dumps(result), cost, time.time()),
            )
            self.conn.execute("UPDATE runs SET cost_usd = cost_usd + ? WHERE run_id=?", (cost, run_id))

    def trace(self, run_id: str) -> list[tuple]:
        with self._lock:
            return self.conn.execute(
                "SELECT step_key, kind, detail, result_json, cost_usd FROM steps WHERE run_id=? ORDER BY ts ASC",
                (run_id,),
            ).fetchall()


class ApprovalPaused(Exception):
    def __init__(self, run_id: str, message: str):
        super().__init__(f"PAUSED (run_id={run_id}): waiting for approval: {message}")
        self.run_id = run_id
        self.message = message


class ApprovalRejected(Exception):
    def __init__(self, run_id: str, message: str):
        super().__init__(f"REJECTED (run_id={run_id}): {message}")
        self.run_id = run_id
        self.message = message


class ReturnSignal(Exception):
    def __init__(self, val: Any):
        self.val = val


class Runtime:
    def __init__(
        self,
        prog: Program,
        store: DurableStore,
        run_id: str,
        approved: bool = False,
        interactive: bool = False,
        input_fn: Callable[[str], str] = input,
        mock_llm: Any = None,
        key_prefix: str = "",
        expected_return: TypeRef | None = None,
    ):
        self.prog = prog
        self.store = store
        self.run_id = run_id
        self.approved = approved
        self.interactive = interactive
        self.input_fn = input_fn
        self.mock_llm = mock_llm
        self.key_prefix = key_prefix
        self.expected_return = expected_return
        self.step_counter = 0
        self.active_allows: list[tuple[str, str]] | None = None

    def next_key(self, tag: str) -> str:
        self.step_counter += 1
        return f"{self.key_prefix}{self.step_counter:03d}:{tag}"

    def check_cap(self, cap: str, target: str, line: int) -> None:
        if self.active_allows is None:
            raise TigoError(f"Line {line}: capability {cap!r} called outside tool")
        if cap.startswith("fs."):
            base = Path.cwd().resolve()
            resolved = (base / target).resolve()
            try:
                rel = str(resolved.relative_to(base))
            except ValueError:
                raise TigoError(f"Line {line}: sandbox blocked path traversal outside workspace: {target!r}")
            for acap, pat in self.active_allows:
                norm_pat = pat[2:] if pat.startswith("./") else pat
                if acap == cap and (fnmatch.fnmatch(rel, norm_pat) or fnmatch.fnmatch(target, pat)):
                    return
        else:
            for acap, pat in self.active_allows:
                if acap == cap and fnmatch.fnmatch(target, pat):
                    return
        raise TigoError(f"Line {line}: sandbox denied {cap}({target!r})")

    def run_workflow(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self.prog.workflows:
            raise TigoError(f"Unknown workflow {name!r}")
        wf = self.prog.workflows[name]
        env = {}
        for pname, pt in wf.params:
            if pname not in args:
                raise TigoError(f"Missing workflow argument {pname!r}")
            env[pname] = validate_type(self.prog, pt, args[pname], pname)
        try:
            self.exec_block(wf.body, env)
        except ReturnSignal as r:
            res = validate_type(self.prog, wf.ret, r.val, f"{name}.return")
            self.store.set_status(self.run_id, "completed")
            return res
        self.store.set_status(self.run_id, "completed")
        return None

    def exec_block(self, stmts: list[tuple], env: dict[str, Any]) -> None:
        for s in stmts:
            kind, line = s[0], s[-1]
            if kind == "let":
                _, name, tref, expr, _ = s
                if expr[0] == "ask":
                    val = self.eval_ask(expr[1], expr[2], env, tref, line)
                elif expr[0] == "parfor":
                    elem_t = tref.arg if (tref and tref.base == "list") else None
                    val = self.exec_parfor(expr[1], expr[2], expr[3], env, elem_t, line)
                else:
                    val = self.eval_expr(expr, env)
                env[name] = validate_type(self.prog, tref, val, name)
            elif kind == "if":
                _, cond, tb, eb, _ = s
                if self.eval_expr(cond, env):
                    self.exec_block(tb, env)
                elif eb:
                    self.exec_block(eb, env)
            elif kind == "for":
                _, var, it, body, _ = s
                for item in self.eval_expr(it, env):
                    env[var] = item
                    self.exec_block(body, env)
            elif kind == "parfor":
                _, var, it, body, _ = s
                self.exec_parfor(var, it, body, env, None, line)
            elif kind == "return":
                expr = s[1]
                val = (
                    self.eval_ask(expr[1], expr[2], env, self.expected_return, line)
                    if expr[0] == "ask" and self.expected_return is not None
                    else self.eval_expr(expr, env)
                )
                raise ReturnSignal(val)
            elif kind == "approve":
                self._handle_approval(str(self.eval_expr(s[1], env)))
            elif kind == "expr":
                self.eval_expr(s[1], env)

    def _handle_approval(self, msg: str) -> None:
        key = self.next_key("approve")
        cached = self.store.get_step(self.run_id, key)
        if cached is not None and cached.get("approved"):
            return
        if self.approved:
            self.store.save_step(self.run_id, key, "approve", msg, {"approved": True})
            return
        if self.interactive:
            ans = self.input_fn(f"[Tigo Approval] {msg} [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                self.store.save_step(self.run_id, key, "approve", msg, {"approved": True})
                return
            self.store.set_status(self.run_id, "rejected")
            raise ApprovalRejected(self.run_id, msg)
        self.store.set_status(self.run_id, "waiting_approval")
        raise ApprovalPaused(self.run_id, msg)

    def eval_expr(self, e: tuple, env: dict[str, Any]) -> Any:
        kind, line = e[0], e[-1]
        if kind == "lit":
            return e[1]
        if kind == "str":
            def repl(m: re.Match) -> str:
                parts = m.group(1).strip().split(".")
                v = env[parts[0]]
                for p in parts[1:]:
                    v = v[p] if isinstance(v, dict) else getattr(v, p)
                return str(v)
            return INTERP_RE.sub(repl, e[1])
        if kind == "var":
            return e[1] if e[1] in ("fs", "http", "sh") else env[e[1]]
        if kind == "list":
            return [self.eval_expr(x, env) for x in e[1]]
        if kind == "dict":
            return {k: self.eval_expr(v, env) for k, v in e[1].items()}
        if kind == "attr":
            obj = self.eval_expr(e[1], env)
            if not isinstance(obj, dict) or e[2] not in obj:
                raise TigoError(f"Line {line}: missing field {e[2]!r}")
            return obj[e[2]]
        if kind == "binop":
            op, l, r = e[1], self.eval_expr(e[2], env), self.eval_expr(e[3], env)
            if op == "+": return str(l) + str(r) if isinstance(l, str) or isinstance(r, str) else l + r
            if op == "-": return l - r
            if op == "*": return l * r
            if op == "/": return l / r
            if op == "==": return l == r
            if op == "!=": return l != r
            if op == "<": return l < r
            if op == "<=": return l <= r
            if op == ">": return l > r
            if op == ">=": return l >= r
            if op == "&&": return bool(l and r)
            if op == "||": return bool(l or r)
        if kind == "mcall":
            target, method = self.eval_expr(e[1], env), e[2]
            args = [self.eval_expr(a, env) for a in e[3]]
            if target == "fs" and method == "read_text":
                self.check_cap("fs.read", args[0], line)
                return Path(args[0]).read_text(encoding="utf-8")
            if target == "fs" and method == "write_text":
                self.check_cap("fs.write", args[0], line)
                Path(args[0]).parent.mkdir(parents=True, exist_ok=True)
                Path(args[0]).write_text(str(args[1]), encoding="utf-8")
                return "ok"
            if target == "http" and method == "get":
                host = urllib.request.urlparse(args[0]).netloc
                self.check_cap("net", host, line)
                with urllib.request.urlopen(args[0], timeout=10) as resp:
                    return resp.read().decode("utf-8")
            if target == "sh" and method == "run":
                cmd_str = str(args[0]).strip()
                self.check_cap("sh", cmd_str, line)
                proc = subprocess.run(shlex.split(cmd_str), capture_output=True, text=True, timeout=15, check=False)
                return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
            raise TigoError(f"Line {line}: unknown capability method {target}.{method}")
        if kind == "call":
            fn, args = e[1], [self.eval_expr(a, env) for a in e[2]]
            if fn == "print":
                print(*args)
                return None
            if fn == "assert":
                if not args or not args[0]:
                    raise TigoError(f"Line {line}: AssertError: {args[1] if len(args) > 1 else 'assertion failed'}")
                return None
            if fn in self.prog.tools:
                return self.invoke_tool(fn, args, line)
            if fn in self.prog.workflows:
                wf = self.prog.workflows[fn]
                return self.run_workflow(fn, {k: v for (k, _), v in zip(wf.params, args)})
        if kind == "parfor":
            return self.exec_parfor(e[1], e[2], e[3], env, None, line)
        if kind == "ask":
            return self.eval_ask(e[1], e[2], env, None, line)
        raise TigoError(f"Line {line}: unsupported expression {kind}")

    def exec_parfor(
        self, var: str, it_expr: tuple, body: list[tuple], env: dict[str, Any], elem_tref: TypeRef | None, line: int
    ) -> list[Any]:
        items = self.eval_expr(it_expr, env)
        if not isinstance(items, list):
            raise TigoError(f"Line {line}: parallel for expects a list, got {type(items).__name__}")
        if not items:
            return []
        base_key = self.next_key("par")

        def run_branch(idx: int, item: Any) -> Any:
            child = Runtime(
                self.prog,
                self.store,
                self.run_id,
                approved=self.approved,
                interactive=False,  # avoid interleaved stdin prompts across threads
                input_fn=self.input_fn,
                mock_llm=self.mock_llm,
                key_prefix=f"{base_key}[{idx}]/",
                expected_return=elem_tref,
            )
            branch_env = dict(env)
            branch_env[var] = item
            try:
                child.exec_block(body, branch_env)
                return None
            except ReturnSignal as r:
                return validate_type(self.prog, elem_tref, r.val, f"{base_key}[{idx}]")

        with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
            futs = [pool.submit(run_branch, i, item) for i, item in enumerate(items)]
            return [f.result() for f in futs]

    def invoke_tool(self, name: str, args: list[Any], line: int) -> Any:
        tool = self.prog.tools[name]
        key = self.next_key(f"tool:{name}")
        cached = self.store.get_step(self.run_id, key)
        if cached is not None:
            return cached
        tenv = {pname: validate_type(self.prog, pt, val, pname) for (pname, pt), val in zip(tool.params, args)}
        prev, self.active_allows = self.active_allows, tool.allows
        try:
            self.exec_block(tool.body, tenv)
            res = None
        except ReturnSignal as r:
            res = validate_type(self.prog, tool.ret, r.val, f"{name}.return")
        finally:
            self.active_allows = prev
        self.store.save_step(self.run_id, key, "tool", f"{name}({args})", res, 0.0)
        return res

    def eval_ask(self, ag_name: str, prompt_expr: tuple, env: dict[str, Any], expected: TypeRef | None, line: int) -> Any:
        ag = self.prog.agents[ag_name]
        prompt = str(self.eval_expr(prompt_expr, env))
        key = self.next_key(f"ask:{ag_name}")
        cached = self.store.get_step(self.run_id, key)
        if cached is not None:
            return validate_type(self.prog, expected, cached, ag_name)

        budget, max_steps, spent = float(ag.config["budget"]), int(ag.config["max_steps"]), 0.0
        schema_hint = type_to_schema_hint(self.prog, expected)
        history = [{"role": "user", "content": prompt}]

        for step in range(1, max_steps + 1):
            if spent >= budget:
                raise TigoError(f"Line {line}: agent {ag_name!r} exceeded budget ${budget:.4f} (spent ${spent:.4f})")

            if self.mock_llm is not None:
                raw, step_cost = self.mock_llm(ag.name, history, schema_hint, step), DEFAULT_STEP_COST
            else:
                res = call_provider(ag, history, schema_hint, expected)
                raw, step_cost = res if res is not None else (self._offline_synth(expected), DEFAULT_STEP_COST)

            if spent + step_cost > budget:
                raise TigoError(f"Line {line}: agent {ag_name!r} exceeded budget ${budget:.4f} (spent ${spent + step_cost:.4f})")
            spent += step_cost

            if isinstance(raw, dict) and "tool" in raw and raw["tool"] in ag.config.get("tools", []):
                tname, targs = raw["tool"], raw.get("args", [])
                tres = self.invoke_tool(tname, targs, line)
                history.append({"role": "assistant", "content": json.dumps(raw)})
                history.append({"role": "tool", "name": tname, "content": json.dumps(tres)})
                continue

            try:
                val = validate_type(self.prog, expected, raw, ag_name)
                self.store.save_step(self.run_id, key, "ask", f"{ag_name}: {prompt[:80]}", val, round(spent, 6))
                return val
            except ValueError as ve:
                history.append({"role": "assistant", "content": json.dumps(raw)})
                history.append({"role": "user", "content": f"Schema validation error: {ve}. Return valid JSON."})

        raise TigoError(f"Line {line}: agent {ag_name!r} exceeded max_steps ({max_steps}) without valid output")

    def _offline_synth(self, tref: TypeRef | None) -> Any:
        if not tref or tref.base == "string":
            return "mock_response"
        if tref.base == "int":
            return 1
        if tref.base == "float":
            return 0.9
        if tref.base == "bool":
            return True
        if tref.base == "union":
            return tref.arg[0]
        if tref.base == "list":
            return [self._offline_synth(tref.arg)]
        if tref.base in self.prog.types:
            return {k: self._offline_synth(ft) for k, ft in self.prog.types[tref.base].fields.items()}
        return "mock"
