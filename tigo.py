#!/usr/bin/env python3
"""Tigo (.tg): Minimal language & durable runtime for safe, typed AI agents."""
import fnmatch, hashlib, json, os, re, sqlite3, sys, time, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- 1. Lexer & AST ---

TOKEN_SPEC = [
    ("SKIP", r"[ \t\r\n]+|//[^\n]*"),
    ("MONEY", r"\$\d+(?:\.\d+)?"),
    ("FLOAT", r"\d+\.\d+"),
    ("INT", r"\d+"),
    ("STR", r'"(?:\\.|[^"\\])*"'),
    ("OP", r"->|==|!=|<=|>=|&&|\|\||[{}()\[\]:,.;=+\-*/<>|]"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
]
TOK_RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in TOKEN_SPEC))
INTERP_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass
class Tok:
    kind: str
    val: str
    line: int


class TigoError(Exception):
    pass


def lex(src: str) -> list[Tok]:
    toks, line = [], 1
    pos = 0
    while pos < len(src):
        m = TOK_RE.match(src, pos)
        if not m:
            raise TigoError(f"Line {line}: unexpected character {src[pos]!r}")
        kind, val = m.lastgroup, m.group()
        if kind != "SKIP":
            toks.append(Tok(kind, val, line))
        line += val.count("\n")
        pos = m.end()
    toks.append(Tok("EOF", "", line))
    return toks


@dataclass
class TypeRef:
    base: str  # "string", "int", "float", "bool", "list", "union", or custom type name
    arg: Any = None  # inner TypeRef for list, or list[str] for union


@dataclass
class TypeDecl:
    name: str
    fields: dict[str, TypeRef]
    line: int


@dataclass
class ToolDecl:
    name: str
    params: list[tuple[str, TypeRef]]
    ret: TypeRef
    allows: list[tuple[str, str]]  # [(cap, pattern)]
    body: list[Any]
    line: int


@dataclass
class AgentDecl:
    name: str
    config: dict[str, Any]
    line: int


@dataclass
class WorkflowDecl:
    name: str
    params: list[tuple[str, TypeRef]]
    ret: TypeRef
    body: list[Any]
    line: int


@dataclass
class Program:
    types: dict[str, TypeDecl] = field(default_factory=dict)
    tools: dict[str, ToolDecl] = field(default_factory=dict)
    agents: dict[str, AgentDecl] = field(default_factory=dict)
    workflows: dict[str, WorkflowDecl] = field(default_factory=dict)


# --- 2. Recursive-Descent Parser ---
# shortcut: single-file grammar without imports/modules, add 'import' when multi-file projects are needed.

class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks, self.i = toks, 0

    def cur(self) -> Tok:
        return self.toks[self.i]

    def eat(self, val: str | None = None, kind: str | None = None) -> Tok:
        t = self.cur()
        if (val and t.val != val) or (kind and t.kind != kind):
            exp = val or kind
            raise TigoError(f"Line {t.line}: expected {exp!r}, got {t.val!r}")
        self.i += 1
        return t

    def match(self, val: str) -> bool:
        if self.cur().val == val:
            self.i += 1
            return True
        return False

    def parse(self) -> Program:
        prog = Program()
        while self.cur().kind != "EOF":
            t = self.cur()
            if t.val == "type":
                d = self.parse_type_decl()
                prog.types[d.name] = d
            elif t.val == "tool":
                d = self.parse_tool_decl()
                prog.tools[d.name] = d
            elif t.val == "agent":
                d = self.parse_agent_decl()
                prog.agents[d.name] = d
            elif t.val in ("workflow", "fn"):
                d = self.parse_workflow_decl()
                prog.workflows[d.name] = d
            else:
                raise TigoError(f"Line {t.line}: unexpected top-level token {t.val!r}")
        return prog

    def parse_type_ref(self) -> TypeRef:
        if self.cur().kind == "STR":
            opts = [json.loads(self.eat(kind="STR").val)]
            while self.match("|"):
                opts.append(json.loads(self.eat(kind="STR").val))
            return TypeRef("union", opts)
        name = self.eat(kind="ID").val
        if name == "list":
            self.eat("<")
            inner = self.parse_type_ref()
            self.eat(">")
            return TypeRef("list", inner)
        return TypeRef(name)

    def parse_type_decl(self) -> TypeDecl:
        line = self.eat("type").line
        name = self.eat(kind="ID").val
        self.eat("{")
        fields = {}
        while not self.match("}"):
            fname = self.eat(kind="ID").val
            self.eat(":")
            fields[fname] = self.parse_type_ref()
            self.match(",")
        return TypeDecl(name, fields, line)

    def parse_params(self) -> list[tuple[str, TypeRef]]:
        self.eat("(")
        params = []
        if self.cur().val != ")":
            while True:
                pname = self.eat(kind="ID").val
                self.eat(":")
                params.append((pname, self.parse_type_ref()))
                if not self.match(","):
                    break
        self.eat(")")
        return params

    def parse_tool_decl(self) -> ToolDecl:
        line = self.eat("tool").line
        name = self.eat(kind="ID").val
        params = self.parse_params()
        ret = TypeRef("void")
        if self.match("->"):
            ret = self.parse_type_ref()
        allows = []
        while self.match("allow"):
            cap = self.eat(kind="ID").val
            if self.match("."):
                cap += "." + self.eat(kind="ID").val
            self.eat("(")
            pat = json.loads(self.eat(kind="STR").val)
            self.eat(")")
            allows.append((cap, pat))
            self.match(",")
        body = self.parse_block()
        return ToolDecl(name, params, ret, allows, body, line)

    def parse_agent_decl(self) -> AgentDecl:
        line = self.eat("agent").line
        name = self.eat(kind="ID").val
        self.eat("{")
        cfg: dict[str, Any] = {}
        while not self.match("}"):
            k = self.eat(kind="ID").val
            self.eat(":")
            t = self.cur()
            if t.kind == "STR":
                val: Any = json.loads(self.eat(kind="STR").val)
                if k == "model" and self.match("fallback"):
                    cfg["fallback"] = json.loads(self.eat(kind="STR").val)
            elif t.kind == "MONEY":
                val = float(self.eat(kind="MONEY").val[1:])
            elif t.kind == "INT":
                val = int(self.eat(kind="INT").val)
            elif t.kind == "FLOAT":
                val = float(self.eat(kind="FLOAT").val)
            elif t.val == "[":
                self.eat("[")
                val = []
                if self.cur().val != "]":
                    while True:
                        val.append(self.eat(kind="ID").val)
                        if not self.match(","):
                            break
                self.eat("]")
            else:
                raise TigoError(f"Line {t.line}: invalid agent config value {t.val!r}")
            cfg[k] = val
            self.match(",")
        return AgentDecl(name, cfg, line)

    def parse_workflow_decl(self) -> WorkflowDecl:
        line = self.eat().line
        name = self.eat(kind="ID").val
        params = self.parse_params()
        ret = TypeRef("void")
        if self.match("->"):
            ret = self.parse_type_ref()
        body = self.parse_block()
        return WorkflowDecl(name, params, ret, body, line)

    def parse_block(self) -> list[Any]:
        self.eat("{")
        stmts = []
        while not self.match("}"):
            stmts.append(self.parse_stmt())
            self.match(";")
        return stmts

    def parse_stmt(self) -> tuple:
        t = self.cur()
        if t.val == "let":
            self.eat("let")
            name = self.eat(kind="ID").val
            tref = self.parse_type_ref() if self.match(":") else None
            self.eat("=")
            return ("let", name, tref, self.parse_expr(), t.line)
        if t.val == "if":
            self.eat("if")
            cond = self.parse_expr()
            then_b = self.parse_block()
            else_b = self.parse_block() if self.match("else") else []
            return ("if", cond, then_b, else_b, t.line)
        if t.val == "for":
            self.eat("for")
            var = self.eat(kind="ID").val
            self.eat("in")
            it = self.parse_expr()
            return ("for", var, it, self.parse_block(), t.line)
        if t.val == "return":
            self.eat("return")
            return ("return", self.parse_expr(), t.line)
        if t.val == "require":
            self.eat("require")
            self.eat("approval")
            self.eat("(")
            msg = self.parse_expr()
            self.eat(")")
            return ("approve", msg, t.line)
        return ("expr", self.parse_expr(), t.line)

    def parse_expr(self) -> tuple:
        return self.parse_binop(0)

    PREC = [("||",), ("&&",), ("==", "!=", "<", "<=", ">", ">="), ("+", "-"), ("*", "/")]

    def parse_binop(self, lvl: int) -> tuple:
        if lvl >= len(self.PREC):
            return self.parse_postfix()
        left = self.parse_binop(lvl + 1)
        while self.cur().val in self.PREC[lvl]:
            op = self.eat().val
            right = self.parse_binop(lvl + 1)
            left = ("binop", op, left, right, left[-1])
        return left

    def parse_postfix(self) -> tuple:
        node = self.parse_primary()
        while True:
            if self.match("."):
                attr = self.eat(kind="ID").val
                if self.cur().val == "(":
                    args = self.parse_args()
                    node = ("mcall", node, attr, args, node[-1])
                else:
                    node = ("attr", node, attr, node[-1])
            elif self.cur().val == "(":
                args = self.parse_args()
                if node[0] != "var":
                    raise TigoError(f"Line {node[-1]}: cannot call {node[0]}")
                node = ("call", node[1], args, node[-1])
            else:
                break
        return node

    def parse_args(self) -> list[tuple]:
        self.eat("(")
        args = []
        if self.cur().val != ")":
            while True:
                args.append(self.parse_expr())
                if not self.match(","):
                    break
        self.eat(")")
        return args

    def parse_primary(self) -> tuple:
        t = self.cur()
        if t.val == "ask":
            self.eat("ask")
            ag = self.eat(kind="ID").val
            self.eat("(")
            prompt = self.parse_expr()
            self.eat(")")
            return ("ask", ag, prompt, t.line)
        if t.kind == "STR":
            self.eat()
            return ("str", json.loads(t.val), t.line)
        if t.kind == "INT":
            self.eat()
            return ("lit", int(t.val), t.line)
        if t.kind == "FLOAT":
            self.eat()
            return ("lit", float(t.val), t.line)
        if t.val in ("true", "false"):
            self.eat()
            return ("lit", t.val == "true", t.line)
        if t.val == "[":
            self.eat("[")
            items = []
            if self.cur().val != "]":
                while True:
                    items.append(self.parse_expr())
                    if not self.match(","):
                        break
            self.eat("]")
            return ("list", items, t.line)
        if t.val == "{":
            self.eat("{")
            kv = {}
            if self.cur().val != "}":
                while True:
                    k = self.eat(kind="ID").val if self.cur().kind == "ID" else json.loads(self.eat(kind="STR").val)
                    self.eat(":")
                    kv[k] = self.parse_expr()
                    if not self.match(","):
                        break
            self.eat("}")
            return ("dict", kv, t.line)
        if t.val == "(":
            self.eat("(")
            e = self.parse_expr()
            self.eat(")")
            return e
        if t.kind == "ID":
            self.eat()
            return ("var", t.val, t.line)
        raise TigoError(f"Line {t.line}: unexpected token {t.val!r} in expression")


# --- 3. Static Type, Prompt & Capability Checker ---

VALID_CAPS = {"fs.read", "fs.write", "net", "sh"}
PRIMITIVES = {"string", "int", "float", "bool", "void", "any"}


def check_program(prog: Program) -> None:
    def check_type_exists(tr: TypeRef, line: int) -> None:
        if tr.base in PRIMITIVES or tr.base == "union":
            return
        if tr.base == "list":
            check_type_exists(tr.arg, line)
            return
        if tr.base not in prog.types:
            raise TigoError(f"Line {line}: unknown type {tr.base!r}")

    for td in prog.types.values():
        for ft in td.fields.values():
            check_type_exists(ft, td.line)

    for tool in prog.tools.values():
        for _, pt in tool.params:
            check_type_exists(pt, tool.line)
        check_type_exists(tool.ret, tool.line)
        for cap, pat in tool.allows:
            if cap not in VALID_CAPS:
                raise TigoError(f"Line {tool.line}: tool {tool.name!r} uses invalid permission {cap!r}")
            if not pat:
                raise TigoError(f"Line {tool.line}: tool {tool.name!r} has empty permission pattern")
        scope = {k: v for k, v in tool.params}
        check_stmts(prog, tool.body, scope, is_tool=True)

    for ag in prog.agents.values():
        for req in ("model", "budget", "max_steps"):
            if req not in ag.config:
                raise TigoError(f"Line {ag.line}: agent {ag.name!r} missing required guardrail {req!r}")
        if ag.config["budget"] <= 0:
            raise TigoError(f"Line {ag.line}: agent {ag.name!r} budget must be > $0")
        if ag.config["max_steps"] <= 0 or ag.config["max_steps"] > 50:
            raise TigoError(f"Line {ag.line}: agent {ag.name!r} max_steps must be between 1 and 50")
        for tname in ag.config.get("tools", []):
            if tname not in prog.tools:
                raise TigoError(f"Line {ag.line}: agent {ag.name!r} references unknown tool {tname!r}")

    for wf in prog.workflows.values():
        for _, pt in wf.params:
            check_type_exists(pt, wf.line)
        check_type_exists(wf.ret, wf.line)
        scope = {k: v for k, v in wf.params}
        check_stmts(prog, wf.body, scope, is_tool=False)


def check_stmts(prog: Program, stmts: list[tuple], scope: dict[str, TypeRef], is_tool: bool) -> None:
    for s in stmts:
        kind, line = s[0], s[-1]
        if kind == "let":
            _, name, tref, expr, _ = s
            if tref:
                if tref.base not in PRIMITIVES and tref.base not in ("list", "union") and tref.base not in prog.types:
                    raise TigoError(f"Line {line}: unknown type {tref.base!r}")
            et = check_expr(prog, expr, scope, is_tool)
            scope[name] = tref or et
        elif kind == "if":
            _, cond, tb, eb, _ = s
            check_expr(prog, cond, scope, is_tool)
            check_stmts(prog, tb, dict(scope), is_tool)
            check_stmts(prog, eb, dict(scope), is_tool)
        elif kind == "for":
            _, var, it, body, _ = s
            it_t = check_expr(prog, it, scope, is_tool)
            inner = it_t.arg if it_t.base == "list" else TypeRef("any")
            sub = dict(scope)
            sub[var] = inner
            check_stmts(prog, body, sub, is_tool)
        elif kind in ("return", "approve", "expr"):
            check_expr(prog, s[1], scope, is_tool)


def check_expr(prog: Program, e: tuple, scope: dict[str, TypeRef], is_tool: bool) -> TypeRef:
    kind, line = e[0], e[-1]
    if kind == "lit":
        v = e[1]
        if isinstance(v, bool):
            return TypeRef("bool")
        return TypeRef("int" if isinstance(v, int) else "float")
    if kind == "str":
        for var in INTERP_RE.findall(e[1]):
            root = var.strip().split(".")[0]
            if root not in scope:
                raise TigoError(f"Line {line}: prompt/string interpolation references undefined variable {root!r}")
        return TypeRef("string")
    if kind == "var":
        if e[1] in ("fs", "http"):
            return TypeRef(e[1])
        if e[1] not in scope:
            raise TigoError(f"Line {line}: undefined variable {e[1]!r}")
        return scope[e[1]]
    if kind == "list":
        for item in e[1]:
            check_expr(prog, item, scope, is_tool)
        return TypeRef("list", TypeRef("any"))
    if kind == "dict":
        for item in e[1].values():
            check_expr(prog, item, scope, is_tool)
        return TypeRef("any")
    if kind == "binop":
        check_expr(prog, e[2], scope, is_tool)
        check_expr(prog, e[3], scope, is_tool)
        return TypeRef("bool" if e[1] in ("==", "!=", "<", "<=", ">", ">=", "&&", "||") else "any")
    if kind == "attr":
        obj_t = check_expr(prog, e[1], scope, is_tool)
        attr = e[2]
        if obj_t.base in prog.types:
            td = prog.types[obj_t.base]
            if attr not in td.fields:
                raise TigoError(f"Line {line}: type {obj_t.base!r} has no field {attr!r}")
            return td.fields[attr]
        return TypeRef("any")
    if kind == "mcall":
        obj_t = check_expr(prog, e[1], scope, is_tool)
        for a in e[3]:
            check_expr(prog, a, scope, is_tool)
        if obj_t.base in ("fs", "http") and not is_tool:
            raise TigoError(f"Line {line}: capability call '{obj_t.base}.{e[2]}' is only allowed inside a tool")
        return TypeRef("string")
    if kind == "call":
        fn, args = e[1], e[2]
        for a in args:
            check_expr(prog, a, scope, is_tool)
        if fn == "print":
            return TypeRef("void")
        if fn in prog.tools:
            tdecl = prog.tools[fn]
            if len(args) != len(tdecl.params):
                raise TigoError(f"Line {line}: tool {fn!r} expects {len(tdecl.params)} args, got {len(args)}")
            return tdecl.ret
        if fn in prog.workflows:
            wdecl = prog.workflows[fn]
            if len(args) != len(wdecl.params):
                raise TigoError(f"Line {line}: workflow {fn!r} expects {len(wdecl.params)} args, got {len(args)}")
            return wdecl.ret
        raise TigoError(f"Line {line}: unknown function or tool {fn!r}")
    if kind == "ask":
        ag_name, prompt = e[1], e[2]
        if ag_name not in prog.agents:
            raise TigoError(f"Line {line}: unknown agent {ag_name!r}")
        check_expr(prog, prompt, scope, is_tool)
        return TypeRef("any")
    return TypeRef("any")


# --- 4. Schema Validation & Durable SQLite Store ---

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
        self.conn = sqlite3.connect(db_path)
        with self.conn:
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
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, 'running', 0.0, ?)",
                (run_id, file, workflow, json.dumps(args), time.time()),
            )
            self.conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (run_id,))

    def get_run(self, run_id: str) -> tuple | None:
        return self.conn.execute(
            "SELECT run_id, file, workflow, args_json, status, cost_usd FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()

    def set_status(self, run_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))

    def get_step(self, run_id: str, step_key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT result_json FROM steps WHERE run_id=? AND step_key=?", (run_id, step_key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_step(self, run_id: str, step_key: str, kind: str, detail: str, result: Any, cost: float = 0.0) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, step_key, kind, detail, json.dumps(result), cost, time.time()),
            )
            self.conn.execute("UPDATE runs SET cost_usd = cost_usd + ? WHERE run_id=?", (cost, run_id))

    def trace(self, run_id: str) -> list[tuple]:
        return self.conn.execute(
            "SELECT step_key, kind, detail, result_json, cost_usd FROM steps WHERE run_id=? ORDER BY ts ASC",
            (run_id,),
        ).fetchall()


# --- 5. Runtime, Sandbox & LLM Execution ---

class ApprovalPaused(Exception):
    def __init__(self, run_id: str, message: str):
        super().__init__(f"PAUSED (run_id={run_id}): waiting for approval: {message}")
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
        mock_llm: Any = None,
    ):
        self.prog = prog
        self.store = store
        self.run_id = run_id
        self.approved = approved
        self.mock_llm = mock_llm
        self.step_counter = 0
        self.active_allows: list[tuple[str, str]] | None = None

    def next_key(self, tag: str) -> str:
        self.step_counter += 1
        return f"{self.step_counter:03d}:{tag}"

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
            elif kind == "return":
                raise ReturnSignal(self.eval_expr(s[1], env))
            elif kind == "approve":
                msg = str(self.eval_expr(s[1], env))
                key = self.next_key("approve")
                cached = self.store.get_step(self.run_id, key)
                if cached is not None and cached.get("approved"):
                    continue
                if self.approved:
                    self.store.save_step(self.run_id, key, "approve", msg, {"approved": True})
                    continue
                self.store.set_status(self.run_id, "waiting_approval")
                raise ApprovalPaused(self.run_id, msg)
            elif kind == "expr":
                self.eval_expr(s[1], env)

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
            return e[1] if e[1] in ("fs", "http") else env[e[1]]
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
            target = self.eval_expr(e[1], env)
            method = e[2]
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
            raise TigoError(f"Line {line}: unknown capability method {target}.{method}")
        if kind == "call":
            fn = e[1]
            args = [self.eval_expr(a, env) for a in e[2]]
            if fn == "print":
                print(*args)
                return None
            if fn in self.prog.tools:
                return self.invoke_tool(fn, args, line)
            if fn in self.prog.workflows:
                wf = self.prog.workflows[fn]
                return self.run_workflow(fn, {k: v for (k, _), v in zip(wf.params, args)})
        if kind == "ask":
            return self.eval_ask(e[1], e[2], env, None, line)
        raise TigoError(f"Line {line}: unsupported expression {kind}")

    def invoke_tool(self, name: str, args: list[Any], line: int) -> Any:
        tool = self.prog.tools[name]
        key = self.next_key(f"tool:{name}")
        cached = self.store.get_step(self.run_id, key)
        if cached is not None:
            return cached
        tenv = {pname: validate_type(self.prog, pt, val, pname) for (pname, pt), val in zip(tool.params, args)}
        prev = self.active_allows
        self.active_allows = tool.allows
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

        budget: float = float(ag.config["budget"])
        max_steps: int = int(ag.config["max_steps"])
        spent = 0.0
        schema_hint = type_to_schema_hint(self.prog, expected)
        history = [{"role": "user", "content": prompt}]

        for step in range(1, max_steps + 1):
            step_cost = 0.002  # estimated per-turn cost unless provider returns exact usage
            if spent + step_cost > budget:
                raise TigoError(
                    f"Line {line}: agent {ag_name!r} exceeded budget ${budget:.4f} (spent ${spent:.4f})"
                )
            spent += step_cost
            raw = self.call_model(ag, history, schema_hint, expected, step)

            # Support tool call directives from model: {"tool": "name", "args": [...]}
            if isinstance(raw, dict) and "tool" in raw and raw["tool"] in ag.config.get("tools", []):
                tname = raw["tool"]
                targs = raw.get("args", [])
                tres = self.invoke_tool(tname, targs, line)
                history.append({"role": "assistant", "content": json.dumps(raw)})
                history.append({"role": "tool", "name": tname, "content": json.dumps(tres)})
                continue

            try:
                val = validate_type(self.prog, expected, raw, ag_name)
                self.store.save_step(self.run_id, key, "ask", f"{ag_name}: {prompt[:80]}", val, round(spent, 6))
                return val
            except ValueError as ve:
                # Auto-repair loop: feed schema validation error back to model within max_steps
                history.append({"role": "assistant", "content": json.dumps(raw)})
                history.append({"role": "user", "content": f"Schema validation error: {ve}. Return valid JSON."})

        raise TigoError(f"Line {line}: agent {ag_name!r} exceeded max_steps ({max_steps}) without valid output")

    def call_model(self, ag: AgentDecl, history: list[dict], schema: dict, expected: TypeRef | None, step: int) -> Any:
        if self.mock_llm is not None:
            return self.mock_llm(ag.name, history, schema, step)

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TIGO_API_KEY")
        if not api_key:
            # Deterministic offline fallback when no API key is set
            return self._offline_synth(expected)

        api_url = os.environ.get("TIGO_API_URL", "https://api.openai.com/v1/chat/completions")
        sys_msg = ag.config.get("system", "You are a helpful AI agent.")
        if expected and expected.base != "string":
            sys_msg += f"\nRespond ONLY with JSON matching this schema: {json.dumps(schema)}"
        payload = {
            "model": ag.config["model"],
            "messages": [{"role": "system", "content": sys_msg}] + history,
            "temperature": 0,
        }
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        if expected and expected.base != "string":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text  # let validate_type trigger the auto-repair loop
        return text

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
            td = self.prog.types[tref.base]
            return {k: self._offline_synth(ft) for k, ft in td.fields.items()}
        return "mock"


# --- 6. Public Entry Points & CLI ---

def compile_source(src: str) -> Program:
    prog = Parser(lex(src)).parse()
    check_program(prog)
    return prog


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage:\n  tigo.py check <file.tg>\n  tigo.py run <file.tg> [workflow] [key=val ...] [--approve]\n  tigo.py replay <run_id> [--approve]\n  tigo.py trace <run_id>")
        return 0

    cmd = args[0]
    store = DurableStore(os.environ.get("TIGO_DB", ".tigo/state.db"))

    try:
        if cmd == "check":
            path = args[1]
            prog = compile_source(Path(path).read_text(encoding="utf-8"))
            print(f"OK: {path} ({len(prog.types)} types, {len(prog.tools)} tools, {len(prog.agents)} agents, {len(prog.workflows)} workflows)")
            return 0

        if cmd == "run":
            path = args[1]
            prog = compile_source(Path(path).read_text(encoding="utf-8"))
            rest = [a for a in args[2:] if a != "--approve"]
            approved = "--approve" in args[2:]
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
            rt = Runtime(prog, store, run_id, approved=approved)
            out = rt.run_workflow(wf_name, wf_inputs)
            print(json.dumps({"run_id": run_id, "status": "completed", "result": out}, indent=2))
            return 0

        if cmd == "replay":
            run_id = args[1]
            approved = "--approve" in args[2:]
            row = store.get_run(run_id)
            if not row:
                raise TigoError(f"Unknown run_id {run_id!r}")
            _, path, wf_name, args_json, _, _ = row
            prog = compile_source(Path(path).read_text(encoding="utf-8"))
            store.set_status(run_id, "running")
            rt = Runtime(prog, store, run_id, approved=approved)
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
    except TigoError as te:
        print(f"TigoError: {te}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
