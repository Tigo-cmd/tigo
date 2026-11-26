"""Lexer, AST, recursive-descent parser, and multi-file import resolver for Tigo."""
import json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOKEN_SPEC = [
    ("SKIP", r"[ \t\r\n]+|//[^\n]*"),
    ("MONEY", r"\$\d+(?:\.\d+)?"),
    ("FLOAT", r"\d+\.\d+"),
    ("INT", r"\d+"),
    ("MSTR", r'"""[\s\S]*?"""'),
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


def format_error(err: Exception, src: str, filename: str = "<input>") -> str:
    msg = str(err)
    m = re.match(r"Line (\d+):\s*(.*)", msg)
    if not m:
        return f"TigoError ({filename}): {msg}"
    line_no, detail = int(m.group(1)), m.group(2)
    lines = src.splitlines()
    if 1 <= line_no <= len(lines):
        code_line = lines[line_no - 1]
        indent = len(code_line) - len(code_line.lstrip(" "))
        pointer = " " * indent + "^" * max(1, len(code_line.strip()))
        return f"{filename}:{line_no}: TigoError: {detail}\n  {line_no:4d} | {code_line}\n       | {pointer}"
    return f"{filename}:{line_no}: TigoError: {detail}"


def decode_str(tok: Tok) -> str:
    if tok.kind == "MSTR":
        body = tok.val[3:-3]
        return body[1:] if body.startswith("\n") else body
    return json.loads(tok.val)


def lex(src: str) -> list[Tok]:
    toks, line, pos = [], 1, 0
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
    base: str
    arg: Any = None


@dataclass
class TypeDecl:
    name: str
    fields: dict[str, TypeRef]
    constraints: dict[str, tuple]
    line: int


@dataclass
class ToolDecl:
    name: str
    params: list[tuple[str, TypeRef]]
    ret: TypeRef
    allows: list[tuple[str, str]]
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
    imports: list[str] = field(default_factory=list)
    types: dict[str, TypeDecl] = field(default_factory=dict)
    tools: dict[str, ToolDecl] = field(default_factory=dict)
    agents: dict[str, AgentDecl] = field(default_factory=dict)
    workflows: dict[str, WorkflowDecl] = field(default_factory=dict)
    tests: dict[str, WorkflowDecl] = field(default_factory=dict)


class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks, self.i = toks, 0

    def cur(self) -> Tok:
        return self.toks[self.i]

    def eat(self, val: str | None = None, kind: str | None = None) -> Tok:
        t = self.cur()
        if (val and t.val != val) or (kind and t.kind != kind):
            raise TigoError(f"Line {t.line}: expected {val or kind!r}, got {t.val!r}")
        self.i += 1
        return t

    def eat_str(self) -> str:
        t = self.cur()
        if t.kind not in ("STR", "MSTR"):
            raise TigoError(f"Line {t.line}: expected string, got {t.val!r}")
        self.i += 1
        return decode_str(t)

    def match(self, val: str) -> bool:
        if self.cur().val == val:
            self.i += 1
            return True
        return False

    def parse(self) -> Program:
        prog = Program()
        while self.cur().kind != "EOF":
            t = self.cur()
            if t.val == "import":
                self.eat("import")
                prog.imports.append(self.eat_str())
            elif t.val == "type":
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
            elif t.val == "test":
                d = self.parse_workflow_decl()
                prog.tests[d.name] = d
            else:
                raise TigoError(f"Line {t.line}: unexpected top-level token {t.val!r}")
        return prog

    def parse_type_ref(self) -> TypeRef:
        if self.cur().kind == "STR":
            opts = [self.eat_str()]
            while self.match("|"):
                opts.append(self.eat_str())
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
        fields, constraints = {}, {}
        while not self.match("}"):
            fname = self.eat(kind="ID").val
            self.eat(":")
            fields[fname] = self.parse_type_ref()
            if self.match("where"):
                constraints[fname] = self.parse_expr()
            self.match(",")
        return TypeDecl(name, fields, constraints, line)

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
        ret = self.parse_type_ref() if self.match("->") else TypeRef("void")
        allows = []
        while self.match("allow"):
            cap = self.eat(kind="ID").val
            if self.match("."):
                cap += "." + self.eat(kind="ID").val
            self.eat("(")
            pat = self.eat_str()
            self.eat(")")
            allows.append((cap, pat))
            self.match(",")
        return ToolDecl(name, params, ret, allows, self.parse_block(), line)

    def parse_agent_decl(self) -> AgentDecl:
        line = self.eat("agent").line
        name = self.eat(kind="ID").val
        self.eat("{")
        cfg: dict[str, Any] = {}
        while not self.match("}"):
            k = self.eat(kind="ID").val
            self.eat(":")
            t = self.cur()
            if t.kind in ("STR", "MSTR"):
                val: Any = self.eat_str()
                if k == "model" and self.match("fallback"):
                    cfg["fallback"] = self.eat_str()
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
        ret = self.parse_type_ref() if self.match("->") else TypeRef("void")
        return WorkflowDecl(name, params, ret, self.parse_block(), line)

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
            return ("for", var, self.parse_expr(), self.parse_block(), t.line)
        if t.val == "parallel":
            self.eat("parallel")
            self.eat("for")
            var = self.eat(kind="ID").val
            self.eat("in")
            return ("parfor", var, self.parse_expr(), self.parse_block(), t.line)
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

    PREC = [("||",), ("&&",), ("==", "!=", "<", "<=", ">", ">="), ("+", "-"), ("*", "/")]

    def parse_expr(self) -> tuple:
        return self.parse_binop(0)

    def parse_binop(self, lvl: int) -> tuple:
        if lvl >= len(self.PREC):
            return self.parse_postfix()
        left = self.parse_binop(lvl + 1)
        while self.cur().val in self.PREC[lvl]:
            op = self.eat().val
            left = ("binop", op, left, self.parse_binop(lvl + 1), left[-1])
        return left

    def parse_postfix(self) -> tuple:
        node = self.parse_primary()
        while True:
            if self.match("."):
                attr = self.eat(kind="ID").val
                if self.cur().val == "(":
                    node = ("mcall", node, attr, self.parse_args(), node[-1])
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
        if t.val == "parallel":
            self.eat("parallel")
            self.eat("for")
            var = self.eat(kind="ID").val
            self.eat("in")
            return ("parfor", var, self.parse_expr(), self.parse_block(), t.line)
        if t.val == "ask":
            self.eat("ask")
            ag = self.eat(kind="ID").val
            self.eat("(")
            prompt = self.parse_expr()
            self.eat(")")
            return ("ask", ag, prompt, t.line)
        if t.kind in ("STR", "MSTR"):
            return ("str", self.eat_str(), t.line)
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
                    k = self.eat(kind="ID").val if self.cur().kind == "ID" else self.eat_str()
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


def compile_source(src: str, base_dir: Path | None = None, seen: set[Path] | None = None) -> Program:
    from .checker import check_program

    prog = Parser(lex(src)).parse()
    base = base_dir or Path.cwd()
    visited = seen if seen is not None else set()
    for imp in prog.imports:
        imp_path = (base / imp).resolve()
        if imp_path in visited:
            continue
        visited.add(imp_path)
        if not imp_path.exists():
            raise TigoError(f"Imported file not found: {imp!r}")
        sub = compile_source(imp_path.read_text(encoding="utf-8"), imp_path.parent, visited)
        prog.types = {**sub.types, **prog.types}
        prog.tools = {**sub.tools, **prog.tools}
        prog.agents = {**sub.agents, **prog.agents}
        prog.workflows = {**sub.workflows, **prog.workflows}
    check_program(prog)
    return prog


def compile_file(path: str | Path) -> Program:
    p = Path(path).resolve()
    return compile_source(p.read_text(encoding="utf-8"), p.parent, {p})
