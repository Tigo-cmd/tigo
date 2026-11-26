"""Static type, prompt interpolation, guardrail, and capability checker for Tigo."""
from .syntax import INTERP_RE, Program, TigoError, TypeRef

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
        for fn, ft in td.fields.items():
            check_type_exists(ft, td.line)
            if fn in td.constraints:
                check_expr(prog, td.constraints[fn], {fn: ft}, is_tool=False)

    for tool in prog.tools.values():
        for _, pt in tool.params:
            check_type_exists(pt, tool.line)
        check_type_exists(tool.ret, tool.line)
        for cap, pat in tool.allows:
            if cap not in VALID_CAPS:
                raise TigoError(f"Line {tool.line}: tool {tool.name!r} uses invalid permission {cap!r}")
            if not pat:
                raise TigoError(f"Line {tool.line}: tool {tool.name!r} has empty permission pattern")
        check_stmts(prog, tool.body, {k: v for k, v in tool.params}, is_tool=True)

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

    for wf in list(prog.workflows.values()) + list(prog.tests.values()):
        for _, pt in wf.params:
            check_type_exists(pt, wf.line)
        check_type_exists(wf.ret, wf.line)
        check_stmts(prog, wf.body, {k: v for k, v in wf.params}, is_tool=False)


def check_stmts(prog: Program, stmts: list[tuple], scope: dict[str, TypeRef], is_tool: bool) -> None:
    for s in stmts:
        kind, line = s[0], s[-1]
        if kind == "let":
            _, name, tref, expr, _ = s
            if tref and tref.base not in PRIMITIVES and tref.base not in ("list", "union") and tref.base not in prog.types:
                raise TigoError(f"Line {line}: unknown type {tref.base!r}")
            et = check_expr(prog, expr, scope, is_tool)
            scope[name] = tref or et
        elif kind == "if":
            _, cond, tb, eb, _ = s
            check_expr(prog, cond, scope, is_tool)
            check_stmts(prog, tb, dict(scope), is_tool)
            check_stmts(prog, eb, dict(scope), is_tool)
        elif kind in ("for", "parfor"):
            _, var, it, body, _ = s
            it_t = check_expr(prog, it, scope, is_tool)
            sub = dict(scope)
            sub[var] = it_t.arg if it_t.base == "list" else TypeRef("any")
            check_stmts(prog, body, sub, is_tool)
        elif kind in ("return", "approve", "expr"):
            check_expr(prog, s[1], scope, is_tool)


def check_expr(prog: Program, e: tuple, scope: dict[str, TypeRef], is_tool: bool) -> TypeRef:
    kind, line = e[0], e[-1]
    if kind == "lit":
        v = e[1]
        return TypeRef("bool") if isinstance(v, bool) else TypeRef("int" if isinstance(v, int) else "float")
    if kind == "str":
        for var in INTERP_RE.findall(e[1]):
            root = var.strip().split(".")[0]
            if root not in scope:
                raise TigoError(f"Line {line}: prompt/string interpolation references undefined variable {root!r}")
        return TypeRef("string")
    if kind == "var":
        if e[1] in ("fs", "http", "sh"):
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
        if obj_t.base in prog.types:
            td = prog.types[obj_t.base]
            if e[2] not in td.fields:
                raise TigoError(f"Line {line}: type {obj_t.base!r} has no field {e[2]!r}")
            return td.fields[e[2]]
        return TypeRef("any")
    if kind == "mcall":
        obj_t = check_expr(prog, e[1], scope, is_tool)
        for a in e[3]:
            check_expr(prog, a, scope, is_tool)
        if obj_t.base in ("fs", "http", "sh") and not is_tool:
            raise TigoError(f"Line {line}: capability call '{obj_t.base}.{e[2]}' is only allowed inside a tool")
        return TypeRef("string")
    if kind == "call":
        fn, args = e[1], e[2]
        for a in args:
            check_expr(prog, a, scope, is_tool)
        if fn in ("print", "assert"):
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
    if kind == "parfor":
        _, var, it, body, _ = e
        it_t = check_expr(prog, it, scope, is_tool)
        sub = dict(scope)
        sub[var] = it_t.arg if it_t.base == "list" else TypeRef("any")
        check_stmts(prog, body, sub, is_tool)
        return TypeRef("list", TypeRef("any"))
    if kind == "ask":
        if e[1] not in prog.agents:
            raise TigoError(f"Line {line}: unknown agent {e[1]!r}")
        check_expr(prog, e[2], scope, is_tool)
        return TypeRef("any")
    return TypeRef("any")
