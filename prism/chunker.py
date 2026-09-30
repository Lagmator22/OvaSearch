"""Split source files into function/class/method chunks with tree-sitter.

Python and JavaScript are parsed with tree-sitter. Code between definitions
(imports, module level statements) is grouped into line windows so nothing in
a file is lost. Any other text file, or a file that fails to parse, falls back
to plain line windows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import PurePosixPath

WINDOW_LINES = 40      # size of fallback / gap windows
MAX_CHUNK_LINES = 150  # definitions longer than this are split into parts

PY_EXT = {".py"}
JS_EXT = {".js", ".mjs", ".cjs", ".jsx"}
TEXT_EXT = {".ts", ".tsx", ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".go", ".rs",
            ".rb", ".php", ".cs", ".kt", ".swift", ".scala", ".sh", ".md", ".json",
            ".yml", ".yaml", ".toml", ".txt"}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "vendor", "__pycache__", ".prism_index"}


@dataclass
class Chunk:
    path: str
    symbol: str
    kind: str        # function | class | method | module | window
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text given to the embedder and BM25: a short header plus the code."""
        return f"# file: {self.path} | {self.kind}: {self.symbol}\n{self.text}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.embed_text.encode("utf-8", "replace")).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hash"] = self.content_hash
        return d


def is_indexable(path: str) -> bool:
    p = PurePosixPath(path)
    if any(part in SKIP_DIRS for part in p.parts):
        return False
    if p.name.endswith(".min.js") or p.name in {"package-lock.json", "yarn.lock"}:
        return False
    return p.suffix.lower() in PY_EXT | JS_EXT | TEXT_EXT


@lru_cache(maxsize=None)
def _parser(lang: str):
    from tree_sitter import Language, Parser
    if lang == "python":
        import tree_sitter_python as mod
    else:
        import tree_sitter_javascript as mod
    return Parser(Language(mod.language()))


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _name(node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    return _text(n, src) if n is not None else None


FUNC_VALUES = {"function_expression", "function", "arrow_function", "generator_function",
               "class"}


def _py_defs(node, src, prefix, out):
    for child in node.named_children:
        target = child
        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition") or child
        if target.type == "function_definition":
            name = prefix + (_name(target, src) or "<anon>")
            out.append((child, name, "method" if prefix else "function"))
        elif target.type == "class_definition":
            name = prefix + (_name(target, src) or "<anon>")
            body = target.child_by_field_name("body")
            methods: list = []
            if body is not None:
                _py_defs(body, src, name + ".", methods)
            out.append((child, name, "class", methods))
            out.extend(methods)


def _js_defs(node, src, prefix, out):
    for child in node.named_children:
        target = child
        if child.type == "export_statement":
            target = child.child_by_field_name("declaration") or child
        t = target.type
        if t in ("function_declaration", "generator_function_declaration"):
            out.append((child, prefix + (_name(target, src) or "<anon>"), "function"))
        elif t == "class_declaration":
            name = prefix + (_name(target, src) or "<anon>")
            methods: list = []
            body = target.child_by_field_name("body")
            if body is not None:
                for m in body.named_children:
                    if m.type == "method_definition":
                        methods.append((m, f"{name}.{_name(m, src)}", "method"))
            out.append((child, name, "class", methods))
            out.extend(methods)
        elif t in ("lexical_declaration", "variable_declaration"):
            for decl in target.named_children:
                if decl.type != "variable_declarator":
                    continue
                val = decl.child_by_field_name("value")
                if val is not None and val.type in FUNC_VALUES:
                    out.append((child, prefix + (_name(decl, src) or "<anon>"), "function"))
                    break
        elif t == "expression_statement":
            expr = target.named_children[0] if target.named_children else None
            if expr is not None and expr.type == "assignment_expression":
                right = expr.child_by_field_name("right")
                if right is not None and right.type in FUNC_VALUES:
                    left = _text(expr.child_by_field_name("left"), src)
                    out.append((child, prefix + left, "function"))


def _leading_comment_start(node) -> int:
    """First row of the comment block (JSDoc, "#" lines) above a definition, one blank line allowed."""
    start = node.start_point[0]
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment" and prev.end_point[0] >= start - 2:
        start = prev.start_point[0]
        prev = prev.prev_sibling
    return start


def _windows(lines, start, end, path, symbol, kind):
    """Line windows over lines[start:end] (0-based, end exclusive); skips blank windows."""
    chunks = []
    for s in range(start, end, WINDOW_LINES):
        e = min(s + WINDOW_LINES, end)
        body = "\n".join(lines[s:e])
        if body.strip():
            chunks.append(Chunk(path, symbol, kind, s + 1, e, body))
    return chunks


def _split_long(chunk: Chunk) -> list[Chunk]:
    n = chunk.end_line - chunk.start_line + 1
    if n <= MAX_CHUNK_LINES:
        return [chunk]
    lines = chunk.text.split("\n")
    parts = []
    for i, s in enumerate(range(0, len(lines), MAX_CHUNK_LINES)):
        seg = lines[s:s + MAX_CHUNK_LINES]
        parts.append(Chunk(chunk.path, f"{chunk.symbol}#part{i + 1}", chunk.kind,
                           chunk.start_line + s, chunk.start_line + s + len(seg) - 1,
                           "\n".join(seg)))
    return parts


def chunk_file(path: str, source: str) -> list[Chunk]:
    suffix = PurePosixPath(path).suffix.lower()
    lang = "python" if suffix in PY_EXT else "javascript" if suffix in JS_EXT else None
    lines = source.split("\n")
    if lang is None:
        return _windows(lines, 0, len(lines), path, PurePosixPath(path).name, "window")

    src = source.encode("utf-8")
    try:
        tree = _parser(lang).parse(src)
    except Exception:
        return _windows(lines, 0, len(lines), path, PurePosixPath(path).name, "window")
    root = tree.root_node
    defs: list = []
    (_py_defs if lang == "python" else _js_defs)(root, src, "", defs)
    if not defs and root.has_error:
        return _windows(lines, 0, len(lines), path, PurePosixPath(path).name, "window")

    chunks: list[Chunk] = []
    covered = [False] * len(lines)
    for d in defs:
        node, name, kind = d[0], d[1], d[2]
        s, e = _leading_comment_start(node), node.end_point[0]
        if kind == "class" and len(d) > 3 and d[3]:
            # Class chunk = header and fields up to the first method; methods are separate.
            first_method = min(m[0].start_point[0] for m in d[3])
            e = max(s, first_method - 1)
        text = "\n".join(lines[s:e + 1])
        chunks.extend(_split_long(Chunk(path, name, kind, s + 1, e + 1, text)))
        for i in range(s, min(e + 1, len(lines))):
            covered[i] = True

    # Module level code that is not inside any definition.
    i = 0
    while i < len(lines):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(lines) and not covered[j]:
            j += 1
        chunks.extend(_windows(lines, i, j, path, "<module>", "module"))
        i = j
    chunks.sort(key=lambda c: c.start_line)
    return chunks
