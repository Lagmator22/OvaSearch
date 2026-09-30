"""Tests for chunking, indexing, incremental reindex and search.

A temporary git repo is built from tests/fixtures/sample_repo with two commits:
v1 is the fixture as is, v2 edits computeTotal and adds a new file.
The embedding model is real (CodeRankEmbed by default, override with PRISM_MODEL).
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from prism.chunker import chunk_file
from prism.index import IndexStore, bm25_tokens, classify_query

FIXTURE = Path(__file__).parent / "fixtures" / "sample_repo"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture(scope="session")
def embedder():
    from prism.embedder import Embedder
    return Embedder()


@pytest.fixture(scope="session")
def repo(tmp_path_factory):
    r = tmp_path_factory.mktemp("repo")
    shutil.copytree(FIXTURE, r, dirs_exist_ok=True)
    git(r, "init", "-q")
    git(r, "config", "user.email", "test@example.com")
    git(r, "config", "user.name", "test")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "v1")
    git(r, "tag", "v1")
    cart = r / "src" / "cart.js"
    cart.write_text(cart.read_text().replace(
        "  return subtotal * (1 + TAX_RATE);",
        "  const total = subtotal * (1 + TAX_RATE);\n  return Math.round(total * 100) / 100;"))
    (r / "src" / "format.js").write_text(
        "/**\n * Format a number of cents as a currency string like $12.50.\n */\n"
        "function formatCurrency(cents) {\n  return '$' + (cents / 100).toFixed(2);\n}\n\n"
        "module.exports = { formatCurrency };\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "v2")
    git(r, "tag", "v2")
    return r


@pytest.fixture(scope="session")
def store(repo, embedder, tmp_path_factory):
    s = IndexStore(tmp_path_factory.mktemp("index"))
    s.stats_v1 = s.index(repo, "v1", embedder)
    s.stats_v2 = s.index(repo, "v2", embedder)
    return s


# ---------------------------------------------------------------- chunking

def test_js_chunks_functions_classes_methods():
    src = (FIXTURE / "src" / "auth.js").read_text()
    chunks = {c.symbol: c for c in chunk_file("src/auth.js", src)}
    for sym in ["hashPassword", "verifyPassword", "SessionStore", "SessionStore.get",
                "SessionStore.set", "SessionStore.destroy"]:
        assert sym in chunks, sym
    hp = chunks["hashPassword"]
    assert hp.kind == "function"
    # JSDoc above the function belongs to the chunk.
    assert hp.text.lstrip().startswith("/**")
    assert "function hashPassword" in hp.text
    lines = src.split("\n")
    assert lines[hp.start_line - 1].startswith("/**")
    assert lines[hp.end_line - 1] == "}"
    assert chunks["SessionStore.get"].kind == "method"


def test_js_arrow_function_and_python_chunks():
    js = {c.symbol for c in chunk_file("src/http.js", (FIXTURE / "src/http.js").read_text())}
    assert {"parseQueryString", "fetchWithRetry"} <= js
    py = {c.symbol: c for c in chunk_file("tools/text_utils.py",
                                          (FIXTURE / "tools/text_utils.py").read_text())}
    assert {"slugify", "levenshtein", "WordCounter", "WordCounter.add",
            "WordCounter.most_common"} <= set(py)
    assert py["WordCounter.add"].kind == "method"


def test_fallback_for_unparseable_and_other_files():
    md = chunk_file("README.md", "\n".join(f"line {i}" for i in range(100)))
    assert [c.kind for c in md] == ["window"] * 3
    assert (md[0].start_line, md[0].end_line) == (1, 40)
    broken = chunk_file("bad.js", "}}}} ((( this is not javascript {{{")
    assert broken and all(c.kind == "window" for c in broken)


def test_every_line_covered():
    src = (FIXTURE / "src" / "cart.js").read_text()
    covered = set()
    for c in chunk_file("src/cart.js", src):
        covered.update(range(c.start_line, c.end_line + 1))
    nonblank = {i + 1 for i, ln in enumerate(src.split("\n")) if ln.strip()}
    assert nonblank <= covered


def test_bm25_tokens_split_identifiers_and_classify():
    toks = bm25_tokens("computeTotal parse_query_string")
    assert {"computetotal", "compute", "total", "parse_query_string", "parse", "query"} <= set(toks)
    assert classify_query("computeTotal") == "identifier"
    assert classify_query("res.send") == "identifier"
    assert classify_query("hash a password with salt") == "natural"


# ---------------------------------------------------------------- indexing

def test_first_index_embeds_everything(store):
    s = store.stats_v1
    assert s["reused"] == 0
    assert s["embedded"] + s["reused_dupes_in_rev"] == s["chunks"]
    assert s["chunks"] >= 15


def test_incremental_reindex_reuses_unchanged_chunks(store):
    s = store.stats_v2
    # Only computeTotal changed and format.js was added; module level chunks of
    # cart.js / format.js may also be new. Everything else must come from cache.
    assert s["embedded"] <= 4
    assert s["reused"] >= store.stats_v1["chunks"] - 2
    assert s["reused"] + s["embedded"] + s["reused_dupes_in_rev"] == s["chunks"]


def test_reindex_same_revision_is_free(store, repo, embedder):
    again = store.index(repo, "v2", embedder)
    assert again["embedded"] == 0
    assert [r["label"] for r in store.revisions()] == ["v1", "v2"]


# ---------------------------------------------------------------- search

@pytest.mark.parametrize("query,path,symbol", [
    ("hash a password with a random salt", "src/auth.js", "hashPassword"),
    ("retry http request with exponential backoff", "src/http.js", "fetchWithRetry"),
    ("edit distance between two strings", "tools/text_utils.py", "levenshtein"),
    ("get a session and return null if it has expired", "src/auth.js", "SessionStore.get"),
    ("computeTotal", "src/cart.js", "computeTotal"),
    ("format cents as a dollar string", "src/format.js", "formatCurrency"),
])
def test_search_finds_expected_chunk(store, embedder, query, path, symbol):
    _, hits = store.search(query, embedder.encode_query(query), rev=None, k=3)
    top = [(h.chunk["path"], h.chunk["symbol"]) for h in hits]
    assert (path, symbol) in top, top


def test_search_respects_revision(store, embedder):
    q = "format cents as a dollar string"
    _, hits = store.search(q, embedder.encode_query(q), rev="v1", k=10)
    assert all(h.chunk["path"] != "src/format.js" for h in hits)


def test_all_versions_groups_near_duplicates(store, embedder):
    q = "compute the cart total including tax"
    fams = store.search_all_versions(q, embedder.encode_query(q), k=5)
    top = fams[0]
    assert top["best"]["symbol"] == "computeTotal"
    revs = [r for v in top["versions"] for r in v["revs"]]
    assert sorted(revs) == ["v1", "v2"]
    assert len(top["versions"]) == 2  # two different texts, grouped as one family
    assert all(v["sim_to_best"] >= 0.95 for v in top["versions"])

    q = "hash a password with a random salt"
    fams = store.search_all_versions(q, embedder.encode_query(q), k=5)
    hp = next(f for f in fams if f["best"]["symbol"] == "hashPassword")
    # Unchanged function: one version present in both revisions.
    assert len(hp["versions"]) == 1 and hp["versions"][0]["revs"] == ["v1", "v2"]
