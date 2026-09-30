"""Per-revision code index with a content-hash embedding cache.

Layout of an index directory:
    meta.json              repo path, model name, list of indexed revisions
    vectors.npy            one row per unique chunk hash (float32, L2 normalised)
    vectors_keys.json      chunk hash for each row of vectors.npy
    revs/<sha>.json        chunks (path, symbol, lines, text, hash) of one revision

Chunks are keyed by the sha1 of their text (plus a path/symbol header), so a new
revision only embeds chunks whose text changed. Everything else is reused.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunker import chunk_file, is_indexable

MAX_FILE_BYTES = 1_000_000
WORKTREE = "WORKTREE"


# ---------------------------------------------------------------- git helpers

def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=input_bytes,
                          capture_output=True, check=True).stdout


def is_git_repo(repo: Path) -> bool:
    try:
        return _git(repo, "rev-parse", "--is-inside-work-tree").strip() == b"true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def resolve_rev(repo: Path, rev: str | None) -> tuple[str, str]:
    """Return (label, commit sha). Non git folders index the working tree."""
    if not is_git_repo(repo):
        if rev not in (None, WORKTREE):
            raise SystemExit(f"{repo} is not a git repo, so --rev {rev} cannot be used")
        return WORKTREE, WORKTREE
    rev = rev or "HEAD"
    sha = _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()
    return rev, sha


def read_files(repo: Path, sha: str) -> dict[str, str]:
    """All indexable text files of a commit (or the working tree)."""
    out: dict[str, str] = {}
    if sha == WORKTREE:
        for p in sorted(repo.rglob("*")):
            rel = p.relative_to(repo).as_posix()
            if p.is_file() and is_indexable(rel) and p.stat().st_size <= MAX_FILE_BYTES:
                out[rel] = p.read_text("utf-8", errors="replace")
        return out

    entries = []
    for rec in _git(repo, "ls-tree", "-r", "-z", "--long", sha).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        _mode, typ, blob, size = meta.split()
        path_s = path.decode("utf-8", "replace")
        if typ == b"blob" and size != b"-" and int(size) <= MAX_FILE_BYTES and is_indexable(path_s):
            entries.append((path_s, blob.decode()))
    if not entries:
        return out
    # One git process for all blobs.
    raw = _git(repo, "cat-file", "--batch",
               input_bytes="\n".join(b for _, b in entries).encode() + b"\n")
    pos = 0
    for path_s, _ in entries:
        header_end = raw.index(b"\n", pos)
        size = int(raw[pos:header_end].split()[2])
        body = raw[header_end + 1: header_end + 1 + size]
        pos = header_end + 1 + size + 1
        out[path_s] = body.decode("utf-8", "replace")
    return out


# ---------------------------------------------------------------- tokenising for BM25

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def bm25_tokens(text: str) -> list[str]:
    """Identifiers plus their camelCase / snake_case parts, lower-cased."""
    toks = []
    for t in _TOKEN.findall(text):
        low = t.lower()
        toks.append(low)
        parts = [p.lower() for piece in t.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            toks.extend(parts)
    return toks


IDENT_LIKE = re.compile(r"^[\w.$]+(\(\))?$")


def classify_query(query: str) -> str:
    """'identifier' for things like getUserById / res.send / parse_args, else 'natural'."""
    q = query.strip()
    if IDENT_LIKE.match(q) and (("_" in q) or ("." in q) or re.search(r"[a-z][A-Z]", q)
                                 or q.endswith("()")):
        return "identifier"
    return "natural"


def rrf(rankings: list[list[int]], weights: list[float], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return scores


# ---------------------------------------------------------------- store

@dataclass
class Hit:
    chunk: dict
    score: float
    dense: float


class IndexStore:
    def __init__(self, index_dir: str | Path):
        self.dir = Path(index_dir)
        self.meta_path = self.dir / "meta.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        self._vec: np.ndarray | None = None
        self._keys: list[str] | None = None
        self._row: dict[str, int] | None = None

    # ---- vectors cache
    def _load_vectors(self):
        if self._vec is not None:
            return
        kp, vp = self.dir / "vectors_keys.json", self.dir / "vectors.npy"
        if kp.exists() and vp.exists():
            self._keys = json.loads(kp.read_text())
            self._vec = np.load(vp)
        else:
            self._keys, self._vec = [], None
        self._row = {h: i for i, h in enumerate(self._keys)}

    def _save_vectors(self):
        np.save(self.dir / "vectors.npy", self._vec)
        (self.dir / "vectors_keys.json").write_text(json.dumps(self._keys))

    def vectors_for(self, hashes: list[str]) -> np.ndarray:
        self._load_vectors()
        return self._vec[[self._row[h] for h in hashes]]

    def _save_meta(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.meta, indent=2, default=str))

    # ---- indexing
    def index(self, repo: str | Path, rev: str | None, embedder) -> dict:
        t0 = time.time()
        repo = Path(repo).resolve()
        label, sha = resolve_rev(repo, rev)
        if self.meta.get("model") and self.meta["model"] != embedder.model_name:
            raise SystemExit(f"index at {self.dir} was built with {self.meta['model']}; "
                             f"use another --index-dir for {embedder.model_name}")
        if self.meta.get("repo") and self.meta["repo"] != str(repo):
            print(f"note: {self.dir} held {self.meta['repo']}, starting a new revision list "
                  f"for {repo} (vector cache is kept)")
            self.meta["revisions"] = []
        self.meta.update(repo=str(repo), model=embedder.model_name)
        self.meta.setdefault("revisions", [])

        files = read_files(repo, sha)
        chunks = [c for path, src in files.items() for c in chunk_file(path, src)]
        t_chunk = time.time() - t0

        self._load_vectors()
        new = {}
        for c in chunks:
            h = c.content_hash
            if h not in self._row and h not in new:
                new[h] = c.embed_text
        reused = sum(1 for c in chunks if c.content_hash in self._row)
        t1 = time.time()
        if new:
            vecs = embedder.encode_docs(list(new.values()))
            start = len(self._keys)
            self._keys.extend(new.keys())
            self._vec = vecs if self._vec is None else np.vstack([self._vec, vecs])
            for i, h in enumerate(new.keys()):
                self._row[h] = start + i
        t_embed = time.time() - t1

        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "revs").mkdir(exist_ok=True)
        (self.dir / "revs" / f"{sha}.json").write_text(
            json.dumps([c.to_dict() for c in chunks]))
        if new:
            self._save_vectors()
        entry = {"label": label, "sha": sha, "n_files": len(files), "n_chunks": len(chunks),
                 "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        revs = self.meta["revisions"]
        pos = next((i for i, r in enumerate(revs) if r["sha"] == sha), None)
        if pos is None:
            revs.append(entry)
        else:
            revs[pos] = entry  # re-indexing keeps the revision's place in the list
        self.meta["default_sha"] = sha
        self._save_meta()
        return {"label": label, "sha": sha, "files": len(files), "chunks": len(chunks),
                "reused": reused, "embedded": len(new),
                "reused_dupes_in_rev": len(chunks) - reused - len(new),
                "chunk_s": t_chunk, "embed_s": t_embed, "total_s": time.time() - t0}

    # ---- search
    def revisions(self) -> list[dict]:
        return self.meta.get("revisions", [])

    def find_rev(self, rev: str | None) -> dict:
        revs = self.revisions()
        if not revs:
            raise SystemExit(f"no index found in {self.dir}; run `python -m prism index <repo>` first")
        if rev is None:
            return next(r for r in revs if r["sha"] == self.meta["default_sha"])
        for r in revs:
            if rev in (r["label"], r["sha"]) or (len(rev) >= 6 and r["sha"].startswith(rev)):
                return r
        # Maybe a ref name that points to an indexed commit.
        try:
            _, sha = resolve_rev(Path(self.meta["repo"]), rev)
            return next(r for r in revs if r["sha"] == sha)
        except (subprocess.CalledProcessError, StopIteration, SystemExit):
            known = ", ".join(r["label"] for r in revs)
            raise SystemExit(f"revision {rev} is not indexed (indexed: {known})")

    def load_chunks(self, sha: str) -> list[dict]:
        return json.loads((self.dir / "revs" / f"{sha}.json").read_text())

    def rank(self, chunks: list[dict], qvec: np.ndarray, query: str, mode: str = "hybrid",
             depth: int = 100) -> list[Hit]:
        from rank_bm25 import BM25Okapi

        vecs = self.vectors_for([c["hash"] for c in chunks])
        dense = vecs @ qvec
        dense_rank = list(np.argsort(-dense)[:depth])
        if mode == "dense":
            fused = {int(i): float(dense[i]) for i in dense_rank}
        else:
            bm25 = BM25Okapi([bm25_tokens(c["path"] + " " + c["symbol"] + "\n" + c["text"])
                              for c in chunks])
            sparse = bm25.get_scores(bm25_tokens(query))
            sparse_rank = [int(i) for i in np.argsort(-sparse)[:depth] if sparse[i] > 0]
            if mode == "bm25":
                fused = {i: float(sparse[i]) for i in sparse_rank}
            else:
                # Identifier-like queries lean on exact token match, sentences on the embedding.
                w = [1.0, 2.0] if classify_query(query) == "identifier" else [1.0, 1.0]
                fused = rrf([[int(i) for i in dense_rank], sparse_rank], w)
        order = sorted(fused, key=lambda i: (-fused[i], -dense[i]))
        return [Hit(chunks[i], fused[i], float(dense[i])) for i in order]

    def search(self, query: str, qvec: np.ndarray, rev: str | None, k: int,
               mode: str = "hybrid") -> tuple[dict, list[Hit]]:
        r = self.find_rev(rev)
        return r, self.rank(self.load_chunks(r["sha"]), qvec, query, mode)[:k]

    def search_all_versions(self, query: str, qvec: np.ndarray, k: int, mode: str = "hybrid",
                            sim_threshold: float = 0.95) -> list[dict]:
        """Search the union of all indexed revisions and group near-duplicates.

        A family is one symbol (same name) whose versions are identical or have
        cosine similarity >= sim_threshold to the best scoring version. Each
        revision contributes at most one version (the one closest to the best).
        """
        revs = self.revisions()
        if not revs:
            raise SystemExit("nothing indexed yet")
        unique: dict[str, dict] = {}
        in_revs: dict[str, list[str]] = {}
        by_symbol: dict[str, list[str]] = {}
        for r in revs:
            for c in self.load_chunks(r["sha"]):
                if c["hash"] not in unique:
                    unique[c["hash"]] = c
                    by_symbol.setdefault(c["symbol"], []).append(c["hash"])
                in_revs.setdefault(c["hash"], []).append(r["label"])
        union = list(unique.values())
        hits = self.rank(union, qvec, query, mode, depth=max(100, 5 * k))

        families: list[dict] = []
        taken: set[str] = set()
        for h in hits:
            best = h.chunk
            if best["hash"] in taken:
                continue
            sim = {best["hash"]: 1.0}
            # Module level windows of one file look alike, so they need the same
            # path and a stricter threshold than named functions/classes.
            generic = best["kind"] in ("module", "window")
            thr = max(sim_threshold, 0.985) if generic else sim_threshold
            cands = [x for x in by_symbol.get(best["symbol"], [])
                     if x != best["hash"] and x not in taken]
            if generic:
                cands = [x for x in cands if unique[x]["path"] == best["path"]]
            if cands:
                bv = self.vectors_for([best["hash"]])[0]
                for x, s in zip(cands, self.vectors_for(cands) @ bv):
                    if s >= thr:
                        sim[x] = float(s)
            # Each revision contributes at most one version: its closest chunk.
            chosen: dict[str, list[str]] = {}
            for r in revs:
                have = [m for m in sim if r["label"] in in_revs[m]]
                if have:
                    m = max(have, key=lambda x: sim[x])
                    chosen.setdefault(m, []).append(r["label"])
            taken.update(sim)
            versions = [{"chunk": unique[m], "revs": rl, "sim_to_best": sim[m]}
                        for m, rl in sorted(chosen.items(), key=lambda kv: -sim[kv[0]])]
            families.append({"best": best, "score": h.score, "dense": h.dense,
                             "versions": versions})
            if len(families) >= k:
                break
        return families
