"""prism command line: index a repo revision, search it, list indexed revisions."""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_INDEX_DIR = os.environ.get("PRISM_INDEX", ".prism_index")


def _preview(text: str, lines: int = 3, width: int = 100) -> str:
    out = []
    for ln in text.strip("\n").split("\n")[:lines]:
        out.append("      | " + (ln if len(ln) <= width else ln[:width - 3] + "..."))
    return "\n".join(out)


def _load_embedder(model: str | None, store=None):
    from .embedder import DEFAULT_MODEL, Embedder

    name = model or (store.meta.get("model") if store else None) or DEFAULT_MODEL
    t0 = time.time()
    emb = Embedder(name)
    print(f"model {name} loaded on CPU via {emb.backend} in {time.time() - t0:.1f}s",
          file=sys.stderr)
    return emb


def cmd_index(args) -> None:
    from .index import IndexStore

    store = IndexStore(args.index_dir)
    emb = _load_embedder(args.model, store)
    for rev in (args.rev or [None]):
        s = store.index(args.repo, rev, emb)
        print(f"indexed {s['label']} ({s['sha'][:10]}): {s['files']} files, {s['chunks']} chunks")
        print(f"  reused {s['reused']} cached chunk vectors, embedded {s['embedded']} new "
              f"chunks ({s['reused_dupes_in_rev']} duplicates inside this revision)")
        print(f"  chunking {s['chunk_s']:.2f}s | embedding {s['embed_s']:.2f}s | "
              f"total rebuild {s['total_s']:.2f}s")


def cmd_search(args) -> None:
    from .index import IndexStore, classify_query

    store = IndexStore(args.index_dir)
    if not store.revisions():
        sys.exit(f"no index in {args.index_dir}; run `python -m prism index <repo>` first")
    emb = _load_embedder(None, store)
    t0 = time.perf_counter()
    qvec = emb.encode_query(args.query)
    t_embed = (time.perf_counter() - t0) * 1000
    print(f'query: "{args.query}"  (type: {classify_query(args.query)}, mode: {args.mode})')

    if args.all_versions:
        fams = store.search_all_versions(args.query, qvec, args.k, args.mode)
        total = (time.perf_counter() - t0) * 1000
        labels = [r["label"] for r in store.revisions()]
        print(f"searched {len(labels)} revisions: {', '.join(labels)}\n")
        for i, f in enumerate(fams, 1):
            b = f["best"]
            print(f"{i:2d}. {b['path']}:{b['start_line']}-{b['end_line']}  {b['symbol']}  "
                  f"score {f['score']:.4f} (cos {f['dense']:.3f})")
            print(_preview(b["text"]))
            for v in f["versions"]:
                c = v["chunk"]
                tag = "best" if c["hash"] == b["hash"] else (
                    f"near-duplicate, sim {v['sim_to_best']:.3f}")
                print(f"      in {', '.join(v['revs'])}: {c['path']}:{c['start_line']}-"
                      f"{c['end_line']} [{tag}]")
            print()
    else:
        r, hits = store.search(args.query, qvec, args.rev, args.k, args.mode)
        total = (time.perf_counter() - t0) * 1000
        print(f"revision {r['label']} ({r['sha'][:10]}), {r['n_chunks']} chunks\n")
        for i, h in enumerate(hits, 1):
            c = h.chunk
            print(f"{i:2d}. {c['path']}:{c['start_line']}-{c['end_line']}  {c['symbol']}  "
                  f"score {h.score:.4f} (cos {h.dense:.3f})")
            print(_preview(c["text"]))
    print(f"latency: {total:.1f} ms total ({t_embed:.1f} ms query embedding)")


def cmd_list(args) -> None:
    from .index import IndexStore

    store = IndexStore(args.index_dir)
    if not store.revisions():
        print(f"nothing indexed in {args.index_dir}")
        return
    print(f"repo: {store.meta['repo']}\nmodel: {store.meta['model']}")
    for r in store.revisions():
        mark = "*" if r["sha"] == store.meta.get("default_sha") else " "
        print(f" {mark} {r['label']:<20} {r['sha'][:10]}  {r['n_chunks']:>6} chunks  "
              f"{r['indexed_at']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m prism",
                                 description="Hybrid code search over git revisions (CPU).")
    ap.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                    help="where the index lives (default: ./.prism_index or $PRISM_INDEX)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="index one or more revisions of a repo")
    p.add_argument("repo")
    p.add_argument("--rev", action="append",
                   help="git commit/tag/branch (repeatable). Default: HEAD, or the working "
                        "tree for a non git folder")
    p.add_argument("--model", default=None, help="embedding model (default: CodeRankEmbed)")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("search", help="search an indexed revision")
    p.add_argument("query")
    p.add_argument("--rev", default=None, help="indexed revision (default: last indexed)")
    p.add_argument("--all-versions", action="store_true",
                   help="search every indexed revision and group near-duplicates")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("list", help="show indexed revisions")
    p.set_defaults(fn=cmd_list)

    args = ap.parse_args(argv)
    args.fn(args)
