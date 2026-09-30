"""Run CoIR AppsRetrieval (MTEB) with a code embedding model on CPU.

Usage:
    python -m prism.eval.run_apps_eval --model sfr-400m --query-mode no_examples_compact --out appsretrieval_results.json

The model is wrapped in PrePostPipelineEncoder (an mteb AbsEncoder subclass),
which can clean queries/code before encoding and adds the prefixes each model
was trained with. Results JSON is written with json.dump(..., default=str)
because the MTEB result object contains datetime fields.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import json
import platform
import re
import time
import tokenize
from pathlib import Path

import numpy as np
import torch

import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

CACHE_DIR = Path(os.environ.get("PRISM_EVAL_CACHE", Path.home() / ".cache" / "prism_eval"))

# name -> settings. Prefixes come from each model card.
MODELS = {
    "bge-small": dict(
        hf="BAAI/bge-small-en-v1.5", query_prefix="", doc_prefix="", max_len=512,
    ),
    "bge-small-inst": dict(
        hf="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="", max_len=512,
    ),
    "coderank": dict(
        hf="nomic-ai/CodeRankEmbed",
        query_prefix="Represent this query for searching relevant code: ",
        doc_prefix="", max_len=1024, trust_remote_code=True,
    ),
    "sfr-400m": dict(
        hf="Salesforce/SFR-Embedding-Code-400M_R",
        query_prefix="Instruct: Given Code or Text, retrieval relevant content\nQuery: ",
        doc_prefix="", max_len=1024, trust_remote_code=True,
    ),
    "jina-code": dict(
        hf="jinaai/jina-embeddings-v2-base-code", query_prefix="", doc_prefix="",
        max_len=1024, trust_remote_code=True,
    ),
}

EXAMPLE_HEADER = re.compile(r"\n\s*-{3,}\s*(Examples?|Sample|Note)s?\s*-{3,}.*", re.S | re.I)
SECTION_RULE = re.compile(r"-{3,}\s*([A-Za-z ]+?)\s*-{3,}")


def clean_query(text: str, mode: str) -> str:
    """Problem statement cleanup. 'full' leaves text as is."""
    if mode == "full":
        return text
    if mode in ("no_examples", "no_examples_compact"):
        text = EXAMPLE_HEADER.sub("", text)
    if mode in ("compact", "no_examples_compact"):
        text = SECTION_RULE.sub(r"\1:", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def strip_py_comments(code: str) -> str:
    """Drop '#' comments with the tokenizer. Falls back to the input on bad code."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code


def clean_doc(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    if mode == "nocomment":
        text = strip_py_comments(text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln.strip())
    return text


class PrePostPipelineEncoder(AbsEncoder):
    """Pre-processing -> sentence-transformers encode -> L2 normalise."""

    def __init__(self, key: str, query_mode: str, doc_mode: str, max_len: int | None,
                 batch_size: int, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        cfg = MODELS[key]
        self.cfg = cfg
        self.query_mode, self.doc_mode = query_mode, doc_mode
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            cfg["hf"], device=device, trust_remote_code=cfg.get("trust_remote_code", False))
        self.model.max_seq_length = max_len or cfg["max_len"]
        self.mteb_model_meta = ModelMeta(
            loader=None, name=f"prism/{key}", revision="local", release_date=None,
            languages=["eng-Latn", "python-Code"], n_parameters=None, memory_usage_mb=None,
            max_tokens=self.model.max_seq_length, embed_dim=None, license=None,
            open_weights=True, public_training_code=None, public_training_data=None,
            framework=["Sentence Transformers", "PyTorch"],
            similarity_fn_name=ScoringFunction.COSINE, use_instructions=True,
            training_datasets=None, reference=f"https://huggingface.co/{cfg['hf']}",
        )

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        is_query = prompt_type is not None and prompt_type.value == "query"
        if is_query:
            texts = [self.cfg["query_prefix"] + clean_query(t, self.query_mode) for t in texts]
        else:
            texts = [self.cfg["doc_prefix"] + clean_doc(t, self.doc_mode) for t in texts]
        # Cache per (model, max_len, exact input texts) so that changing only the
        # query cleanup does not re-encode the 8765 corpus documents.
        key = hashlib.sha1(json.dumps([self.cfg["hf"], self.model.max_seq_length, texts])
                           .encode("utf-8", "replace")).hexdigest()[:16]
        cache = CACHE_DIR / f"{key}.npy"
        if cache.exists():
            print(f"[cache] {len(texts)} {'queries' if is_query else 'docs'} from {cache}")
            return np.load(cache)
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sfr-400m", choices=sorted(MODELS))
    ap.add_argument("--query-mode", default="no_examples_compact",
                    choices=["full", "no_examples", "compact", "no_examples_compact"])
    ap.add_argument("--doc-mode", default="raw", choices=["raw", "strip", "nocomment"])
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="appsretrieval_results.json")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    enc = PrePostPipelineEncoder(args.model, args.query_mode, args.doc_mode, args.max_len,
                                 args.batch_size)
    task = mteb.get_task("AppsRetrieval")
    t0 = time.time()
    res = mteb.evaluate(enc, task, cache=None, overwrite_strategy="always")
    wall = time.time() - t0

    task_res = res.task_results[0]
    scores = task_res.scores["test"][0]
    payload = {
        "task": "AppsRetrieval",
        "model": MODELS[args.model]["hf"],
        "settings": {"query_mode": args.query_mode, "doc_mode": args.doc_mode,
                     "max_seq_length": enc.model.max_seq_length,
                     "query_prefix": MODELS[args.model]["query_prefix"],
                     "doc_prefix": MODELS[args.model]["doc_prefix"],
                     "device": "cpu", "torch_threads": torch.get_num_threads()},
        "hardware": f"{platform.processor() or platform.machine()} | {platform.platform()}",
        "wall_time_s": round(wall, 1),
        "ndcg_at_10": scores["ndcg_at_10"],
        "mrr_at_10": scores["mrr_at_10"],
        "mteb_result": task_res.model_dump() if hasattr(task_res, "model_dump") else task_res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"model={MODELS[args.model]['hf']} query_mode={args.query_mode} "
          f"doc_mode={args.doc_mode} max_len={enc.model.max_seq_length}")
    print(f"NDCG@10 = {scores['ndcg_at_10']:.5f}")
    print(f"MRR@10  = {scores['mrr_at_10']:.5f}")
    print(f"wall time = {wall:.1f}s  -> {args.out}")


if __name__ == "__main__":
    main()
