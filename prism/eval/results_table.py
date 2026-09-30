"""Print a markdown table of every run JSON in prism/eval/runs/ (sorted by NDCG@10)."""

import json
from pathlib import Path

rows = []
for f in sorted(Path(__file__).parent.glob("runs/*.json")):
    d = json.loads(f.read_text())
    s = d["settings"]
    prefix = "yes" if s["query_prefix"] else "none"
    rows.append((d["ndcg_at_10"], d["mrr_at_10"], d["model"], prefix, s["query_mode"],
                 s["doc_mode"], s["max_seq_length"], d["wall_time_s"], s["torch_threads"], f.name))

print("| Model | Query prefix | Query cleanup | Code cleanup | Max tokens | NDCG@10 | MRR@10 "
      "| Wall time | Threads | Run file |")
print("|---|---|---|---|---|---|---|---|---|---|")
for nd, mr, model, pf, qm, dm, ml, wall, th, name in sorted(rows, reverse=True):
    print(f"| {model} | {pf} | {qm} | {dm} | {ml} | {nd:.5f} | {mr:.5f} | {wall / 60:.1f} min | {th} | "
          f"`runs/{name}` |")
