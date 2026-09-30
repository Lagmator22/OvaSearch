import numpy as np, mteb, glob, os, math
from rank_bm25 import BM25Okapi
from prism.index import bm25_tokens
t = mteb.get_task("AppsRetrieval"); t.load_data()
s = t.dataset["default"]["test"]
qids = list(s["queries"]["id"]); dids = list(s["corpus"]["id"])
rel = s["relevant_docs"]
files = sorted(glob.glob(os.path.expanduser("~/.cache/prism_eval/*.npy")), key=os.path.getmtime)
arrs = [np.load(f) for f in files]
Q = next(a for a in arrs if a.shape[0] == len(qids)); D = next(a for a in arrs if a.shape[0] == len(dids))
S = Q @ D.T
def ndcg_mrr(rank_lists):
    nd, mr = [], []
    for qi, ranked in enumerate(rank_lists):
        r = rel[qids[qi]]
        dcg = sum((1/math.log2(i+2)) for i, d in enumerate(ranked[:10]) if dids[d] in r and r[dids[d]] > 0)
        idcg = sum(1/math.log2(i+2) for i in range(min(10, sum(1 for v in r.values() if v > 0))))
        nd.append(dcg/idcg if idcg else 0)
        m = 0
        for i, d in enumerate(ranked[:10]):
            if dids[d] in r and r[dids[d]] > 0: m = 1/(i+1); break
        mr.append(m)
    return np.mean(nd), np.mean(mr)
dense = [list(np.argsort(-S[i])[:100]) for i in range(len(qids))]
print("dense", ndcg_mrr(dense))
bm = BM25Okapi([bm25_tokens(x) for x in s["corpus"]["text"]])
sparse = []
for q in s["queries"]["text"]:
    sc = bm.get_scores(bm25_tokens(q)); sparse.append(list(np.argsort(-sc)[:100]))
print("bm25", ndcg_mrr(sparse))
for w in [0.2, 0.5, 1.0]:
    fused = []
    for a, b in zip(dense, sparse):
        sc = {}
        for r, d in enumerate(a): sc[d] = sc.get(d, 0) + 1/(61+r)
        for r, d in enumerate(b): sc[d] = sc.get(d, 0) + w/(61+r)
        fused.append(sorted(sc, key=lambda d: -sc[d]))
    print("rrf w_bm25=", w, ndcg_mrr(fused))
