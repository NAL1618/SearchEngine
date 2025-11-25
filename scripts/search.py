import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from heapq import nlargest
from typing import Dict, List, Optional, Set, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_json(path: str):
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as g:
        data = json.load(g)
    t1 = time.time()
    print(f"[TIME] load {os.path.basename(path)}: {t1 - t0:.3f}s")
    return data


def _score_shard(
    fp: str,
    qterms: List[str],
    meta: Dict[str, float],
    doclen: Dict[str, int],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    with open(fp, "r", encoding="utf-8") as g:
        idx = json.load(g)

    N = meta["N"]
    avgdl = meta["avgdl"]

    k1 = 1.2
    b = 0.75

    for t in qterms:
        postings = idx.get(t)
        if not postings:
            continue

        df = postings["df"]
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))

        for did, tf in postings["postings"]:
            dl = doclen.get(did, 1)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            if denom <= 0:
                continue

            s = idf * tf * (k1 + 1) / denom
            scores[did] = scores.get(did, 0.0) + s

    return scores


def score_query(
    qterms: List[str],
    shard_paths: List[str],
    doclen: Dict[str, int],
    meta: Dict[str, float],
    max_workers: Optional[int],
    topk: int,
) -> List[Tuple[str, float]]:
    t0 = time.time()
    merged: Dict[str, float] = {}

    if max_workers is None or max_workers <= 1:
        print(f"[TIME] BM25 start (single-process), shards={len(shard_paths)}")
        for fp in shard_paths:
            shard_scores = _score_shard(fp, qterms, meta, doclen)
            for did, score in shard_scores.items():
                merged[did] = merged.get(did, 0.0) + score
    else:
        print(f"[TIME] BM25 start (workers={max_workers}), shards={len(shard_paths)}")
        worker = partial(_score_shard, qterms=qterms, meta=meta, doclen=doclen)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for shard_scores in ex.map(worker, shard_paths):
                for did, score in shard_scores.items():
                    merged[did] = merged.get(did, 0.0) + score

    t1 = time.time()
    print(f"[TIME] BM25 done: {t1 - t0:.3f}s, scored_docs={len(merged)}")

    t2 = time.time()
    hits = nlargest(topk, merged.items(), key=lambda x: x[1])
    t3 = time.time()
    print(f"[TIME] top-{topk} select: {t3 - t2:.3f}s")

    return hits


def main():
    parser = argparse.ArgumentParser(description="BM25 search on champion-limited index")
    parser.add_argument("query", help="query string")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    overall_start = time.time()

    qterms = tokenize(args.query)
    if not qterms:
        print("Query is empty after tokenization.")
        return

    print("[TIME] load meta/doc stuff")
    meta = load_json(os.path.join("artifacts", "index", "meta.json"))
    doclen = load_json(os.path.join("artifacts", "index", "doclen.json"))
    titles = load_json(os.path.join("artifacts", "index", "titles.json"))
    urls = load_json(os.path.join("artifacts", "index", "urls.json"))

    shard_paths = sorted(
        os.path.join("artifacts", "index", fp)
        for fp in os.listdir("artifacts/index")
        if fp.startswith("pp_") and fp.endswith(".json")
    )
    print(f"[TIME] shards: {len(shard_paths)}")

    hits = score_query(
        qterms=qterms,
        shard_paths=shard_paths,
        doclen=doclen,
        meta=meta,
        max_workers=args.max_workers,
        topk=args.topk,
    )

    overall_end = time.time()
    print(f"[TIME] total query: {overall_end - overall_start:.3f}s")

    if not hits:
        print("No results found.")
        return

    for did, s in hits:
        url = urls.get(did, "")
        if url and not url.startswith("http"):
            url = "https://" + url
        print(f"{s:8.4f}  {titles.get(did, '(no title)')}  {url}")


if __name__ == "__main__":
    main()
