import argparse
import gzip
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from heapq import nlargest
from typing import Dict, Iterable, List, Optional, Set, Tuple

# tokenization
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

# global data used inside worker processes
DOC_LEN: Dict[str, int] = {}
META: Dict[str, float] = {}


def _init_worker(doclen: Dict[str, int], meta: Dict[str, float]):
    global DOC_LEN, META
    DOC_LEN = doclen
    META = meta


def _score_shard(fp: str, qterms: List[str], candidate_set: Optional[Set[str]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with gzip.open(fp, "rt", encoding="utf-8") as g:
        idx = json.load(g)
    N, avgdl = META["N"], META["avgdl"]
    for t in qterms:
        postings = idx.get(t)
        if not postings:
            continue
        df = postings["df"]
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for did, tf in postings["postings"]:
            if candidate_set is not None and did not in candidate_set:
                continue
            dl = DOC_LEN.get(did, 1)
            denom = tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl)
            s = idf * tf * (1.2 + 1) / (denom if denom else 1.0)
            scores[did] = scores.get(did, 0.0) + s
    return scores


def load_json_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as g:
        return json.load(g)


def load_lsh_buckets(path: str) -> Dict[Tuple[int, int], List[str]]:
    raw = load_json_gz(path)
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for key, dids in raw.items():
        band_str, value_str = key.split(":", 1)
        buckets[(int(band_str), int(value_str))] = dids
    return buckets


def simhash(tokens: Iterable[str], bits: int = 64) -> int:
    v = [0] * bits
    for token in tokens:
        h = hash(token)
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight >= 0:
            fingerprint |= (1 << i)
    return fingerprint


def lsh_candidates(qterms: List[str], buckets: Dict[Tuple[int, int], List[str]], max_candidates: int) -> Set[str]:
    sig = simhash(qterms)
    band_size = 16
    cand: Set[str] = set()
    for band in range(0, 64, band_size):
        key = (band // band_size, (sig >> band) & ((1 << band_size) - 1))
        cand.update(buckets.get(key, ()))
        if len(cand) >= max_candidates:
            break
    return cand


def score_query(
    qterms: List[str],
    shard_paths: List[str],
    doclen: Dict[str, int],
    meta: Dict[str, float],
    max_workers: Optional[int],
    candidate_set: Optional[Set[str]],
    topk: int,
) -> List[Tuple[str, float]]:
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(doclen, meta)) as ex:
        worker = partial(_score_shard, qterms=qterms, candidate_set=candidate_set)
        partial_scores = ex.map(worker, shard_paths)
        merged: Dict[str, float] = {}
        for shard_scores in partial_scores:
            for did, score in shard_scores.items():
                merged[did] = merged.get(did, 0.0) + score
    return nlargest(topk, merged.items(), key=lambda x: x[1])


def main():
    parser = argparse.ArgumentParser(description="Parallel BM25 search with optional LSH pruning")
    parser.add_argument("query", help="Query string to search for")
    parser.add_argument("--topk", type=int, default=10, help="Number of results to return")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="Parallel workers (processes)")
    parser.add_argument("--use-lsh", action="store_true", help="Enable LSH-based candidate generation")
    parser.add_argument(
        "--max-candidates", type=int, default=500, help="Maximum LSH candidates before BM25 rerank"
    )
    args = parser.parse_args()

    qterms = tokenize(args.query)
    if not qterms:
        print("Query is empty after tokenization")
        return

    meta = load_json_gz("artifacts/index/meta.json.gz")
    doclen = load_json_gz("artifacts/index/doclen.json.gz")
    titles = load_json_gz("artifacts/index/titles.json.gz")
    urls = load_json_gz("artifacts/index/urls.json.gz")

    lsh_buckets = None
    candidates: Optional[Set[str]] = None
    if args.use_lsh:
        lsh_path = "artifacts/index/simhash_bands.json.gz"
        if not os.path.exists(lsh_path):
            print("LSH data not found; falling back to full BM25 search")
        else:
            lsh_buckets = load_lsh_buckets(lsh_path)
            candidates = lsh_candidates(qterms, lsh_buckets, args.max_candidates)

    shard_paths = sorted(
        fp for fp in os.listdir("artifacts/index") if fp.startswith("pp_") and fp.endswith(".json.gz")
    )
    shard_paths = [os.path.join("artifacts/index", fp) for fp in shard_paths]

    hits = score_query(
        qterms=qterms,
        shard_paths=shard_paths,
        doclen=doclen,
        meta=meta,
        max_workers=args.max_workers,
        candidate_set=candidates,
        topk=args.topk,
    )

    for did, s in hits:
        url = urls.get(did, "")
        if url and not url.startswith("http"):
            url = "https://" + url
        print(f"{s:8.4f}  {titles.get(did, '(no title)')}  {url}")


if __name__ == "__main__":
    main()