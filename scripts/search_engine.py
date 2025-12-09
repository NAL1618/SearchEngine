import json
import math
import os
import re
from heapq import nlargest
from typing import Dict, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as g:
        return json.load(g)


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


def load_index(index_dir: str = "artifacts/index"):
    meta = load_json(os.path.join(index_dir, "meta.json"))
    doclen = load_json(os.path.join(index_dir, "doclen.json"))
    titles = load_json(os.path.join(index_dir, "titles.json"))
    urls = load_json(os.path.join(index_dir, "urls.json"))

    shard_paths = sorted(
        os.path.join(index_dir, fp)
        for fp in os.listdir(index_dir)
        if fp.startswith("pp_") and fp.endswith(".json")
    )

    return meta, doclen, titles, urls, shard_paths


def run_search(
    query: str,
    meta: Dict[str, float],
    doclen: Dict[str, int],
    titles: Dict[str, str],
    urls: Dict[str, str],
    shard_paths: List[str],
    topk: int = 10,
) -> List[Dict[str, str]]:
    qterms = tokenize(query)
    if not qterms:
        return []

    merged: Dict[str, float] = {}

    for fp in shard_paths:
        shard_scores = _score_shard(fp, qterms, meta, doclen)
        for did, score in shard_scores.items():
            merged[did] = merged.get(did, 0.0) + score

    hits: List[Tuple[str, float]] = nlargest(topk, merged.items(), key=lambda x: x[1])

    results: List[Dict[str, str]] = []
    for did, s in hits:
        url = urls.get(did, "")
        if url and not url.startswith("http"):
            url = "https://" + url
        results.append(
            {
                "doc_id": did,
                "score": f"{s:.4f}",
                "title": titles.get(did, "(no title)"),
                "url": url,
            }
        )

    return results
