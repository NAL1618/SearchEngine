# scripts/search_engine.py

import json
import math
import os
import re
from heapq import nlargest
from html import unescape
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
    snippets = load_json(os.path.join(index_dir, "snippets.json"))

    shard_paths = sorted(
        os.path.join(index_dir, fp)
        for fp in os.listdir(index_dir)
        if fp.startswith("pp_") and fp.endswith(".json")
    )

    return meta, doclen, titles, urls, snippets, shard_paths


def _clean_text(text: str) -> str:
    # remove HTML tags and decode entities
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pick_answer(
    query: str,
    results: List[Dict[str, str]],
    snippets: Dict[str, str],
    max_docs: int = 3,
) -> str:
    if not results:
        return ""

    qterms = set(tokenize(query))
    if not qterms:
        return ""

    best = ""
    best_score = 0.0

    for r in results[:max_docs]:
        did = r["doc_id"]
        raw = snippets.get(did, "").strip()
        if not raw:
            continue

        text = _clean_text(raw)
        if not text:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            tokens = tokenize(s)
            if not tokens:
                continue

            overlap = sum(1 for t in tokens if t in qterms)
            if overlap == 0:
                continue

            # favor sentences with more overlap but not insanely long
            score = overlap / (len(tokens) ** 0.5)
            if score > best_score:
                best_score = score
                best = s

    if not best:
        # fallback: just take a cleaned prefix from the top doc
        top_did = results[0]["doc_id"]
        raw = snippets.get(top_did, "")
        best = _clean_text(raw)[:200]

    return best


def run_search(
    query: str,
    meta: Dict[str, float],
    doclen: Dict[str, int],
    titles: Dict[str, str],
    urls: Dict[str, str],
    snippets: Dict[str, str],
    shard_paths: List[str],
    topk: int = 10,
) -> Tuple[str, List[Dict[str, str]]]:
    qterms = tokenize(query)
    if not qterms:
        return "", []

    merged: Dict[str, float] = {}

    for fp in shard_paths:
        shard_scores = _score_shard(fp, qterms, meta, doclen)
        for did, score in shard_scores.items():
            merged[did] = merged.get(did, 0.0) + score

    hits = nlargest(topk, merged.items(), key=lambda x: x[1])

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

    answer = _pick_answer(query, results, snippets)
    return answer, results
