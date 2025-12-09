import re
import json
import os
import collections
import shutil

CORPUS = "artifacts/corpus.jsonl"
IDXDIR = "artifacts/index"

TOK_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str):
    return [t.lower() for t in TOK_RE.findall(text)]


if os.path.isdir(IDXDIR):
    existing = os.listdir(IDXDIR)
    if existing:
        print(f"Clearing existing index directory {IDXDIR} ({len(existing)} files)...")
        for name in existing:
            path = os.path.join(IDXDIR, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
else:
    os.makedirs(IDXDIR, exist_ok=True)

postings = {} 
doclen = {}
titles = {}
urls = {}
snippets = {}
N = 0

with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        did = obj["id"]
        title = obj.get("title", "")
        text = obj.get("text", "")

        toks = tokenize(title + " " + text)
        if not toks:
            continue

        N += 1
        titles[did] = title
        urls[did] = obj.get("url", "")
        doclen[did] = len(toks)

        snippet_text = text.replace("\n", " ").strip()
        snippets[did] = snippet_text[:500]

        tf = collections.Counter(toks)
        for term, freq in tf.items():
            postings.setdefault(term, []).append((did, freq))

avgdl = sum(doclen.values()) / max(1, len(doclen))

with open(os.path.join(IDXDIR, "meta.json"), "w", encoding="utf-8") as g:
    json.dump({"N": N, "avgdl": avgdl}, g)

with open(os.path.join(IDXDIR, "doclen.json"), "w", encoding="utf-8") as g:
    json.dump(doclen, g)

with open(os.path.join(IDXDIR, "titles.json"), "w", encoding="utf-8") as g:
    json.dump(titles, g)

with open(os.path.join(IDXDIR, "urls.json"), "w", encoding="utf-8") as g:
    json.dump(urls, g)

with open(os.path.join(IDXDIR, "snippets.json"), "w", encoding="utf-8") as g:
    json.dump(snippets, g)

CHAMP_K = 50
for term, plist in postings.items():
    plist.sort(key=lambda x: x[1], reverse=True)
    postings[term] = plist[:CHAMP_K]

items = sorted(postings.items(), key=lambda x: x[0])

shard = 0
buf = {}
cap = 200_000

for i, (term, plist) in enumerate(items):
    buf[term] = {"df": len(plist), "postings": plist}
    if (i + 1) % cap == 0:
        with open(os.path.join(IDXDIR, f"pp_{shard:03d}.json"), "w", encoding="utf-8") as g:
            json.dump(buf, g)
        buf = {}
        shard += 1

if buf:
    with open(os.path.join(IDXDIR, f"pp_{shard:03d}.json"), "w", encoding="utf-8") as g:
        json.dump(buf, g)

print(f"Indexed docs: {N}, avgdl: {avgdl:.2f}, shards: {shard + 1}, champions per term: {CHAMP_K}")
