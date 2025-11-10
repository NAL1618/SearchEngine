import sys, json, gzip, math, glob, re

# this part gets the search query from command line 
q = "test" if len(sys.argv) < 2 else sys.argv[1]
# this part tokenizes text by splitting into lowercase words
tok = lambda s:[t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)]

# this part loads metadata about the index for total docs and the average length
with gzip.open("artifacts/index/meta.json.gz","rt") as g:
    meta = json.load(g)
N, avgdl = meta["N"], meta["avgdl"]

# this part loads document lengths for each document
with gzip.open("artifacts/index/doclen.json.gz","rt") as g: doclen = json.load(g)
#loads document titles
with gzip.open("artifacts/index/titles.json.gz","rt") as g: titles = json.load(g)
#loads document URLs
with gzip.open("artifacts/index/urls.json.gz","rt") as g: urls = json.load(g)

# this part defines the scoring function using BM25 algorithm
def score_query(qterms, k1=1.2, b=0.75):
    scores = {}
    # this part loops through all the posting list files
    for fp in glob.glob("artifacts/index/pp_*.json.gz"):
        with gzip.open(fp,"rt",encoding="utf-8") as g:
            idx = json.load(g)
        # this part processes each query term
        for t in qterms:
            if t not in idx: 
                continue
            # this part calculates document frequency
            df = idx[t]["df"]
            idf = math.log(1 + (N - df + 0.5)/(df + 0.5))
            # this part scores each document containing the term
            for did, tf in idx[t]["postings"]:
                dl = doclen.get(did, 1)
                denom = tf + k1*(1 - b + b*dl/avgdl)
                s = idf * tf * (k1+1) / (denom if denom else 1.0)
                scores[did] = scores.get(did, 0.0) + s
    # this part returns the top 10 documents sorted by score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

# uses the scoring function to get search results
hits = score_query(tok(q))
# this part displays each result with score, title, and URL
for did, s in hits:
    url = urls.get(did,"")
    if url and not url.startswith("http"):
        url = "https://" + url
    print(f"{s:8.4f}  {titles.get(did,'(no title)')}  {url}")
