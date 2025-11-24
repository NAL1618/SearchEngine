import re, json, os, gzip, collections


def simhash(tokens, bits=64):
    """Compute a simple SimHash fingerprint for a multiset of tokens."""
    v = [0] * bits
    for t, freq in collections.Counter(tokens).items():
        h = hash(t)
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += freq if bit else -freq
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight >= 0:
            fingerprint |= (1 << i)
    return fingerprint

# this part sets the paths for corpus input and index output
CORPUS = "artifacts/corpus.jsonl"
IDXDIR = "artifacts/index"
# this part creates the pattern for tokenizing
tok_re = re.compile(r"[A-Za-z0-9]+")

# this part defines the tokenization function
def tokenize(text):
    return [t.lower() for t in tok_re.findall(text)]

#index directory
os.makedirs(IDXDIR, exist_ok=True)

# this part initializes data structures
postings = {}
doclen = {}
titles = {}
urls = {}
simhash_bands = collections.defaultdict(list)       
N = 0

# this part processes each document
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        # this part extracts document ID and combines title and text
        did = obj["id"]
        text = (obj.get("title","") + " " + obj.get("text",""))
        # this part tokenizes the document text
        toks = tokenize(text)
        if not toks:
            continue
        #increment
        N += 1
        # this part stores document metadata
        titles[did] = obj.get("title","")
        urls[did]   = obj.get("url","")
        doclen[did] = len(toks)
        # this part counts term frequencies in the document
        tf = collections.Counter(toks)
        for term, freq in tf.items():
            postings.setdefault(term, []).append((did, freq))
        
         # compute simhash bands for LSH-based candidate retrieval
        sig = simhash(toks)
        band_size = 16
        for band in range(0, 64, band_size):
            bucket = (band // band_size, (sig >> band) & ((1 << band_size) - 1))
            simhash_bands[bucket].append(did)

# calculate the average document length
avgdl = sum(doclen.values())/max(1,len(doclen))

#saves metadata to a compressed file
with gzip.open(f"{IDXDIR}/meta.json.gz","wt",encoding="utf-8") as g:
    json.dump({"N":N,"avgdl":avgdl}, g)
#saves document lengths to a compressed file
with gzip.open(f"{IDXDIR}/doclen.json.gz","wt",encoding="utf-8") as g:
    json.dump(doclen, g)
#saves document titles to a compressed file
with gzip.open(f"{IDXDIR}/titles.json.gz","wt",encoding="utf-8") as g:
    json.dump(titles, g)
#saves document URLs to a compressed file
with gzip.open(f"{IDXDIR}/urls.json.gz","wt",encoding="utf-8") as g:
    json.dump(urls, g)
# Saves simhash buckets to support approximate candidate search.
# Each bucket key is stored as "band:value" to keep JSON small.
with gzip.open(f"{IDXDIR}/simhash_bands.json.gz","wt",encoding="utf-8") as g:
    json.dump({f"{b}:{v}": dids for (b, v), dids in simhash_bands.items()}, g)
# this part sorts all terms alphabetically
items = sorted(postings.items(), key=lambda x: x[0])
# this part sets up variables for splitting postings into shards
shard, buf, cap = 0, {}, 200000
# this part processes each term and creates sharded posting files
for i,(term, plist) in enumerate(items):
    # this part adds the term with its document frequency and postings to the buffer
    buf[term] = {"df": len(plist), "postings": plist}
    # this part writes the buffer to a shard file when it reaches capacity
    if (i+1)%cap==0:
        with gzip.open(f"{IDXDIR}/pp_{shard:03d}.json.gz","wt",encoding="utf-8") as g:
            json.dump(buf, g)
        buf, shard = {}, shard+1
# this part writes any remaining terms to the final shard
if buf:
    with gzip.open(f"{IDXDIR}/pp_{shard:03d}.json.gz","wt",encoding="utf-8") as g:
        json.dump(buf, g)

# this part prints a summary of the indexing process
print(f"Indexed docs: {N}, avgdl: {avgdl:.2f}, shards: {shard+1}")
