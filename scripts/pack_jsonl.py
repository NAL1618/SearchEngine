import json, glob, os

SRC = "data/extracted"
OUT = "artifacts/corpus.jsonl"
limit = int(os.environ.get("DOC_LIMIT", "0")) 

count = 0
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as out:

    for fp in glob.glob(f"{SRC}/**/wiki_*", recursive=True):
        with open(fp, encoding="utf-8") as f:

            for line in f:
                obj = json.loads(line)
                if not obj.get("text"): 
                    continue

                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

                if limit and count >= limit:
                    break
        if limit and count >= limit:
            break
print(f"Wrote {count} docs to {OUT}")
