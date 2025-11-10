import json, glob, os

# sets up the source and output paths
SRC = "data/extracted"
OUT = "artifacts/corpus.jsonl"
limit = int(os.environ.get("DOC_LIMIT", "0")) 

#document counter
count = 0
# this part creates the output directory
os.makedirs(os.path.dirname(OUT), exist_ok=True)
# this part opens the output file for writing
with open(OUT, "w", encoding="utf-8") as out:
    # this part loops through all wiki files in the extracted data folders
    for fp in glob.glob(f"{SRC}/**/wiki_*", recursive=True):
        with open(fp, encoding="utf-8") as f:
            #reads each line from the wiki file
            for line in f:
                obj = json.loads(line)
                if not obj.get("text"): 
                    continue
                # this part writes the document to the output file
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
                # this part stops processing if we've reached the document limit
                if limit and count >= limit:
                    break
        if limit and count >= limit:
            break
# this part prints out how many documents were written
print(f"Wrote {count} docs to {OUT}")
