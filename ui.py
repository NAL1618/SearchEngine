# scripts/ui.py

from flask import Flask, render_template, request
from scripts.search_engine import load_index, run_search

app = Flask(__name__)

meta, doclen, titles, urls, snippets, shard_paths = load_index()


@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "").strip()
    results = []
    answer = ""
    searching = False

    if query:
        searching = True
        answer, results = run_search(
            query=query,
            meta=meta,
            doclen=doclen,
            titles=titles,
            urls=urls,
            snippets=snippets,
            shard_paths=shard_paths,
            topk=10,
        )

    return render_template(
        "search.html",
        query=query,
        results=results,
        searching=searching,
        answer=answer,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
