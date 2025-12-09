from flask import Flask, render_template, request
from google import genai
from scripts.search_engine import load_index, run_search

app = Flask(__name__)
client = genai.Client()


meta, doclen, titles, urls, shard_paths = load_index()


def get_ai_answer(query: str) -> str:
    prompt = (
        "Explain the following in 50 words or less using simple, clear language. "
        "Do not mention that you are an AI.\n\n"
        f"{query}"
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",  
            contents=prompt,
        )
        return (resp.text or "").strip()
    except Exception as e:
        print("Gemini error:", e)
        return "AI answer unavailable right now. Here are some relevant links instead:"


@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "").strip()
    results = []
    answer = ""
    searching = False

    if query:
        searching = True
        answer = get_ai_answer(query)
        results = run_search(
            query=query,
            meta=meta,
            doclen=doclen,
            titles=titles,
            urls=urls,
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
