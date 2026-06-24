"""
compare.py — sends the same queries to both servers and prints results side-by-side.
  main.py  (with reranking)    → http://localhost:8000
  main2.py (without reranking) → http://localhost:8001
"""
import sys, io, time, requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

URLS = {
    "WITH_RERANKER  (port 8000)": "http://localhost:8000/chat",
    "NO_RERANKER    (port 8001)": "http://localhost:8001/chat",
}

QUERIES = [
    "when was Rabindranath Tagore born?",
    "who were the major contributors to GDP growth in India?",
    "write about Mobile Payments and UPI in about 150 words",
    "who invented the telephone?",
]

SEP = "=" * 70

def ask(url: str, query: str) -> tuple[str, float]:
    payload = {"session_id": "compare_test", "message": query}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        answer = "".join(chunk for chunk in resp.iter_content(chunk_size=None, decode_unicode=True) if chunk)
        elapsed = time.perf_counter() - t0
        return answer.strip(), elapsed
    except Exception as e:
        return f"[ERROR: {e}]", time.perf_counter() - t0

for query in QUERIES:
    print(f"\n{SEP}")
    print(f"QUERY: {query}")
    print(SEP)
    results = {}
    for label, url in URLS.items():
        answer, elapsed = ask(url, query)
        results[label] = (answer, elapsed)

    for label, (answer, elapsed) in results.items():
        words = len(answer.split())
        print(f"\n▶ {label}  [{elapsed:.1f}s | {words} words]")
        print(f"  {answer[:500]}{'...' if len(answer) > 500 else ''}")

print(f"\n{SEP}\nDone.\n")
