import requests
import json
import time

URL_WITH = "http://localhost:8000/chat"
URL_WITHOUT = "http://localhost:8001/chat"

# 10 fresh evaluation queries mapped to their ground truth correct answers (based on the PDFs)
EVAL_DATA = [
    {
        "query": "Who is the author of 'Think Like a Monk' and what guide does he offer in the book summarized in download?",
        "gold": "Jay Shetty is the author. In '8 Rules of Love', he offers a transformative guide to navigating the complexities of romance, blending ancient wisdom with contemporary insights."
    },
    {
        "query": "According to download2, what systems did banks introduce in the mid-20th century to improve transaction efficiency and reduce dependency on physical cash?",
        "gold": "Banks introduced credit cards and debit cards."
    },
    {
        "query": "What is the official target percentage of the fiscal deficit mentioned in Union Budget 2026?",
        "gold": "The official fiscal deficit target is 4.3% of GDP."
    },
    {
        "query": "Who are the two countries whose national anthems were composed by Rabindranath Tagore, and what are the titles of those compositions?",
        "gold": "India ('Jana Gana Mana') and Bangladesh ('Amar Sonar Bangla')."
    },
    {
        "query": "What does Rule 1 say about the stages from loneliness to solitude?",
        "gold": "The stages are: 1. Presence (identifying personal values), 2. Discomfort (facing it with small challenges), and 3. Confidence (building self-confidence without seeking validation)."
    },
    {
        "query": "What are the four major relationship deal-breakers identified in Rule 7?",
        "gold": "The four major deal-breakers are abuse, infidelity, inertia, and disinterest."
    },
    {
        "query": "What is the specific value of retail inflation in India in April 2026?",
        "gold": "Retail inflation rose to approximately 3.48% in April 2026."
    },
    {
        "query": "What are the four pursuits in life led by Dharma as described in the Vedic pursuits?",
        "gold": "The four pursuits are: Dharma (purpose), Artha (work and finance), Kama (pleasure and connection), and Moksha (spiritual liberation)."
    },
    {
        "query": "How did Rabindranath Tagore protest against colonial brutality after the Jallianwala Bagh Massacre?",
        "gold": "He renounced the knighthood awarded to him by the British government."
    },
    {
        "query": "What stage of life focuses on reflection, forgiveness, and healing in relationships under Rule 6?",
        "gold": "The Vanaprastha stage of life."
    }
]

def get_answer(url, query):
    payload = {"session_id": "eval_test", "message": query}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        # Handle SSE stream extraction
        text = resp.text
        return text.strip()
    except Exception as e:
        return f"Error: {e}"

print("Starting evaluation across both pipelines...")
results = []

for idx, item in enumerate(EVAL_DATA, 1):
    q = item["query"]
    gold = item["gold"]
    print(f"Running query {idx}/10...")
    
    ans_with = get_answer(URL_WITH, q)
    ans_without = get_answer(URL_WITHOUT, q)
    
    results.append({
        "index": idx,
        "query": q,
        "gold": gold,
        "with_reranker": ans_with,
        "without_reranker": ans_without
    })

# Format results directly to a Markdown file
with open("eval_results.md", "w", encoding="utf-8") as f:
    f.write("# RAG Pipeline Evaluation Report\n\n")
    f.write("Evaluation comparing `main.py` (With Reranking) vs. `main2.py` (Without Reranking).\n\n")
    
    for r in results:
        f.write(f"### Q{r['index']}: {r['query']}\n\n")
        f.write(f"**Correct Answer (Ground Truth):**\n{r['gold']}\n\n")
        f.write(f"**With Reranker Output:**\n{r['with_reranker']}\n\n")
        f.write(f"**Without Reranker Output:**\n{r['without_reranker']}\n\n")
        f.write("---\n\n")

print("Evaluation finished! Results saved to eval_results.md")
