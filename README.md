---
title: Multi Agent RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# 🤖 Multi-Agent RAG Chatbot

Fact-grounded multi-agent system powered by:
- **RAG Agent**: Hybrid BM25 + Vector Retrieval (`BAAI/bge-m3`) + CrossEncoder Reranking (`BAAI/bge-reranker-v2-m3`)
- **Evaluation Agent**: Context sufficiency reasoning
- **Web Agent**: Supplementary Tavily Web search
- **Answer Agent**: Fact-checking Critic verification pass

## Setup for Hugging Face Space
1. Add `GOOGLE_API_KEY` (and optionally `Tavily_API_KEY`) under **Space Settings → Repository Secrets**.
2. Space SDK: **Gradio**
3. Hardware: **ZeroGPU (Free)** or **CPU Basic (Free)**
