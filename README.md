---
title: Multi Agent RAG Chatbot
emoji: A
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

# Atlas - Multi-Agent Retrieval-Augmented Generation System

Atlas is a production-ready, multi-agent AI chatbot built on a Retrieval-Augmented Generation (RAG) pipeline. It is designed to answer questions grounded in indexed documents, with automatic fallback to live web search when local context is insufficient. Every response is verified by a second-pass Critic agent before being delivered to the user.

---

## System Overview

The system is composed of six specialized agents that work together in a coordinated pipeline. Each agent has a well-defined role and communicates structured results to the next stage. The pipeline runs end-to-end on every user query, from query expansion through to streamed, fact-checked output.

---

## Agent Pipeline

### Stage 1 - Query Rewriter Agent

**Module:** `multi_agent/agents/query_rewriter_agent.py`

The Query Rewriter receives the raw user query and rewrites it into two optimized versions: one tailored for dense vector and BM25 document retrieval, and another for broad web search. This expansion improves recall across both sources and helps disambiguate short or vague queries.

### Stage 2 - RAG Agent

**Module:** `multi_agent/agents/rag_agent.py`

The RAG Agent performs hybrid document retrieval by combining BM25 sparse retrieval and dense vector search using the `BAAI/bge-m3` embedding model. Retrieved candidates are then reranked using `BAAI/bge-reranker-v2-m3`, a cross-encoder model that scores each document chunk against the query for relevance. The top-ranked chunks are passed downstream as structured context.

### Stage 3 - Context Evaluation Agent

**Module:** `multi_agent/agents/evaluation_agent.py`

The Evaluation Agent assesses whether the retrieved document chunks are sufficient to answer the user query. It returns a structured result containing a boolean sufficiency flag and a floating-point confidence score. If the context is deemed insufficient, the pipeline routes to the Web Agent instead of proceeding directly to answer generation.

### Stage 4 - Web Search Agent (Conditional)

**Module:** `multi_agent/agents/web_agent.py`

The Web Agent is triggered only when the Evaluation Agent returns insufficient context. It performs a live Tavily web search using the expanded web query from Stage 1, fetches the top result pages, cleans the HTML content, and returns structured web context including source URLs and relevance scores. This stage is skipped entirely when local document context is sufficient.

### Stage 5 - Answer Agent and Critic

**Module:** `multi_agent/agents/answer_agent.py`

The Answer Agent synthesizes a draft response from the available context (either document chunks or web content). After the draft is generated, a second Critic pass reviews it for factual accuracy, value consistency, and grounding in the provided context. If inconsistencies are detected, the Critic corrects them before the response is streamed to the user.

### Stage 6 - Supervisor Agent

**Module:** `multi_agent/agents/supervisor_agent.py`

The Supervisor orchestrates all five agents in sequence, manages routing logic between RAG and web fallback paths, maintains per-session conversation history, and streams the final verified response token-by-token to the client via Server-Sent Events (SSE).

---

## Retrieval Architecture

The hybrid retrieval system combines two complementary search methods:

**BM25 Sparse Retrieval** uses term frequency and inverse document frequency to score keyword matches between the query and document chunks. It performs well on exact name matches, technical terms, and short factual queries.

**Dense Vector Search** uses the `BAAI/bge-m3` multilingual embedding model to encode both the query and documents into high-dimensional vector representations. It performs well on semantic similarity, paraphrased queries, and conceptual questions.

**Cross-Encoder Reranking** with `BAAI/bge-reranker-v2-m3` scores each retrieved chunk directly against the query in a joint attention pass, producing a more accurate relevance ranking than either retrieval method alone.

---

## Document Indexing

Documents placed in the `docs_multi/` directory are automatically loaded, chunked, embedded, and stored in a persistent ChromaDB vector store on startup. Supported formats include PDF, CSV, TXT, and Markdown. A fingerprint-based hash system detects document changes and triggers re-indexing only when necessary, avoiding redundant processing on repeated restarts.

---

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| GET | / | Serves the custom Atlas web workspace (index.html) |
| POST | /chat | Accepts a query and streams the agent response via SSE |
| GET | /api/documents | Returns a list of all indexed documents with metadata |
| POST | /clear | Clears the conversation history for a given session |
| GET | /documents/{name} | Serves or downloads a specific indexed document file |

---

## Project Structure

```
multi_agent/
    agents/
        query_rewriter_agent.py   - Query expansion for retrieval and web search
        rag_agent.py              - Hybrid BM25 and vector retrieval with reranking
        evaluation_agent.py       - Context sufficiency scoring
        web_agent.py              - Tavily web search fallback
        answer_agent.py           - Response generation and Critic verification
        supervisor_agent.py       - Pipeline orchestration and SSE streaming
    retrieval/
        ingestion.py              - Document loading, chunking, and indexing
        retriever.py              - Hybrid retriever with cross-encoder reranking
        table_serialization.py    - PDF table extraction and serialization
    memory/
        history.py                - Per-session conversation history management
    evaluation/
        routing_logger.py         - Agent routing decision logging
    config.py                     - Centralized configuration
    api.py                        - FastAPI application and endpoint definitions

evaluate_rag/                     - Single-agent RAG evaluation scripts
evaluate_multi_rag/               - Multi-agent pipeline evaluation scripts
docs_multi/                       - Document knowledge base directory
app.py                            - Hugging Face Space entrypoint
index.html                        - Custom web frontend
requirements.txt                  - Python package dependencies
```

---

## Setup and Deployment

### Hugging Face Space

This application is deployed as a Gradio Space on Hugging Face. To configure:

1. Open your Space settings and navigate to Repository Secrets.
2. Add a secret named `GOOGLE_API_KEY` with your Gemini API key value.
3. Optionally add `TAVILY_API_KEY` to enable live web search fallback.
4. The Space will restart automatically and pick up the new secrets.

Once the Space is running, upload documents to the `docs_multi/` directory via the Files tab on Hugging Face to make them available to the RAG pipeline.

### Environment Variables

| Variable | Required | Description |
| :--- | :--- | :--- |
| GOOGLE_API_KEY | Yes | Gemini API key for LLM inference |
| TAVILY_API_KEY | No | Tavily API key for web search fallback |

---

## Evaluation

The `evaluate_rag/` and `evaluate_multi_rag/` directories contain scripts for benchmarking the retrieval and generation pipeline against standard datasets including SciFactk, HotpotQA, SQuAD, RAGBench, and WikiTableQuestions. Metrics reported include NDCG@10, Recall@5, Context Precision, Faithfulness, and Answer Relevancy.
