---
layout: post
title:  "Pick Vector Store"
date:   2025-12-02 14:34:15 -0600
categories: AI
published: true
---
# How do you pick vector stores for different AI applications?

- **LangChain Community Stack (`HuggingFaceEmbeddings`, `FAISS`, `PGVector`)**
  - Unified abstractions for embedding, retrieval, and chains; easy to swap FAISS ↔︎ pgvector with shared `Document` objects.
  - Vibrant ecosystem (MCP adapters, structured output, agent frameworks) targeting the 0.3.x split packages.
  - Supports fully local pipelines: offline Hugging Face models, file-backed FAISS, or production pgvector.
  - Cons: rapid API churn with frequent deprecations, hands-on wiring for custom ingestion, docs sometimes lag behind releases.

- **LlamaIndex**
  - Higher-level orchestration: composable graph indices, query engines, built-in evaluation/observability tools.
  - Rich document loaders and auto-summarization accelerate prototyping complex RAG flows.
  - Larger dependency footprint and slower cold starts; advanced modules often assume managed/cloud components.
  - Tighter coupling to LlamaIndex APIs makes migrating back to raw vector stores or LangChain patterns more involved.