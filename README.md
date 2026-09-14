# RAG & Agentic AI: Zero to Research Level
### University Textbook • Production Systems Engineering Canon • Research Survey • Hands-On Curriculum

[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg?logo=typescript)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-6.4-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

**RAG & Agentic AI: Zero to Research Level** is an exhaustive, rigorous, university-grade learning system and production engineering platform covering:

* **Retrieval-Augmented Generation (RAG):** Foundations, Layout-Aware Ingestion, Dense vs Sparse, ColPali, Table Lattice Extraction.
* **Advanced RAG Architectures:** Self-RAG, Corrective RAG (CRAG), Adaptive RAG, GraphRAG, Hybrid Search & Reciprocal Rank Fusion (RRF).
* **AI Agents & Agentic Workflows:** POMDP Formalisms, Yao et al. (2022) ReAct, Reflexion, Tree of Thoughts (ToT), Model Context Protocol (MCP).
* **Systems & Production Engineering:** Vector DB Benchmarks (Qdrant, pgvector, Milvus, Weaviate, Pinecone, FAISS), Semantic Caching, OpenTelemetry Distributed Tracing.
* **Adversarial Security & Hardening:** Indirect Prompt Injection Mitigation, Sandboxing (Firecracker microVMs), In-Engine Roaring Bitmap Pre-filtering.
* **Evaluation Frameworks:** RAGAS Triad (Faithfulness, Answer Relevance, Context Precision), Pairwise LLM-as-a-Judge with Position Bias Swapping.

---

## 📚 What's Inside

### 1. 42 Comprehensive Curriculum Modules (Levels 0 to 13)
* **Level 0 (Math & ML):** Linear Algebra, Cosine Equivalence, Softmax Temperature Bounds, Transformer QKV Attention.
* **Level 1–3 (LLM Engineering & Document Ingestion):** Layout-Aware Vision Ingestion, ColPali, Table Lattice Extraction, Chunking Paradigms.
* **Level 4–6 (Retrieval, Reranking & Advanced RAG):** BM25, Dense Bi-encoders, Cross-encoder Reranking, Self-RAG, CRAG, GraphRAG.
* **Level 7–9 (Agent Architectures, Memory & Tool Use):** ReAct Interleaved Loops, Grammar-Constrained Decoding, Letta/MemGPT OS Paging, MCP.
* **Level 10–12 (Multi-Agent Swarms & Production Systems):** StateGraphs, Compounding Error Modeling, SSE Streaming, Redis Semantic Caching, Vector DB Matrix.
* **Level 13 (Capstone & Frontier Research):** 6-Phase Autonomous Academic Research Assistant, Hallucination Mitigations, Future Horizons.

Each module includes:
1. Formal Definition & Technical Intuition
2. Rigorous Technical & Scientific Explanation
3. Mathematical Derivations & LaTeX Formulas
4. Complete Production Python Implementation
5. Concrete Production Failure Modes & Engineering Tradeoffs
6. Hands-on Coding Exercises with Hints and Reference Solutions
7. Interactive Self-Assessment Quiz with Instant Explanations

### 2. Interactive Generative-UI Research Labs
* **Decision Calculator:** Multi-slider trade-off engine evaluating RAG vs Fine-Tuning vs Long Context vs Agents with real-time cost and latency estimates.
* **Vector Math Sandbox:** Live 2D coordinate plane rendering vector arrows, dot products, Euclidean L2 norms, angular separation $\theta$, and cosine equivalence proofs.
* **ReAct Simulator:** Step-by-step interactive trace stepper through Thought $\to$ Action $\to$ Observation $\to$ Reflection $\to$ Final Answer.
* **Chunking Visualizer:** Visual comparison of Fixed Token, Recursive Character, Semantic, Anthropic Contextual, and Jina Late Chunking.
* **Vector DB Matrix:** In-depth benchmark of 7 vector engines (Qdrant, pgvector, Milvus, Weaviate, Pinecone, FAISS, OpenSearch).
* **Mythbusters Laboratory:** 8 empirically debunked industry fallacies with academic citations.
* **Interview Trainer:** 15 elite interview questions across Junior, Senior, and Staff Architect tiers with revealable red flags and ideal answers.
* **Capstone Roadmap:** 12-week milestone tracker with acceptance criteria for the Autonomous Research Assistant.

### 3. 11 Verified Hands-On Projects (100% Test Suite Pass)
1. **Resilient Streaming LLM Client from Scratch** (Decorrelated jitter, backoff, SSE)
2. **Hybrid Search Engine** (Dense BGE-Large + Sparse BM25 + Reciprocal Rank Fusion $k=60$)
3. **Advanced Chunking Pipeline** (Parent-child + Anthropic Contextual Retrieval)
4. **ColBERT / Late Interaction Scorer** (MaxSim token alignment)
5. **CRAG Corrective RAG Pipeline** (Confidence evaluation & web search routing)
6. **Self-RAG with Reflection Tokens** (`[Retrieve]`, `[Critique]`, `[No-Retrieve]`)
7. **ReAct Agent from Scratch with Tool Calling** (Sandboxed Python execution)
8. **Hierarchical Multi-Agent Supervisor** (LangGraph state machine)
9. **Production RAG with Semantic Cache & Observability** (Redis cosine threshold + OpenTelemetry)
10. **LLM Evaluation Harness** (RAGAS Triad + LLM judge position swapping)
11. **MCP (Model Context Protocol) Server & Agent Client** (JSON-RPC 2.0 stdio protocol)

---

## 🚀 Getting Started Locally

### Prerequisites
* Node.js v18+ (tested on Node v24.12.0)
* npm v9+

### Installation

```bash
# Clone the repository
git clone https://github.com/Hasan-8326/rag-agentic-ai-learning-system.git
cd rag-agentic-ai-learning-system

# Install dependencies
npm install

# Start local development server
npm run dev
```

Visit `http://localhost:5173` to explore the system.

### Build for Production

```bash
npm run build
```

The optimized static assets will be output to `dist/`.

---

## 🔬 Research Foundations (30+ Verified Papers)
This curriculum is built on direct peer-reviewed citations including:
* **Vaswani et al. (2017):** *Attention Is All You Need* (NeurIPS)
* **Karpukhin et al. (2020):** *Dense Passage Retrieval for Open-Domain QA* (EMNLP)
* **Lewis et al. (2020):** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS)
* **Khattab & Zaharia (2020):** *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction* (SIGIR)
* **Yao et al. (2022):** *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR)
* **Shinn et al. (2023):** *Reflexion: Language Agents with Verbal Reinforcement Learning* (NeurIPS)
* **Asai et al. (2023):** *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (ICLR)
* **Yan et al. (2024):** *Corrective Retrieval Augmented Generation (CRAG)* (arXiv:2401.15884)
* **Edge et al. (2024):** *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (Microsoft Research)
* **Faysse et al. (2024):** *ColPali: Efficient Document Retrieval with Vision Language Models* (Hugging Face)

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
