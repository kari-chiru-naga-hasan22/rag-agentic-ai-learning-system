import { InterviewQuestion } from '../types/curriculum';

export const interviewData: InterviewQuestion[] = [
  {
    id: 1,
    tier: 'Junior AI Engineer',
    topic: 'Vector Search & Metric Spaces',
    question: 'Under what exact mathematical condition is Cosine Similarity identical to the Euclidean Dot Product, and why do production vector databases normalize embeddings upon ingestion?',
    competenciesTested: ['Linear algebra', 'Vector normalization', 'Database indexing optimization'],
    redFlags: [
      'Believing cosine similarity is always identical to dot product',
      'Thinking normalization modifies the semantic meaning or orientation of vectors',
      'Not knowing that dot products are computationally faster than cosine similarity'
    ],
    idealAnswer: 'Cosine similarity is defined as cos(u, v) = (u . v) / (||u|| * ||v||). When vectors u and v are L2-normalized such that ||u|| = 1 and ||v|| = 1, the denominator evaluates to exactly 1, making cos(u, v) = u . v. Production vector databases normalize vectors during ingestion so that at query time, similarity search computes a pure dot product (d multiplications and d-1 additions) without calculating square roots and division on every comparison, accelerating SIMD/AVX-512 hardware execution by 2x to 3x.',
    followUpProbes: ['How does this relate to Euclidean (L2) distance on the unit sphere? (||u - v||^2 = 2 - 2(u . v))', 'What happens if you index unnormalized vectors and query with dot product?']
  },
  {
    id: 2,
    tier: 'Junior AI Engineer',
    topic: 'Chunking & Context Boundaries',
    question: 'Why does naive fixed-token chunking (e.g. 500 tokens with 50-token overlap) degrade retrieval quality on structured documents, and what alternative would you implement?',
    competenciesTested: ['Document parsing', 'Chunking algorithms', 'Information retrieval fundamentals'],
    redFlags: [
      'Recommending simply increasing chunk size to 2,000 tokens as a universal fix',
      'Ignoring structural boundaries like table rows, markdown headers, or code blocks',
      'Unfamiliarity with semantic or hierarchical chunking'
    ],
    idealAnswer: 'Fixed-token chunking ignores syntactic and semantic boundaries, slicing mid-sentence, across table rows, or between an entity definition and its attributes. This damages dense embedding representations and splits co-dependent facts into separate vectors. Better alternatives include: (1) Recursive Character Splitting respecting structural separators (\\n\\n, \\n, markdown headers), (2) Hierarchical Parent-Child retrieval where small leaf chunks (128 tokens) are indexed for precision but their enclosing parent document (1,024 tokens) is returned for generation, and (3) Anthropic Contextual Retrieval where an LLM prepends document-level context to each chunk prior to embedding.',
    followUpProbes: ['How would you handle tables specifically? (Extract to HTML/Markdown and keep rows intact)', 'What is the trade-off of contextual retrieval? (Higher ingestion cost and storage)']
  },
  {
    id: 3,
    tier: 'Junior AI Engineer',
    topic: 'RAG Evaluation Metrics',
    question: 'In the RAG Triad, what is the precise difference between "Context Relevance", "Groundedness / Faithfulness", and "Answer Relevance"?',
    competenciesTested: ['Evaluation methodologies', 'RAGAS metrics', 'Failure attribution'],
    redFlags: [
      'Confusing Faithfulness with Answer Relevance',
      'Relying only on subjective human vibes rather than formal metric definitions',
      'Inability to isolate retriever errors from generator errors'
    ],
    idealAnswer: '(1) Context Relevance measures retriever quality: the proportion of retrieved chunks that are directly necessary to answer the user query. (2) Groundedness / Faithfulness measures generator truthfulness: whether every factual claim in the generated answer is mathematically entailed by the retrieved context alone (zero hallucinations beyond context). (3) Answer Relevance measures generator comprehension: whether the generated answer actually addresses the user query rather than drifting into irrelevant though grounded facts. Tracking all three allows immediate localization of pipeline failures: low Context Relevance = retriever failure; high Context Relevance but low Faithfulness = model hallucination.',
    followUpProbes: ['How do frameworks like RAGAS compute Faithfulness programmatically? (Decomposing answer into atomic claims and checking NLI entailment against context)', 'What are the main biases when using an LLM to judge Faithfulness?']
  },
  {
    id: 4,
    tier: 'Senior AI Engineer',
    topic: 'Hybrid Search & Reciprocal Rank Fusion',
    question: 'Explain how Reciprocal Rank Fusion (RRF) works to combine dense vector search with BM25 sparse search. Why is RRF preferred over weighted score combination in production?',
    competenciesTested: ['Information retrieval', 'Hybrid fusion algorithms', 'Score normalization pitfalls'],
    redFlags: [
      'Suggesting simple addition of raw BM25 scores and cosine similarities',
      'Unawareness of score distribution differences between sparse and dense models',
      'Inability to write the RRF equation'
    ],
    idealAnswer: 'RRF is an unsupervised rank aggregation method that combines multiple retrieval lists using item positions rather than raw scores. For each document d, its score is RRF(d) = sum_{m in models} 1 / (k + rank_m(d)), where k is a smoothing constant (conventionally k=60). RRF is strongly preferred over weighted score combination because BM25 produces unbounded positive scores (0 to +inf) while cosine similarity is bounded (-1 to +1). Direct score combination requires min-max or z-score calibration which drifts wildly across queries, while RRF is completely immune to score scaling differences and outliers.',
    followUpProbes: ['Why is k=60 chosen in the literature? (Mitigates top-rank dominance so rank 1 does not completely overwhelm rank 2-5)', 'When would you use a neural Cross-Encoder instead of RRF?']
  },
  {
    id: 5,
    tier: 'Senior AI Engineer',
    topic: 'Autonomous Agent State & Recovery',
    question: 'Design a resilient tool-calling loop for an agent that executes shell and SQL tools. How do you prevent catastrophic state loops, handle non-zero exit codes, and guarantee idempotency?',
    competenciesTested: ['Agent architectures', 'Error recovery', 'State machines', 'Security sandboxing'],
    redFlags: [
      'Allowing an unconstrained while(True) loop without step limits or timeouts',
      'Feeding full multi-megabyte stack traces or data outputs directly back into LLM context',
      'Lack of permission controls on destructive tools (DROP TABLE, rm -rf)'
    ],
    idealAnswer: 'A resilient tool execution loop must be structured as a formal StateGraph (e.g. LangGraph): (1) Maximum Step Horizon & Budget Governor: Hard cap of N=10 steps and token budget kill-switch. (2) Cycle Detection: Maintain an action hash history; if the exact same tool and arguments are invoked twice consecutively with identical failures, force a strategy pivot. (3) Structured Error Reflection: Catch exceptions and non-zero exit codes, truncate outputs to 1,000 characters, and feed structured error feedback (exit code, stderr snippet, suggested remedy) into the next Thought turn. (4) Sandboxing & Idempotency: All shell commands run inside an ephemeral container with read-only rootfs; database connections use read-only transactions with query timeouts.',
    followUpProbes: ['How do you handle tools that produce side effects (e.g. sending an email)? (Human-In-The-Loop confirmation breakpoint before state transition)', 'What is grammar-constrained decoding and how does it prevent JSON syntax errors in tool calling?']
  },
  {
    id: 6,
    tier: 'Senior AI Engineer',
    topic: 'Memory Hierarchies & Context Degradation',
    question: 'How do you architect an agent memory system that maintains conversational coherence across 6 months of daily interactions without blowing up context window limits or latency?',
    competenciesTested: ['Agent memory systems', 'MemGPT architecture', 'Episodic vs Semantic memory'],
    redFlags: [
      'Suggesting shoving all historical chat logs into a 1M context window on every turn',
      'Using naive FIFO message trimming that deletes early critical instructions or user profile details',
      'Unfamiliarity with hierarchical memory paging'
    ],
    idealAnswer: 'Implement a multi-tier hierarchical memory architecture inspired by MemGPT/Letta: (1) Working Memory (In-Context RAM): Fixed system prompt containing a core user profile block (key preferences, constraints, facts) + sliding window of the last 6-10 turns. (2) Summarization Layer: When the sliding window reaches 80% capacity, an asynchronous background task summarizes older turns into concise episodic narratives and archives them. (3) Episodic Long-Term Memory (Archival Disk): Stored in a vector database embedded with conversational timestamps. Retrieved dynamically via hybrid search when the current conversation references past events. (4) Reflection & Consolidation: Nightly agent job clustering episodic memories and updating core semantic knowledge graphs.',
    followUpProbes: ['How do you score memories for retrieval? (Using Park et al. formula: Score = a*recency + b*importance + c*relevance)', 'What happens when retrieved memories conflict with current conversation facts?']
  },
  {
    id: 7,
    tier: 'Staff/Principal AI Architect',
    topic: 'Distributed Vector Systems & Multi-Tenancy at Scale',
    question: 'You are architecting an enterprise search engine indexing 500 million documents across 50,000 enterprise tenants with strict document-level ACLs. How do you design the storage, indexing, and query path to guarantee < 100ms p95 latency and zero data leakage?',
    competenciesTested: ['Distributed systems', 'Vector DB internals', 'Multi-tenant isolation', 'Security architecture'],
    redFlags: [
      'Creating 50,000 independent vector DB instances or collections (management & resource collapse)',
      'Using naive post-filtering on query results (leads to catastrophic Recall Collapse)',
      'Ignoring horizontal sharding and index memory footprints'
    ],
    idealAnswer: '(1) Sharding & Storage Architecture: Multi-tenant shared cluster partitioned by tenant groups into shards. High-volume tenants receive dedicated shards. Chunks are compressed using Scalar Quantization (SQ8) to reduce RAM by 75% while retaining 98%+ recall. (2) In-Engine Pre-Filtering: Document-level ACLs are stored as tenant_id and allowed_group_ids inside vector payload metadata. During query execution, the engine evaluates permissions using Roaring Bitmaps *during* HNSW graph traversal. Chunks outside the user\'s authorized bitmask are pruned before distance computation, completely avoiding post-filtering recall collapse. (3) Two-Tier Routing: Gateway validates tenant JWT, attaches authorized security bitmask, and routes exclusively to shards housing that tenant\'s data.',
    followUpProbes: ['How do you handle real-time permission revocations? (Update Roaring Bitmaps via Kafka CDC stream)', 'How do you benchmark whether HNSW pre-filtering is dropping recall when filters are extremely restrictive (<0.1% matching)?']
  },
  {
    id: 8,
    tier: 'Staff/Principal AI Architect',
    topic: '10,000 QPS Low-Latency High-Availability RAG System',
    question: 'Design an end-to-end RAG platform serving 10,000 queries per second with a strict p95 Time-To-First-Token (TTFT) budget of 800ms. Detail the caching, retrieval, inference batching, and cost optimization strategy.',
    competenciesTested: ['High-throughput architecture', 'GPU inference economics', 'Continuous batching', 'Semantic caching'],
    redFlags: [
      'Relying solely on external API providers without local inference clusters at 10k QPS',
      'Ignoring the difference between TTFT and Inter-Token Latency (ITL)',
      'Unfamiliarity with semantic caching and continuous batching engines (vLLM / TensorRT-LLM)'
    ],
    idealAnswer: '(1) Multi-Tier Caching: Tier 1 Exact Hash Cache (Redis) intercepting frequent identical queries in < 2ms. Tier 2 Semantic Cache (Redis Vector Search) matching queries with cosine similarity >= 0.96 with pre-computed validated responses, resolving 25-35% of traffic in < 15ms. (2) Ultra-Fast Retrieval: Distributed Qdrant cluster on NVMe with SQ8 quantization; candidate retrieval restricted to top-20 in < 12ms. Skip heavy cross-encoder reranking on simple queries using a lightweight ModernBERT classifier. (3) Inference Fleet: Self-hosted vLLM cluster running LLaMA-3.1-8B-Instruct with FP8 quantization and continuous batching (PagedAttention). Speculative decoding with a 1B draft model reduces TTFT to < 220ms. Total p95 TTFT is ~480ms, well within the 800ms budget.',
    followUpProbes: ['What is the monthly cost comparison between OpenAI API vs Self-Hosted H100 cluster at 10k QPS? ($3.8M/mo vs $280k/mo)', 'How do you handle streaming over HTTP without buffering in reverse proxies? (X-Accel-Buffering: no with SSE)']
  }
];
