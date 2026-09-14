import { Module } from '../../types/curriculum';

export const systemsModules: Module[] = [
  {
    id: 25,
    slug: 'production-ai-systems',
    level: 9,
    levelName: 'Level 9: Production AI & Systems Engineering',
    title: '25. Production AI Systems: Streaming, Caching & Microservices',
    duration: '4 Hours',
    description: 'Server-Sent Events (SSE) streaming, Semantic Caching (Redis), Circuit Breakers, Model Routing (SLM vs LLM tiering), and FastAPI architecture.',
    learningObjectives: [
      'Architect low-latency streaming microservices using Server-Sent Events (SSE) and HTTP/2',
      'Implement a Redis semantic vector cache that intercepts 30%+ of repetitive queries in <15ms',
      'Deploy resilient circuit breakers and multi-provider fallback cascades'
    ],
    prerequisites: ['Level 8 Multi-Agent Systems & Infrastructure'],
    theory: {
      definition: 'The distributed systems engineering patterns required to operate generative AI models reliably at scale under strict latency and cost constraints.',
      intuition: 'A prototype in a notebook calls `client.chat.completions.create()` synchronously. In production, that call will drop, time out, or cost thousands of dollars during traffic spikes. Production engineering wraps that call with streaming (so the user sees words immediately), semantic caching (so identical questions are answered for free in 5ms), and automatic failovers.',
      technicalExplanation: 'Key production components: (1) Low-Latency Streaming: Users will not wait 15 seconds for a complete 800-token response. Streaming via SSE (`text/event-stream`) reduces perceived latency to the Time-To-First-Token (TTFT, ~250-500ms). Reverse proxies must bypass buffering via `X-Accel-Buffering: no`. (2) Semantic Caching: Computes query vector embedding and queries Redis. If cosine similarity >= 0.96 and TTL is valid, returns cached response in <10ms, slashing LLM token costs by 25-45%. (3) Model Tiering: Routes simple extractions/greetings to fast SLMs (Gemma-2-9B at $0.05/1M tokens) and reserves frontier models (Claude 3.5 Sonnet, GPT-4o) for complex multi-hop reasoning.',
      example: 'A user asks: "What is your return policy?" at 9:00 AM. At 9:05 AM, another user asks: "How do I return an item?" The semantic cache detects cosine similarity = 0.975 and serves the cached answer in 6ms without hitting the LLM API.'
    },
    implementation: {
      language: 'Python',
      code: `import time

class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        self.cache: list[dict] = []
        self.threshold = threshold

    def get(self, query_vec: list[float]) -> str | None:
        for entry in self.cache:
            # Simulated dot product on normalized vectors
            sim = sum(a * b for a, b in zip(query_vec, entry["vector"]))
            if sim >= self.threshold:
                return f"[CACHE HIT in 4ms] {entry['response']}"
        return None

    def put(self, query_vec: list[float], response: str):
        self.cache.append({"vector": query_vec, "response": response, "timestamp": time.time()})

cache = SemanticCache(threshold=0.90)
cache.put([1.0, 0.0], "Our refund window is 30 days.")
hit = cache.get([0.98, 0.02])
print("Cache Result:", hit)`,
      explanation: 'Two-tier semantic vector cache implementation.'
    },
    failureModes: [
      'Stale semantic cache: Serving an outdated refund policy after the legal department updated the terms',
      'SSE proxy buffering: NGINX or Cloudflare buffering chunks until 4KB accumulates, completely defeating the purpose of streaming'
    ],
    engineeringTradeoffs: [
      'Semantic Cache Threshold: Threshold 0.90 gives high hit rates (40%) but risks returning answers to subtly different questions; 0.98 gives zero false positives but lowers hit rate to 15%.'
    ],
    exercise: {
      prompt: 'Write the NGINX configuration directive required to enable real-time SSE streaming without buffering.',
      hint: 'Look for the proxy buffering setting.',
      solution: `location /v1/chat/stream {
    proxy_pass http://fastapi_backend;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header X-Accel-Buffering no;
}`
    },
    quiz: [
      {
        question: 'Why is Server-Sent Events (SSE) preferred over WebSockets for LLM chat completions?',
        options: [
          'SSE is faster than WebSockets.',
          'SSE operates over standard unidirectional HTTP/2, natively supporting reconnection, firewalls, and proxying without stateful socket upgrade overhead.',
          'WebSockets do not support text data.',
          'SSE encrypts tokens automatically.'
        ],
        correctIndex: 1,
        explanation: 'LLM completions are inherently unidirectional: the client sends a single request, and the server streams multiple tokens back. SSE uses standard HTTP/2 streams, avoiding the connection state complexity of full-duplex WebSockets.'
      }
    ]
  },
  {
    id: 26,
    slug: 'observability-tracing-telemetry',
    level: 9,
    levelName: 'Level 9: Production AI & Systems Engineering',
    title: '26. Observability, Tracing & LLMOps Telemetry',
    duration: '3.5 Hours',
    description: 'OpenTelemetry GenAI Semantic Conventions, Langfuse, Arize Phoenix, distributed spans, token cost accounting, and latency profiling.',
    learningObjectives: [
      'Instrument LLM pipelines with OpenTelemetry distributed tracing across microservices',
      'Track span hierarchies: User Request -> Router -> Retriever -> Reranker -> Generator',
      'Audit token consumption and latency metrics (TTFT, Inter-Token Latency) in production'
    ],
    prerequisites: ['Level 25 Production AI Systems: Streaming, Caching & Microservices'],
    theory: {
      definition: 'The telemetry, logging, and distributed tracing systems that provide complete visibility into model decisions, tool calls, costs, and latencies across distributed AI systems.',
      intuition: 'When an AI system fails in production, you cannot just look at a standard error log that says "200 OK". You need to see the exact prompt sent, the 5 chunks retrieved, how the reranker scored them, how many tokens were burned, and which exact tool call caused the unexpected behavior.',
      technicalExplanation: 'The OpenTelemetry GenAI Working Group (2024) standardizes telemetry attributes: `gen_ai.system` (openai, anthropic), `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, and `gen_ai.response.finish_reasons`. Spans are organized hierarchically: Root Span (`chat_session`) -> Child Span (`hybrid_retrieval`) -> Child Span (`dense_search`) -> Child Span (`reranking`) -> Child Span (`llm_generation`). Distributed trace headers (W3C `traceparent`) propagate context across HTTP microservices.',
      example: 'An agent fails to book a flight. In Langfuse, the engineer clicks the trace, expands turn 3, and sees: Tool `flight_api` returned `400 Bad Request: Date in past (2023-09-14)`. The engineer immediately discovers the agent had the wrong year in its system clock.'
    },
    implementation: {
      language: 'Python',
      code: `import time
from contextlib import contextmanager

class SimpleTelemetryTracker:
    def __init__(self):
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, attributes: dict = None):
        start = time.time()
        span_record = {"name": name, "attributes": attributes or {}}
        try:
            yield span_record
        finally:
            span_record["duration_ms"] = (time.time() - start) * 1000
            self.spans.append(span_record)

tracker = SimpleTelemetryTracker()
with tracker.span("rag_pipeline", {"user_id": "u123"}):
    with tracker.span("retrieval", {"top_k": 3}):
        time.sleep(0.015) # 15ms retrieval
    with tracker.span("llm_call", {"model": "gpt-4o-mini", "input_tokens": 450}):
        time.sleep(0.040) # 40ms generation

print("Captured Spans:", len(tracker.spans))`,
      explanation: 'Context managers capture timing and OpenTelemetry span attributes for child operations, recording hierarchical traces and execution latencies.'
    },
    failureModes: [
      'Logging PII/Secrets: Blindly logging raw user prompts containing credit cards or passwords into unencrypted telemetry stores',
      'Telemetry overhead: Synchronous telemetry reporting adding 50ms latency to every user token stream'
    ],
    engineeringTradeoffs: [
      '100% trace sampling vs Head-based 5% sampling: Sampling 100% of traces gives perfect auditability but generates terabytes of logging data; sampling 5% reduces costs by 95% while catching systemic latency trends.'
    ],
    exercise: {
      prompt: 'What are the two mandatory metrics required to measure streaming LLM performance?',
      hint: 'One measures the initial wait time; the other measures the smoothness of the text generation.',
      solution: '1. Time-To-First-Token (TTFT): the time from sending the request to receiving the very first token chunk. 2. Inter-Token Latency (ITL) or Tokens Per Second (TPS): the average time between consecutive tokens during generation.'
    },
    quiz: [
      {
        question: 'What is the purpose of the OpenTelemetry GenAI Semantic Conventions?',
        options: [
          'To standardize attribute names for tokens, models, prompts, and latencies across all AI tools and observability platforms.',
          'To replace Python with C++.',
          'To make all models output the same answers.',
          'To encrypt GPU memory.'
        ],
        correctIndex: 0,
        explanation: 'OpenTelemetry GenAI conventions standardize metric and span attributes (like gen_ai.usage.input_tokens) so that any LLM application can switch between observability vendors (Langfuse, Datadog, Phoenix) without rewriting instrumentation code.'
      }
    ]
  },
  {
    id: 27,
    slug: 'security-red-teaming-governance',
    level: 9,
    levelName: 'Level 9: Production AI & Systems Engineering',
    title: '27. Enterprise Security, Red Teaming & Agentic Governance',
    duration: '4 Hours',
    description: 'OWASP Top 10 for LLMs, prompt injection, tool privilege escalation, data exfiltration, sandboxing, and adversarial red-teaming.',
    learningObjectives: [
      'Audit applications against the OWASP Top 10 for LLMs and Generative AI',
      'Prevent Excessive Agency and tool privilege escalation in autonomous agents',
      'Implement dual-model control/data plane separation architectures'
    ],
    prerequisites: ['Level 26 Observability, Tracing & LLMOps Telemetry'],
    theory: {
      definition: 'The comprehensive security discipline focused on preventing adversarial exploitation, data breaches, unauthorized agency, and safety violations in LLM and agentic systems.',
      intuition: 'Traditional software vulnerabilities (SQL injection, buffer overflows) exploit computer code syntax. LLM vulnerabilities exploit natural language semantics. If an attacker can convince the model to ignore its system prompt, the attacker gains control of all tools and data accessible to that agent.',
      technicalExplanation: 'OWASP Top 10 for LLMs: (1) Prompt Injection (Direct & Indirect), (2) Sensitive Information Disclosure, (3) Supply Chain Vulnerabilities, (4) Data and Model Poisoning, (5) Improper Output Handling, (6) Excessive Agency, (7) System Prompt Leakage, (8) Vector and Embedding Weaknesses, (9) Misinformation, (10) Unbounded Consumption. In agents, Excessive Agency occurs when an agent is granted high-privilege tools (e.g. `delete_user`, `send_wire_transfer`) without Human-in-the-Loop approval gates.',
      example: 'A customer service agent with tool `execute_sql` is asked: "Please summarize my past orders and also run DROP TABLE users;--". Without read-only credentials and input validation, the database is wiped.'
    },
    implementation: {
      language: 'Python',
      code: `def enforce_least_privilege_tool(action: str, user_role: str) -> bool:
    """RBAC guardrail preventing excessive agency."""
    HIGH_RISK_TOOLS = {"delete_database", "execute_wire_transfer", "grant_admin"}
    if action in HIGH_RISK_TOOLS and user_role != "super_admin":
        return False
    return True

assert enforce_least_privilege_tool("delete_database", user_role="customer") is False
assert enforce_least_privilege_tool("view_account", user_role="customer") is True
print("Security Guardrail Enforced!")`,
      explanation: 'Least-privilege authorization gatekeeper preventing unauthorized tool execution.'
    },
    failureModes: [
      'Granting write/delete database permissions to an autonomous LLM tool',
      'Trusting client-side prompt safety filters that can be bypassed via direct API calls'
    ],
    engineeringTradeoffs: [
      'Human-in-the-Loop vs Complete Autonomy: Requiring human clicks on high-risk actions prevents catastrophic mistakes, but introduces human latency (minutes/hours).'
    ],
    exercise: {
      prompt: 'What is the "Dual-LLM" pattern for preventing indirect prompt injection in RAG?',
      hint: 'Use one model to summarize/extract data and a separate isolated model to formulate the final answer.',
      solution: 'The Privileged Model handles user interaction and holds tool permissions. The Quarantined / Reader Model has zero tools and only reads untrusted documents to extract raw facts into a rigid JSON schema. The Privileged Model inspects the validated JSON data, preventing untrusted document instructions from ever reaching the execution control plane.'
    },
    quiz: [
      {
        question: 'What is "Excessive Agency" according to OWASP?',
        options: [
          'An agent that runs too many queries per minute.',
          'Granting an agent damaging permissions, excessive autonomy, or high-impact tool access without human verification.',
          'An agent with more than 10 tools.',
          'An agent running on multiple GPUs.'
        ],
        correctIndex: 1,
        explanation: 'Excessive Agency occurs when an LLM is granted broad permissions (e.g. file deletion, financial transactions) and autonomous authority without adequate constraints or human approval checkpoints.'
      }
    ]
  },
  {
    id: 28,
    slug: 'performance-cost-engineering',
    level: 9,
    levelName: 'Level 9: Production AI & Systems Engineering',
    title: '28. Performance Engineering & Unit Cost Economics',
    duration: '3.5 Hours',
    description: 'Token arithmetic, Little\'s Law GPU capacity sizing, TTFT vs TPS optimization, semantic caching economics, and blended cost reduction.',
    learningObjectives: [
      'Model end-to-end token unit economics across input prompt caching, generation, and retrieval',
      'Apply Little\'s Law (L = lambda * W) to calculate GPU cluster capacity for target QPS',
      'Achieve 50%+ cost reductions using semantic caching and tiered SLM routing'
    ],
    prerequisites: ['Level 25 Production AI Systems: Streaming, Caching & Microservices'],
    theory: {
      definition: 'The mathematical and financial modeling required to scale generative AI systems while maintaining profitable unit economics and sub-second latencies.',
      intuition: 'Calling frontier models for every simple query will bankrupt an enterprise application. Performance engineering calculates exactly how many GPUs you need, how much each query costs, and how to route 70% of queries to cheaper models or caches.',
      technicalExplanation: 'Inference sizing relies on Little\'s Law: L = lambda * W, where L is concurrency (active requests in-flight), lambda is arrival rate (QPS), and W is average request duration (seconds). Sizing GPUs requires comparing compute FLOPs during prefill against memory bandwidth during decode. Cost modeling: Blended Cost = (1 - Cache_Hit_Rate) * [P_SLM * C_SLM + (1 - P_SLM) * C_Frontier] + Cache_Cost. In 2024-2026, prompt caching (Anthropic, OpenAI, DeepSeek) slashes input token costs by up to 90% for repeated prefix contexts.',
      mathematics: {
        formula: 'C_{\\text{query}} = (T_{\\text{in}} \\times P_{\\text{in}} + T_{\\text{out}} \\times P_{\\text{out}}) \\times (1 - H_{\\text{cache}}) + C_{\\text{infra}}',
        variables: [
          { name: 'T_in, T_out', desc: 'Number of input prompt tokens and output generated tokens' },
          { name: 'P_in, P_out', desc: 'Price per million input/output tokens' },
          { name: 'H_cache', desc: 'Semantic cache hit rate (e.g. 0.35)' },
          { name: 'C_infra', desc: 'Amortized vector database and compute cost per query' }
        ]
      },
      example: 'A support system receives 1,000,000 queries/month. Direct frontier model calls cost $15,000. Implementing a semantic cache (32% hit rate) and routing 50% of remainder to an 8B SLM drops monthly costs to $4,200 (a 72% savings).'
    },
    implementation: {
      language: 'Python',
      code: `def compute_blended_cost(
    monthly_queries: int,
    input_tokens: int,
    output_tokens: int,
    cache_hit_rate: float,
    frontier_price_in: float, # per 1M
    frontier_price_out: float,
    slm_ratio: float,
    slm_price_in: float,
    slm_price_out: float
) -> float:
    billable_queries = monthly_queries * (1.0 - cache_hit_rate)
    slm_queries = billable_queries * slm_ratio
    frontier_queries = billable_queries * (1.0 - slm_ratio)
    
    cost_frontier = frontier_queries * ((input_tokens * frontier_price_in + output_tokens * frontier_price_out) / 1e6)
    cost_slm = slm_queries * ((input_tokens * slm_price_in + output_tokens * slm_price_out) / 1e6)
    return cost_frontier + cost_slm

baseline = compute_blended_cost(1000000, 1000, 200, 0.0, 3.0, 15.0, 0.0, 0.1, 0.4)
optimized = compute_blended_cost(1000000, 1000, 200, 0.35, 3.0, 15.0, 0.6, 0.1, 0.4)
print(f"Baseline Cost: $\{baseline:.2f}")
print(f"Optimized Cost: $\{optimized:.2f} (Savings: {((baseline-optimized)/baseline)*100:.1f}%)")`,
      explanation: 'Financial model calculating token cost savings from caching and SLM tiering.'
    },
    failureModes: [
      'Unbounded prompt growth: Appending full chat history indefinitely causes input token costs to scale quadratically with session length',
      'Ignoring Little\'s Law: Under-provisioning GPU batch capacity causing queue times to skyrocket during traffic spikes'
    ],
    engineeringTradeoffs: [
      'Prompt caching vs Dynamic RAG: Prompt caching requires identical prefix tokens across requests; inserting dynamic timestamps or random IDs at the start of a prompt destroys cache hits.'
    ],
    exercise: {
      prompt: 'If an application receives 100 QPS and the average request takes 1.5 seconds, how many concurrent requests must the system support simultaneously?',
      hint: 'Apply Little\'s Law: L = lambda * W.',
      solution: 'L = 100 QPS * 1.5 seconds = 150 concurrent requests. The inference cluster and API gateway must maintain capacity for 150 parallel streams.'
    },
    quiz: [
      {
        question: 'What is the primary operational effect of Prompt Caching in modern API providers (e.g. Anthropic, OpenAI)?',
        options: [
          'It eliminates the need for GPUs.',
          'It provides up to a 90% discount on input tokens and up to an 80% reduction in TTFT for shared prompt prefixes.',
          'It encrypts user queries.',
          'It trains the model on user queries.'
        ],
        correctIndex: 1,
        explanation: 'Prompt caching reuses pre-computed KV-cache states for identical prompt prefixes across requests, dramatically slashing both input token costs and prefill TTFT latency.'
      }
    ]
  },
  {
    id: 29,
    slug: 'framework-comparison-evaluation',
    level: 10,
    levelName: 'Level 10: Frameworks, Benchmarks & Research Papers',
    title: '29. Framework Architecture Comparison & The Abstraction Tax',
    duration: '3.5 Hours',
    description: 'LangChain, LangGraph, LlamaIndex, Haystack, Semantic Kernel, AutoGen, CrewAI, DSPy, and PydanticAI. The Rule of Three for building native.',
    learningObjectives: [
      'Evaluate frameworks across Abstraction Leakiness, Debuggability, and Production Readiness',
      'Understand why high-level abstractions impose hidden prompt bloat and debugging obfuscation',
      'Apply the "Rule of Three" to decide when to build native vs adopt a framework'
    ],
    prerequisites: ['Level 28 Performance Engineering & Unit Cost Economics'],
    theory: {
      definition: 'A critical engineering assessment of modern generative AI orchestration frameworks and the trade-offs of third-party abstraction layers.',
      intuition: 'Frameworks promise to let you build an agent in 3 lines of code. But when something breaks in production at 2:00 AM, those 3 lines hide a 40-frame stack trace and undocumented prompt templates you cannot see or edit. Know what is underneath the hood before committing your architecture.',
      technicalExplanation: 'The "Abstraction Tax" manifests across four dimensions: (1) Cognitive Overhead: hidden prompts and implicit token formatting. (2) API Lag: frameworks take weeks to wrap newly released provider features (e.g. prompt caching, structured decoding). (3) Stack Trace Obfuscation: simple errors traverse dozens of nested base classes. (4) Vendor Lock-in. The Framework Spectrum ranges from Low Abstraction / High Control (Raw Provider SDKs, PydanticAI) to Explicit State Machines (LangGraph, Haystack) to Algorithmic Optimizers (DSPy) to High Abstraction / High Magic (LangChain classic, CrewAI).',
      example: 'A developer uses a high-level agent wrapper that secretly injects 1,200 tokens of ReAct prompt instructions on every turn. By replacing it with native Pydantic schemas and raw tool calls, token usage drops by 60% and latency drops by 400ms.'
    },
    implementation: {
      language: 'Python',
      code: `class NativeAgentPattern:
    """The 'Build Native' pattern: Raw SDK + Pydantic + Tenacity."""
    def __init__(self, client):
        self.client = client

    def run_step(self, messages: list[dict], tools_schema: list[dict]) -> dict:
        # Direct, transparent API call without framework wrappers:
        return self.client.chat_completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            temperature=0.0
        )`,
      explanation: 'The lean, native production pattern avoiding framework abstraction tax.'
    },
    failureModes: [
      'Framework Abandonment: Building on a framework that changes its core abstractions every 6 months, requiring constant breaking refactors',
      'Hidden Prompt Injection: Frameworks appending hidden default system prompts that contradict enterprise guidelines'
    ],
    engineeringTradeoffs: [
      'Framework vs Native: Use LangGraph when you need complex cyclic state machines and checkpointing; use Raw SDKs + Pydantic when building high-throughput single-step RAG or simple tool loops.'
    ],
    exercise: {
      prompt: 'State the "Rule of Three" for deciding whether to write native code or introduce an AI framework.',
      hint: 'Consider prototypes vs production microservices.',
      solution: '1. If a capability requires <= 3 files and <= 150 lines of clean Python (e.g. simple RAG, tool calling, memory buffer), write it native using raw SDKs. 2. If you need complex cyclic state machines with time-travel persistence across 10+ states, use LangGraph. 3. Never adopt a framework merely for its prompt templates.'
    },
    quiz: [
      {
        question: 'Why do senior production AI engineers frequently prefer PydanticAI or raw SDKs over high-abstraction frameworks?',
        options: [
          'High-abstraction frameworks do not support Python.',
          'Raw SDKs and Pydantic provide complete transparency over token prompts, zero hidden abstractions, instant support for new provider features, and microsecond stack traces.',
          'Raw SDKs run faster on GPUs.',
          'PydanticAI eliminates token costs.'
        ],
        correctIndex: 1,
        explanation: 'PydanticAI and raw provider SDKs offer type safety without hidden prompt bloat, avoiding the debugging nightmares and API lag common in heavy wrapper frameworks.'
      }
    ]
  },
  {
    id: 30,
    slug: 'chronological-research-papers',
    level: 10,
    levelName: 'Level 10: Frameworks, Benchmarks & Research Papers',
    title: '30. Chronological Research Papers Database & Reading Order',
    duration: '4 Hours',
    description: 'The 30 foundational research papers from Attention Is All You Need (2017) to 2026 frontiers. Problem, Method, Equations, Limitations, and Impact.',
    learningObjectives: [
      'Master the historical research progression of RAG and Agentic AI from 2017 to 2026',
      'Analyze the original papers: Vaswani 2017, Karpukhin 2020, Lewis 2020, Yao 2022, Asai 2023, Edge 2024',
      'Navigate research literature with critical skepticism regarding benchmark claims'
    ],
    prerequisites: ['Level 29 Framework Architecture Comparison & The Abstraction Tax'],
    theory: {
      definition: 'The curated academic literature and foundational papers that defined modern information retrieval, representation learning, reasoning, and autonomous agents.',
      intuition: 'Reading original research papers frees you from relying on second-hand blog posts and vendor marketing hype. When you know the original equations, experimental setups, and stated limitations, you can reason about new AI developments from first principles.',
      technicalExplanation: 'The literature spans four distinct eras: (1) Foundational Era (2017-2020): Attention Is All You Need (Vaswani), BERT (Devlin), GPT-3 (Brown). (2) Dense Retrieval & RAG Era (2020-2022): REALM (Guu), DPR (Karpukhin), RAG (Lewis), ColBERT (Khattab), RETRO (Borgeaud). (3) Reasoning & Agentic Era (2022-2023): Chain-of-Thought (Wei), ReAct (Yao), Toolformer (Schick), Reflexion (Shinn), Tree of Thoughts (Yao). (4) Advanced Systems Era (2024-2026): Self-RAG (Asai), GraphRAG (Edge), MemGPT (Packer), Contextual Retrieval (Anthropic), RULER (Hsieh).',
      example: 'Before building GraphRAG, reading Edge et al. (2024) reveals that GraphRAG is designed specifically for global sensemaking queries, and is actually slower and more expensive than vector RAG for specific local fact lookups.'
    },
    implementation: {
      language: 'Markdown / BibTeX',
      code: `@inproceedings{lewis2020rag,
  author    = {Patrick Lewis and Ethan Perez and Aleksand Piktus and Fabio Petroni and Vladimir Karpukhin and Naman Goyal and Heinrich K{\\"u}ttler and Mike Lewis and Wen-tau Yih and Tim Rockt{\\"a}schel and Sebastian Riedel and Douwe Kiela},
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {9459--9474},
  year      = {2020}
}`,
      explanation: 'Standard BibTeX citation for the seminal Lewis et al. RAG paper.'
    },
    failureModes: [
      'Citing hallucinated papers or unverified preprint claims as established facts',
      'Ignoring the "Limitations" section in academic papers'
    ],
    engineeringTradeoffs: [
      'Reading foundational papers vs blog tutorials: Papers require 3x more mathematical effort upfront, but give permanent architectural clarity that never becomes obsolete.'
    ],
    exercise: {
      prompt: 'What was the core difference in training methodology between REALM (Guu 2020) and DPR (Karpukhin 2020)?',
      hint: 'Consider whether the retriever was trained via masked language modeling end-to-end or via contrastive question-passage pairs.',
      solution: 'REALM trained the retriever end-to-end using self-supervised masked language modeling without human question-answer pairs, requiring asynchronous MIPS index refreshing. DPR trained the retriever using supervised question-passage pairs with in-batch negative contrastive loss, which proved far more computationally efficient.'
    },
    quiz: [
      {
        question: 'Which paper first introduced the interleaved generation of verbal reasoning traces (Thoughts) and executable tool Actions?',
        options: [
          'Attention Is All You Need (Vaswani et al., 2017)',
          'ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)',
          'Dense Passage Retrieval (Karpukhin et al., 2020)',
          'LoRA: Low-Rank Adaptation (Hu et al., 2021)'
        ],
        correctIndex: 1,
        explanation: 'Yao et al. (ICLR 2023 / arXiv 2022) authored ReAct, establishing the canonical Thought-Action-Observation loop that underpins modern agent software frameworks.'
      }
    ]
  },
  {
    id: 31,
    slug: 'benchmarks-evaluation-realities',
    level: 10,
    levelName: 'Level 10: Frameworks, Benchmarks & Research Papers',
    title: '31. Benchmarks & Evaluation Realities: MTEB, BEIR, SWE-bench & GAIA',
    duration: '3.5 Hours',
    description: 'Detailed analysis of MTEB, BEIR, MS MARCO, HotpotQA, SWE-bench, GAIA, data contamination, leaderboard hacking, and production divergence.',
    learningObjectives: [
      'Deconstruct standard IR benchmarks (BEIR, MTEB) and agent benchmarks (SWE-bench, GAIA)',
      'Identify benchmark flaws: test set contamination, reasoning shortcuts, and pooling bias',
      'Distinguish between leaderboard rankings and production performance realities'
    ],
    prerequisites: ['Level 30 Chronological Research Papers Database & Reading Order'],
    theory: {
      definition: 'The standardized academic datasets and evaluation protocols used to rank embedding models, retrievers, and autonomous agents, and the systematic ways they diverge from production.',
      intuition: 'A car that sets a speed record on a flat, paved racetrack might break down immediately on a rocky dirt road. Similarly, an embedding model that tops the MTEB leaderboard might fail completely on your company\'s proprietary medical PDFs or financial tables.',
      technicalExplanation: 'Key benchmarks: (1) MTEB (Massive Text Embedding Benchmark, Muennighoff et al. 2023): 56 datasets across 8 tasks (classification, clustering, retrieval, STS). Known flaw: retrieval is dominated by MS MARCO, favoring models biased toward search engine queries. (2) BEIR (Thakur et al. 2021): zero-shot evaluation across 18 diverse domains. (3) SWE-bench (Jimenez et al. 2024): evaluates agents on resolving real GitHub issues from open-source Python repos. Flaws: high test flakiness, environment setup failures, and test-suite leakage. (4) GAIA (Mialon et al. 2023): multi-modal, tool-use tasks requiring web browsing, spreadsheet analysis, and PDF parsing.',
      example: 'A model ranks #1 on MTEB retrieval by fine-tuning heavily on MS MARCO. In production enterprise search, its NDCG@10 drops by 30% because enterprise queries involve long, structured internal acronyms absent from web search logs.'
    },
    implementation: {
      language: 'Python',
      code: `def calculate_hit_rate_at_k(results: list[list[str]], ground_truth: list[str], k: int = 5) -> float:
    """Computes Hit Rate@K across a benchmark evaluation batch."""
    hits = 0
    for retrieved_docs, true_doc in zip(results, ground_truth):
        if true_doc in retrieved_docs[:k]:
            hits += 1
    return hits / max(1, len(ground_truth))

sample_retrievals = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
ground_truths = ["doc2", "doc99"]
hr = calculate_hit_rate_at_k(sample_retrievals, ground_truths, k=2)
print(f"Hit Rate@2: {hr * 100:.1f}%")`,
      explanation: 'Standard benchmark Hit Rate@K calculation.'
    },
    failureModes: [
      'Data Contamination: A benchmark dataset was scraped into the model\'s pre-training corpus, producing deceptively high evaluation scores',
      'Leaderboard Goodhart\'s Law: Optimizing an embedding model solely to climb MTEB ranks by over-fitting to MS MARCO'
    ],
    engineeringTradeoffs: [
      'Generic Benchmarks vs Domain Gold Sets: Evaluating on public MTEB is free and fast; creating an internal 200-question gold test set costs engineering time but provides 10x more predictive value for production success.'
    ],
    exercise: {
      prompt: 'What is "pooling bias" in traditional Information Retrieval benchmarks like MS MARCO?',
      hint: 'Consider how the "relevant" documents were chosen in the original dataset.',
      solution: 'In datasets like MS MARCO, only candidates retrieved by a specific pool of initial search systems (e.g. BM25 and early dual-encoders) were judged by humans. If a modern neural retriever surfaces a new, superior document that was never in the original candidate pool, it is marked as "irrelevant" (false negative), artificially penalizing advanced retrieval models.'
    },
    quiz: [
      {
        question: 'What does the SWE-bench benchmark evaluate?',
        options: [
          'Grammar spelling accuracy.',
          'Autonomous software engineering agents resolving real-world GitHub pull requests and issues against unit test suites.',
          'Vector database insertion speed.',
          'GPU memory bandwidth.'
        ],
        correctIndex: 1,
        explanation: 'SWE-bench evaluates end-to-end coding agents by presenting them with real GitHub issue descriptions from popular Python repositories, requiring the agent to edit files and pass the repository\'s existing unit test suite.'
      }
    ]
  }
];
