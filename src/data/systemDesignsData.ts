import { SystemDesignCase } from '../types/curriculum';

export const systemDesignsData: SystemDesignCase[] = [
  {
    id: 1,
    title: 'Chatbot over Complex Technical PDFs (Tables, Equations, Multi-Column)',
    scale: '500,000 documents (~10M pages), 1,000 QPS peak',
    sla: 'p95 TTFT < 800ms, Accuracy > 92%',
    requirements: [
      'High-fidelity parsing of complex 2-column academic & engineering PDFs',
      'Accurate retrieval of financial and statistical tables with cell preservation',
      'LaTeX equation preservation without token corruption',
      'Direct page and bounding-box attribution in generated citations'
    ],
    architectureComponents: [
      { name: 'Vision / Ingestion Service', role: 'Converts PDF pages to high-res image tiles and extracts layout trees', tech: 'PyMuPDF + ColPali / LayoutLMv3' },
      { name: 'Table Extractor', role: 'Dual-pass lattice & stream table detection, serializing to Markdown/HTML', tech: 'Table Transformer (TATR) + Camelot' },
      { name: 'Multi-Vector Store', role: 'Stores page patch embeddings and chunk vectors with page metadata', tech: 'Qdrant / Milvus (HNSW + SQ8)' },
      { name: 'Neural Reranker', role: 'Cross-encoder scoring top-50 candidates down to top-5', tech: 'Cohere Rerank v3 / BGE-Reranker-Large' },
      { name: 'Generator Gateway', role: 'Streams answer with bounding box highlight coordinates for PDF viewer', tech: 'FastAPI + Claude 3.5 Sonnet / GPT-4o' }
    ],
    dataFlowSteps: [
      'User uploads PDF -> S3 bucket triggers async ingest task via SQS/Celery worker',
      'Worker parses document layout, tables into structured HTML, and body text into contextual chunks',
      'Chunks and table summaries are embedded via BGE-M3 and indexed into Qdrant with doc_id & page metadata',
      'User query enters API gateway -> HyDE + Sub-question decomposition expands query',
      'Hybrid search (Dense 768-d + BM25 sparse) retrieves top-50 candidates -> Cross-encoder reranks top-5',
      'Top-5 chunks with table HTML injected into prompt template -> SSE streamed answer with page citations'
    ],
    failureModesAndRecovery: [
      { failure: 'OCR misreads numbers in financial tables', mitigation: 'Store raw HTML table representation alongside embeddings and verify checksums on numeric outputs' },
      { failure: 'Late interaction token index memory bloat', mitigation: 'Apply ColBERTv2 centroid-based residual quantization (compressing 128-d vectors to 2 bits/dim)' }
    ],
    costAndLatencyModel: 'Monthly infrastructure: ~$2,400 (Vector DB + Ingest workers). Per-query token cost: ~$0.008. p50 latency: 620ms.'
  },
  {
    id: 2,
    title: 'Enterprise Multi-Tenant Semantic Search with Strict Document-Level ACLs',
    scale: '100,000 tenants, 50,000,000 indexed chunks, 5,000 QPS',
    sla: 'p99 latency < 250ms, Zero unauthorized document leakage',
    requirements: [
      'Hard tenant data isolation with zero cross-tenant query contamination',
      'Dynamic user permission checks (RBAC/ABAC) evaluated at query time',
      'Zero "Recall Collapse" from naive post-filtering',
      'Instant permission revocation synchronization'
    ],
    architectureComponents: [
      { name: 'Auth & Policy Gateway', role: 'Resolves user claims and tenant token into permission bitmasks', tech: 'OAuth2 / OPA (Open Policy Agent)' },
      { name: 'Vector DB Cluster', role: 'Distributed vector search with in-engine pre-filtering bitmasks', tech: 'Qdrant / Milvus (Payload Roaring Bitmaps)' },
      { name: 'Permission Sync Bus', role: 'CDC stream updating document ACL bitsets in real time', tech: 'Kafka + Debezium' },
      { name: 'Hybrid Inverted Index', role: 'Sparse keyword search restricted to tenant namespace', tech: 'OpenSearch / Elasticsearch' }
    ],
    dataFlowSteps: [
      'Client query arrives with JWT bearer token containing user_id and tenant_id',
      'Gateway verifies JWT with OPA -> Fetches user group IDs: [g_sales, g_exec_eu]',
      'Gateway constructs vector search query: payload filter (tenant_id == "T123" AND acl_groups OVERLAPS [g_sales, g_exec_eu])',
      'Vector engine evaluates filter via Roaring Bitmaps *during* HNSW graph traversal (pre-filtering)',
      'Engine returns top-K authorized candidates; cross-tenant chunks never enter candidate beam',
      'Query response returns authorized snippets with security audit trail logged to ClickHouse'
    ],
    failureModesAndRecovery: [
      { failure: 'Post-filtering Recall Collapse (all top-10 chunks filtered out by ACL)', mitigation: 'Strictly enforce in-engine pre-filtering so HNSW explores alternate graph paths within the authorized bitmask' },
      { failure: 'Permission sync lag exposing newly restricted documents', mitigation: 'Synchronous ACL cache invalidation in Redis with optimistic lock versioning on documents' }
    ],
    costAndLatencyModel: 'Monthly compute: ~$6,500 for distributed 3-node Qdrant cluster + Redis. Query latency: 85ms p50, 190ms p99.'
  },
  {
    id: 3,
    title: 'Real-Time Customer Support RAG with Self-Correction & Human-in-the-Loop',
    scale: '2,000,000 daily tickets, 250 QPS peak',
    sla: 'p90 response < 1.5s, Hallucination rate < 0.1% on refund/billing policies',
    requirements: [
      'Self-correction mechanism verifying refund policy compliance before sending',
      'Automatic confidence scoring routing low-confidence answers to human agents',
      'Live synchronization with CRM customer account state',
      'Support agent override & feedback learning loop'
    ],
    architectureComponents: [
      { name: 'Query Classifier', role: 'Classifies intent, policy sensitivity, and customer sentiment', tech: 'Fine-tuned ModernBERT / FastText' },
      { name: 'CRM State Fetcher', role: 'Retrieves active subscription, past tickets, and billing tier', tech: 'gRPC client to Salesforce/Stripe' },
      { name: 'Corrective RAG Pipeline', role: 'Retrieves SOPs and evaluates retrieved passage relevance', tech: 'Self-RAG / CRAG evaluator' },
      { name: 'Groundedness Verifier', role: 'Secondary fast LLM verifying factual entailment of response', tech: 'vLLM (Llama-3.1-8B-Instruct)' },
      { name: 'Zendesk Bridge', role: 'Dispatches automated response or escalates to human inbox', tech: 'Webhooks + Celery' }
    ],
    dataFlowSteps: [
      'Inbound ticket arrives -> Intent classifier categorizes: "Billing Inquiry (High Risk)"',
      'CRM fetcher attaches customer state: { tier: "Enterprise", mrr: 1200, status: "Active" }',
      'Retriever fetches latest Billing SOP v4.2 -> CRAG evaluator assigns confidence gamma = 0.94',
      'Draft generator synthesizes response -> Entailment verifier checks: All claims supported by SOP?',
      'If verified: response auto-sent to customer within 1.2s; If verification score < 0.85: routed to Human Agent with pre-filled draft'
    ],
    failureModesAndRecovery: [
      { failure: 'Hallucinated promise of unauthorized refund', mitigation: 'Hard deterministic guardrail checking for refund dollar amounts exceeding customer tier limit' },
      { failure: 'Customer service policy updated mid-day with cache staleness', mitigation: 'Instant TTL invalidation on policy knowledge base using Redis pub/sub' }
    ],
    costAndLatencyModel: 'Reduces human support load by 68%. Average cost per ticket: $0.012 vs $3.50 for human agent.'
  },
  {
    id: 4,
    title: 'Autonomous Deep-Research Web Agent with Source Verification',
    scale: '10,000 comprehensive research dossiers/day',
    sla: '10-page synthesized report with 30+ verified citations delivered in < 3 minutes',
    requirements: [
      'Recursive query planning and sub-question decomposition',
      'Stealth headless browsing, web scraping, and PDF whitepaper extraction',
      'Cross-verification across multiple independent primary sources',
      'Detection and elimination of commercial SEO marketing spam'
    ],
    architectureComponents: [
      { name: 'Planning Agent', role: 'Constructs dynamic research tree and sub-hypotheses', tech: 'LangGraph StateGraph + Claude 3.5 Sonnet' },
      { name: 'Web Scraping Fleet', role: 'Headless browser execution and clean markdown parsing', tech: 'Playwright + Trafilatura / Firecrawl' },
      { name: 'Source Quality Filter', role: 'Domain reputation scoring (.edu, .gov, peer-reviewed, corporate PR)', tech: 'Custom PageRank + Whitelist DB' },
      { name: 'Synthesis & Fact-Checker', role: 'Cross-checks claims across sources and writes report with citations', tech: 'GPT-4o / DeepSeek-V3' }
    ],
    dataFlowSteps: [
      'User inputs research topic: "Solid-state battery commercialization timeline 2026-2030"',
      'Planner generates 6 sub-questions covering chemistry, OEM partnerships, and manufacturing costs',
      'Worker agents dispatch parallel web searches -> Download 45 web articles and 12 technical PDFs',
      'Scraping fleet extracts clean text; source quality filter discards 20 low-quality SEO articles',
      'Fact-checker builds claim matrix: requires >= 2 independent primary sources per major milestone date',
      'Report synthesizer structures executive brief, comparison table, technical deep-dive, and bibliography'
    ],
    failureModesAndRecovery: [
      { failure: 'Agent gets trapped in infinite web browsing loop', mitigation: 'Strict depth horizon limit (max_depth=3) and token budget governor' },
      { failure: 'Indirect prompt injection hidden in scraped webpage', mitigation: 'Content sandbox isolating raw page HTML in data plane, parsing with strict schema' }
    ],
    costAndLatencyModel: 'Cost per report: ~$0.85 (LLM tokens + browser compute). End-to-end time: 90-140 seconds.'
  },
  {
    id: 5,
    title: 'Multi-Modal Data Analysis Agent (Text, Code Interpreter, SQL, Charting)',
    scale: '50,000 enterprise business analysts',
    sla: 'p95 query-to-chart latency < 5s',
    requirements: [
      'Automatic schema introspection over Snowflake/BigQuery data warehouses',
      'Safe sandbox code execution for Python pandas/numpy data transformation',
      'High-quality interactive chart generation (Vega-Lite / Plotly)',
      'Natural language explanation of statistical significance and anomalies'
    ],
    architectureComponents: [
      { name: 'NL-to-SQL Engine', role: 'Converts business query into optimized read-only SQL', tech: 'Vanna.ai / Custom Few-shot SQL Agent' },
      { name: 'Data Warehouse Gateway', role: 'Executes queries with query timeout and row limit constraints', tech: 'Snowflake / BigQuery REST API' },
      { name: 'Code Sandbox', role: 'Isolated container executing Python analysis scripts with zero internet access', tech: 'E2B / Firecracker MicroVMs' },
      { name: 'Visualization Generator', role: 'Outputs declarative JSON chart specs', tech: 'Plotly / Vega-Lite' }
    ],
    dataFlowSteps: [
      'Analyst asks: "Compare churn rate between annual and monthly cohorts over last 6 quarters"',
      'Agent introspects schema metadata -> Generates optimized SQL query with window functions',
      'Warehouse returns 500-row aggregated result set to isolated backend',
      'Agent writes Python analysis script: calculates retention curves and generates Plotly JSON',
      'Script executes in Firecracker microVM -> Produces interactive chart JSON and statistical summary',
      'Frontend renders Generative UI widget with interactive toggles and CSV export'
    ],
    failureModesAndRecovery: [
      { failure: 'LLM generates destructive SQL (DROP/DELETE) or expensive full-table scan', mitigation: 'Enforce read-only database credentials and strict query budget estimators' },
      { failure: 'Malicious Python code attempts container breakout', mitigation: 'MicroVM isolation (Firecracker) with read-only root filesystem and seccomp syscall filters' }
    ],
    costAndLatencyModel: 'Compute cost: ~$0.03 per analysis run. Query latency: 2.8s p50, 4.9s p95.'
  },
  {
    id: 6,
    title: 'Hierarchical Multi-Agent Enterprise Assistant (Supervisor-Worker Architecture)',
    scale: '150,000 corporate employees across HR, IT, Finance, and Legal',
    sla: 'p90 request completion < 8s',
    requirements: [
      'Unified conversational entry point routing to specialized departmental agents',
      'Shared episodic conversation context with department-specific permission isolation',
      'Transactional multi-step task execution (e.g. submit expense + notify manager)',
      'Deterministic audit logging for compliance'
    ],
    architectureComponents: [
      { name: 'Supervisor Agent', role: 'Triages user requests, maintains task state, and delegates to workers', tech: 'LangGraph StateGraph' },
      { name: 'HR Worker Agent', role: 'Handles PTO, benefits, policy questions via HR knowledge base', tech: 'Workday API + HR RAG' },
      { name: 'IT Worker Agent', role: 'Automates password resets, VPN access, software provisioning', tech: 'ServiceNow + Okta API' },
      { name: 'Finance Worker Agent', role: 'Processes invoices, expense approvals, budget inquiries', tech: 'SAP / NetSuite API' },
      { name: 'Blackboard State Store', role: 'Stores shared session state with encrypted field-level permissions', tech: 'Redis Enterprise / PostgreSQL' }
    ],
    dataFlowSteps: [
      'Employee inputs: "I\'m traveling to London next week. Book flight approval and check per diem policy"',
      'Supervisor decomposes into two parallel sub-tasks: [Task A: HR per diem lookup], [Task B: Finance travel request]',
      'Supervisor dispatches Task A to HR Agent and Task B to Finance Agent with user identity token',
      'HR Agent retrieves London tier-1 per diem ($120/day) from RAG SOPs',
      'Finance Agent checks employee department budget and scaffolds expense authorization draft',
      'Supervisor merges worker outputs into single cohesive approval summary with action button'
    ],
    failureModesAndRecovery: [
      { failure: 'Worker agent fails or encounters API timeout', mitigation: 'Supervisor implements circuit breaker and returns partial success with graceful fallback message' },
      { failure: 'Sycophancy cascade where workers confirm false assumptions', mitigation: 'Structured communication schema with strict validation rather than unconstrained freeform chat' }
    ],
    costAndLatencyModel: 'Saves estimated 2.4 hours/employee/week. Total platform cost: $0.18/user/month.'
  },
  {
    id: 7,
    title: 'Low-Latency High-Throughput Streaming RAG Pipeline (<800ms TTFT at 10,000 QPS)',
    scale: '10,000 queries per second, 100,000,000 active users',
    sla: 'p95 TTFT < 800ms, p99 ITL < 40ms, 99.99% Availability',
    requirements: [
      'Ultra-fast vector search completing in < 15ms at 10k QPS',
      'Semantic caching intercepting >= 30% of repetitive queries in < 5ms',
      'Zero-buffering Server-Sent Events (SSE) streaming direct to browser',
      'Tiered model routing (SLM for simple queries, Frontier for complex)'
    ],
    architectureComponents: [
      { name: 'Edge Gateway', role: 'SSL termination, token bucket rate limiting, and SSE streaming', tech: 'Envoy / Cloudflare Workers' },
      { name: 'Semantic Cache', role: 'Redis vector search caching exact & near-identical queries (cos sim >= 0.96)', tech: 'Redis Enterprise (RediSearch)' },
      { name: 'Distributed Vector Cluster', role: 'Sharded HNSW indices with SQ8 scalar quantization across 32 nodes', tech: 'Qdrant Distributed / Milvus' },
      { name: 'Model Router', role: 'Routes to Gemma-2-9B (simple) or LLaMA-3.3-70B (complex)', tech: 'FastText intent classifier + vLLM cluster' },
      { name: 'Inference Engine', role: 'Continuous batching with PagedAttention and FP8 quantization', tech: 'vLLM on NVIDIA H100s' }
    ],
    dataFlowSteps: [
      'Query hits Edge Gateway -> Checks Semantic Cache with query embedding (3ms)',
      'Cache Hit (32% of traffic): Immediately streams pre-computed validated response -> TTFT: 18ms!',
      'Cache Miss: Dispatches parallel dual-retrieval (Dense 768-d in Qdrant + BM25 in OpenSearch) -> 14ms',
      'RRF fuses top-10 chunks -> Model router evaluates query complexity (2ms)',
      'Routes to vLLM continuous batching pool -> PagedAttention computes first token in 180ms',
      'Tokens streamed via SSE to client over HTTP/2 connection with zero buffering'
    ],
    failureModesAndRecovery: [
      { failure: 'Sudden spike in query volume saturating GPU memory', mitigation: 'Dynamic speculative decoding with SLM draft model and graceful degradation of top-K chunks' },
      { failure: 'Semantic cache returns outdated answer for updated document', mitigation: 'Invalidate cache keys via document ID inverted index upon any CMS update event' }
    ],
    costAndLatencyModel: 'Semantic cache saves $48,000/month in GPU inference costs. Blended TTFT: 240ms p50, 780ms p95.'
  },
  {
    id: 8,
    title: 'Enterprise Agent Platform: Sandboxing, Tool Registry, RBAC, and Observability',
    scale: 'Platform for 500 internal developer teams deploying 2,000 custom AI agents',
    sla: '99.95% platform availability, < 50ms sandbox startup overhead',
    requirements: [
      'Model Context Protocol (MCP) standardized tool interface',
      'Isolated ephemeral execution sandboxes with zero host access',
      'Fine-grained role-based access control (RBAC) on external API tool calls',
      'OpenTelemetry distributed tracing capturing all agent thoughts, tool calls, and costs'
    ],
    architectureComponents: [
      { name: 'MCP Tool Registry', role: 'Catalog of verified enterprise tools with JSON Schema contracts', tech: 'MCP Server Registry + Vault for secrets' },
      { name: 'Sandbox Orchestrator', role: 'Spins up microVM sandboxes in < 40ms with copy-on-write rootfs', tech: 'Firecracker MicroVMs / Kata Containers' },
      { name: 'Agent Runtime Engine', role: 'Stateless execution engine driving agent decision loops', tech: 'FastAPI + Temporal Workflow Engine' },
      { name: 'LLMOps Observability', role: 'Tracks spans, token costs, latency, and evaluations across all agents', tech: 'OpenTelemetry + Langfuse / Arize Phoenix' }
    ],
    dataFlowSteps: [
      'Developer deploys agent spec YAML: defines allowed MCP tools, model provider, and token limits',
      'User triggers agent run -> Platform provisions ephemeral Firecracker microVM with pre-warmed snapshot',
      'Agent runtime queries MCP Registry -> Injects validated tool schemas into model context',
      'Agent invokes tool `finance_db.run_query` -> Gateway checks user RBAC permissions before execution',
      'Tool executes inside sandbox -> Results streamed back to agent loop',
      'Every thought, tool call, latency millisecond, and token expenditure emitted to OpenTelemetry collector'
    ],
    failureModesAndRecovery: [
      { failure: 'Rogue agent enters infinite recursive tool execution loop', mitigation: 'Enforce hard execution timeouts (max 120s), max step limits (max 20), and token budget kill-switches' },
      { failure: 'Tool execution leaks sensitive API keys or credentials', mitigation: 'Zero secret exposure to LLM context: LLM receives only placeholder tokens resolved by the execution gateway' }
    ],
    costAndLatencyModel: 'Platform amortizes infrastructure across 500 teams. Replaces custom boilerplate with 10x developer velocity.'
  }
];
