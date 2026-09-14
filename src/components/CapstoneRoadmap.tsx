import React, { useState, useEffect } from 'react';
import { Target, CheckCircle2, Circle, Trophy, Award, Sparkles, BookCheck, Terminal, Shield, Cpu } from 'lucide-react';

interface Phase {
  id: number;
  title: string;
  duration: string;
  objective: string;
  deliverables: string[];
  evaluationCriteria: string;
}

const CAPSTONE_PHASES: Phase[] = [
  {
    id: 1,
    title: 'Phase 1: Ingestion & Parsing Engine',
    duration: 'Week 1–2',
    objective: 'Construct a resilient PDF and multimodal ingestion pipeline capable of parsing two-column academic layouts, complex LaTeX mathematical proofs, and nested tabular data without cell corruption.',
    deliverables: [
      'Implement PyMuPDF layout-aware segmentation script',
      'Integrate Table Transformer (TATR) or Camelot for HTML/Markdown table extraction',
      'Preserve raw LaTeX delimiters ($...$ and $$...$$) through tokenization pipelines',
      'Compute SHA-256 chunk deduplication hashes to prevent redundant vector storage'
    ],
    evaluationCriteria: 'Achieve >95% cell accuracy on PubTables-1M test suite and zero LaTeX delimiter corruption.'
  },
  {
    id: 2,
    title: 'Phase 2: Hybrid Retrieval & Vector Indexing',
    duration: 'Week 3–4',
    objective: 'Build a production dual-index vector and sparse retrieval system with late chunking and reciprocal rank fusion.',
    deliverables: [
      'Index chunks in Qdrant/pgvector using BGE-Large-en-v1.5 (1024-d) with HNSW (M=16, ef_search=64)',
      'Construct a sparse inverted index using BM25 with tuned parameters (k1=1.5, b=0.75)',
      'Implement Reciprocal Rank Fusion (RRF with constant k=60) combining sparse and dense candidate sets',
      'Integrate a Cross-Encoder reranker (BGE-Reranker-Large) to re-score top-50 candidates down to top-5'
    ],
    evaluationCriteria: 'Achieve NDCG@10 > 0.82 and Recall@5 > 0.88 on BEIR benchmark dataset.'
  },
  {
    id: 3,
    title: 'Phase 3: Adaptive Agentic Retrieval Router',
    duration: 'Week 5–6',
    objective: 'Implement Self-RAG reflection tokens and CRAG confidence grading to dynamically decide whether retrieval is necessary, sufficient, or requires external web search fallback.',
    deliverables: [
      'Train or prompt router for [Retrieve], [No-Retrieve], and [Critique] tokens',
      'Implement CRAG confidence thresholds (High Confidence -> Direct synthesis; Low Confidence -> Web query expansion; Ambiguous -> Hybrid fusion)',
      'Add HyDE (Hypothetical Document Embeddings) generation for ambiguous query clarification',
      'Construct query decomposition module for multi-hop comparative queries'
    ],
    evaluationCriteria: 'Reduce hallucinated answers on ambiguous queries by >40% compared to static naive RAG.'
  },
  {
    id: 4,
    title: 'Phase 4: Autonomous Reasoning & Tool Execution Loop',
    duration: 'Week 7–8',
    objective: 'Develop an agentic ReAct loop with persistent episodic memory, sandboxed Python code execution, and strict halting invariants.',
    deliverables: [
      'Implement stateful agent runner using LangGraph or native StateGraph pattern',
      'Integrate Model Context Protocol (MCP) server exposing tools (arXiv search, SEC EDGAR, Python REPL)',
      'Enforce syntactic stop tokens and a hard step quota (T=10) with cyclic argument detection',
      'Construct Letta/MemGPT-style tiered memory (Working memory buffer + Episodic vector store)'
    ],
    evaluationCriteria: 'Pass 100% of complex multi-hop reasoning tasks with exact numerical precision verified via Python sandbox.'
  },
  {
    id: 5,
    title: 'Phase 5: Security Hardening & Pre-Filtering Guardrails',
    duration: 'Week 9–10',
    objective: 'Harden the architecture against indirect prompt injection, data poisoning, and unauthorized document access.',
    deliverables: [
      'Implement Roaring Bitmap pre-filtering in vector engine enforcing strict user RBAC/ABAC permissions',
      'Deploy dual-LLM input/output sanitizer checking for delimiter hijacking and jailbreak payloads',
      'Containerize Python code execution in ephemeral Firecracker microVMs or Docker gVisor sandboxes',
      'Configure cryptographic HMAC signing on all external tool callbacks'
    ],
    evaluationCriteria: 'Zero document leakage under multi-tenant penetration testing and 100% containment of sandbox escape exploits.'
  },
  {
    id: 6,
    title: 'Phase 6: Continuous CI/CD Evaluation & RAGAS Benchmarking',
    duration: 'Week 11–12',
    objective: 'Establish automated evaluation pipelines measuring the RAG Triad and end-to-end task completion rates before production deployment.',
    deliverables: [
      'Automate RAGAS evaluation pipeline: Faithfulness, Answer Relevance, Context Precision',
      'Implement pairwise LLM-as-a-judge with position-bias swapping and CoT justifications',
      'Instrument OpenTelemetry distributed tracing across all LLM calls, embeddings, and vector queries',
      'Deploy on Kubernetes / Vercel with Redis semantic caching and circuit breaker failover'
    ],
    evaluationCriteria: 'RAGAS Faithfulness > 0.94, Answer Relevance > 0.92, p95 latency < 1.2s under 50 concurrent requests.'
  }
];

export const CapstoneRoadmap: React.FC = () => {
  const [completedPhases, setCompletedPhases] = useState<number[]>(() => {
    try {
      const saved = localStorage.getItem('capstone_completed');
      return saved ? JSON.parse(saved) : [1];
    } catch {
      return [1];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('capstone_completed', JSON.stringify(completedPhases));
    } catch {}
  }, [completedPhases]);

  const togglePhase = (phaseId: number) => {
    setCompletedPhases((prev) =>
      prev.includes(phaseId) ? prev.filter((id) => id !== phaseId) : [...prev, phaseId]
    );
  };

  const progressPercent = Math.round((completedPhases.length / CAPSTONE_PHASES.length) * 100);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Trophy className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">Capstone Project Roadmap</h2>
            <p className="text-sm text-slate-500">
              Autonomous Academic Research Assistant & Synthesis Engine (12-Week Milestone Blueprint).
            </p>
          </div>
        </div>

        {/* Overall Progress Gauge */}
        <div className="flex items-center space-x-3 bg-slate-50 border border-slate-200 px-4 py-2 rounded-xl">
          <div className="text-right">
            <div className="text-xs font-bold text-slate-500 uppercase">Capstone Progress</div>
            <div className="text-sm font-extrabold text-indigo-600 font-mono">
              {completedPhases.length} / {CAPSTONE_PHASES.length} Phases ({progressPercent}%)
            </div>
          </div>
          <div className="w-16 bg-slate-200 h-2.5 rounded-full overflow-hidden">
            <div
              className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Overview Banner */}
      <div className="bg-indigo-50/70 border border-indigo-200 rounded-xl p-5 space-y-2 text-xs text-indigo-950">
        <h3 className="text-sm font-bold text-indigo-950 flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <span>Capstone Goal: Production Multi-Agent Research System</span>
        </h3>
        <p className="text-slate-700 leading-relaxed text-xs">
          This 6-phase capstone synthesizes every core competency taught in this curriculum: layout-aware ingestion, dense/sparse hybrid search, agentic routing, tool sandboxing, security pre-filtering, and continuous RAGAS evaluation into a single enterprise-ready system.
        </p>
      </div>

      {/* Phase Cards */}
      <div className="space-y-6">
        {CAPSTONE_PHASES.map((phase) => {
          const isDone = completedPhases.includes(phase.id);
          return (
            <div
              key={phase.id}
              className={`rounded-xl border-2 transition-all p-6 ${
                isDone
                  ? 'border-emerald-500 bg-emerald-50/20'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-3 mb-3">
                <div className="flex items-center space-x-3">
                  <button
                    onClick={() => togglePhase(phase.id)}
                    className="flex-shrink-0 text-slate-400 hover:text-emerald-600 transition"
                  >
                    {isDone ? (
                      <CheckCircle2 className="w-6 h-6 text-emerald-600" />
                    ) : (
                      <Circle className="w-6 h-6" />
                    )}
                  </button>
                  <div>
                    <h3 className={`text-base font-bold ${isDone ? 'text-emerald-950 line-through' : 'text-slate-900'}`}>
                      {phase.title}
                    </h3>
                    <span className="text-xs font-semibold text-slate-500 font-mono">{phase.duration}</span>
                  </div>
                </div>

                <button
                  onClick={() => togglePhase(phase.id)}
                  className={`text-xs font-bold px-3 py-1.5 rounded-lg border transition ${
                    isDone
                      ? 'bg-emerald-100 border-emerald-300 text-emerald-800'
                      : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                  }`}
                >
                  {isDone ? 'Completed' : 'Mark Complete'}
                </button>
              </div>

              <p className="text-xs text-slate-600 leading-relaxed mb-4">{phase.objective}</p>

              {/* Deliverables checklist */}
              <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-2 mb-3 text-xs">
                <div className="font-bold text-[11px] uppercase tracking-wider text-slate-500">
                  Key Technical Deliverables:
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-slate-700">
                  {phase.deliverables.map((del, idx) => (
                    <div key={idx} className="flex items-start space-x-2">
                      <span className="text-indigo-600 font-bold">•</span>
                      <span>{del}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Evaluation Criteria */}
              <div className="text-xs text-indigo-900 bg-indigo-50/60 p-3 rounded-lg border border-indigo-100 flex items-center space-x-2">
                <BookCheck className="w-4 h-4 text-indigo-600 flex-shrink-0" />
                <span><strong>Acceptance Rubric:</strong> {phase.evaluationCriteria}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
