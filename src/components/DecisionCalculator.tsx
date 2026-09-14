import React, { useState, useMemo } from 'react';
import { Sliders, CheckCircle2, AlertTriangle, ArrowRight, Zap, DollarSign, Clock, Cpu } from 'lucide-react';

export const DecisionCalculator: React.FC = () => {
  const [updateFreq, setUpdateFreq] = useState<number>(3); // 1: Static -> 5: Real-time/Seconds
  const [corpusSize, setCorpusSize] = useState<number>(3); // 1: <100 pages -> 5: 10M+ documents
  const [reasoningComplexity, setReasoningComplexity] = useState<number>(3); // 1: Factoid lookup -> 5: Multi-step tool use
  const [budgetLimit, setBudgetLimit] = useState<number>(2); // 1: Very tight (<$100/mo) -> 4: Enterprise (> $5k/mo)
  const [latencyTolerance, setLatencyTolerance] = useState<number>(2); // 1: Real-time (<500ms) -> 4: Async/Minutes acceptable

  // Calculate scores (0 to 100) for each architecture
  const scores = useMemo(() => {
    // Naive / Advanced RAG: high score when data updates frequently, corpus is medium to large, latency is interactive, budget is reasonable
    let ragScore = 40 + updateFreq * 10 + corpusSize * 6 - (reasoningComplexity > 4 ? 15 : 0) - (latencyTolerance === 1 ? 5 : 0);
    
    // Fine-Tuning: high score when domain vocabulary / style is unique, corpus is static/slow-moving, low latency needed, budget allows training
    let ftScore = 20 + (6 - updateFreq) * 12 - corpusSize * 5 - reasoningComplexity * 4 + (latencyTolerance === 1 ? 25 : 0) + (budgetLimit >= 3 ? 15 : -15);
    
    // Long Context: high score when corpus is compact (< 200k tokens), updates don't matter (just feed in prompt), budget allows high token costs, latency allows TTFT
    let lcScore = 70 - corpusSize * 15 + (latencyTolerance >= 3 ? 15 : -10) - (budgetLimit <= 2 ? 15 : -5);
    
    // Agentic RAG / Multi-Agent: high score when reasoning complexity is high, tool access is required, latency tolerance is relaxed
    let agentScore = 15 + reasoningComplexity * 16 + (latencyTolerance >= 3 ? 20 : -20) + (budgetLimit >= 3 ? 10 : -10);

    // Normalize
    ragScore = Math.max(10, Math.min(98, ragScore));
    ftScore = Math.max(5, Math.min(95, ftScore));
    lcScore = Math.max(5, Math.min(92, lcScore));
    agentScore = Math.max(5, Math.min(99, agentScore));

    return {
      rag: Math.round(ragScore),
      fineTuning: Math.round(ftScore),
      longContext: Math.round(lcScore),
      agents: Math.round(agentScore),
    };
  }, [updateFreq, corpusSize, reasoningComplexity, budgetLimit, latencyTolerance]);

  // Determine top recommendation
  const recommendation = useMemo(() => {
    const entries = [
      { name: 'Modular / Advanced RAG', score: scores.rag, id: 'rag' },
      { name: 'Autonomous Agentic RAG', score: scores.agents, id: 'agents' },
      { name: 'Model Fine-Tuning (LoRA / SFT)', score: scores.fineTuning, id: 'ft' },
      { name: 'Long-Context Direct Prompting', score: scores.longContext, id: 'lc' },
    ].sort((a, b) => b.score - a.score);

    const top = entries[0];
    let rationale = '';
    let stack = '';
    let estimatedCost = '';
    let estimatedLatency = '';

    if (top.id === 'rag') {
      rationale = 'Your requirements favor fast-changing external knowledge and medium-to-large retrieval corpora with interactive latency. Fine-tuning cannot keep up with continuous document sync without catastrophic forgetting.';
      stack = 'Hybrid Search (BM25 + BGE-Large-en-v1.5) + Qdrant/pgvector + Cohere-Rerank-v3 + Claude 3.5 Sonnet';
      estimatedCost = '$0.002 - $0.006 per query';
      estimatedLatency = '350ms - 800ms p95';
    } else if (top.id === 'agents') {
      rationale = 'High reasoning complexity and multi-step workflows require dynamic tool invocation, state graphs, and self-reflection loops. A standard single-shot retrieval pipeline will produce hallucinated synthesis.';
      stack = 'LangGraph / Native StateGraph + MCP Tools + Hybrid RAG Tool + Tool Call Reflection Invariants';
      estimatedCost = '$0.03 - $0.15 per completed task';
      estimatedLatency = '3.5s - 12.0s p95';
    } else if (top.id === 'ft') {
      rationale = 'Static domain terminology, specialized formatting, or extreme low latency requirements make parameter adaptation via SFT or DPO optimal over constant context injection.';
      stack = 'Llama 3.3 70B / Qwen 2.5 32B + Unsloth / Axolotl (QLoRA r=16) + vLLM serving on FP8';
      estimatedCost = '$500 one-time train + $0.0008 per inference';
      estimatedLatency = '60ms - 150ms p95';
    } else {
      rationale = 'Your corpus fits well within the 200k - 1M token context window, query volume is modest, and zero vector indexing overhead is preferred over complex retrieval pipelines.';
      stack = 'Gemini 1.5 Pro (2M context) or Claude 3.5 Sonnet (200k context) with Prompt Caching';
      estimatedCost = '$0.015 - $0.05 per query (cached: $0.004)';
      estimatedLatency = '1.8s - 4.5s TTFT';
    }

    return {
      name: top.name,
      score: top.score,
      rationale,
      stack,
      estimatedCost,
      estimatedLatency,
    };
  }, [scores]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      <div className="border-b border-slate-100 pb-5">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Sliders className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">System Architecture Decision Calculator</h2>
            <p className="text-sm text-slate-500">
              Tune your operational constraints to evaluate tradeoffs between RAG, Fine-Tuning, Long Context, and Agents.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Sliders Column */}
        <div className="lg:col-span-6 space-y-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Operational Constraints</h3>

          {/* Slider 1 */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium text-slate-700">Knowledge Update Frequency</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                {updateFreq === 1 && 'Static / Infrequent'}
                {updateFreq === 2 && 'Weekly / Monthly'}
                {updateFreq === 3 && 'Daily Ingestion'}
                {updateFreq === 4 && 'Hourly Sync'}
                {updateFreq === 5 && 'Real-time (Seconds/Streaming)'}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              value={updateFreq}
              onChange={(e) => setUpdateFreq(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Static</span>
              <span>Real-time</span>
            </div>
          </div>

          {/* Slider 2 */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium text-slate-700">Knowledge Corpus Size</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                {corpusSize === 1 && '< 100 Pages (< 50k tokens)'}
                {corpusSize === 2 && '1,000 Pages (100k - 500k tokens)'}
                {corpusSize === 3 && '50,000 Docs (10M tokens)'}
                {corpusSize === 4 && '1M+ Docs (Vector Scale)'}
                {corpusSize === 5 && '10M+ Enterprise Corpus'}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              value={corpusSize}
              onChange={(e) => setCorpusSize(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Single Document</span>
              <span>Massive Enterprise</span>
            </div>
          </div>

          {/* Slider 3 */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium text-slate-700">Reasoning & Tool Complexity</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                {reasoningComplexity === 1 && 'Simple Factoid / Direct Q&A'}
                {reasoningComplexity === 2 && 'Semantic Synthesis (2-3 chunks)'}
                {reasoningComplexity === 3 && 'Query Decomposition / Comparative'}
                {reasoningComplexity === 4 && 'Tool Calling (SQL + Web + APIs)'}
                {reasoningComplexity === 5 && 'Autonomous Multi-step Planning'}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              value={reasoningComplexity}
              onChange={(e) => setReasoningComplexity(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Direct Lookup</span>
              <span>Multi-step Autonomous</span>
            </div>
          </div>

          {/* Slider 4 */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium text-slate-700">Budget / Cost Sensitivity</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                {budgetLimit === 1 && 'Extreme Lean (< $100/mo)'}
                {budgetLimit === 2 && 'Moderate ($500 - $1,500/mo)'}
                {budgetLimit === 3 && 'Growth ($2,000 - $10,000/mo)'}
                {budgetLimit === 4 && 'Enterprise ($10k+/mo)'}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="4"
              value={budgetLimit}
              onChange={(e) => setBudgetLimit(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Budget Conscious</span>
              <span>Performance Driven</span>
            </div>
          </div>

          {/* Slider 5 */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="font-medium text-slate-700">Latency Tolerance</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
                {latencyTolerance === 1 && 'Hard Real-Time (< 400ms)'}
                {latencyTolerance === 2 && 'Interactive UI (800ms - 2s)'}
                {latencyTolerance === 3 && 'Background Analysis (5s - 15s)'}
                {latencyTolerance === 4 && 'Asynchronous Batch (Minutes)'}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="4"
              value={latencyTolerance}
              onChange={(e) => setLatencyTolerance(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>Sub-second</span>
              <span>Batch Mode</span>
            </div>
          </div>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-6 space-y-6">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Architectural Fit Scores</h3>

          {/* Score Bars */}
          <div className="space-y-3 bg-slate-50 p-4 rounded-xl border border-slate-200">
            <div>
              <div className="flex justify-between text-sm font-semibold text-slate-800 mb-1">
                <span>Modular / Advanced RAG</span>
                <span className="text-indigo-600">{scores.rag}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                <div className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${scores.rag}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm font-semibold text-slate-800 mb-1">
                <span>Autonomous Agentic RAG</span>
                <span className="text-emerald-600">{scores.agents}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                <div className="bg-emerald-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${scores.agents}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm font-semibold text-slate-800 mb-1">
                <span>Model Fine-Tuning (SFT/LoRA)</span>
                <span className="text-amber-600">{scores.fineTuning}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                <div className="bg-amber-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${scores.fineTuning}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm font-semibold text-slate-800 mb-1">
                <span>Long-Context Direct Prompting</span>
                <span className="text-purple-600">{scores.longContext}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden">
                <div className="bg-purple-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${scores.longContext}%` }}></div>
              </div>
            </div>
          </div>

          {/* Primary Recommendation Card */}
          <div className="bg-indigo-50/70 border border-indigo-200 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-700 bg-indigo-100 px-2.5 py-1 rounded-full">
                Primary Recommendation
              </span>
              <span className="text-xl font-extrabold text-indigo-900">{recommendation.score}% Match</span>
            </div>
            <h4 className="text-lg font-bold text-slate-900">{recommendation.name}</h4>
            <p className="text-sm text-slate-700 leading-relaxed">{recommendation.rationale}</p>
            
            <div className="pt-2 border-t border-indigo-200/60 grid grid-cols-2 gap-3 text-xs">
              <div className="flex items-center space-x-1.5 text-slate-700">
                <DollarSign className="w-4 h-4 text-emerald-600" />
                <span><strong>Est. Cost:</strong> {recommendation.estimatedCost}</span>
              </div>
              <div className="flex items-center space-x-1.5 text-slate-700">
                <Clock className="w-4 h-4 text-amber-600" />
                <span><strong>Est. Latency:</strong> {recommendation.estimatedLatency}</span>
              </div>
            </div>

            <div className="pt-2 text-xs text-slate-600">
              <span className="font-semibold text-slate-800">Recommended Reference Stack:</span>
              <div className="mt-1 font-mono text-[11px] bg-white p-2 rounded border border-indigo-100 text-indigo-950">
                {recommendation.stack}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
