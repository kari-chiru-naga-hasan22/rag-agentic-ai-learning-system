import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Bot, Terminal, CheckCircle2, AlertOctagon, Sparkles } from 'lucide-react';

interface Step {
  stepIndex: number;
  phase: 'THOUGHT' | 'ACTION' | 'OBSERVATION' | 'REFLECTION' | 'FINAL_ANSWER';
  content: string;
  toolCall?: {
    name: string;
    args: Record<string, any>;
    result: Record<string, any> | string;
  };
  tokenCost: number;
  cumulativeReliability: number;
}

interface Scenario {
  id: string;
  title: string;
  userQuery: string;
  steps: Step[];
}

const SCENARIOS: Scenario[] = [
  {
    id: 'finance',
    title: 'Financial SEC 10-K Comparison',
    userQuery: 'Calculate the year-over-year growth rate of AWS revenue compared to Microsoft Azure for fiscal year 2024.',
    steps: [
      {
        stepIndex: 1,
        phase: 'THOUGHT',
        content: 'I need to find verified revenue figures for Amazon AWS and Microsoft Cloud/Azure for FY2024. I will first query the SEC EDGAR 10-K database for Amazon FY2024 segment reporting.',
        tokenCost: 180,
        cumulativeReliability: 0.98,
      },
      {
        stepIndex: 2,
        phase: 'ACTION',
        content: 'Calling tool `sec_edgar_search` with ticker AMZN and filing type 10-K.',
        toolCall: {
          name: 'sec_edgar_search',
          args: { ticker: 'AMZN', form: '10-K', fiscal_year: 2024, section: 'Segment Reporting' },
          result: {
            ticker: 'AMZN',
            segment: 'AWS',
            fy2023_revenue_usd_billions: 90.76,
            fy2024_revenue_usd_billions: 107.56,
            status: 'Filed Feb 2025'
          }
        },
        tokenCost: 320,
        cumulativeReliability: 0.96,
      },
      {
        stepIndex: 3,
        phase: 'OBSERVATION',
        content: 'Amazon AWS generated $107.56B in FY24 compared to $90.76B in FY23. Next, I need Microsoft Azure revenue figures. Microsoft combines Azure inside Intelligent Cloud, so I must query MSFT 10-K footnote disclosures.',
        tokenCost: 490,
        cumulativeReliability: 0.94,
      },
      {
        stepIndex: 4,
        phase: 'ACTION',
        content: 'Calling tool `sec_edgar_search` with ticker MSFT for Intelligent Cloud segment and Azure percentage growth.',
        toolCall: {
          name: 'sec_edgar_search',
          args: { ticker: 'MSFT', form: '10-K', fiscal_year: 2024, section: 'Intelligent Cloud' },
          result: {
            ticker: 'MSFT',
            intelligent_cloud_revenue_usd_billions: 105.36,
            azure_growth_rate_constant_currency: '33% YoY',
            estimated_azure_cloud_portion_billions: 75.20
          }
        },
        tokenCost: 650,
        cumulativeReliability: 0.92,
      },
      {
        stepIndex: 5,
        phase: 'THOUGHT',
        content: 'Now I must calculate the exact AWS percentage growth: ((107.56 - 90.76) / 90.76) * 100. I will execute this via the sandboxed Python calculator to avoid floating-point math hallucinations.',
        tokenCost: 780,
        cumulativeReliability: 0.91,
      },
      {
        stepIndex: 6,
        phase: 'ACTION',
        content: 'Calling tool `python_eval` for verified arithmetic.',
        toolCall: {
          name: 'python_eval',
          args: { code: 'aws_diff = 107.56 - 90.76; growth = (aws_diff / 90.76) * 100; print(f"{growth:.2f}%")' },
          result: { stdout: '18.51%', returncode: 0 }
        },
        tokenCost: 910,
        cumulativeReliability: 0.90,
      },
      {
        stepIndex: 7,
        phase: 'REFLECTION',
        content: 'All required parameters obtained: AWS grew 18.51% YoY to $107.56B, while Azure grew 33% YoY in constant currency to ~$75.2B. Both numbers are verified against official SEC 10-K filings. Termination condition met.',
        tokenCost: 1040,
        cumulativeReliability: 0.89,
      },
      {
        stepIndex: 8,
        phase: 'FINAL_ANSWER',
        content: 'For Fiscal Year 2024:\n1. **Amazon AWS Revenue:** Grew from $90.76B to $107.56B, representing a **+18.51% YoY growth rate**.\n2. **Microsoft Azure:** Reported **+33% YoY constant currency growth** (estimated base revenue reached ~$75.2B within the $105.36B Intelligent Cloud segment).\n\n**Conclusion:** While AWS maintains larger absolute revenue ($107.56B vs ~$75.2B), Azure exhibited significantly higher relative YoY growth (33% vs 18.51%).',
        tokenCost: 1220,
        cumulativeReliability: 0.88,
      }
    ]
  },
  {
    id: 'devops',
    title: 'Site Reliability Engineering Incident Triage',
    userQuery: 'Diagnose why production API p95 latency spiked to 4,200ms at 14:02 UTC.',
    steps: [
      {
        stepIndex: 1,
        phase: 'THOUGHT',
        content: 'A sudden p95 latency jump indicates downstream bottlenecking (database lock, third-party timeout, or connection pool exhaustion). I will query Prometheus metrics at 14:02 UTC.',
        tokenCost: 160,
        cumulativeReliability: 0.98,
      },
      {
        stepIndex: 2,
        phase: 'ACTION',
        content: 'Calling tool `query_prometheus` for database connection pool wait durations.',
        toolCall: {
          name: 'query_prometheus',
          args: { query: 'rate(pg_stat_activity_pool_wait_seconds[5m])', time: '14:02:00Z' },
          result: { metric_value: 3.84, alert_state: 'CRITICAL', active_connections: 100, max_connections: 100 }
        },
        tokenCost: 340,
        cumulativeReliability: 0.96,
      },
      {
        stepIndex: 3,
        phase: 'OBSERVATION',
        content: 'PostgreSQL connection pool hit 100% saturation (100/100 connections). Queries were queueing for 3.84s before obtaining a connection. Now I must check Postgres pg_stat_activity to identify the blocking transaction.',
        tokenCost: 520,
        cumulativeReliability: 0.94,
      },
      {
        stepIndex: 4,
        phase: 'ACTION',
        content: 'Calling tool `postgres_query` for long-running unindexed queries and table locks.',
        toolCall: {
          name: 'postgres_query',
          args: { sql: 'SELECT pid, query, state, age(clock_timestamp(), query_start) FROM pg_stat_activity WHERE state != \'idle\' ORDER BY query_start ASC LIMIT 3;' },
          result: {
            pid: 48192,
            query: 'ALTER TABLE customer_events ADD COLUMN metadata jsonb DEFAULT \'{}\';',
            duration: '14 minutes',
            lock_type: 'AccessExclusiveLock'
          }
        },
        tokenCost: 710,
        cumulativeReliability: 0.93,
      },
      {
        stepIndex: 5,
        phase: 'REFLECTION',
        content: 'Root cause identified: An ALTER TABLE migration deployed at 14:00 UTC acquired an AccessExclusiveLock on customer_events table, blocking all concurrent API reads and exhausting the HikariCP connection pool.',
        tokenCost: 880,
        cumulativeReliability: 0.92,
      },
      {
        stepIndex: 6,
        phase: 'FINAL_ANSWER',
        content: '### Incident Root Cause Analysis (14:02 UTC Latency Spike)\n* **Primary Fault:** Migration job PID 48192 executed `ALTER TABLE customer_events ADD COLUMN metadata jsonb;` which acquired an `AccessExclusiveLock`.\n* **Secondary Failure:** All incoming API reads queued behind the exclusive lock, completely exhausting the database connection pool (100/100 active connections; wait duration: 3.84s).\n* **Immediate Remediation:** Terminate blocking migration `SELECT pg_cancel_backend(48192);` and rerun migration asynchronously using PostgreSQL concurrent DDL patterns without default value rewrite.',
        tokenCost: 1100,
        cumulativeReliability: 0.91,
      }
    ]
  }
];

export const ReActSimulator: React.FC = () => {
  const [selectedScenarioIndex, setSelectedScenarioIndex] = useState<number>(0);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  const scenario = SCENARIOS[selectedScenarioIndex];
  const currentStep = scenario.steps[currentStepIndex];

  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setTimeout(() => {
        if (currentStepIndex < scenario.steps.length - 1) {
          setCurrentStepIndex((prev) => prev + 1);
        } else {
          setIsPlaying(false);
        }
      }, 1600);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStepIndex, scenario.steps.length]);

  const handleNext = () => {
    if (currentStepIndex < scenario.steps.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex((prev) => prev - 1);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStepIndex(0);
  };

  const getPhaseBadge = (phase: Step['phase']) => {
    switch (phase) {
      case 'THOUGHT':
        return <span className="bg-purple-100 text-purple-800 text-xs font-bold px-2.5 py-1 rounded-full">THOUGHT (Internal Reasoning)</span>;
      case 'ACTION':
        return <span className="bg-blue-100 text-blue-800 text-xs font-bold px-2.5 py-1 rounded-full">ACTION (Tool Call)</span>;
      case 'OBSERVATION':
        return <span className="bg-amber-100 text-amber-800 text-xs font-bold px-2.5 py-1 rounded-full">OBSERVATION (Environment Feedback)</span>;
      case 'REFLECTION':
        return <span className="bg-indigo-100 text-indigo-800 text-xs font-bold px-2.5 py-1 rounded-full">REFLECTION (Self-Audit)</span>;
      case 'FINAL_ANSWER':
        return <span className="bg-emerald-100 text-emerald-800 text-xs font-bold px-2.5 py-1 rounded-full">FINAL ANSWER (Terminated)</span>;
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">ReAct Reasoning & Tool Execution Simulator</h2>
            <p className="text-sm text-slate-500">
              Interactive execution trace demonstrating Yao et al. (2022) interleaved Thought-Action-Observation cycles.
            </p>
          </div>
        </div>

        {/* Scenario Toggle */}
        <div className="flex items-center space-x-2">
          {SCENARIOS.map((sc, idx) => (
            <button
              key={sc.id}
              onClick={() => {
                setSelectedScenarioIndex(idx);
                setCurrentStepIndex(0);
                setIsPlaying(false);
              }}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition ${
                selectedScenarioIndex === idx
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm'
                  : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
              }`}
            >
              {sc.title}
            </button>
          ))}
        </div>
      </div>

      {/* User Prompt Banner */}
      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">User Instruction</span>
        <p className="text-sm font-medium text-slate-800 font-mono">"{scenario.userQuery}"</p>
      </div>

      {/* Stepper Controls & System Metrics */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-50/80 p-4 rounded-xl border border-slate-200">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg shadow-sm transition"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isPlaying ? 'Pause' : 'Auto Play'}</span>
          </button>
          <button
            onClick={handlePrev}
            disabled={currentStepIndex === 0}
            className="p-1.5 rounded-lg border border-slate-300 bg-white text-slate-700 disabled:opacity-40 hover:bg-slate-100 transition"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-xs font-semibold text-slate-600 px-2">
            Step {currentStepIndex + 1} of {scenario.steps.length}
          </span>
          <button
            onClick={handleNext}
            disabled={currentStepIndex === scenario.steps.length - 1}
            className="p-1.5 rounded-lg border border-slate-300 bg-white text-slate-700 disabled:opacity-40 hover:bg-slate-100 transition"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 rounded-lg border border-slate-300 bg-white text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition"
            title="Reset to Step 1"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Compounding Reliability & Cost Metric */}
        <div className="flex items-center space-x-6 text-xs">
          <div>
            <span className="text-slate-500">Context Tokens:</span>{' '}
            <strong className="text-slate-800 font-mono">{currentStep.tokenCost} tokens</strong>
          </div>
          <div>
            <span className="text-slate-500">Compounding Reliability (∏ p_i):</span>{' '}
            <strong className={`font-mono ${currentStep.cumulativeReliability > 0.9 ? 'text-emerald-600' : 'text-amber-600'}`}>
              {(currentStep.cumulativeReliability * 100).toFixed(1)}%
            </strong>
          </div>
        </div>
      </div>

      {/* Main Execution Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Current Step Card */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-5 bg-white border-2 border-indigo-100 rounded-xl shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              {getPhaseBadge(currentStep.phase)}
              <span className="text-xs font-mono text-slate-400">Step #{currentStep.stepIndex}</span>
            </div>

            <p className="text-sm text-slate-800 whitespace-pre-line leading-relaxed">
              {currentStep.content}
            </p>

            {/* Tool payload if ACTION */}
            {currentStep.toolCall && (
              <div className="mt-4 pt-3 border-t border-slate-100 space-y-2">
                <div className="flex items-center space-x-1.5 text-xs font-bold text-slate-700">
                  <Terminal className="w-4 h-4 text-indigo-600" />
                  <span>Tool Invocation: <code className="text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">{currentStep.toolCall.name}</code></span>
                </div>
                <div className="bg-slate-900 text-slate-100 p-3 rounded-lg font-mono text-xs overflow-x-auto">
                  <div className="text-slate-400 text-[10px] uppercase font-semibold mb-1">// Arguments:</div>
                  <pre className="text-emerald-400">{JSON.stringify(currentStep.toolCall.args, null, 2)}</pre>
                  <div className="text-slate-400 text-[10px] uppercase font-semibold mt-2 mb-1">// Observation Returned:</div>
                  <pre className="text-amber-300">{JSON.stringify(currentStep.toolCall.result, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>

          {/* Stepper Timeline bar */}
          <div className="flex items-center space-x-1 pt-2">
            {scenario.steps.map((st, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentStepIndex(idx)}
                className={`h-2 flex-1 rounded-full transition-all ${
                  idx === currentStepIndex
                    ? 'bg-indigo-600 ring-2 ring-indigo-300'
                    : idx < currentStepIndex
                    ? 'bg-indigo-200'
                    : 'bg-slate-200'
                }`}
                title={`Step ${idx + 1}: ${st.phase}`}
              />
            ))}
          </div>
        </div>

        {/* Right: POMDP Formalism & Agent Invariants */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 space-y-3 text-xs">
            <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              <span>POMDP Mathematical State</span>
            </h4>
            <div className="font-mono space-y-1 text-slate-700 bg-white p-3 rounded border border-slate-200">
              <div>S_t = Environment Ground Truth</div>
              <div>O_t = Tool Return Output</div>
              <div>H_t = [q, t_1, a_1, o_1, ..., t_k]</div>
              <div>a_(t+1) ~ π(a | H_t, θ_LLM)</div>
            </div>
            <p className="text-slate-600 leading-relaxed">
              Unlike a pure Markov Decision Process (MDP) where full system state is visible, LLM agents operate under <strong>Partial Observability (POMDP)</strong>. The agent constructs beliefs solely from its historical context buffer $H_t$.
            </p>
          </div>

          <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-5 space-y-2 text-xs text-emerald-950">
            <div className="font-bold flex items-center space-x-1.5 text-emerald-900">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Production Halting Invariant</span>
            </div>
            <p className="leading-relaxed">
              To prevent infinite agent looping, production systems enforce three strict bounds:
              <br />1. <strong>Hard Step Quota:</strong> Maximum $T=10$ iterations before forced termination.
              <br />2. <strong>Syntactic Stop Token:</strong> Halting triggered only when the model outputs `FINAL_ANSWER:`.
              <br />3. <strong>Cycle Detection:</strong> SHA-256 hashing of identical tool arguments across consecutive steps.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
