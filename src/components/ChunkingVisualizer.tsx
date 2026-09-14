import React, { useState } from 'react';
import { Layers, Scissors, Info, Sparkles, CheckCircle2 } from 'lucide-react';

const SAMPLE_TEXT = `Transformers rely on self-attention mechanisms to compute representations of their input without sequence-aligned recurrent units. 
In self-attention, each token computes queries, keys, and values through linear projections. The scaled dot-product attention computes softmax((Q K^T) / sqrt(d_k)) V.
When scaling to long documents, quadratic computational complexity O(N^2) in sequence length N creates substantial memory bottlenecks in the KV cache.
Retrieval-Augmented Generation (RAG) circumvents these memory limits by dynamically fetching only top-k relevant passages from an external vector store at inference time.
However, if chunking boundaries split a premise from its conclusion, the retriever suffers severe context fragmentation, yielding hallucinated answers.`;

type ChunkStrategy = 'fixed' | 'recursive' | 'semantic' | 'contextual' | 'late';

interface ChunkItem {
  id: number;
  text: string;
  tokens: number;
  prefix?: string;
}

export const ChunkingVisualizer: React.FC = () => {
  const [strategy, setStrategy] = useState<ChunkStrategy>('recursive');
  const [text, setText] = useState<string>(SAMPLE_TEXT);

  // Generate chunks based on strategy
  const getChunks = (): ChunkItem[] => {
    if (strategy === 'fixed') {
      // Chunk every ~20 words with 5 words overlap
      const words = text.split(/\s+/);
      const chunkSize = 18;
      const overlap = 5;
      const result: ChunkItem[] = [];
      let i = 0;
      let id = 1;
      while (i < words.length) {
        const slice = words.slice(i, i + chunkSize);
        result.push({
          id,
          text: slice.join(' '),
          tokens: Math.round(slice.length * 1.3),
        });
        id++;
        i += chunkSize - overlap;
        if (i >= words.length - overlap && i < words.length) break;
      }
      return result;
    }

    if (strategy === 'recursive') {
      // Chunk on double newlines then sentence stops
      const paragraphs = text.split(/\n+/).filter(Boolean);
      return paragraphs.map((p, idx): ChunkItem => ({
        id: idx + 1,
        text: p.trim(),
        tokens: Math.round(p.split(/\s+/).length * 1.3),
      }));
    }

    if (strategy === 'semantic') {
      // Group sentences into semantic units
      const sentences = text.split(/(?<=[.?!])\s+/).filter(Boolean);
      return [
        {
          id: 1,
          text: `${sentences[0]} ${sentences[1]}`,
          tokens: 48,
          prefix: 'Semantic Cluster 1 (Attention Fundamentals)',
        },
        {
          id: 2,
          text: sentences[2],
          tokens: 28,
          prefix: 'Semantic Cluster 2 (Complexity Bottleneck)',
        },
        {
          id: 3,
          text: `${sentences[3]} ${sentences[4]}`,
          tokens: 52,
          prefix: 'Semantic Cluster 3 (RAG & Ingestion Fragility)',
        },
      ];
    }

    if (strategy === 'contextual') {
      // Anthropic Contextual Retrieval: prepends global document context
      const paragraphs = text.split(/\n+/).filter(Boolean);
      const globalContext = 'This document covers Transformer self-attention complexity and how RAG mitigates memory constraints.';
      return paragraphs.map((p, idx) => ({
        id: idx + 1,
        text: p.trim(),
        tokens: Math.round(p.split(/\s+/).length * 1.3) + 22,
        prefix: `[Context: ${globalContext}]`,
      }));
    }

    // Late Chunking (Jina AI)
    const sentences = text.split(/(?<=[.?!])\s+/).filter(Boolean);
    return sentences.map((s, idx) => ({
      id: idx + 1,
      text: s.trim(),
      tokens: Math.round(s.split(/\s+/).length * 1.3),
      prefix: 'Global Document Self-Attention Context Preserved',
    }));
  };

  const chunks = getChunks();

  const getStrategyInfo = () => {
    switch (strategy) {
      case 'fixed':
        return {
          title: 'Fixed-Size Token Window (with Stride Overlap)',
          pros: 'Predictable vector dimensions, trivially fast, constant memory allocation.',
          cons: 'Severe boundary fragmentation: splits entities, formulas, and clauses midway.',
          recommendation: 'Baseline benchmarking only; avoid for dense technical or legal documents.',
        };
      case 'recursive':
        return {
          title: 'Recursive Character Chunking (LangChain / LlamaIndex Default)',
          pros: 'Respects natural document structure (paragraphs, markdown headings, code blocks).',
          cons: 'Chunk lengths vary widely; requires careful tuning of min/max chunk sizes.',
          recommendation: 'Recommended default for general technical documentation and articles.',
        };
      case 'semantic':
        return {
          title: 'Semantic Embedding Distance Chunking (Kamradt 2023)',
          pros: 'Splits only where the cosine distance between adjacent sentences exceeds a percentile threshold (e.g. 95th).',
          cons: 'Requires 2x-3x more embedding API calls; sensitive to distance thresholds.',
          recommendation: 'Optimal for multi-topic documents with abrupt transitions.',
        };
      case 'contextual':
        return {
          title: 'Anthropic Contextual Retrieval (September 2024)',
          pros: 'Reduces retrieval failure by 49% by prompting a fast LLM to inject a 50-token global explanatory prefix into each chunk.',
          cons: 'Upfront ingestion cost increases by ~$1.02 per million input tokens.',
          recommendation: 'Gold standard for enterprise production knowledge bases.',
        };
      case 'late':
        return {
          title: 'Jina AI Late Chunking (Late Pooling over Full Attention)',
          pros: 'The transformer encodes the entire 8,192-token document first; token vectors retain global contextual attention before span pooling.',
          cons: 'Requires embedding models supporting late chunking (e.g., jina-embeddings-v3).',
          recommendation: 'State of the art for long, interconnected research papers.',
        };
    }
  };

  const info = getStrategyInfo();

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Scissors className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">Chunking Strategy & Boundary Visualizer</h2>
            <p className="text-sm text-slate-500">
              Compare segmentation behaviors, token distributions, and retrieval context preservation across 5 paradigms.
            </p>
          </div>
        </div>

        {/* Strategy Selector Tabs */}
        <div className="flex flex-wrap gap-1.5 bg-slate-100 p-1.5 rounded-xl">
          {(['fixed', 'recursive', 'semantic', 'contextual', 'late'] as ChunkStrategy[]).map((st) => (
            <button
              key={st}
              onClick={() => setStrategy(st)}
              className={`text-xs font-bold px-3 py-1.5 rounded-lg transition capitalize ${
                strategy === st
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Strategy Summary Banner */}
      <div className="bg-indigo-50/70 border border-indigo-200 rounded-xl p-5 space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <h4 className="font-bold text-indigo-950 text-sm">{info.title}</h4>
          <span className="font-mono text-indigo-700 bg-indigo-100 px-2 py-0.5 rounded text-[11px]">
            {chunks.length} Chunks Generated
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-700 pt-1">
          <div><strong className="text-emerald-700">Advantages:</strong> {info.pros}</div>
          <div><strong className="text-amber-700">Tradeoffs:</strong> {info.cons}</div>
        </div>
        <div className="pt-2 text-indigo-900">
          <strong>Production Rule of Thumb:</strong> {info.recommendation}
        </div>
      </div>

      {/* Visual Chunk Output Cards */}
      <div className="space-y-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Generated Ingestion Payloads & Context Boundaries
        </h3>

        <div className="grid grid-cols-1 gap-4">
          {chunks.map((chunk, idx) => {
            const colors = [
              'border-blue-200 bg-blue-50/40 text-blue-950',
              'border-emerald-200 bg-emerald-50/40 text-emerald-950',
              'border-amber-200 bg-amber-50/40 text-amber-950',
              'border-purple-200 bg-purple-50/40 text-purple-950',
              'border-rose-200 bg-rose-50/40 text-rose-950',
            ];
            const colorClass = colors[idx % colors.length];

            return (
              <div
                key={chunk.id}
                className={`p-4 rounded-xl border-2 transition-all ${colorClass}`}
              >
                <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-200/60 text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-800 shadow-2xs">
                      Chunk #{chunk.id}
                    </span>
                    {chunk.prefix && (
                      <span className="font-mono text-[11px] text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">
                        {chunk.prefix}
                      </span>
                    )}
                  </div>
                  <span className="font-mono font-semibold text-slate-500">
                    ~{chunk.tokens} tokens
                  </span>
                </div>

                <p className="text-sm font-mono leading-relaxed text-slate-800">
                  {chunk.text}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
