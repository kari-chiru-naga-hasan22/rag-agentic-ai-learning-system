import React, { useState, useMemo } from 'react';
import { Database, Filter, Check, X, Shield, Sparkles, ExternalLink } from 'lucide-react';

interface VectorDb {
  id: string;
  name: string;
  category: 'Library' | 'SQL Extension' | 'Standalone' | 'Serverless Cloud' | 'Search Engine';
  primaryLanguage: string;
  license: string;
  indices: string[];
  filteringStrategy: string;
  scaleSweetSpot: string;
  ramFootprint: string;
  strengths: string[];
  weaknesses: string[];
  verdict: string;
}

const VECTOR_DBS: VectorDb[] = [
  {
    id: 'qdrant',
    name: 'Qdrant',
    category: 'Standalone',
    primaryLanguage: 'Rust',
    license: 'Apache 2.0',
    indices: ['HNSW', 'Scalar Quantization', 'Product Quantization'],
    filteringStrategy: 'In-engine payload pre-filtering with roaring bitmaps during HNSW graph traversal',
    scaleSweetSpot: '1M – 100M+ vectors with rich metadata attributes',
    ramFootprint: 'Moderate (mmap payload storage allows zero-copy disk caching)',
    strengths: [
      'Fastest filtered vector search in BEIR benchmarks',
      'Native geo, text, and numerical payload indexing',
      'Built-in support for sparse vectors (BM25/SPLADE)',
      'Rust memory safety and deterministic p99 latency'
    ],
    weaknesses: [
      'Smaller third-party plugin ecosystem compared to PostgreSQL',
      'Clustering requires Raft consensus setup for self-hosted high availability'
    ],
    verdict: 'The top recommendation for production AI applications requiring hybrid dense+sparse retrieval with strict low-latency metadata filtering.'
  },
  {
    id: 'pgvector',
    name: 'pgvector (PostgreSQL)',
    category: 'SQL Extension',
    primaryLanguage: 'C / PostgreSQL',
    license: 'PostgreSQL License (Open Source)',
    indices: ['HNSW', 'IVFFlat'],
    filteringStrategy: 'Relational planner pre-filtering / index scans (improves greatly with iterative HNSW in v0.7+)',
    scaleSweetSpot: '< 10M vectors alongside existing transactional relational data',
    ramFootprint: 'High (requires generous shared_buffers and work_mem for HNSW construction)',
    strengths: [
      'ACID transactions and foreign keys linking vectors directly to business tables',
      'Zero new infrastructure if your organization already operates PostgreSQL',
      'Full SQL ecosystem: joins, row-level security (RLS), triggers, and backups'
    ],
    weaknesses: [
      'Index build times on > 5M vectors are significantly slower than dedicated C++/Rust engines',
      'Vacuuming overhead and RAM saturation can affect concurrent OLTP workloads'
    ],
    verdict: 'Best choice for existing Postgres deployments with fewer than 5-10 million vectors. Do not introduce a separate vector database until this reaches scaling limits.'
  },
  {
    id: 'milvus',
    name: 'Milvus',
    category: 'Standalone',
    primaryLanguage: 'Go (Coordination) / C++ (Knowhere Core)',
    license: 'Apache 2.0',
    indices: ['HNSW', 'IVF-FLAT', 'IVF-PQ', 'DiskANN', 'SCaNN', 'GPU-IVF'],
    filteringStrategy: 'Boolean expression parser compiled to Bitset mask before graph exploration',
    scaleSweetSpot: '50M – Billions of vectors across distributed Kubernetes clusters',
    ramFootprint: 'Configurable (DiskANN and Knowhere support offloading vectors to NVMe SSD)',
    strengths: [
      'True distributed architecture separating query nodes, index nodes, and data coordinators',
      'First-class GPU index acceleration (NVIDIA RAFT / Knowhere)',
      'Scales to multi-billion vector datasets'
    ],
    weaknesses: [
      'High operational complexity: requires Kubernetes, MinIO/S3, Pulsar/Kafka, and etcd',
      'Overkill for small to medium projects'
    ],
    verdict: 'The enterprise standard for ultra-high-throughput, multi-million to multi-billion vector architectures managed by dedicated DevOps/platform teams.'
  },
  {
    id: 'pinecone',
    name: 'Pinecone',
    category: 'Serverless Cloud',
    primaryLanguage: 'Proprietary (Rust/C++ Core)',
    license: 'Proprietary SaaS',
    indices: ['Custom HNSW / Graph-based', 'Serverless Blob-backed'],
    filteringStrategy: 'Managed serverless metadata filter matching',
    scaleSweetSpot: 'Zero-ops managed deployment from prototype to enterprise scale',
    ramFootprint: 'Zero local overhead (fully managed cloud abstraction)',
    strengths: [
      'Zero infrastructure management; instant provisioning and autoscaling',
      'Pinecone Serverless architecture decouples storage from compute, slashing idle costs',
      'Simple client SDKs with built-in reranking partnerships'
    ],
    weaknesses: [
      'Closed source, vendor lock-in, and unpredictable egress/query cost spikes at high sustained QPS',
      'Cannot be deployed on-premises or inside isolated air-gapped VPCs'
    ],
    verdict: 'Optimal for startups and teams prioritizing rapid time-to-market with zero DevOps maintenance overhead.'
  },
  {
    id: 'weaviate',
    name: 'Weaviate',
    category: 'Standalone',
    primaryLanguage: 'Go',
    license: 'BSD-3-Clause',
    indices: ['HNSW', 'Flat', 'Dynamic Vector Indexing', 'Product Quantization'],
    filteringStrategy: 'Inverted index payload filtering integrated with HNSW search',
    scaleSweetSpot: '1M – 50M vectors with multimodal objects (images, text, video)',
    ramFootprint: 'Moderate to high (dynamic cache compression available)',
    strengths: [
      'GraphQL and REST APIs with native schema modeling and semantic vectors',
      'Native modules for automated vectorization (OpenAI, Cohere, HuggingFace)',
      'Strong hybrid search combining BM25 and vector search out of the box'
    ],
    weaknesses: [
      'Go garbage collection can introduce occasional p99 latency spikes under heavy concurrent write loads',
      'Complex configuration flags for fine-tuning memory thresholds'
    ],
    verdict: 'Excellent for multimodal applications and teams that value built-in inference vectorizers and GraphQL object schemas.'
  },
  {
    id: 'faiss',
    name: 'FAISS (Meta AI)',
    category: 'Library',
    primaryLanguage: 'C++ with Python bindings',
    license: 'MIT License',
    indices: ['IndexFlatL2', 'IndexIVFFlat', 'IndexIVFPQ', 'IndexHNSWFlat'],
    filteringStrategy: 'IDSelector post-filtering or pre-filtering via manual partitioning',
    scaleSweetSpot: 'Offline batch similarity, research experimentation, or custom embedded services',
    ramFootprint: 'Extremely efficient (direct C++ memory layouts and low-level quantization)',
    strengths: [
      'Raw indexing algorithms authored by Meta AI research; benchmark baseline for speed',
      'Native GPU acceleration (supports multi-GPU clustering with CUDA)',
      'Zero network overhead when run in-process'
    ],
    weaknesses: [
      'Not a database: lacks ACID transactions, replication, concurrent write handling, and distributed consensus',
      'Manual serialization and custom metadata mapping required'
    ],
    verdict: 'The gold standard for algorithm benchmarks, batch clustering, and custom in-memory recommendation systems, but not a full database.'
  },
  {
    id: 'opensearch',
    name: 'OpenSearch / Elasticsearch',
    category: 'Search Engine',
    primaryLanguage: 'Java',
    license: 'Apache 2.0 (OpenSearch) / SSPL (Elastic)',
    indices: ['k-NN plugin (FAISS or NMSLIB engine)', 'Lucene HNSW'],
    filteringStrategy: 'Lucene inverted index pre-filtering combined with approximate k-NN',
    scaleSweetSpot: 'Enterprise log analytics and lexical search ecosystems adding vector capabilities',
    ramFootprint: 'High (Java JVM heap management + off-heap native memory for vector indices)',
    strengths: [
      'World-class BM25 lexical tokenization, analyzers, stemmers, and multilingual support',
      'Seamless Reciprocal Rank Fusion (RRF) between lexical queries and vector embeddings',
      'Mature enterprise tooling, access controls, and dashboard integrations'
    ],
    weaknesses: [
      'High memory footprint (JVM heap + off-heap C++ k-NN buffers)',
      'Vector query throughput is lower than specialized engines like Qdrant or Milvus'
    ],
    verdict: 'Ideal if your organization already relies heavily on Elasticsearch/OpenSearch for text search and wants to upgrade to hybrid retrieval without introducing a new vector silo.'
  }
];

export const VectorDbComparator: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [selectedDb, setSelectedDb] = useState<VectorDb>(VECTOR_DBS[0]);

  const filteredDbs = useMemo(() => {
    if (selectedCategory === 'All') return VECTOR_DBS;
    return VECTOR_DBS.filter((db) => db.category === selectedCategory);
  }, [selectedCategory]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">Vector Database Benchmark Matrix</h2>
            <p className="text-sm text-slate-500">
              Rigorous architectural comparison across 7 vector engines for indexing, filtering, RAM, and scale.
            </p>
          </div>
        </div>

        {/* Filter Category */}
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-700 outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="All">All Architectures (7)</option>
            <option value="Standalone">Dedicated Standalone (Qdrant, Milvus, Weaviate)</option>
            <option value="SQL Extension">SQL Extension (pgvector)</option>
            <option value="Serverless Cloud">Serverless Cloud (Pinecone)</option>
            <option value="Library">In-Memory Library (FAISS)</option>
            <option value="Search Engine">Search Engine (OpenSearch)</option>
          </select>
        </div>
      </div>

      {/* Grid of Database Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredDbs.map((db) => {
          const isSelected = selectedDb.id === db.id;
          return (
            <div
              key={db.id}
              onClick={() => setSelectedDb(db)}
              className={`p-5 rounded-xl border-2 cursor-pointer transition-all ${
                isSelected
                  ? 'border-indigo-600 bg-indigo-50/30 shadow-md ring-1 ring-indigo-600'
                  : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-slate-900 text-base">{db.name}</h3>
                <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
                  {db.category}
                </span>
              </div>

              <div className="text-xs text-slate-500 space-y-1 mb-3">
                <div>Language: <strong className="text-slate-700">{db.primaryLanguage}</strong></div>
                <div>Scale: <span className="text-slate-700">{db.scaleSweetSpot}</span></div>
              </div>

              <div className="pt-2 border-t border-slate-100 flex flex-wrap gap-1">
                {db.indices.slice(0, 3).map((idx) => (
                  <span key={idx} className="text-[10px] font-mono bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded">
                    {idx}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* In-Depth Selected DB Inspector */}
      {selectedDb && (
        <div className="bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          <div className="flex flex-col sm:flex-row justify-between sm:items-center gap-2 border-b border-slate-200 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-lg font-bold text-slate-900">{selectedDb.name} Architectural Profile</h3>
                <span className="text-xs font-mono bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded font-semibold">
                  {selectedDb.license}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-0.5">Written in {selectedDb.primaryLanguage} | Category: {selectedDb.category}</p>
            </div>
            <div className="text-xs font-semibold px-3 py-1 rounded-lg bg-white border border-slate-200 text-slate-700">
              RAM: {selectedDb.ramFootprint}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Supported Index Types</h4>
              <div className="flex flex-wrap gap-1.5">
                {selectedDb.indices.map((idx) => (
                  <span key={idx} className="text-xs font-mono bg-white border border-slate-200 px-2 py-1 rounded-md text-slate-800">
                    {idx}
                  </span>
                ))}
              </div>

              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 pt-2">Metadata Filtering Strategy</h4>
              <p className="text-xs text-slate-700 bg-white p-3 rounded-lg border border-slate-200 leading-relaxed font-mono">
                {selectedDb.filteringStrategy}
              </p>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-700 mb-1.5 flex items-center space-x-1">
                  <Check className="w-3.5 h-3.5" />
                  <span>Key Architectural Strengths</span>
                </h4>
                <ul className="space-y-1 text-xs text-slate-700">
                  {selectedDb.strengths.map((s, idx) => (
                    <li key={idx} className="flex items-start space-x-1.5">
                      <span className="text-emerald-500 font-bold">•</span>
                      <span>{s}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-amber-700 mb-1.5 flex items-center space-x-1">
                  <X className="w-3.5 h-3.5" />
                  <span>Limitations & Caveats</span>
                </h4>
                <ul className="space-y-1 text-xs text-slate-700">
                  {selectedDb.weaknesses.map((w, idx) => (
                    <li key={idx} className="flex items-start space-x-1.5">
                      <span className="text-amber-500 font-bold">•</span>
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-indigo-50/80 border border-indigo-200 rounded-xl p-4 text-xs text-indigo-950">
            <strong>Production Engineering Recommendation:</strong> {selectedDb.verdict}
          </div>
        </div>
      )}
    </div>
  );
};
