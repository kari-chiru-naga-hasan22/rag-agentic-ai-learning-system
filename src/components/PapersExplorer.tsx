import React, { useState, useMemo } from 'react';
import { papersData } from '../data/papersData';
import { ResearchPaper } from '../types/curriculum';
import { BookOpen, Search, ExternalLink, Filter, Calendar, Award, AlertCircle, Zap } from 'lucide-react';

export const PapersExplorer: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedEra, setSelectedEra] = useState<string>('All');
  const [selectedPaper, setSelectedPaper] = useState<ResearchPaper>(papersData[0]);

  const eras = [
    'All',
    'Foundational (2017-2020)',
    'Dense Retrieval & RAG (2020-2022)',
    'Reasoning & Agents (2022-2023)',
    'Advanced Systems (2024-2026)',
  ];

  const filteredPapers = useMemo(() => {
    return papersData.filter((p) => {
      const matchesEra = selectedEra === 'All' || p.era === selectedEra;
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        p.title.toLowerCase().includes(q) ||
        p.authors.toLowerCase().includes(q) ||
        p.venue.toLowerCase().includes(q) ||
        p.problem.toLowerCase().includes(q) ||
        p.method.toLowerCase().includes(q);
      return matchesEra && matchesSearch;
    });
  }, [searchQuery, selectedEra]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <BookOpen className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">Canonical Research Papers Library (30+)</h2>
            <p className="text-sm text-slate-500">
              Rigorous, peer-verified breakdowns of seminal literature from Vaswani et al. (2017) to 2026.
            </p>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative w-full md:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search papers, authors, venues..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-800"
          />
        </div>
      </div>

      {/* Era Tabs */}
      <div className="flex flex-wrap gap-1.5">
        {eras.map((era) => (
          <button
            key={era}
            onClick={() => setSelectedEra(era)}
            className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition ${
              selectedEra === era
                ? 'bg-indigo-600 border-indigo-600 text-white shadow-xs'
                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
            }`}
          >
            {era}
          </button>
        ))}
      </div>

      {/* Main Grid: Papers List & Detail View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Paper List */}
        <div className="lg:col-span-5 space-y-2.5 max-h-[680px] overflow-y-auto pr-1">
          {filteredPapers.map((paper) => {
            const isSelected = selectedPaper.id === paper.id;
            return (
              <div
                key={paper.id}
                onClick={() => setSelectedPaper(paper)}
                className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/40 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                    {paper.year} • {paper.venue}
                  </span>
                </div>
                <h3 className="text-sm font-bold text-slate-900 line-clamp-2 mt-1">{paper.title}</h3>
                <p className="text-xs text-slate-500 line-clamp-1 mt-1">{paper.authors}</p>
                <p className="text-xs text-slate-600 line-clamp-2 mt-1.5">{paper.problem}</p>
              </div>
            );
          })}
          {filteredPapers.length === 0 && (
            <div className="text-center py-12 text-xs text-slate-400">
              No research papers matched your search filter.
            </div>
          )}
        </div>

        {/* Selected Paper Deep Dive */}
        <div className="lg:col-span-7 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          <div className="border-b border-slate-200 pb-4 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold text-indigo-700 bg-indigo-100 px-2.5 py-1 rounded-full">
                {selectedPaper.venue} ({selectedPaper.year})
              </span>
              <span className="text-xs font-medium text-slate-500">{selectedPaper.era}</span>
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 leading-snug">{selectedPaper.title}</h3>
            <p className="text-xs text-slate-600 font-medium">Authors: {selectedPaper.authors}</p>
          </div>

          {/* Section 1: Problem & Method */}
          <div className="space-y-3 text-xs">
            <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-1">
              <strong className="text-slate-900 uppercase tracking-wider text-[11px] block text-rose-700">
                Core Problem Addressed:
              </strong>
              <p className="text-slate-700 leading-relaxed">{selectedPaper.problem}</p>
            </div>

            <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-1">
              <strong className="text-slate-900 uppercase tracking-wider text-[11px] block text-indigo-700">
                Novel Method & Architectural Contribution:
              </strong>
              <p className="text-slate-700 leading-relaxed">{selectedPaper.method}</p>
            </div>
          </div>

          {/* Section 2: Key Equation */}
          {selectedPaper.keyEquation && (
            <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs space-y-1.5 overflow-x-auto">
              <span className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">// Canonical Equation</span>
              <div className="text-emerald-400 text-sm py-1 font-semibold">{selectedPaper.keyEquation}</div>
            </div>
          )}

          {/* Section 3: Empirical Findings & Limitations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="bg-emerald-50/70 border border-emerald-200 p-4 rounded-xl space-y-1">
              <strong className="text-emerald-950 font-bold block flex items-center space-x-1">
                <Award className="w-3.5 h-3.5 text-emerald-600" />
                <span>Empirical Findings</span>
              </strong>
              <p className="text-emerald-900 leading-relaxed">{selectedPaper.keyFindings}</p>
            </div>

            <div className="bg-amber-50/70 border border-amber-200 p-4 rounded-xl space-y-1">
              <strong className="text-amber-950 font-bold block flex items-center space-x-1">
                <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                <span>Known Limitations</span>
              </strong>
              <p className="text-amber-900 leading-relaxed">{selectedPaper.limitations}</p>
            </div>
          </div>

          {/* Section 4: Production Impact */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 text-xs space-y-1">
            <strong className="text-slate-900 font-bold block flex items-center space-x-1 text-indigo-900">
              <Zap className="w-3.5 h-3.5 text-indigo-600" />
              <span>Production Engineering Impact (2026 Perspective):</span>
            </strong>
            <p className="text-slate-700 leading-relaxed">{selectedPaper.productionImpact}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
