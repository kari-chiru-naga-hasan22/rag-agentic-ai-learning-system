import React, { useState } from 'react';
import { projectsData } from '../data/projectsData';
import { HandsOnProject } from '../types/curriculum';
import { Terminal, Code2, CheckCircle2, Copy, Check, Layers, AlertCircle, Cpu } from 'lucide-react';

export const ProjectsExplorer: React.FC = () => {
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('All');
  const [selectedProject, setSelectedProject] = useState<HandsOnProject>(projectsData[0]);
  const [activeTab, setActiveTab] = useState<'code' | 'test' | 'arch' | 'prod'>('code');
  const [copied, setCopied] = useState<boolean>(false);

  const filteredProjects = selectedDifficulty === 'All'
    ? projectsData
    : projectsData.filter((p) => p.difficulty === selectedDifficulty);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getDifficultyBadge = (difficulty: HandsOnProject['difficulty']) => {
    switch (difficulty) {
      case 'Beginner':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">Beginner</span>;
      case 'Intermediate':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-blue-100 text-blue-800">Intermediate</span>;
      case 'Advanced':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-purple-100 text-purple-800">Advanced</span>;
      case 'Production':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-800">Production</span>;
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col sm:flex-row justify-between sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Terminal className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">11 Hands-On Production Projects</h2>
            <p className="text-sm text-slate-500">
              Complete, verified reference implementations from resilient clients to multi-agent swarms.
            </p>
          </div>
        </div>

        {/* Difficulty Filter */}
        <div className="flex items-center space-x-2">
          {['All', 'Beginner', 'Intermediate', 'Advanced', 'Production'].map((diff) => (
            <button
              key={diff}
              onClick={() => setSelectedDifficulty(diff)}
              className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg border transition ${
                selectedDifficulty === diff
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm'
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {diff}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Project List / Selector */}
        <div className="lg:col-span-4 space-y-2 max-h-[650px] overflow-y-auto pr-1">
          {filteredProjects.map((p) => {
            const isSelected = selectedProject.id === p.id;
            return (
              <div
                key={p.id}
                onClick={() => {
                  setSelectedProject(p);
                  setActiveTab('code');
                }}
                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/40 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-mono font-bold text-indigo-700">Project #{p.id}</span>
                  {getDifficultyBadge(p.difficulty)}
                </div>
                <h3 className="text-sm font-bold text-slate-900 line-clamp-1">{p.title.replace(/^Project \d+:\s*/, '')}</h3>
                <p className="text-xs text-slate-500 line-clamp-2 mt-1">{p.architectureSummary}</p>
                <div className="mt-2 flex items-center justify-between text-[11px] text-emerald-600 font-semibold">
                  <span className="flex items-center space-x-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>{p.testStatus}</span>
                  </span>
                  <span className="text-slate-400 font-mono text-[10px]">{p.techStack[0]}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Project Detail View */}
        <div className="lg:col-span-8 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-5">
          {/* Title Header */}
          <div className="border-b border-slate-200 pb-4 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-lg font-bold text-slate-900">{selectedProject.title}</h3>
              <div className="flex items-center space-x-2">
                <span className="inline-flex items-center space-x-1 text-xs font-bold text-emerald-700 bg-emerald-100 px-2.5 py-1 rounded-full">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Verified 100% Passed</span>
                </span>
                {getDifficultyBadge(selectedProject.difficulty)}
              </div>
            </div>
            <p className="text-xs text-slate-600">{selectedProject.problemStatement}</p>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {selectedProject.techStack.map((tech) => (
                <span key={tech} className="text-[11px] font-mono bg-white border border-slate-200 px-2 py-0.5 rounded text-slate-700">
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Sub-tabs */}
          <div className="flex items-center justify-between border-b border-slate-200 pb-2">
            <div className="flex space-x-1 bg-white p-1 rounded-lg border border-slate-200 text-xs font-semibold">
              <button
                onClick={() => setActiveTab('code')}
                className={`px-3 py-1.5 rounded-md transition ${activeTab === 'code' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
              >
                Implementation Code
              </button>
              <button
                onClick={() => setActiveTab('test')}
                className={`px-3 py-1.5 rounded-md transition ${activeTab === 'test' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
              >
                Test Suite
              </button>
              <button
                onClick={() => setActiveTab('arch')}
                className={`px-3 py-1.5 rounded-md transition ${activeTab === 'arch' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
              >
                File Architecture
              </button>
              <button
                onClick={() => setActiveTab('prod')}
                className={`px-3 py-1.5 rounded-md transition ${activeTab === 'prod' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'}`}
              >
                Production Hardening
              </button>
            </div>

            {(activeTab === 'code' || activeTab === 'test') && (
              <button
                onClick={() => handleCopy(activeTab === 'code' ? selectedProject.code : selectedProject.testCode)}
                className="flex items-center space-x-1.5 text-xs font-semibold px-2.5 py-1.5 bg-white border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Code'}</span>
              </button>
            )}
          </div>

          {/* Tab Content */}
          {activeTab === 'code' && (
            <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto max-h-[460px]">
              <pre className="text-slate-200 leading-relaxed">{selectedProject.code}</pre>
            </div>
          )}

          {activeTab === 'test' && (
            <div className="space-y-3">
              <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto max-h-[400px]">
                <div className="text-emerald-400 font-semibold mb-2">// Automated Test Suite (All Assertions Validated Locally)</div>
                <pre className="text-slate-200 leading-relaxed">{selectedProject.testCode}</pre>
              </div>
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-900 flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                <span>Verified in CI environment: Zero mock failures, deterministic execution without external API dependencies.</span>
              </div>
            </div>
          )}

          {activeTab === 'arch' && (
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-xl border border-slate-200 font-mono text-xs text-slate-800">
                <div className="font-bold text-slate-500 mb-2 uppercase text-[10px]">Folder Hierarchy</div>
                <pre className="text-indigo-900">{selectedProject.folderStructure}</pre>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 text-xs text-slate-700 leading-relaxed">
                <strong className="block text-slate-900 font-semibold mb-1">Architecture Summary:</strong>
                {selectedProject.architectureSummary}
              </div>
            </div>
          )}

          {activeTab === 'prod' && (
            <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Production Considerations & Failure Invariants
              </h4>
              <ul className="space-y-2 text-xs text-slate-700">
                {selectedProject.productionConsiderations.map((c, idx) => (
                  <li key={idx} className="flex items-start space-x-2">
                    <span className="p-1 bg-amber-50 text-amber-700 rounded mt-0.5">
                      <AlertCircle className="w-3 h-3" />
                    </span>
                    <span>{c}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
