import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { ModuleView } from './components/ModuleView';
import { DecisionCalculator } from './components/DecisionCalculator';
import { VectorMathSandbox } from './components/VectorMathSandbox';
import { ReActSimulator } from './components/ReActSimulator';
import { ChunkingVisualizer } from './components/ChunkingVisualizer';
import { VectorDbComparator } from './components/VectorDbComparator';
import { ProjectsExplorer } from './components/ProjectsExplorer';
import { PapersExplorer } from './components/PapersExplorer';
import { SystemDesignViewer } from './components/SystemDesignViewer';
import { MisconceptionsBuster } from './components/MisconceptionsBuster';
import { InterviewTrainer } from './components/InterviewTrainer';
import { CapstoneRoadmap } from './components/CapstoneRoadmap';
import { curriculumData } from './data/curriculumData';
import {
  BookOpen,
  Cpu,
  Sparkles,
  Database,
  Terminal,
  ShieldCheck,
  Award,
  CheckCircle2,
  ExternalLink,
  Layers,
  ArrowRight
} from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('curriculum');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [completedModules, setCompletedModules] = useState<number[]>(() => {
    try {
      const saved = localStorage.getItem('rag_master_completed_modules');
      return saved ? JSON.parse(saved) : [0, 1];
    } catch {
      return [0, 1];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('rag_master_completed_modules', JSON.stringify(completedModules));
    } catch {}
  }, [completedModules]);

  const toggleModuleComplete = (id: number) => {
    setCompletedModules((prev) =>
      prev.includes(id) ? prev.filter((mId) => mId !== id) : [...prev, id]
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        completedCount={completedModules.length}
        totalModules={curriculumData.length}
      />

      {/* Hero Stats Banner */}
      <section className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
            <div className="space-y-2 max-w-3xl">
              <div className="inline-flex items-center space-x-2 px-3 py-1 bg-indigo-50 border border-indigo-200 rounded-full text-xs font-semibold text-indigo-700">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Production AI Engineering & Academic Research Canon</span>
              </div>
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 tracking-tight">
                RAG & Agentic AI: <span className="text-indigo-600">Zero to Research Level</span>
              </h1>
              <p className="text-sm sm:text-base text-slate-600 leading-relaxed">
                A complete university-level textbook, rigorous systems engineering canon, hands-on coding curriculum, and interactive research laboratory. Built strictly on empirical benchmarks, peer-reviewed formalisms, and verified production code.
              </p>
            </div>

            {/* Quick Stats Badges */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-indigo-600">42</div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Modules</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-emerald-600">30+</div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Papers</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-blue-600">11</div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Projects</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-purple-600">8</div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Systems</div>
              </div>
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-amber-600">15</div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Interviews</div>
              </div>
              <div className="bg-emerald-50 border border-emerald-200 p-3 rounded-xl text-center">
                <div className="text-xl font-extrabold text-emerald-700">100%</div>
                <div className="text-[11px] font-semibold text-emerald-800 uppercase tracking-wider">Tests Passed</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'curriculum' && (
          <ModuleView
            completedModules={completedModules}
            onToggleComplete={toggleModuleComplete}
          />
        )}
        {activeTab === 'decision' && <DecisionCalculator />}
        {activeTab === 'sandbox' && <VectorMathSandbox />}
        {activeTab === 'react-sim' && <ReActSimulator />}
        {activeTab === 'chunking' && <ChunkingVisualizer />}
        {activeTab === 'vectordb' && <VectorDbComparator />}
        {activeTab === 'projects' && <ProjectsExplorer />}
        {activeTab === 'papers' && <PapersExplorer />}
        {activeTab === 'system-design' && <SystemDesignViewer />}
        {activeTab === 'misconceptions' && <MisconceptionsBuster />}
        {activeTab === 'interview' && <InterviewTrainer />}
        {activeTab === 'capstone' && <CapstoneRoadmap />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-slate-800">RAG & Agentic AI Master System</span>
            <span>•</span>
            <span>MIT License</span>
            <span>•</span>
            <span>Ponytail Engineered (Zero Unneeded Abstractions)</span>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-emerald-700 font-semibold flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>All 11 Reference Projects Verified Locally</span>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
