import React, { useState, useMemo, useEffect } from 'react';
import { curriculumData } from '../data/curriculumData';
import { Module } from '../types/curriculum';
import {
  BookOpen,
  Code2,
  CheckCircle2,
  Circle,
  Copy,
  Check,
  Search,
  AlertTriangle,
  HelpCircle,
  Clock,
  GraduationCap,
  Sparkles,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

interface ModuleViewProps {
  completedModules: number[];
  onToggleComplete: (id: number) => void;
}

export const ModuleView: React.FC<ModuleViewProps> = ({
  completedModules,
  onToggleComplete,
}) => {
  const [selectedModuleId, setSelectedModuleId] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'theory' | 'code' | 'failures' | 'exercise' | 'quiz'>('theory');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedLevel, setSelectedLevel] = useState<string>('All');
  const [copied, setCopied] = useState<boolean>(false);
  const [showHint, setShowHint] = useState<boolean>(false);
  const [showSolution, setShowSolution] = useState<boolean>(false);
  const [selectedQuizAnswers, setSelectedQuizAnswers] = useState<Record<number, number>>({});

  // Reset quiz & exercise toggles on module change
  useEffect(() => {
    setShowHint(false);
    setShowSolution(false);
    setSelectedQuizAnswers({});
    setActiveTab('theory');
  }, [selectedModuleId]);

  const levels = useMemo(() => {
    const set = new Set<string>();
    curriculumData.forEach((m) => set.add(`Level ${m.level}: ${m.levelName}`));
    return ['All', ...Array.from(set)];
  }, []);

  const filteredModules = useMemo(() => {
    return curriculumData.filter((m) => {
      const matchesLevel =
        selectedLevel === 'All' || `Level ${m.level}: ${m.levelName}` === selectedLevel;
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        m.title.toLowerCase().includes(q) ||
        m.description.toLowerCase().includes(q) ||
        m.levelName.toLowerCase().includes(q) ||
        m.theory.definition.toLowerCase().includes(q);
      return matchesLevel && matchesSearch;
    });
  }, [searchQuery, selectedLevel]);

  const currentModule =
    curriculumData.find((m) => m.id === selectedModuleId) || curriculumData[0];

  const handleCopyCode = (codeText: string) => {
    navigator.clipboard.writeText(codeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleQuizSelect = (quizIdx: number, optionIdx: number) => {
    setSelectedQuizAnswers((prev) => ({ ...prev, [quizIdx]: optionIdx }));
  };

  const isCompleted = completedModules.includes(currentModule.id);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header Controls */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">University Textbook & Curriculum (42 Modules)</h2>
            <p className="text-sm text-slate-500">
              Rigorous, level-by-level mastery from mathematical foundations to frontier multi-agent systems.
            </p>
          </div>
        </div>

        {/* Global Module Search */}
        <div className="relative w-full md:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search all 42 modules..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-800"
          />
        </div>
      </div>

      {/* Level Selector Dropdown / Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1">
        <span className="text-xs font-bold uppercase text-slate-400 whitespace-nowrap">Filter Level:</span>
        <div className="flex space-x-1.5">
          {levels.slice(0, 6).map((lvl) => (
            <button
              key={lvl}
              onClick={() => setSelectedLevel(lvl)}
              className={`text-xs font-semibold px-2.5 py-1 rounded-lg border whitespace-nowrap transition ${
                selectedLevel === lvl
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-xs'
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {lvl === 'All' ? 'All (42 Modules)' : lvl.split(':')[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Modules Navigation & Reader View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Module Sidebar List */}
        <div className="lg:col-span-4 space-y-2 max-h-[750px] overflow-y-auto pr-1">
          {filteredModules.map((m) => {
            const isSelected = m.id === currentModule.id;
            const isDone = completedModules.includes(m.id);
            return (
              <div
                key={m.id}
                onClick={() => setSelectedModuleId(m.id)}
                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/40 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono font-bold text-indigo-700">
                    Mod {m.id} • L{m.level}
                  </span>
                  {isDone && (
                    <span className="flex items-center space-x-1 text-[10px] font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>Mastered</span>
                    </span>
                  )}
                </div>
                <h3 className="text-xs font-bold text-slate-900 line-clamp-1">{m.title}</h3>
                <p className="text-[11px] text-slate-500 line-clamp-2 mt-1">{m.description}</p>
                <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>{m.duration}</span>
                  <span>{m.levelName}</span>
                </div>
              </div>
            );
          })}
          {filteredModules.length === 0 && (
            <div className="text-center py-12 text-xs text-slate-400">
              No modules matched your search criteria.
            </div>
          )}
        </div>

        {/* Selected Module Textbook View */}
        <div className="lg:col-span-8 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          {/* Module Banner */}
          <div className="border-b border-slate-200 pb-4 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-indigo-700 bg-indigo-100 px-2.5 py-1 rounded-full font-mono">
                  Module #{currentModule.id} (Level {currentModule.level})
                </span>
                <span className="text-xs font-semibold text-slate-500">{currentModule.levelName}</span>
              </div>
              <button
                onClick={() => onToggleComplete(currentModule.id)}
                className={`flex items-center space-x-1.5 text-xs font-bold px-3 py-1.5 rounded-lg border transition ${
                  isCompleted
                    ? 'bg-emerald-100 border-emerald-300 text-emerald-800'
                    : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                }`}
              >
                {isCompleted ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <Circle className="w-4 h-4" />}
                <span>{isCompleted ? 'Marked Mastered' : 'Mark as Mastered'}</span>
              </button>
            </div>

            <h3 className="text-xl font-extrabold text-slate-900">{currentModule.title}</h3>
            <p className="text-xs text-slate-600 leading-relaxed">{currentModule.description}</p>

            {/* Learning Objectives & Prerequisites */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
              <div className="bg-white p-3 rounded-lg border border-slate-200">
                <strong className="text-indigo-900 uppercase tracking-wider text-[10px] block mb-1">
                  Learning Objectives:
                </strong>
                <ul className="space-y-1 text-slate-700 text-[11px]">
                  {currentModule.learningObjectives.map((obj, idx) => (
                    <li key={idx} className="flex items-start space-x-1.5">
                      <span className="text-indigo-500 font-bold">•</span>
                      <span>{obj}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white p-3 rounded-lg border border-slate-200">
                <strong className="text-slate-500 uppercase tracking-wider text-[10px] block mb-1">
                  Prerequisites:
                </strong>
                <div className="flex flex-wrap gap-1 mt-1">
                  {currentModule.prerequisites.map((prereq, idx) => (
                    <span key={idx} className="text-[10px] bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-mono">
                      {prereq}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Module Sub-Tabs */}
          <div className="flex items-center justify-between border-b border-slate-200 pb-2">
            <div className="flex flex-wrap gap-1 bg-white p-1 rounded-lg border border-slate-200 text-xs font-semibold">
              <button
                onClick={() => setActiveTab('theory')}
                className={`px-3 py-1.5 rounded-md transition ${
                  activeTab === 'theory' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Theory & Derivations
              </button>
              <button
                onClick={() => setActiveTab('code')}
                className={`px-3 py-1.5 rounded-md transition ${
                  activeTab === 'code' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Code Implementation
              </button>
              <button
                onClick={() => setActiveTab('failures')}
                className={`px-3 py-1.5 rounded-md transition ${
                  activeTab === 'failures' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Failure Modes
              </button>
              <button
                onClick={() => setActiveTab('exercise')}
                className={`px-3 py-1.5 rounded-md transition ${
                  activeTab === 'exercise' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Hands-On Exercise
              </button>
              <button
                onClick={() => setActiveTab('quiz')}
                className={`px-3 py-1.5 rounded-md transition ${
                  activeTab === 'quiz' ? 'bg-indigo-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Knowledge Quiz
              </button>
            </div>

            {activeTab === 'code' && (
              <button
                onClick={() => handleCopyCode(currentModule.implementation.code)}
                className="flex items-center space-x-1 text-xs font-semibold px-2.5 py-1 bg-white border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 transition"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            )}
          </div>

          {/* Tab 1: Theory */}
          {activeTab === 'theory' && (
            <div className="space-y-5 text-xs">
              <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-2">
                <strong className="text-indigo-900 font-bold uppercase tracking-wider text-[11px] block">
                  Formal Definition & Technical Intuition
                </strong>
                <p className="text-slate-800 leading-relaxed text-sm font-medium">
                  {currentModule.theory.definition}
                </p>
                <p className="text-slate-600 leading-relaxed pt-1">
                  <strong>Intuition:</strong> {currentModule.theory.intuition}
                </p>
              </div>

              {/* Technical explanation */}
              <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-2">
                <strong className="text-slate-900 font-bold uppercase tracking-wider text-[11px] block">
                  Detailed Engineering & Scientific Explanation
                </strong>
                <p className="text-slate-700 leading-relaxed whitespace-pre-line">
                  {currentModule.theory.technicalExplanation}
                </p>
              </div>

              {/* Mathematical formulation if present */}
              {currentModule.theory.mathematics && (
                <div className="bg-slate-900 text-slate-100 p-4 rounded-xl space-y-3 font-mono">
                  <div className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">// Mathematical Formulation</div>
                  <div className="text-emerald-400 text-sm font-bold bg-slate-950 p-2.5 rounded border border-slate-800 overflow-x-auto">
                    {currentModule.theory.mathematics.formula}
                  </div>
                  <div className="space-y-1 text-slate-300 text-[11px]">
                    <span className="text-slate-400 font-bold">Variable Legend:</span>
                    {currentModule.theory.mathematics.variables.map((v, idx) => (
                      <div key={idx} className="flex space-x-2">
                        <span className="text-indigo-300 font-semibold">{v.name}:</span>
                        <span>{v.desc}</span>
                      </div>
                    ))}
                  </div>
                  {currentModule.theory.mathematics.derivation && (
                    <div className="pt-2 border-t border-slate-800 text-slate-400 text-[11px]">
                      <strong>Derivation Note:</strong> {currentModule.theory.mathematics.derivation}
                    </div>
                  )}
                </div>
              )}

              {/* Concrete Example */}
              <div className="bg-emerald-50/70 border border-emerald-200 p-4 rounded-xl text-xs space-y-1 text-emerald-950">
                <strong className="font-bold block uppercase tracking-wider text-[11px] text-emerald-900">
                  Concrete Production Example:
                </strong>
                <p className="leading-relaxed">{currentModule.theory.example}</p>
              </div>
            </div>
          )}

          {/* Tab 2: Code Implementation */}
          {activeTab === 'code' && (
            <div className="space-y-4">
              <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto max-h-[440px]">
                <div className="text-slate-400 text-[10px] uppercase font-bold mb-2">// Language: {currentModule.implementation.language}</div>
                <pre className="text-slate-200 leading-relaxed">{currentModule.implementation.code}</pre>
              </div>
              <div className="bg-white p-4 rounded-xl border border-slate-200 text-xs text-slate-700 leading-relaxed">
                <strong className="block text-slate-900 font-semibold mb-1">Code Explanation & Invariants:</strong>
                {currentModule.implementation.explanation}
              </div>
            </div>
          )}

          {/* Tab 3: Failure Modes & Tradeoffs */}
          {activeTab === 'failures' && (
            <div className="space-y-5 text-xs">
              <div className="bg-amber-50/60 border border-amber-200 p-5 rounded-xl space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-amber-800 flex items-center space-x-1.5">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  <span>Production Failure Modes & Edge Cases</span>
                </h4>
                <ul className="space-y-1.5 pl-2 text-amber-950">
                  {currentModule.failureModes.map((fm, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-amber-600 font-bold">✕</span>
                      <span>{fm}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  Engineering Tradeoffs
                </h4>
                <ul className="space-y-1.5 pl-2 text-slate-700">
                  {currentModule.engineeringTradeoffs.map((tradeoff, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-indigo-600 font-bold">⇄</span>
                      <span>{tradeoff}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Tab 4: Hands-On Exercise */}
          {activeTab === 'exercise' && (
            <div className="space-y-5 text-xs">
              <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-2">
                <strong className="text-slate-900 font-bold uppercase tracking-wider text-[11px] block">
                  Exercise Assignment
                </strong>
                <p className="text-slate-800 leading-relaxed text-sm">
                  {currentModule.exercise.prompt}
                </p>
              </div>

              {/* Hint button */}
              <div className="space-y-2">
                <button
                  onClick={() => setShowHint(!showHint)}
                  className="text-xs font-bold text-indigo-600 hover:text-indigo-800 transition flex items-center space-x-1"
                >
                  <HelpCircle className="w-3.5 h-3.5" />
                  <span>{showHint ? 'Hide Implementation Hint' : 'Show Implementation Hint'}</span>
                </button>
                {showHint && (
                  <div className="p-3 bg-indigo-50 border border-indigo-100 rounded-lg text-indigo-950">
                    {currentModule.exercise.hint}
                  </div>
                )}
              </div>

              {/* Solution reveal */}
              <div className="space-y-2 pt-2 border-t border-slate-200">
                <button
                  onClick={() => setShowSolution(!showSolution)}
                  className="text-xs font-bold text-slate-700 hover:text-slate-900 transition flex items-center space-x-1"
                >
                  <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                  <span>{showSolution ? 'Hide Reference Solution' : 'Reveal Reference Solution'}</span>
                </button>
                {showSolution && (
                  <div className="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto">
                    <pre className="text-slate-200">{currentModule.exercise.solution}</pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Tab 5: Quiz */}
          {activeTab === 'quiz' && (
            <div className="space-y-6 text-xs">
              {currentModule.quiz.map((q, qIdx) => {
                const selected = selectedQuizAnswers[qIdx];
                const isAnswered = selected !== undefined;
                const isCorrect = selected === q.correctIndex;

                return (
                  <div key={qIdx} className="bg-white p-5 rounded-xl border border-slate-200 space-y-3">
                    <div className="font-bold text-slate-900 text-sm">
                      Q{qIdx + 1}: {q.question}
                    </div>

                    <div className="space-y-2">
                      {q.options.map((opt, optIdx) => {
                        let btnStyle = 'border-slate-200 hover:bg-slate-50 text-slate-700 bg-white';
                        if (isAnswered) {
                          if (optIdx === q.correctIndex) {
                            btnStyle = 'border-emerald-500 bg-emerald-50 text-emerald-950 font-bold';
                          } else if (selected === optIdx) {
                            btnStyle = 'border-rose-500 bg-rose-50 text-rose-950 font-bold';
                          }
                        }

                        return (
                          <button
                            key={optIdx}
                            disabled={isAnswered}
                            onClick={() => handleQuizSelect(qIdx, optIdx)}
                            className={`w-full text-left p-3 rounded-lg border text-xs transition flex items-center justify-between ${btnStyle}`}
                          >
                            <span>{opt}</span>
                            {isAnswered && optIdx === q.correctIndex && (
                              <Check className="w-4 h-4 text-emerald-600 flex-shrink-0 ml-2" />
                            )}
                          </button>
                        );
                      })}
                    </div>

                    {isAnswered && (
                      <div
                        className={`p-3 rounded-lg text-xs leading-relaxed ${
                          isCorrect ? 'bg-emerald-50 border border-emerald-200 text-emerald-950' : 'bg-rose-50 border border-rose-200 text-rose-950'
                        }`}
                      >
                        <strong>{isCorrect ? 'Correct! ' : 'Incorrect. '}</strong>
                        {q.explanation}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
