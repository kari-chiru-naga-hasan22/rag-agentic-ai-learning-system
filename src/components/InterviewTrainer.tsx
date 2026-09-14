import React, { useState, useMemo } from 'react';
import { interviewData } from '../data/interviewData';
import { InterviewQuestion } from '../types/curriculum';
import { GraduationCap, Eye, EyeOff, AlertTriangle, CheckCircle2, ChevronRight, HelpCircle, Sparkles } from 'lucide-react';

export const InterviewTrainer: React.FC = () => {
  const [selectedTier, setSelectedTier] = useState<string>('All');
  const [activeQuestionId, setActiveQuestionId] = useState<number>(1);
  const [showAnswer, setShowAnswer] = useState<boolean>(false);
  const [showRedFlags, setShowRedFlags] = useState<boolean>(false);

  const filteredQuestions = useMemo(() => {
    if (selectedTier === 'All') return interviewData;
    return interviewData.filter((q) => q.tier === selectedTier);
  }, [selectedTier]);

  const activeQuestion = interviewData.find((q) => q.id === activeQuestionId) || interviewData[0];

  const getTierBadge = (tier: InterviewQuestion['tier']) => {
    switch (tier) {
      case 'Junior AI Engineer':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-blue-100 text-blue-800">Junior (L3/L4)</span>;
      case 'Senior AI Engineer':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800">Senior (L5)</span>;
      case 'Staff/Principal AI Architect':
        return <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-purple-100 text-purple-800">Staff / Principal (L6+)</span>;
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">15 Elite Interview Questions & Rubrics</h2>
            <p className="text-sm text-slate-500">
              Battle-tested technical interview questions spanning Junior, Senior, and Staff Architect tiers.
            </p>
          </div>
        </div>

        {/* Tier filter buttons */}
        <div className="flex flex-wrap gap-1.5">
          {['All', 'Junior AI Engineer', 'Senior AI Engineer', 'Staff/Principal AI Architect'].map((tier) => (
            <button
              key={tier}
              onClick={() => {
                setSelectedTier(tier);
                setShowAnswer(false);
                setShowRedFlags(false);
              }}
              className={`text-xs font-semibold px-2.5 py-1.5 rounded-lg border transition ${
                selectedTier === tier
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-xs'
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {tier === 'Staff/Principal AI Architect' ? 'Staff / Principal' : tier.replace(' AI Engineer', '')}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Question Selector List */}
        <div className="lg:col-span-5 space-y-2 max-h-[640px] overflow-y-auto pr-1">
          {filteredQuestions.map((q) => {
            const isSelected = q.id === activeQuestion.id;
            return (
              <div
                key={q.id}
                onClick={() => {
                  setActiveQuestionId(q.id);
                  setShowAnswer(false);
                  setShowRedFlags(false);
                }}
                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/40 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono font-bold text-indigo-700">Q#{q.id}</span>
                  {getTierBadge(q.tier)}
                </div>
                <div className="text-xs font-semibold text-slate-500">{q.topic}</div>
                <h3 className="text-xs font-bold text-slate-900 line-clamp-2 mt-1">{q.question}</h3>
              </div>
            );
          })}
        </div>

        {/* Question Detail Flashcard */}
        <div className="lg:col-span-7 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          <div className="border-b border-slate-200 pb-4 space-y-2">
            <div className="flex items-center justify-between">
              {getTierBadge(activeQuestion.tier)}
              <span className="text-xs font-mono font-bold text-slate-400">{activeQuestion.topic}</span>
            </div>
            <h3 className="text-base font-bold text-slate-900 leading-relaxed">{activeQuestion.question}</h3>
          </div>

          {/* Competencies Tested */}
          <div>
            <h4 className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
              Target Competencies Tested
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {activeQuestion.competenciesTested.map((comp) => (
                <span key={comp} className="text-xs font-semibold bg-white border border-slate-200 text-indigo-700 px-2.5 py-1 rounded-md">
                  {comp}
                </span>
              ))}
            </div>
          </div>

          {/* Red Flags Toggle */}
          <div className="space-y-2">
            <button
              onClick={() => setShowRedFlags(!showRedFlags)}
              className="flex items-center space-x-1.5 text-xs font-bold text-amber-700 hover:text-amber-800 transition"
            >
              <AlertTriangle className="w-4 h-4 text-amber-600" />
              <span>{showRedFlags ? 'Hide Candidate Red Flags' : 'Reveal Candidate Red Flags (Fatal Flaws)'}</span>
            </button>
            {showRedFlags && (
              <div className="p-4 bg-amber-50/70 border border-amber-200 rounded-xl space-y-2 text-xs text-amber-950">
                <div className="font-bold text-[11px] uppercase tracking-wider text-amber-800">Interviewer Watchlist (Instant Reject Criteria):</div>
                <ul className="space-y-1.5 pl-2">
                  {activeQuestion.redFlags.map((rf, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-amber-600 font-bold">✕</span>
                      <span>{rf}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Ideal Answer Section */}
          <div className="space-y-3 pt-2 border-t border-slate-200">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Architectural Model Answer
              </span>
              <button
                onClick={() => setShowAnswer(!showAnswer)}
                className="flex items-center space-x-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition shadow-xs"
              >
                {showAnswer ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                <span>{showAnswer ? 'Hide Ideal Answer' : 'Reveal Ideal Answer'}</span>
              </button>
            </div>

            {showAnswer ? (
              <div className="p-5 bg-white border border-slate-200 rounded-xl space-y-4 text-xs">
                <p className="text-slate-800 leading-relaxed text-sm whitespace-pre-line">
                  {activeQuestion.idealAnswer}
                </p>

                {/* Follow up probes */}
                {activeQuestion.followUpProbes.length > 0 && (
                  <div className="pt-3 border-t border-slate-100 space-y-2">
                    <div className="font-bold text-[11px] uppercase tracking-wider text-indigo-700 flex items-center space-x-1">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Interviewer Follow-Up Probes (Leveling to Staff/Principal):</span>
                    </div>
                    <ul className="space-y-1 pl-2 text-slate-600">
                      {activeQuestion.followUpProbes.map((probe, idx) => (
                        <li key={idx} className="flex items-start space-x-1.5">
                          <span className="text-indigo-500 font-bold">→</span>
                          <span>{probe}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 bg-white border border-dashed border-slate-200 rounded-xl text-xs text-slate-400">
                Click "Reveal Ideal Answer" above to view the staff-level technical synthesis.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
