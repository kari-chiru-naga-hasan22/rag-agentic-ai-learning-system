import React, { useState } from 'react';
import { misconceptionsData } from '../data/misconceptionsData';
import { Misconception } from '../types/curriculum';
import { AlertCircle, CheckCircle2, ShieldAlert, BookOpen, Flame, HelpCircle } from 'lucide-react';

export const MisconceptionsBuster: React.FC = () => {
  const [activeMythId, setActiveMythId] = useState<number>(1);

  const activeMyth = misconceptionsData.find((m) => m.id === activeMythId) || misconceptionsData[0];

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-rose-50 text-rose-600 rounded-xl">
            <Flame className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">8 Industry Misconceptions & Mythbusters</h2>
            <p className="text-sm text-slate-500">
              Adversarial, research-backed debunking of common hype, marketing claims, and architectural fallacies.
            </p>
          </div>
        </div>
      </div>

      {/* Grid: Myth selector and debunk card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Myths list */}
        <div className="lg:col-span-5 space-y-2 max-h-[620px] overflow-y-auto pr-1">
          {misconceptionsData.map((m) => {
            const isSelected = activeMythId === m.id;
            return (
              <div
                key={m.id}
                onClick={() => setActiveMythId(m.id)}
                className={`p-3.5 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-rose-500 bg-rose-50/30 shadow-sm ring-1 ring-rose-500'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono font-bold text-rose-700 bg-rose-50 px-2 py-0.5 rounded">
                    Myth #{m.id}
                  </span>
                </div>
                <h3 className="text-xs font-bold text-slate-900 line-clamp-2 mt-1">"{m.myth}"</h3>
              </div>
            );
          })}
        </div>

        {/* Selected Myth In-Depth Debunk View */}
        <div className="lg:col-span-7 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          {/* Myth Header */}
          <div className="bg-rose-50/70 border border-rose-200 p-5 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 text-rose-700 text-xs font-bold uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-rose-600" />
              <span>Pervasive Industry Myth #{activeMyth.id}</span>
            </div>
            <h3 className="text-base font-bold text-rose-950 leading-snug">
              "{activeMyth.myth}"
            </h3>
          </div>

          {/* Ground Truth Reality */}
          <div className="bg-emerald-50/70 border border-emerald-200 p-5 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 text-emerald-800 text-xs font-bold uppercase tracking-wider">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Rigorous Ground Truth Reality</span>
            </div>
            <p className="text-sm font-semibold text-emerald-950 leading-relaxed">
              {activeMyth.reality}
            </p>
          </div>

          {/* Empirical Evidence & Citation */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 space-y-3 text-xs">
            <strong className="text-slate-900 font-bold block uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
              <BookOpen className="w-3.5 h-3.5 text-indigo-600" />
              <span>Empirical Research Evidence</span>
            </strong>
            <p className="text-slate-700 leading-relaxed font-normal">
              {activeMyth.empiricalEvidence}
            </p>
            <div className="pt-2 border-t border-slate-100 text-[11px] font-mono text-indigo-700">
              <strong>Citation:</strong> {activeMyth.paperReference}
            </div>
          </div>

          {/* Counter-Proof Derivation */}
          <div className="bg-slate-900 text-slate-100 p-5 rounded-xl text-xs space-y-2 font-mono">
            <div className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">// Mathematical & Algorithmic Counter-Proof</div>
            <p className="text-slate-200 leading-relaxed">
              {activeMyth.counterProof}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
