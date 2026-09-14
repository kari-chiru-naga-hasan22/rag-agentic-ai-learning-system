import React, { useState } from 'react';
import { systemDesignsData } from '../data/systemDesignsData';
import { SystemDesignCase } from '../types/curriculum';
import { Network, Server, ArrowRight, ShieldCheck, AlertTriangle, DollarSign, Clock, Layers, CheckCircle2 } from 'lucide-react';

export const SystemDesignViewer: React.FC = () => {
  const [selectedCase, setSelectedCase] = useState<SystemDesignCase>(systemDesignsData[0]);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-100 pb-5 flex flex-col md:flex-row justify-between md:items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
            <Network className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900">8 Production System Design Blueprints</h2>
            <p className="text-sm text-slate-500">
              End-to-end architectures, scale metrics, failure mitigations, and latency budgets for enterprise systems.
            </p>
          </div>
        </div>
      </div>

      {/* Grid: Architecture selector & details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Cases List */}
        <div className="lg:col-span-4 space-y-2 max-h-[680px] overflow-y-auto pr-1">
          {systemDesignsData.map((sysCase) => {
            const isSelected = selectedCase.id === sysCase.id;
            return (
              <div
                key={sysCase.id}
                onClick={() => setSelectedCase(sysCase)}
                className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/40 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50/50'
                }`}
              >
                <span className="text-[11px] font-mono font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                  System Design #{sysCase.id}
                </span>
                <h3 className="text-sm font-bold text-slate-900 mt-1 line-clamp-2">{sysCase.title}</h3>
                <p className="text-xs text-slate-500 mt-1">Scale: {sysCase.scale}</p>
                <div className="mt-2 text-[11px] text-emerald-700 font-semibold flex items-center space-x-1">
                  <Clock className="w-3.5 h-3.5 text-emerald-600" />
                  <span>SLA: {sysCase.sla}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Architecture Blueprint View */}
        <div className="lg:col-span-8 bg-slate-50 rounded-2xl border border-slate-200 p-6 space-y-6">
          <div className="border-b border-slate-200 pb-4 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold text-indigo-700 bg-indigo-100 px-2.5 py-1 rounded-full">
                System Blueprint #{selectedCase.id}
              </span>
              <span className="text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 rounded-md">
                SLA: {selectedCase.sla}
              </span>
            </div>
            <h3 className="text-xl font-extrabold text-slate-900">{selectedCase.title}</h3>
            <p className="text-xs font-medium text-slate-500"><strong>Scale Profile:</strong> {selectedCase.scale}</p>
          </div>

          {/* Key Requirements */}
          <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Core Engineering Requirements
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-700">
              {selectedCase.requirements.map((req, idx) => (
                <div key={idx} className="flex items-start space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600 mt-0.5 flex-shrink-0" />
                  <span>{req}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Architecture Components */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              System Architecture Components
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {selectedCase.architectureComponents.map((comp, idx) => (
                <div key={idx} className="bg-white p-3.5 rounded-xl border border-slate-200 space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <strong className="text-slate-900 font-bold">{comp.name}</strong>
                    <span className="font-mono text-[10px] bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded">
                      {comp.tech}
                    </span>
                  </div>
                  <p className="text-slate-600 text-[11px] leading-relaxed">{comp.role}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Data Flow Pipeline */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              End-to-End Execution Data Flow
            </h4>
            <div className="bg-white p-4 rounded-xl border border-slate-200 space-y-2.5 text-xs">
              {selectedCase.dataFlowSteps.map((step, idx) => (
                <div key={idx} className="flex items-start space-x-2.5">
                  <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-50 text-indigo-600 font-bold flex items-center justify-center text-[10px]">
                    {idx + 1}
                  </span>
                  <p className="text-slate-700 font-mono text-[11px] pt-0.5 leading-relaxed">{step}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Failure Modes & Recovery */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-700">
              Failure Invariants & Mitigations
            </h4>
            <div className="space-y-2 text-xs">
              {selectedCase.failureModesAndRecovery.map((fm, idx) => (
                <div key={idx} className="bg-amber-50/60 border border-amber-200 p-3.5 rounded-xl space-y-1">
                  <div className="flex items-center space-x-1.5 text-amber-950 font-bold">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
                    <span>Failure Mode: {fm.failure}</span>
                  </div>
                  <p className="text-amber-900 text-[11px] pl-5 leading-relaxed">
                    <strong>Production Mitigation:</strong> {fm.mitigation}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Cost & Latency Model */}
          <div className="bg-indigo-50/70 border border-indigo-200 p-4 rounded-xl text-xs space-y-1">
            <div className="flex items-center space-x-1.5 text-indigo-950 font-bold">
              <DollarSign className="w-4 h-4 text-emerald-600" />
              <span>Cost & Latency Projections:</span>
            </div>
            <p className="text-indigo-900 leading-relaxed pl-5 font-mono text-[11px]">
              {selectedCase.costAndLatencyModel}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
