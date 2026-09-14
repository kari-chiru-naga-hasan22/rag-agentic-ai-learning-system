import React from 'react';
import { BookOpen, Cpu, Sparkles, Database, Terminal, ShieldAlert, Award, Compass, Search, Github, ExternalLink } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  completedCount: number;
  totalModules: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  searchQuery,
  setSearchQuery,
  completedCount,
  totalModules,
}) => {
  const navItems = [
    { id: 'curriculum', label: 'Curriculum', icon: BookOpen },
    { id: 'decision', label: 'Decision Engine', icon: Compass },
    { id: 'sandbox', label: 'Vector Sandbox', icon: Cpu },
    { id: 'react-sim', label: 'ReAct Simulator', icon: Terminal },
    { id: 'chunking', label: 'Chunking Lab', icon: Sparkles },
    { id: 'vectordb', label: 'Vector DBs', icon: Database },
    { id: 'projects', label: '11 Projects', icon: Terminal },
    { id: 'papers', label: 'Papers (30+)', icon: BookOpen },
    { id: 'system-design', label: 'System Design', icon: Cpu },
    { id: 'misconceptions', label: 'Myth Buster', icon: ShieldAlert },
    { id: 'interview', label: 'Interview Prep', icon: Award },
    { id: 'capstone', label: 'Capstone', icon: Award },
  ];

  const progressPercent = Math.round((completedCount / totalModules) * 100);

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('curriculum')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-700 to-sky-500 flex items-center justify-center text-white shadow-sm font-bold text-lg">
              RA
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-slate-900 text-base tracking-tight">RAG & Agentic AI</span>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-full">
                  Zero to Research
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">University Textbook + Systems Engineering Canon</p>
            </div>
          </div>

          {/* Search bar */}
          <div className="hidden md:flex items-center flex-1 max-w-xs mx-6">
            <div className="relative w-full">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search 42 modules, papers, code..."
                className="w-full pl-9 pr-4 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 text-slate-900 placeholder:text-slate-400 transition"
              />
            </div>
          </div>

          {/* Actions & Progress */}
          <div className="flex items-center space-x-3">
            {/* Progress indicator */}
            <div className="hidden lg:flex items-center space-x-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg">
              <div className="w-20 bg-slate-200 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <span className="text-xs font-semibold text-slate-700">{progressPercent}%</span>
              <span className="text-[11px] text-slate-400">({completedCount}/{totalModules})</span>
            </div>

            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="p-2 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition"
              title="View on GitHub"
            >
              <Github className="w-4 h-4" />
            </a>
          </div>
        </div>

        {/* Scrollable Navigation Sub-bar */}
        <div className="flex items-center space-x-1 overflow-x-auto py-2 border-t border-slate-100 no-scrollbar">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
