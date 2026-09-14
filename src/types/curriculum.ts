export interface Module {
  id: number;
  slug: string;
  level: number;
  levelName: string;
  title: string;
  duration: string;
  description: string;
  learningObjectives: string[];
  prerequisites: string[];
  theory: {
    definition: string;
    intuition: string;
    technicalExplanation: string;
    mathematics?: {
      formula: string;
      variables: { name: string; desc: string }[];
      derivation?: string;
    };
    example: string;
  };
  implementation: {
    language: string;
    code: string;
    explanation: string;
  };
  failureModes: string[];
  engineeringTradeoffs: string[];
  exercise: {
    prompt: string;
    hint: string;
    solution: string;
  };
  quiz: {
    question: string;
    options: string[];
    correctIndex: number;
    explanation: string;
  }[];
}

export interface ResearchPaper {
  id: string;
  title: string;
  authors: string;
  year: number;
  venue: string;
  era: 'Foundational (2017-2020)' | 'Dense Retrieval & RAG (2020-2022)' | 'Reasoning & Agents (2022-2023)' | 'Advanced Systems (2024-2026)';
  problem: string;
  method: string;
  keyEquation?: string;
  keyFindings: string;
  limitations: string;
  productionImpact: string;
  url?: string;
}

export interface HandsOnProject {
  id: number;
  title: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Production';
  techStack: string[];
  architectureSummary: string;
  problemStatement: string;
  folderStructure: string;
  code: string;
  testCode: string;
  testStatus: 'Passed 100%' | 'Verified';
  productionConsiderations: string[];
}

export interface SystemDesignCase {
  id: number;
  title: string;
  scale: string;
  sla: string;
  requirements: string[];
  architectureComponents: { name: string; role: string; tech: string }[];
  dataFlowSteps: string[];
  failureModesAndRecovery: { failure: string; mitigation: string }[];
  costAndLatencyModel: string;
}

export interface Misconception {
  id: number;
  myth: string;
  reality: string;
  empiricalEvidence: string;
  counterProof: string;
  paperReference: string;
}

export interface InterviewQuestion {
  id: number;
  tier: 'Junior AI Engineer' | 'Senior AI Engineer' | 'Staff/Principal AI Architect';
  topic: string;
  question: string;
  competenciesTested: string[];
  redFlags: string[];
  idealAnswer: string;
  followUpProbes: string[];
}
