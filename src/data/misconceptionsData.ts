import { Misconception } from '../types/curriculum';

export const misconceptionsData: Misconception[] = [
  {
    id: 1,
    myth: 'RAG completely eliminates hallucinations in LLM applications.',
    reality: 'RAG grounds responses in external text, but hallucinations persist through unfaithfulness, misattribution, and parametric override.',
    empiricalEvidence: 'The RGB Benchmark (Chen et al., 2024) demonstrated that when presented with 100% relevant retrieved passages, frontier models (GPT-4, LLaMA-2) still hallucinated in 12% to 28% of responses due to parametric memory conflicts.',
    counterProof: 'If a model has strong prior parametric weights associating a celebrity with a city, presenting an opposing passage often causes the model to either reject the passage or hallucinate a hybrid false compromise.',
    paperReference: 'Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented Generation" (AAAI 2024)'
  },
  {
    id: 2,
    myth: 'Bigger chunks and retrieving more passages (high Top-K) always improves generation accuracy.',
    reality: 'Context saturation triggers the "Lost in the Middle" phenomenon, diluting self-attention and degrading answer fidelity.',
    empiricalEvidence: 'Liu et al. (TACL 2024) proved that LLM performance follows a U-shaped curve: accuracy drops by 20-35% when critical evidence is placed in the middle of a 20-chunk prompt compared to the first or last chunk.',
    counterProof: 'Increasing Top-K from 5 to 25 triples the Time-To-First-Token (TTFT) and token costs while increasing the probability of injecting contradictory or irrelevant distractors that confuse the generator.',
    paperReference: 'Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (TACL 2024)'
  },
  {
    id: 3,
    myth: 'Vector embeddings capture semantic meaning completely and make keyword search obsolete.',
    reality: 'Dense embeddings suffer from high-dimensional hubness, out-of-domain vocabulary collapse, and complete blindness to negation.',
    empiricalEvidence: 'BEIR benchmark evaluations (Thakur et al., NeurIPS 2021) showed BM25 outperformed DPR and dense bi-encoders on 6 out of 18 datasets, particularly in technical, medical (BioASQ), and financial domains.',
    counterProof: 'A query for "drugs without caffeine" embedded via pure cosine distance will score highest with passages describing "caffeine drugs" because vector dot-products measure topic co-occurrence rather than Boolean logic.',
    paperReference: 'Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (NeurIPS 2021)'
  },
  {
    id: 4,
    myth: 'Autonomous agents are strictly superior to deterministic DAG workflows.',
    reality: 'Autonomous agent reliability decays exponentially with step count, whereas deterministic DAGs guarantee predictable execution and bounded latency.',
    empiricalEvidence: 'In a 10-step autonomous agent loop where each tool call has a 95% success rate, overall task completion is only (0.95)^10 = 59.8%. At a 90% per-step rate, success drops to 34.8%.',
    counterProof: 'For structured enterprise pipelines (e.g. invoice extraction, report generation), deterministic LangGraph DAGs achieve >99.2% reliability compared to <65% for open-ended autonomous agents.',
    paperReference: 'Anthropic Engineering, "Building Effective Agents" (2024)'
  },
  {
    id: 5,
    myth: 'More agents communicating in a multi-agent system automatically produce better results.',
    reality: 'Multi-agent loops introduce compounding latency, sycophancy cascades, and Byzantine hallucination amplification.',
    empiricalEvidence: 'SWE-bench evaluations showed that single-agent scaffolding (SWE-agent, Agentless) consistently outperformed complex 10-agent conversational swarms while consuming 80% fewer tokens.',
    counterProof: 'In unconstrained debate protocols, agents frequently defer to hallucinated assertions made by peer agents, reinforcing incorrect beliefs through peer confirmation bias rather than error-correcting.',
    paperReference: 'Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (ICML 2024)'
  },
  {
    id: 6,
    myth: '1M+ token context windows make RAG obsolete.',
    reality: 'Long-context models are financially prohibitive at scale, suffer from multi-needle reasoning degradation, and cannot enforce document-level ACLs.',
    empiricalEvidence: 'RULER benchmark (Hsieh et al., 2024) demonstrated that retrieval accuracy drops from 95% at 4k tokens to under 65% at 128k tokens for complex multi-hop reasoning tasks.',
    counterProof: 'Running 10,000 queries per day through a 500k context prompt costs ~$30,000/month with 15-second TTFT latencies, compared to ~$120/month with 450ms TTFT using a hybrid RAG pipeline.',
    paperReference: 'Hsieh et al., "RULER: What\'s the Real Context Size of Your Long-Context Language Model?" (2024)'
  },
  {
    id: 7,
    myth: 'RAG and Fine-Tuning are interchangeable techniques for teaching models new facts.',
    reality: 'Fine-tuning modifies style, format, and behavior; it is an inefficient and unreliable mechanism for factual knowledge memorization.',
    empiricalEvidence: 'Gekhman et al. (2024) demonstrated that fine-tuning an LLM on new factual knowledge causes catastrophic forgetting of general reasoning and accelerates hallucination on adjacent unlearned topics.',
    counterProof: 'Fine-tuned weights cannot be queried with access-control security filters (ACLs), cannot be updated in real-time when documents change, and provide zero citation transparency.',
    paperReference: 'Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" (EMNLP 2024)'
  },
  {
    id: 8,
    myth: 'LLM-as-a-Judge is an unbiased, fully calibrated replacement for human evaluation.',
    reality: 'Judge models suffer from severe position bias, verbosity bias, self-enhancement bias, and egocentric calibration drift.',
    empiricalEvidence: 'Zheng et al. (NeurIPS 2023) showed that LLM judges favor the first response presented up to 70% of the time (position bias) and assign 15-25% higher scores to verbose answers regardless of factual accuracy.',
    counterProof: 'A judge model created by Provider A will systematically score its own model completions 10-15% higher than equivalent completions from Provider B due to familiarity with its own token generation distributions.',
    paperReference: 'Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023)'
  }
];
