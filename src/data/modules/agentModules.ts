import { Module } from '../../types/curriculum';

export const agentModules: Module[] = [
  {
    id: 17,
    slug: 'ai-agents-foundations',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '17. AI Agents: Formal Definitions, Agency Spectrum & Loops',
    duration: '4 Hours',
    description: 'Workflows vs DAGs vs Agents, the Anthropic Agency Spectrum, POMDP state machines, agent loops, halting invariants, and non-determinism.',
    learningObjectives: [
      'Distinguish between deterministic workflows, DAG pipelines, and true autonomous agents',
      'Model agent interaction formally as a Partially Observable Markov Decision Process (POMDP)',
      'Design unbreakable halting invariants: step horizons, token budgets, and cycle detectors'
    ],
    prerequisites: ['Level 6 Advanced & Agentic RAG'],
    theory: {
      definition: 'An AI Agent is an autonomous entity that perceives its environment through observations, maintains internal state, makes decisions via a reasoning policy (LLM), and executes actions using tools to achieve a specified goal over multiple iterative steps.',
      intuition: 'A workflow is a train on a track: it follows a fixed sequence of stops, and if a branch is blocked, it throws an error. An agent is a taxi driver: given a destination, it checks traffic, takes detours when a road is closed, asks for clarification when lost, and navigates obstacles dynamically until arriving at the destination.',
      technicalExplanation: 'Anthropic\'s "Building Effective Agents" (2024) formalizes the agency spectrum: (1) Basic Prompt -> (2) Routing / Branching -> (3) Parallelization -> (4) Orchestrator-Workers -> (5) Autonomous Evaluator-Optimizer Loop -> (6) Open-Ended Autonomous Agent. Formally, an agent operates within a Partially Observable Markov Decision Process (POMDP): M = <S, A, T, R, Omega, O, gamma>. Because the agent cannot observe the entire world state s_t, it maintains a belief state b_t in its autoregressive context window: b_t = P(s_t | o_1:t, a_1:t-1). Without strict halting invariants (finite step horizons, financial budgets, action hashing), agents risk entering non-terminating loops.',
      mathematics: {
        formula: 'b_{t+1}(s\') = \\eta \\cdot \\mathcal{O}(o_{t+1} \\mid s\', a_t) \\sum_{s \\in \\mathcal{S}} \\mathcal{T}(s\' \\mid s, a_t) b_t(s)',
        variables: [
          { name: 'b_t(s)', desc: 'Belief state distribution over environment states s at step t' },
          { name: 'a_t', desc: 'Action chosen by the agent policy' },
          { name: 'o_{t+1}', desc: 'Observation returned from tool execution' },
          { name: 'T', desc: 'State transition probability distribution' },
          { name: 'O', desc: 'Observation emission probability distribution' }
        ]
      },
      example: 'User asks: "Check if the server is down. If it is, restart the docker container and ping me on Slack." A deterministic script breaks if the container name changed. An agent runs `docker ps`, observes the real container name `api-prod-v2`, restarts it, verifies health, and sends the notification.'
    },
    implementation: {
      language: 'Python',
      code: `class MinimalAgentLoop:
    def __init__(self, policy_llm, environment_tools, max_steps: int = 5):
        self.policy = policy_llm
        self.tools = environment_tools
        self.max_steps = max_steps

    def execute(self, goal: str) -> dict:
        history = [f"Goal: {goal}"]
        for step in range(1, self.max_steps + 1):
            decision = self.policy.decide_next_step(history)
            if decision["type"] == "FINAL_ANSWER":
                return {"status": "SUCCESS", "answer": decision["content"], "steps": step}
            
            # Execute tool action
            action_name = decision["tool"]
            args = decision["arguments"]
            observation = self.tools[action_name](**args)
            history.append(f"Action {step}: {action_name}({args}) -> Observation: {observation}")
        return {"status": "HALTED_STEP_LIMIT", "history": history}`,
      explanation: 'The fundamental autonomous Sense-Plan-Act execution loop with step cap invariants.'
    },
    failureModes: [
      'Infinite Cyclic Loops: The agent repeats the exact same failing tool call continuously',
      'Goal Drift: After 8 steps, the agent loses track of the original user prompt and solves irrelevant sub-problems'
    ],
    engineeringTradeoffs: [
      'Workflows vs Agents: Deterministic DAG workflows achieve >99% reliability with low latency; autonomous agents solve unpredictable open-ended tasks but their error rate compounds exponentially with step count.'
    ],
    exercise: {
      prompt: 'If an agent has a 92% per-step success rate, calculate the probability that it successfully completes an 8-step autonomous task.',
      hint: 'P(complete) = (0.92)^8.',
      solution: '(0.92)^8 = 0.5132 (51.3%). This proves why production systems minimize agent step count or use deterministic DAGs wherever possible.'
    },
    quiz: [
      {
        question: 'Why must autonomous agents always have hard step-count and token-budget limits in production?',
        options: [
          'Because Python cannot run loops longer than 10 iterations.',
          'To prevent runaway infinite loops, budget exhaustion, and hanging processes when agents fail to reach stopping conditions.',
          'Because LLM context windows only support 10 tokens.',
          'To speed up network connections.'
        ],
        correctIndex: 1,
        explanation: 'Due to non-deterministic model completions and unexpected tool errors, unconstrained agent loops can easily loop indefinitely, draining thousands of dollars in API credits and freezing server resources.'
      }
    ]
  },
  {
    id: 18,
    slug: 'tool-calling-function-execution',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '18. Tool Calling, Structured Outputs & Safe Execution',
    duration: '4 Hours',
    description: 'JSON Schema generation, Grammar-Constrained Decoding (Outlines, vLLM), sandboxing (Docker, Firecracker), error retries, and idempotency.',
    learningObjectives: [
      'Understand how Grammar-Constrained Decoding (GCD) uses FSM logit masking to eliminate JSON syntax errors',
      'Build dynamic tool registries from native Python type hints and Pydantic schemas',
      'Isolate tool execution inside ephemeral sandboxes to prevent remote code execution attacks'
    ],
    prerequisites: ['Level 17 AI Agents: Formal Definitions, Agency Spectrum & Loops'],
    theory: {
      definition: 'The protocol and infrastructure that enables LLMs to emit structured, validated tool invocations and safely execute them in isolated computational environments.',
      intuition: 'An LLM cannot directly talk to your database or run Python code. Tool calling is the translator: the model generates a precise JSON payload adhering to a contract, your code runs the tool safely in a sandbox, and hands the result back to the model as text.',
      technicalExplanation: 'Traditional tool use used prompt engineering ("Please output JSON: ...") which frequently produced malformed syntax, broken quotes, and missing fields. Modern tool calling utilizes Grammar-Constrained Decoding (GCD, Willard & Louf 2023 / Outlines). GCD compiles a Pydantic JSON schema into a Finite State Machine (FSM). At each token step t, the FSM determines the exact subset of vocabulary tokens that maintain valid JSON syntax, setting the logits of all invalid tokens to -inf: P_constrained(w_t | w_<t). Syntax errors drop to exactly 0.0% with <1.5ms overhead. For execution, tools with side effects must be idempotent and isolated inside microVMs (Firecracker) or containers with restricted syscalls (seccomp).',
      mathematics: {
        formula: 'z_{t, i}^{\\text{masked}} = \\begin{cases} z_{t, i}, & \\text{if } w_i \\in \\text{ValidNextTokens}(\\mathcal{FSM}, w_{<t}) \\\\ -\\infty, & \\text{otherwise} \\end{cases}',
        variables: [
          { name: 'z_{t, i}', desc: 'Raw model logit for vocabulary token i at step t' },
          { name: 'FSM', desc: 'Finite State Machine compiled from target JSON Schema' },
          { name: 'ValidNextTokens', desc: 'Grammar transition function returning allowed token IDs' }
        ]
      },
      example: 'When asking a weather tool, the LLM emits `{"city": "Paris", "units": "celsius"}`. If the model attempts to generate an unquoted string or missing bracket, the FSM masks those tokens, making syntax violations mathematically impossible.'
    },
    implementation: {
      language: 'Python',
      code: `import json
from pydantic import BaseModel, Field

class StockPriceLookup(BaseModel):
    ticker: str = Field(..., description="Stock symbol, e.g. AAPL")
    include_after_hours: bool = Field(default=False)

def mock_execute_tool(raw_json_str: str) -> str:
    try:
        # Validate against Pydantic contract
        payload = StockPriceLookup.model_validate_json(raw_json_str)
        return f"SUCCESS: Price of {payload.ticker.upper()} is $182.40"
    except Exception as err:
        # Structured error feedback for agent reflection
        return f"TOOL_VALIDATION_ERROR: {str(err)}. Correct format: {StockPriceLookup.model_json_schema()}"

valid_call = '{"ticker": "NVDA", "include_after_hours": true}'
invalid_call = '{"ticker": 12345}'
print("Call 1:", mock_execute_tool(valid_call))
print("Call 2:", mock_execute_tool(invalid_call))`,
      explanation: 'Pydantic contract validation with structured error feedback for LLM self-correction.'
    },
    failureModes: [
      'Unsandboxed code execution: Running `eval()` or `subprocess.run()` without container isolation allows prompt injection to run `rm -rf /` or steal AWS credentials',
      'Non-idempotent tool retries: Retrying a failed charge_credit_card tool call multiple times, double-billing the customer'
    ],
    engineeringTradeoffs: [
      'Schema size vs Context Window: Giving an agent 50 detailed tool schemas consumes 3,000+ tokens on every turn. Solution: Use Retrieval-Augmented Tool Selection (RATS) to inject only the top-5 relevant tool schemas.'
    ],
    exercise: {
      prompt: 'Design an idempotent API wrapper for a tool that creates a customer support ticket.',
      hint: 'Include an idempotency key generated by hashing the user prompt and timestamp.',
      solution: `def create_ticket(title: str, body: str, idempotency_key: str):
    if idempotency_key in processed_keys:
        return existing_ticket_cache[idempotency_key]
    ticket = db.insert(title, body)
    existing_ticket_cache[idempotency_key] = ticket
    return ticket`
    },
    quiz: [
      {
        question: 'How does Grammar-Constrained Decoding (GCD) eliminate JSON syntax errors during LLM generation?',
        options: [
          'It retries the query until the JSON is valid.',
          'It compiles the schema into an FSM and sets logits of invalid syntax tokens to -infinity at each sampling step.',
          'It fine-tunes the model on 10 million JSON files.',
          'It converts JSON to XML before generation.'
        ],
        correctIndex: 1,
        explanation: 'GCD constructs a pushdown automaton or finite state machine from the schema and masks invalid token logits at the vocabulary projection layer, ensuring every emitted token preserves valid syntax.'
      }
    ]
  },
  {
    id: 19,
    slug: 'planning-reasoning-paradigms',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '19. Planning & Reasoning: CoT, ReAct, Reflexion & LATS',
    duration: '4.5 Hours',
    description: 'Chain-of-Thought (Wei 2022), ReAct (Yao 2022), Reflexion (Shinn 2023), Tree of Thoughts (Yao 2023), Plan-and-Solve, and Language Agent Tree Search (LATS).',
    learningObjectives: [
      'Implement the ReAct Thought-Action-Observation loop with stateful scratchpads',
      'Apply Reflexion verbal reinforcement learning to self-correct from past trajectory failures',
      'Understand Language Agent Tree Search (LATS) combining Monte Carlo Tree Search with LLM reflection'
    ],
    prerequisites: ['Level 18 Tool Calling, Structured Outputs & Safe Execution'],
    theory: {
      definition: 'The cognitive architectures that structure how an agent decomposes goals into actionable sub-tasks, reflects on intermediate results, and searches for optimal decision paths.',
      intuition: 'If you ask someone to immediately answer a complex puzzle, they make mistakes. If you tell them: "Think out loud, write down your steps, check your work after each step, and if something looks wrong, cross it out and try a different angle," their accuracy skyrockets. That is reasoning and planning.',
      technicalExplanation: 'Paradigms: (1) Chain-of-Thought (CoT, Wei et al. 2022): eliciting step-by-step reasoning via few-shot exemplars or "Think step-by-step". (2) ReAct (Yao et al. 2022): interleaves verbal Reasoning Traces (Thoughts) with task-specific Actions and environment Observations. (3) Reflexion (Shinn et al. 2023): evaluates failed trajectories and appends explicit verbal critiques into episodic memory to guide subsequent trials (improving HumanEval from 68% to 91%). (4) Tree of Thoughts (ToT, Yao et al. 2023): explores multiple reasoning paths via BFS/DFS, evaluating each thought state with heuristic scores. (5) Language Agent Tree Search (LATS, Zhou et al. 2024): unifies LLM reasoning with Monte Carlo Tree Search (MCTS), using value functions, rollouts, and backpropagation of reward.',
      mathematics: {
        formula: 'UCT(\\text{node}) = \\frac{Q(\\text{node})}{N(\\text{node})} + c \\sqrt{\\frac{\\ln N(\\text{parent})}{N(\\text{node})}}',
        variables: [
          { name: 'Q(node)', desc: 'Accumulated evaluation reward of the reasoning node' },
          { name: 'N(node)', desc: 'Visit count of the current reasoning state' },
          { name: 'c', desc: 'Exploration vs exploitation constant (typically sqrt(2))' }
        ]
      },
      example: 'In mathematical problem solving, standard prompting fails on: "A store has 15 apples, sells 6, gets a shipment of 20, and 3 rot. How many are left?" ReAct generates: Thought 1: Start with 15. Action 1: Subtract 6 (9). Thought 2: Add 20 (29). Thought 3: Subtract 3 (26). Final Answer: 26.'
    },
    implementation: {
      language: 'Python',
      code: `class ReflexionAgent:
    """Agent using verbal self-reflection memory across attempts."""
    def __init__(self, evaluator_fn):
        self.reflections: list[str] = []
        self.evaluator = evaluator_fn

    def solve(self, task: str, max_trials: int = 3) -> str:
        for trial in range(1, max_trials + 1):
            # Formulate prompt including previous failure critiques:
            ref_context = "\\n".join([f"- Previous mistake: {r}" for r in self.reflections])
            attempt = f"Solution attempt for: {task} (incorporating:\\n{ref_context})"
            
            # Evaluate solution:
            passed, critique = self.evaluator(attempt)
            if passed:
                return f"SUCCESS (Trial {trial}): {attempt}"
            self.reflections.append(critique)
        return "FAILED_ALL_TRIALS"

# Simple evaluator that demands 'def fibonacci'
eval_fn = lambda code: (True, "") if "def fibonacci" in code else (False, "Code failed to declare 'def fibonacci'")
agent = ReflexionAgent(eval_fn)
print(agent.solve("Write a fibonacci sequence generator"))`,
      explanation: 'Reflexion pattern capturing verbal critiques to improve subsequent execution trials.'
    },
    failureModes: [
      'Hallucinated verification: An agent claims its code passed unit tests when it never actually executed the test runner',
      'ToT exponential state explosion: Generating 5 thoughts per branch with depth 5 requires 3,125 LLM evaluations!'
    ],
    engineeringTradeoffs: [
      'ReAct vs LATS: ReAct is fast (1-3s, 1 LLM call per step) and handles 90% of business tasks; LATS achieves higher SOTA on competitive coding/math but takes 60-180s and costs 20x-50x more tokens.'
    ],
    exercise: {
      prompt: 'Write out the trajectory steps of a ReAct agent tasked with finding the age difference between two historical figures.',
      hint: 'Deconstruct into: Thought 1, Action 1, Observation 1, Thought 2, Action 2, Observation 2, Thought 3, Final Answer.',
      solution: `Goal: Age difference between Ada Lovelace and Alan Turing.
Thought 1: Look up birth year of Ada Lovelace.
Action 1: search("Ada Lovelace birth year") -> Obs 1: Born Dec 10, 1815.
Thought 2: Look up birth year of Alan Turing.
Action 2: search("Alan Turing birth year") -> Obs 2: Born June 23, 1912.
Thought 3: Calculate difference 1912 - 1815 = 97 years.
Final Answer: The age difference is 96 years and approximately 6 months (born 97 years apart).`
    },
    quiz: [
      {
        question: 'What is the core insight of the Reflexion framework (Shinn et al., 2023)?',
        options: [
          'It updates neural network weights using reinforcement learning.',
          'It uses verbal self-reflection stored in episodic memory to guide future trials without updating model weights.',
          'It forces models to output pure Python code.',
          'It replaces LLMs with decision trees.'
        ],
        correctIndex: 1,
        explanation: 'Reflexion demonstrates that language models can perform verbal reinforcement learning: by analyzing past execution errors and storing textual critiques in memory, the model avoids repeating identical mistakes across subsequent trials.'
      }
    ]
  },
  {
    id: 20,
    slug: 'agent-memory-architectures',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '20. Agent Memory: Working, Episodic, Semantic & MemGPT',
    duration: '4 Hours',
    description: 'Working memory, token sliding windows, summarization, episodic memory retrieval (Park et al.), semantic knowledge graphs, and MemGPT/Letta OS memory paging.',
    learningObjectives: [
      'Architect a multi-tier memory system separating Working Memory (RAM) from Archival Memory (Disk)',
      'Implement the Generative Agents tripartite retrieval formula (Recency, Importance, Relevance)',
      'Understand the MemGPT/Letta operating system architecture for perpetual state agents'
    ],
    prerequisites: ['Level 19 Planning & Reasoning: CoT, ReAct, Reflexion & LATS'],
    theory: {
      definition: 'The persistent data structures and retrieval mechanisms that allow autonomous agents to store, update, recall, and consolidate knowledge across long temporal horizons.',
      intuition: 'Humans do not keep every conversation they have had in active working memory. We keep the current discussion in mind (working memory), recall past events when relevant (episodic memory), store permanent facts like our home address (semantic memory), and know how to drive a car (procedural memory). Agents require the exact same architecture.',
      technicalExplanation: 'Memory taxonomy: (1) Working Memory (In-Context RAM): The immediate LLM context window. Managed via token sliding windows and dynamic summarize-on-overflow. (2) Episodic Memory: Records past interactions and tool outputs with timestamps. Scored for retrieval using the Generative Agents formula (Park et al., 2023): Score(m) = a*Recency + b*Importance + c*Relevance. (3) Semantic Memory: Core factual profiles and knowledge graph triplets. (4) Procedural Memory: Tool definitions, few-shot exemplars, and stored executable code workflows. (5) MemGPT (Packer et al., 2023 / Letta): Models the LLM as an operating system. Core memory is permanently resident in the prompt; Archival and Recall memory live in external vector and relational databases, accessed via self-directed paging function calls (`core_memory_append`, `archival_memory_search`).',
      mathematics: {
        formula: 'S(m, q) = \\alpha_{\\text{rec}} e^{-\\lambda \\Delta t} + \\alpha_{\\text{imp}} I(m) + \\alpha_{\\text{rel}} \\cos(\\mathbf{e}_m, \\mathbf{e}_q)',
        variables: [
          { name: 'Delta t', desc: 'Time elapsed since memory creation in hours/days' },
          { name: 'lambda', desc: 'Exponential decay rate constant' },
          { name: 'I(m)', desc: 'Normalized importance score of the memory (1 to 10)' },
          { name: 'cos(e_m, e_q)', desc: 'Cosine similarity between memory embedding and query' }
        ]
      },
      example: 'In a personal assistant, the user mentions: "My daughter is allergic to peanuts" 3 months ago. This fact is archived. Today the user asks: "Find a Thai restaurant near me." The agent retrieves the peanut allergy episodic memory and automatically checks peanut oil safety in recommendations.'
    },
    implementation: {
      language: 'Python',
      code: `import time
import math

class HierarchicalAgentMemory:
    def __init__(self, working_limit: int = 3):
        self.working_memory: list[dict] = []
        self.archival_memory: list[dict] = []
        self.working_limit = working_limit

    def add_message(self, role: str, text: str, importance: float = 1.0):
        msg = {"role": role, "text": text, "timestamp": time.time(), "importance": importance}
        self.working_memory.append(msg)
        if len(self.working_memory) > self.working_limit:
            # Evict oldest message to archival vector storage:
            evicted = self.working_memory.pop(0)
            self.archival_memory.append(evicted)

    def retrieve_memories(self, query: str) -> list[str]:
        # Scored via recency and lexical relevance:
        now = time.time()
        scored = []
        for m in self.archival_memory:
            dt = now - m["timestamp"]
            recency = math.exp(-0.01 * dt)
            overlap = sum(1 for w in query.lower().split() if w in m["text"].lower())
            score = 0.5 * recency + 0.5 * overlap
            scored.append((score, m["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:2]]`,
      explanation: 'Hierarchical agent memory managing working context limits and archival eviction.'
    },
    failureModes: [
      'Memory Contradiction: Outdated memories (e.g. "I live in Chicago") overriding updated current facts ("I moved to London")',
      'Context Thrashing: Repeatedly paging memories into and out of working context across multi-turn queries'
    ],
    engineeringTradeoffs: [
      'In-Context Long-Term Storage vs Vector Retrieval: Putting 100k tokens of chat history into a long context window costs $0.30 per query and adds 5s latency; hierarchical vector memory retrieves only top-3 relevant facts for $0.005 in 15ms.'
    ],
    exercise: {
      prompt: 'How would you handle GDPR "Right to be Forgotten" requests in an agent with episodic vector memory?',
      hint: 'Every memory object must include user_id metadata and a deletion tombstone API.',
      solution: 'All memory chunks must store user_id as a primary indexed metadata field. Upon receiving a deletion request, the system executes a hard metadata deletion query (`delete where user_id = X`) in the vector database and purges all cached summaries and checkpoints.'
    },
    quiz: [
      {
        question: 'In the MemGPT / Letta architecture, how does the agent interact with its long-term archival memory?',
        options: [
          'All memories are loaded into the prompt on every turn.',
          'The agent issues autonomous function calls (e.g. archival_memory_search) to page relevant chunks into working memory when needed.',
          'The operating system restarts the model.',
          'Memories are permanently hardcoded into the model weights.'
        ],
        correctIndex: 1,
        explanation: 'MemGPT treats the LLM as an OS: the model autonomously decides when to invoke memory search functions to page historical facts from external disk storage into its active working context.'
      }
    ]
  },
  {
    id: 21,
    slug: 'agent-architectures-stategraphs',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '21. Agent Topologies: Routers, Supervisors & LangGraph StateGraphs',
    duration: '4 Hours',
    description: 'Single-agent patterns, Router/Dispatcher, Planner-Executor, StateGraph state machines, cyclic transitions, persistence, and checkpointing.',
    learningObjectives: [
      'Construct deterministic StateGraph agents with cyclic loops and conditional routing',
      'Implement state persistence and time-travel debugging with checkpoint savers',
      'Compare LangGraph StateGraphs against raw ReAct loops for enterprise predictability'
    ],
    prerequisites: ['Level 20 Agent Memory: Working, Episodic, Semantic & MemGPT'],
    theory: {
      definition: 'The structural wiring and flow control that dictates how an agent traverses intermediate states, executes conditional branches, and persists execution history.',
      intuition: 'A simple agent is a single loop. An enterprise agent is a state machine: a flowchart with defined rooms (nodes) and doors (edges). You can only go from Room A to Room B if specific criteria are met, and you can save your progress at any door to resume later or undo a mistake.',
      technicalExplanation: 'Single-loop ReAct agents struggle with complex branching, human-in-the-loop approvals, and recovery. LangGraph formalizes agents as Cyclic State Graphs: Nodes are pure or impure functions f: State -> Delta State that transform shared state. Edges are deterministic transitions or conditional router functions. The state schema uses reducer annotations (e.g. `operator.add` for message append). Checkpointers (Postgres/Redis) snapshot state after every node execution, enabling: (1) Human-in-the-loop pauses, (2) Fault-tolerant resume after server crashes, and (3) Time-travel debugging (replaying execution from step 3 with modified inputs).',
      example: 'A customer support bot enters node `triage`. If the user wants a refund > $100, the conditional edge routes to node `human_approval`, pausing execution until a manager clicks "Approve" in an admin dashboard.'
    },
    implementation: {
      language: 'Python',
      code: `class StateGraphEngine:
    """Minimal deterministic StateGraph implementation."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, name: str, fn):
        self.nodes[name] = fn

    def add_conditional_edges(self, source: str, router_fn, mapping: dict):
        self.edges[source] = (router_fn, mapping)

    def run(self, initial_state: dict, entry_point: str, max_steps: int = 5) -> dict:
        state = initial_state.copy()
        curr = entry_point
        for _ in range(max_steps):
            if curr == "END": break
            # Execute node:
            state = self.nodes[curr](state)
            if curr in self.edges:
                router, mapping = self.edges[curr]
                decision = router(state)
                curr = mapping[decision]
            else:
                curr = "END"
        return state

# Example usage:
graph = StateGraphEngine()
graph.add_node("analyze", lambda s: {**s, "score": 0.95})
graph.add_conditional_edges(
    "analyze",
    lambda s: "PASS" if s["score"] >= 0.8 else "FAIL",
    {"PASS": "approved", "FAIL": "rejected"}
)
graph.add_node("approved", lambda s: {**s, "status": "APPROVED"})
graph.add_node("rejected", lambda s: {**s, "status": "REJECTED"})
final = graph.run({"input": "audit"}, entry_point="analyze")
print("Final State:", final)`,
      explanation: 'Deterministic state-machine flow with conditional branching and state reducers.'
    },
    failureModes: [
      'State Mutation Race Conditions: Concurrent nodes mutating non-isolated state variables simultaneously',
      'Unchecked cycles: A conditional edge routing back to an earlier node creating an infinite loop'
    ],
    engineeringTradeoffs: [
      'LangGraph StateGraph vs Raw Python script: StateGraph introduces dependency abstractions but provides automated checkpointing, observability tracing, and human-in-the-loop pauses.'
    ],
    exercise: {
      prompt: 'Diagram the nodes and conditional edges for a code-generation agent that loops up to 3 times if tests fail.',
      hint: 'Nodes: Draft, TestRunner. Conditional Edge: If tests pass -> END; if tests fail and count < 3 -> Draft.',
      solution: `Nodes: [GenerateCode] -> [RunUnitTests]
Conditional Edge from [RunUnitTests]:
  - If exit_code == 0 -> [END]
  - If exit_code != 0 and retry_count < 3 -> [GenerateCode] (increment retry_count)
  - If retry_count >= 3 -> [EscalateToHuman]`
    },
    quiz: [
      {
        question: 'What is the primary benefit of state checkpointing in graph-based agent architectures like LangGraph?',
        options: [
          'It compiles Python to C++.',
          'It saves a snapshot of the agent state at every step, enabling fault tolerance, time-travel debugging, and human-in-the-loop pauses.',
          'It deletes previous chat history to save RAM.',
          'It encrypts the model prompt.'
        ],
        correctIndex: 1,
        explanation: 'Checkpointing persists state transitions to disk/database after every node execution. If a process crashes or requires human manager approval, execution can be seamlessly resumed from the exact checkpoint.'
      }
    ]
  },
  {
    id: 22,
    slug: 'agentic-rag-architectures',
    level: 7,
    levelName: 'Level 7: Agentic AI & Autonomous Systems',
    title: '22. Agentic RAG: Dynamic Retrieval & Iterative Reasoning',
    duration: '4.5 Hours',
    description: 'Routing agents, query decomposition, active iterative retrieval, tool-augmented verification, and handling multi-step reasoning over complex corpora.',
    learningObjectives: [
      'Build an Agentic RAG pipeline that decomposes multi-hop queries into sub-questions',
      'Implement active retrieval where the agent dynamically chooses *when* and *what* to retrieve',
      'Verify retrieved documents and autonomously execute web search fallbacks'
    ],
    prerequisites: ['Level 21 Agent Topologies: Routers, Supervisors & LangGraph StateGraphs'],
    theory: {
      definition: 'An advanced RAG architecture where an autonomous agent drives the retrieval process dynamically: formulating search queries, evaluating retrieved passage sufficiency, executing multi-hop lookups, and verifying answers.',
      intuition: 'Standard RAG does one retrieval pass: Question in -> Top-3 chunks out -> Answer. But what if the question is: "Did the founder of the company that created ChatGPT graduate from the same university as the founder of Tesla?" Standard RAG fails. Agentic RAG breaks it into: Step 1: Who founded OpenAI? (Sam Altman). Step 2: Where did he go to college? (Stanford). Step 3: Who founded Tesla? (Elon Musk). Step 4: Where did he go? (Penn). Step 5: Compare and answer.',
      technicalExplanation: 'Agentic RAG transforms retrieval from a passive data-pipe into an active tool set: `vector_search()`, `keyword_search()`, `document_lookup()`, and `web_search()`. The agent policy plans sub-queries, inspects intermediate findings, detects missing information, reformulates search terms, and aggregates evidence across multiple independent documents. A document grading node checks whether evidence is sufficient to answer the prompt; if not, it queries alternative sources.',
      example: 'In financial analysis: "Compare Apple\'s Q4 2024 gross margin to Microsoft\'s." Agentic RAG queries Apple\'s 10-K, extracts 46.2%, then queries Microsoft\'s 10-K, extracts 69.4%, computes the delta (23.2%), and outputs the synthesis.'
    },
    implementation: {
      language: 'Python',
      code: `class AgenticRAGOrchestrator:
    def __init__(self, internal_kb, web_kb):
        self.internal_kb = internal_kb
        self.web_kb = web_kb

    def answer_multihop(self, complex_query: str) -> str:
        # Step 1: Query decomposition
        sub_questions = [
            "What company created ChatGPT and who is its CEO?",
            "Who is the CEO of Tesla?",
            "Compare the universities attended by both CEOs."
        ]
        evidence = {}
        for q in sub_questions[:2]:
            docs = self.internal_kb(q)
            if not docs:
                docs = self.web_kb(q) # Fallback
            evidence[q] = docs[0] if docs else "No data"
        
        # Step 2: Final synthesis
        return f"Synthesized from evidence: {evidence}"`,
      explanation: 'Multi-hop iterative query decomposition in Agentic RAG.'
    },
    failureModes: [
      'Multi-hop Error Compounding: If Step 1 returns the wrong CEO, all subsequent retrieval steps branch into erroneous domains',
      'Token budget explosion on recursive decomposition'
    ],
    engineeringTradeoffs: [
      'Standard RAG vs Agentic RAG: Standard RAG takes 400ms at $0.005; Agentic RAG takes 3-8s and costs $0.05, but is capable of solving multi-hop relational questions that standard RAG completely fails on.'
    ],
    exercise: {
      prompt: 'Identify the 3 sub-queries required to answer: "Is the current prime minister of the UK younger than the current French president?"',
      hint: 'Find identity and age of Person A, Person B, then compare.',
      solution: '1. "Who is the current Prime Minister of the United Kingdom and what is their date of birth?" 2. "Who is the current President of France and what is their date of birth?" 3. "Compare the ages of [PM Name] and [President Name]."'
    },
    quiz: [
      {
        question: 'What is the primary operational difference between Naive RAG and Agentic RAG?',
        options: [
          'Naive RAG uses Python; Agentic RAG uses C++.',
          'Naive RAG executes a single static retrieval step; Agentic RAG dynamically plans, evaluates, rewrites queries, and performs multi-hop lookups iteratively.',
          'Agentic RAG does not use vector databases.',
          'Naive RAG has zero hallucinations.'
        ],
        correctIndex: 1,
        explanation: 'Agentic RAG treats retrieval as an active, iterative tool that an agent policy invokes dynamically across multiple turns, enabling multi-hop reasoning and self-correcting fallbacks.'
      }
    ]
  },
  {
    id: 23,
    slug: 'multi-agent-systems-coordination',
    level: 8,
    levelName: 'Level 8: Multi-Agent Systems & Infrastructure',
    title: '23. Multi-Agent Systems: Coordination, Protocols & Failure Cascades',
    duration: '4 Hours',
    description: 'Supervisor-Worker, Swarm, Multi-Agent Debate (Du 2023), message protocols (A2A JSON), shared vs isolated state, compounding errors, and benchmark realities.',
    learningObjectives: [
      'Architect Supervisor-Worker and Peer-to-Peer Swarm multi-agent topologies',
      'Analyze the compounding error equation P = prod(p_i) and why single agents often beat multi-agent teams',
      'Implement multi-agent debate protocols with independent verification guards'
    ],
    prerequisites: ['Level 22 Agentic RAG: Dynamic Retrieval & Iterative Reasoning'],
    theory: {
      definition: 'A distributed system where multiple specialized AI agents interact, communicate via structured protocols, delegate tasks, and collaborate to solve complex problems.',
      intuition: 'Building software requires a frontend engineer, a backend engineer, and a QA reviewer. A multi-agent system assigns specialized personas and tools to different agents, passing tasks between them like a software team.',
      technicalExplanation: 'Multi-Agent Topologies: (1) Supervisor-Worker (Hub-and-Spoke): A central supervisor routes tasks, aggregates worker outputs, and enforces governance. (2) Swarm / Peer-to-Peer: Agents pass control dynamically via handoff functions (OpenAI Swarm). (3) Multi-Agent Debate (Du et al., 2023): Multiple agents generate independent solutions, critique peer solutions, and vote to reach consensus. Critical Reality: Multi-agent systems suffer from compounding errors: if each agent is 90% reliable, a 5-agent handoff chain has reliability (0.90)^5 = 59.0%. Without rigid communication envelopes and isolated mailboxes, agents enter sycophancy cascades where errors are confirmed rather than corrected.',
      mathematics: {
        formula: 'P_{\\text{system}} = \\prod_{i=1}^N p_i, \\quad \\text{If } p_i = 0.90, N = 10 \\implies P = 0.90^{10} = 0.3486 \\; (34.8\\%)',
        variables: [
          { name: 'p_i', desc: 'Probability of individual agent i executing its sub-task correctly' },
          { name: 'N', desc: 'Number of sequential agent handoffs' },
          { name: 'P_system', desc: 'Total probability of flawless end-to-end task completion' }
        ]
      },
      example: 'A code refactoring team: Agent A (Static Analyzer) flags lint errors -> Agent B (Coder) generates fix -> Agent C (Test Runner) runs pytest in a container. If tests fail, Agent C sends test logs back to Agent B for revision.'
    },
    implementation: {
      language: 'Python',
      code: `class SupervisorTeam:
    def __init__(self, researcher_agent, reviewer_agent):
        self.researcher = researcher_agent
        self.reviewer = reviewer_agent

    def execute_task(self, prompt: str) -> dict:
        # Step 1: Worker executes task
        draft = self.researcher(prompt)
        
        # Step 2: Independent critic evaluates draft
        critique = self.reviewer(draft)
        if critique["verdict"] == "REJECTED":
            # Targeted revision:
            draft = self.researcher(f"{prompt}\\nAddress critique: {critique['feedback']}")
        return {"final_output": draft, "review": critique}`,
      explanation: 'Supervisor pattern coordinating worker and reviewer agents.'
    },
    failureModes: [
      'Sycophancy cascade: Agent B blindly praises and confirms a hallucination made by Agent A',
      'Semantic telephone game: Information loss across 5 agent handoffs distorts the original user instructions'
    ],
    engineeringTradeoffs: [
      'Single-Agent with tools vs Multi-Agent Swarm: Benchmarks (SWE-bench) show single-agent scaffolding (SWE-agent) consistently outperforms multi-agent swarms at 1/5th the token cost and latency.'
    ],
    exercise: {
      prompt: 'Why do unconstrained multi-agent debate protocols often fail to correct errors in small models?',
      hint: 'Consider peer pressure and model agreement bias (sycophancy).',
      solution: 'Small models lack strong internal calibrated confidence. When confronted with an assertive peer agent (even one asserting a falsehood), the second model frequently defers and conforms to the peer assertion rather than critically re-evaluating the underlying mathematical or factual truth.'
    },
    quiz: [
      {
        question: 'What is the primary risk of increasing the number of communicating agents in a sequential multi-agent pipeline?',
        options: [
          'Python memory leaks.',
          'Compounding error decay: overall reliability drops exponentially as P = prod(p_i), while token costs scale linearly with each agent turn.',
          'Database locks.',
          'Embeddings become negative.'
        ],
        correctIndex: 1,
        explanation: 'Each autonomous agent handoff introduces an independent failure probability. At N=10 steps with 90% per-step accuracy, overall completion probability drops to 34.8%, while latency and cost multiply by 10x.'
      }
    ]
  },
  {
    id: 24,
    slug: 'mcp-agent-infrastructure',
    level: 8,
    levelName: 'Level 8: Multi-Agent Systems & Infrastructure',
    title: '24. Model Context Protocol (MCP) & Modern Agent Infrastructure',
    duration: '4 Hours',
    description: 'Anthropic Model Context Protocol (MCP 2024), JSON-RPC 2.0 specs, stdio vs SSE transports, Tools, Resources, Prompts, security sandboxing, and E2B/Firecracker runtimes.',
    learningObjectives: [
      'Master the Model Context Protocol (MCP) specification: Host-Client-Server topology',
      'Implement an MCP Server exposing Tools (`tools/call`) and Resources (`resources/read`) over stdio and SSE',
      'Compare MCP against OpenAPI plugins, OpenAI function calling, and raw REST APIs'
    ],
    prerequisites: ['Level 23 Multi-Agent Systems: Coordination, Protocols & Failure Cascades'],
    theory: {
      definition: 'The open standard released by Anthropic (Nov 2024) enabling secure, bidirectional, standardized communication between AI models/hosts and local or remote data sources, tools, and prompts.',
      intuition: 'Before USB, every computer accessory had a proprietary cable (serial, parallel, PS/2). Before MCP, every AI tool integration required custom SDK wrappers for LangChain, LlamaIndex, OpenAI, and Claude. MCP is the "USB-C for AI": write your tool or data source once as an MCP server, and every AI agent, IDE (Cursor), and model can connect to it instantly.',
      technicalExplanation: 'MCP operates as a client-server architecture using JSON-RPC 2.0 framing. The Host (e.g. Claude Desktop, Antigravity) runs an MCP Client that connects to one or more MCP Servers. Transports: (1) `stdio`: Standard input/output for local, ultra-fast inter-process communication (IPC). (2) `SSE` (Server-Sent Events) over HTTP: for remote, distributed cloud servers. MCP Primitives: (1) Tools (`tools/list`, `tools/call`): Model-controlled executable functions with JSON schemas. (2) Resources (`resources/list`, `resources/read`): Application-controlled context data (files, database tables). (3) Prompts (`prompts/list`, `prompts/get`): User-controlled reusable templates. (4) Sampling: Enables the server to request LLM completions back from the host client.',
      mathematics: {
        formula: '\\text{Complexity Reduction: } \\mathcal{O}(M \\times N) \\xrightarrow{\\text{MCP}} \\mathcal{O}(M + N)',
        variables: [
          { name: 'M', desc: 'Number of AI model client applications' },
          { name: 'N', desc: 'Number of enterprise data tools and storage systems' }
        ]
      },
      example: 'An engineering team builds a `postgres-mcp` server. It instantly works in Cursor, Claude Desktop, Antigravity, and custom internal FastAPI agents without rewriting a single line of integration code.'
    },
    implementation: {
      language: 'Python',
      code: `import json
import sys

def handle_mcp_stdio():
    """Minimal MCP Server responding to tools/list and tools/call over stdio."""
    tools_manifest = {
        "tools": [{
            "name": "get_system_time",
            "description": "Returns current UTC timestamp",
            "inputSchema": {"type": "object", "properties": {}}
        }]
    }
    # Simulated JSON-RPC request handling
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    if req["method"] == "tools/list":
        response = {"jsonrpc": "2.0", "id": req["id"], "result": tools_manifest}
        print("MCP Response:", json.dumps(response))

handle_mcp_stdio()`,
      explanation: 'Minimal JSON-RPC 2.0 handler illustrating the MCP protocol wire format.'
    },
    failureModes: [
      'MCP Prompt Injection via Resources: An untrusted file read via `resources/read` injects jailbreak prompts into the host client context',
      'Unbounded stdio buffer deadlocks when servers emit raw unbuffered debug logs to stdout'
    ],
    engineeringTradeoffs: [
      'stdio vs SSE transport: stdio offers zero-network ultra-fast IPC with native OS process security for local agents; SSE supports distributed cloud deployment but requires HTTPS authentication and network egress management.'
    ],
    exercise: {
      prompt: 'What is the purpose of the roots capability (`roots/list`) in the Model Context Protocol?',
      hint: 'Think about filesystem boundaries and sandboxing.',
      solution: 'The `roots` primitive allows the host client to inform the server about which specific filesystem directories the server is authorized to access, preventing rogue or compromised MCP servers from reading files outside the designated workspace.'
    },
    quiz: [
      {
        question: 'What architectural problem does the Model Context Protocol (MCP) fundamentally solve?',
        options: [
          'It replaces Python with Rust.',
          'It reduces the M x N integration problem between M model clients and N data tools down to an open, standardized M + N protocol.',
          'It speeds up GPU training time.',
          'It eliminates the need for APIs.'
        ],
        correctIndex: 1,
        explanation: 'Before MCP, connecting M model clients to N tools required M x N proprietary integrations. MCP standardizes the transport and schema layer so any model client connects to any tool server (M + N complexity).'
      }
    ]
  }
];
