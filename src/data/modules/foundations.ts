import { Module } from '../../types/curriculum';

export const foundationModules: Module[] = [
  {
    id: 0,
    slug: 'executive-overview',
    level: 0,
    levelName: 'Level 0: Foundation & Orientation',
    title: '0. Executive Overview & The Modern AI Engineering Stack',
    duration: '1.5 Hours',
    description: 'The definitive mental model: from closed-book parametric neural networks to dynamic, grounded, autonomous cognitive architectures.',
    learningObjectives: [
      'Understand the fundamental transition from static pre-trained models to grounded dynamic systems',
      'Deconstruct the four pillars: RAG, Long-Context, Fine-Tuning, and Autonomous Agents',
      'Internalize the Abstraction Tax and develop first-principles engineering intuition',
      'Map the complete curriculum journey from beginner to research-grade architect'
    ],
    prerequisites: ['Basic curiosity about AI systems', 'Familiarity with software concepts'],
    theory: {
      definition: 'Retrieval-Augmented Generation (RAG) and Agentic AI represent the architectural bridge connecting frozen neural network weights (parametric memory) to external, mutable, real-world data and deterministic tools (non-parametric memory).',
      intuition: 'Imagine an open-book exam: an LLM alone relies solely on what it memorized before the test date (which might be outdated or fuzzy). RAG gives the model access to an organized library of textbooks to look up exact facts before answering. An Agent gives the model hands and tools (a calculator, a browser, an API client) to take actions, verify intermediate steps, and solve multi-step problems iteratively.',
      technicalExplanation: 'Pre-trained foundation models capture world knowledge through conditional probability distributions P(w_t | w_<t; theta). However, updating weights theta is computationally prohibitive, slow, and prone to catastrophic forgetting. RAG formalizes an explicit latent variable formulation P(y | x) = sum_z P(z | x) P(y | x, z), conditioning token generation on non-parametric documents z retrieved from an indexed corpus. Agentic AI extends this further by introducing a Partially Observable Markov Decision Process (POMDP), where the model acts as a policy generating actions a_t in response to state beliefs b_t derived from tool observations o_t.',
      mathematics: {
        formula: 'P_{RAG}(y|x; \\theta, \\eta) = \\sum_{z \\in \\text{top-}K(\\mathcal{Z})} P_\\eta(z|x) \\prod_{i=1}^L P_\\theta(y_i | x, z, y_{<i})',
        variables: [
          { name: 'x', desc: 'Input query sequence' },
          { name: 'y', desc: 'Target generated token sequence' },
          { name: 'z', desc: 'Retrieved document passage from corpus Z' },
          { name: 'P_eta(z|x)', desc: 'Probability of selecting passage z given query x via retriever parameters eta' },
          { name: 'P_theta(y|x, z)', desc: 'Probability of generating sequence y conditioned on query x and passage z via generator weights theta' }
        ],
        derivation: 'Derived by Lewis et al. (2020) by treating external documents as latent variables marginalized across sequence-level beam search hypotheses.'
      },
      example: 'When asking "What was Acme Corp\'s Q3 2024 operating margin?", a closed-book model trained in 2023 will hallucinate a plausible number. RAG retrieves the audited 10-Q filing chunk containing "Operating margin was 14.8%", injecting it directly into the prompt context with strict citation instructions.'
    },
    implementation: {
      language: 'Python',
      code: `def rag_mental_model(query: str, retriever, generator) -> dict:
    # 1. Non-Parametric Retrieval:
    relevant_chunks = retriever.search(query, top_k=3)
    
    # 2. Context Injection:
    prompt = f"Context:\\n{relevant_chunks}\\n\\nQuery: {query}\\nAnswer:"
    
    # 3. Parametric Generation:
    answer = generator.generate(prompt)
    return {"answer": answer, "sources": relevant_chunks}`,
      explanation: 'The three-step anatomy of all retrieval-augmented systems: Retrieve from non-parametric index, Inject into context window, Generate grounded output.'
    },
    failureModes: [
      'Retrieval Blindness: The retriever fails to fetch the relevant passage, forcing the generator to guess',
      'Context Distraction: Irrelevant retrieved chunks pollute the context, causing attention dilution',
      'Parametric Override: The model relies on internal memorized biases, ignoring contrary retrieved facts'
    ],
    engineeringTradeoffs: [
      'Latency vs Grounding: Raw LLM calls take ~300ms; adding hybrid retrieval and reranking adds 150-400ms',
      'Cost vs Context: Passing 20 chunks increases input token costs by 5x-10x per query compared to top-3 chunks'
    ],
    exercise: {
      prompt: 'Write down a scenario where an autonomous agent is required instead of a single RAG lookup.',
      hint: 'Think about a task that requires multiple steps, conditional logic, and external tool execution.',
      solution: 'Scenario: "Audit this GitHub repository for outdated dependencies, open a pull request updating vulnerable packages, and verify that unit tests pass in CI." RAG can only look up documentation; an agent must run bash commands, inspect git status, edit files, and invoke the GitHub API.'
    },
    quiz: [
      {
        question: 'What is the primary technical distinction between parametric and non-parametric memory?',
        options: [
          'Parametric memory lives in neural network weights; non-parametric memory lives in external indexable data structures.',
          'Parametric memory is faster to update than non-parametric memory.',
          'Non-parametric memory is stored in RAM; parametric memory is stored on disk.',
          'Parametric memory uses cosine similarity; non-parametric memory uses backpropagation.'
        ],
        correctIndex: 0,
        explanation: 'Parametric memory consists of static weights optimized during training; non-parametric memory consists of external vector/keyword databases that can be updated in milliseconds without model re-training.'
      }
    ]
  },
  {
    id: 1,
    slug: 'prerequisites-environment',
    level: 0,
    levelName: 'Level 0: Foundation & Orientation',
    title: '1. Prerequisites & Production Environment Architecture',
    duration: '2 Hours',
    description: 'Mastering modern Python 3.12+, virtual environments (uv, venv), Docker containerization, POSIX fundamentals, and GPU hardware basics.',
    learningObjectives: [
      'Configure deterministic, reproducible development environments using modern tools (uv, Docker)',
      'Understand the memory and compute requirements for running local embeddings and SLM inference',
      'Structure clean, modular Python packages following production standards'
    ],
    prerequisites: ['Basic command line familiarity'],
    theory: {
      definition: 'The prerequisite software engineering and systems stack required to develop, test, and deploy resilient AI applications.',
      intuition: 'Building AI applications is 80% software engineering and 20% model orchestration. Without strict dependency pinning, type safety, and isolated environments, library version conflicts and breaking API changes will break production pipelines.',
      technicalExplanation: 'Production AI development requires deterministic dependency resolution (using modern lockfiles via uv), asynchronous I/O runtimes (asyncio/uvicorn), containerization (Docker multi-stage builds), and explicit environment variable governance (Pydantic Settings). On the hardware side, understanding GPU memory bandwidth (HBM3 vs GDDR6) and compute precision (FP16, BF16, FP8, INT4) dictates architectural feasibility.',
      example: 'Using `uv venv --python 3.12` creates an isolated environment 10x faster than standard venv, while `uv pip sync requirements.lock` guarantees bit-for-bit reproducibility across local development, CI/CD, and Kubernetes pods.'
    },
    implementation: {
      language: 'Bash / Dockerfile',
      code: `# Multi-stage production Dockerfile for AI microservice
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

FROM python:3.12-slim AS runner
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src/ /app/src/
USER nobody
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]`,
      explanation: 'Multi-stage Docker build separating dependency compilation from the lightweight, non-root runtime container.'
    },
    failureModes: [
      'Unpinned dependencies causing breaking changes on rebuilds',
      'Running containers as root user in production environments',
      'Out-of-Memory (OOM) killer terminating pods due to unbounded NumPy/Torch vector memory allocation'
    ],
    engineeringTradeoffs: [
      'Slim containers vs pre-warmed ML images: Slim images download faster (<150MB) but require pre-downloading model weights; fat images (5GB+) launch immediately without runtime network dependencies.'
    ],
    exercise: {
      prompt: 'Calculate the minimum GPU VRAM required to load an 8-Billion parameter model in FP16 precision.',
      hint: 'Each parameter in FP16 uses 2 bytes. Multiply parameters by bytes, then add 20% overhead for activations and KV cache.',
      solution: '8,000,000,000 params * 2 bytes = 16,000,000,000 bytes = 16 GB base weights. Adding 20% overhead for activations: 16 * 1.2 = 19.2 GB. A 24 GB GPU (such as an NVIDIA RTX 4090 or A10G) is required.'
    },
    quiz: [
      {
        question: 'Why is BF16 (Bfloat16) preferred over FP16 for training and inference in modern foundation models?',
        options: [
          'BF16 uses half as many bytes as FP16.',
          'BF16 has the same dynamic exponent range as FP32 (8 bits), preventing gradient underflow and overflow.',
          'BF16 has higher precision (mantissa) than FP16.',
          'BF16 does not require GPU tensor cores.'
        ],
        correctIndex: 1,
        explanation: 'BF16 allocates 8 bits to the exponent (matching FP32) and 7 bits to the mantissa, whereas FP16 has only 5 exponent bits. This wider range prevents numerical underflow/overflow during attention and loss computation.'
      }
    ]
  },
  {
    id: 2,
    slug: 'python-software-foundations',
    level: 1,
    levelName: 'Level 1: Software & Systems Foundations',
    title: '2. Python & Software Engineering Foundations for AI',
    duration: '3 Hours',
    description: 'Type annotations, Pydantic v2 data contracts, AsyncIO concurrent event loops, structured error handling, and robust HTTP streaming.',
    learningObjectives: [
      'Model rigid data contracts using Pydantic v2 schemas for input validation and LLM structured outputs',
      'Architect concurrent non-blocking pipelines using Python AsyncIO and ThreadPoolExecutors',
      'Implement resilient network retries with exponential backoff and decorrelated jitter'
    ],
    prerequisites: ['Python syntax basics'],
    theory: {
      definition: 'The software engineering design patterns and language features essential for building high-concurrency, type-safe AI backends.',
      intuition: 'LLM responses are probabilistic and non-deterministic. Without strict schemas (Pydantic) and asynchronous I/O (AsyncIO), your backend will crash on malformed outputs and stall during slow LLM generation calls.',
      technicalExplanation: 'Network calls to LLMs and vector databases are I/O-bound with high variance (200ms to 20s). Standard synchronous code blocks the entire thread, reducing throughput to 1 request per worker. Python AsyncIO allows single-threaded cooperative multitasking, enabling thousands of concurrent queries to wait on streaming chunks simultaneously. Pydantic v2 (built on Rust pydantic-core) provides microsecond-fast JSON deserialization and schema generation for function calling.',
      example: 'Using `asyncio.gather()` to query a vector database, a BM25 keyword index, and a user permission service in parallel cuts end-to-end retrieval latency from 45ms to 18ms.'
    },
    implementation: {
      language: 'Python',
      code: `from pydantic import BaseModel, Field
import asyncio
from typing import List

class DocumentChunk(BaseModel):
    id: str = Field(..., description="Unique chunk UUID")
    text: str = Field(..., min_length=1)
    vector_score: float = Field(..., ge=-1.0, le=1.0)
    metadata: dict = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    query: str
    chunks: List[DocumentChunk]
    execution_time_ms: float

async def fetch_chunk(chunk_id: str) -> DocumentChunk:
    await asyncio.sleep(0.02)
    return DocumentChunk(id=chunk_id, text="Grounded content", vector_score=0.92)

async def main():
    chunks = await asyncio.gather(*[fetch_chunk(f"c_{i}") for i in range(5)])
    result = RetrievalResult(query="test", chunks=chunks, execution_time_ms=21.4)
    print(result.model_dump_json(indent=2))

asyncio.run(main())`,
      explanation: 'Async parallel execution coupled with Pydantic v2 schema validation for structured retrieval outputs.'
    },
    failureModes: [
      'Blocking the AsyncIO event loop with CPU-heavy operations',
      'Unbounded concurrency overwhelming downstream provider rate limits'
    ],
    engineeringTradeoffs: [
      'Pydantic validation overhead: Pydantic v2 has near-zero overhead (<5 microseconds), making it suitable for high-throughput inner loops.'
    ],
    exercise: {
      prompt: 'Implement a function that queries two mock APIs concurrently with a hard 500ms timeout.',
      hint: 'Use `asyncio.wait_for()` wrapping `asyncio.gather()`.',
      solution: `async def query_with_timeout(api1, api2):
    try:
        return await asyncio.wait_for(asyncio.gather(api1(), api2()), timeout=0.5)
    except asyncio.TimeoutError:
        return ["FALLBACK_TIMEOUT"]`
    },
    quiz: [
      {
        question: 'What happens if you run a CPU-intensive matrix multiplication directly inside an async Python function?',
        options: [
          'It runs on a background worker thread automatically.',
          'It blocks the single AsyncIO event loop, freezing all other active connections and streams.',
          'It raises an AsyncExecutionError.',
          'It speeds up execution by utilizing all available CPU cores.'
        ],
        correctIndex: 1,
        explanation: 'Python AsyncIO uses a single-threaded event loop. Any synchronous CPU-bound operation blocks the loop, pausing all other concurrent coroutines until it finishes.'
      }
    ]
  },
  {
    id: 3,
    slug: 'machine-learning-foundations',
    level: 2,
    levelName: 'Level 2: Mathematical Foundations of Intelligence',
    title: '3. Machine Learning Foundations: Optimization, Probability & Information Theory',
    duration: '3.5 Hours',
    description: 'Loss functions, gradient descent, cross-entropy, Shannon entropy, KL-divergence, perplexity, and generalization theory.',
    learningObjectives: [
      'Derive and interpret Cross-Entropy Loss and Perplexity for autoregressive language models',
      'Understand Softmax temperature scaling dynamics and its analytical limits (T -> 0, T -> inf)',
      'Apply information theory metrics (Shannon Entropy, KL-Divergence) to measure model uncertainty'
    ],
    prerequisites: ['Basic calculus and matrix multiplication'],
    theory: {
      definition: 'The mathematical and statistical machinery that governs how neural networks represent patterns, optimize parameters, and quantify uncertainty.',
      intuition: 'Language modeling is fundamentally a game of predicting the next word given previous words. Cross-entropy loss measures how surprised the model is by the actual next word; lower loss means the model was confident and correct.',
      technicalExplanation: 'An autoregressive language model parameterizes a categorical probability distribution over a vocabulary V at each position. Given one-hot target y_t and predicted probabilities p_t = softmax(z_t), the cross-entropy loss is L = -sum_i y_{t,i} log(p_{t,i}) = -log(p_{t, target}). Perplexity (PPL) is the exponentiated average loss: PPL = exp(L). Softmax temperature T scales logits prior to normalization: p_i = exp(z_i / T) / sum_j exp(z_j / T). As T -> 0, the distribution collapses to an argmax one-hot vector (deterministic); as T -> inf, it flattens into a uniform distribution (maximum entropy).',
      mathematics: {
        formula: '\\mathcal{L}_{CE} = -\\frac{1}{N} \\sum_{t=1}^N \\log P(w_t \\mid w_{<t}; \\theta), \\quad \\text{PPL} = \\exp(\\mathcal{L}_{CE})',
        variables: [
          { name: 'w_t', desc: 'The ground-truth target token at step t' },
          { name: 'w_{<t}', desc: 'The sequence of tokens preceding step t' },
          { name: 'theta', desc: 'Model parameters' },
          { name: 'PPL', desc: 'Perplexity: effective branching factor of the model' }
        ]
      },
      example: 'If an LLM assigns probability 0.5 to the correct next word, the cross-entropy loss is -log(0.5) = 0.693. If it assigns probability 0.01, loss spikes to -log(0.01) = 4.605.'
    },
    implementation: {
      language: 'Python / NumPy',
      code: `import numpy as np

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

def cross_entropy_and_perplexity(predicted_prob: float) -> tuple[float, float]:
    loss = -np.log(max(1e-12, predicted_prob))
    ppl = np.exp(loss)
    return loss, ppl

logits = np.array([2.0, 1.0, 0.1])
print("T=1.0:", softmax_with_temperature(logits, 1.0))
print("T=0.2:", softmax_with_temperature(logits, 0.2))`,
      explanation: 'Numerically stable Softmax implementation with temperature scaling and perplexity calculator.'
    },
    failureModes: [
      'Numerical overflow in Softmax exp(z) when logits are large',
      'Setting temperature = 0 without handling the division-by-zero singularity'
    ],
    engineeringTradeoffs: [
      'Temperature vs Determinism: T=0 gives reproducible deterministic outputs (essential for code and tool calling); T=0.7 gives varied creative prose but increases risk of ungrounded completions.'
    ],
    exercise: {
      prompt: 'If a model achieves a cross-entropy loss of 2.302 on a test dataset, what is its perplexity?',
      hint: 'PPL = exp(loss). Note that exp(2.3025) is approximately 10.',
      solution: 'PPL = exp(2.302) = 9.994 ~= 10. This means on average, the model is as uncertain as picking uniformly among 10 candidate words.'
    },
    quiz: [
      {
        question: 'What happens to the Softmax output distribution as temperature T approaches infinity?',
        options: [
          'It becomes a one-hot vector with 1.0 at the highest logit.',
          'It converges to a completely uniform distribution where all vocabulary tokens have equal probability 1 / |V|.',
          'It causes all probabilities to become NaN.',
          'It doubles the perplexity score.'
        ],
        correctIndex: 1,
        explanation: 'As T -> inf, z_i / T -> 0 for all i. Since exp(0) = 1, each token probability evaluates to 1 / sum(1) = 1 / |V|, producing uniform randomness.'
      }
    ]
  },
  {
    id: 4,
    slug: 'neural-networks-embeddings',
    level: 2,
    levelName: 'Level 2: Mathematical Foundations of Intelligence',
    title: '4. Neural Networks & Representation Learning',
    duration: '3 Hours',
    description: 'Feedforward networks, backpropagation, non-linear activations (GELU, SwiGLU), high-dimensional vector spaces, and embedding geometry.',
    learningObjectives: [
      'Understand how continuous dense vectors represent discrete semantic concepts',
      'Compare modern activation functions (ReLU vs GELU vs SwiGLU)',
      'Analyze high-dimensional geometry: the curse of dimensionality and vector orthogonality'
    ],
    prerequisites: ['Level 3 Machine Learning Foundations'],
    theory: {
      definition: 'The architecture of deep multi-layer perceptrons and the mechanics of mapping discrete symbols into continuous, differentiable geometric embedding spaces.',
      intuition: 'Words like "king" and "queen" share many properties (royalty, leadership, human). Instead of treating words as isolated arbitrary IDs, embeddings map them to vectors where their spatial direction and distance reflect their real-world conceptual relationships.',
      technicalExplanation: 'An embedding layer is essentially a lookup table that multiplies a one-hot vector of dimension |V| by a weight matrix W of dimension |V| x d. Modern transformer architectures have largely replaced classic ReLU activations with smooth approximations: Gaussian Error Linear Units (GELU) and SwiGLU (Swish Gated Linear Unit, used in LLaMA). High-dimensional spaces (e.g. d=768 to d=3072) exhibit non-intuitive geometric properties: almost all randomly sampled vectors are mutually orthogonal (dot product near zero), and volume concentrates entirely on the surface shell of hyperspheres.',
      mathematics: {
        formula: '\\text{SwiGLU}(x) = \\text{Swish}(x W_1) \\odot (x W_2), \\quad \\text{where } \\text{Swish}(z) = z \\cdot \\sigma(\\beta z)',
        variables: [
          { name: 'W_1, W_2', desc: 'Feedforward projection weight matrices' },
          { name: 'sigma', desc: 'Sigmoid activation function' },
          { name: 'odot', desc: 'Element-wise Hadamard product' }
        ]
      },
      example: 'In a 768-dimensional embedding space, the vector difference vector("king") - vector("man") + vector("woman") lands in close proximity (cosine similarity > 0.82) to vector("queen").'
    },
    implementation: {
      language: 'Python / NumPy',
      code: `import numpy as np

def swiglu(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    swish = lambda z: z / (1.0 + np.exp(-z))
    gate = swish(np.dot(x, W1))
    value = np.dot(x, W2)
    return np.dot(gate * value, W3)

x = np.random.randn(1, 128)
W1 = np.random.randn(128, 256)
W2 = np.random.randn(128, 256)
W3 = np.random.randn(256, 128)
out = swiglu(x, W1, W2, W3)
print("SwiGLU Output Shape:", out.shape)`,
      explanation: 'Implementation of the SwiGLU gating mechanism powering LLaMA, Mistral, and modern LLM feed-forward layers.'
    },
    failureModes: [
      'Dying neurons in standard ReLU networks when activations permanently become negative',
      'High-dimensional hubness: certain vectors become nearest neighbors to an unnaturally large percentage of queries'
    ],
    engineeringTradeoffs: [
      'GELU/SwiGLU vs ReLU: SwiGLU achieves faster convergence and higher final validation accuracy, but requires 3 matrix multiplications per layer instead of 2, increasing FLOPs by ~50%.'
    ],
    exercise: {
      prompt: 'Prove mathematically why two random vectors sampled uniformly from a standard normal distribution in R^d have expected dot product 0.',
      hint: 'Compute E[u . v] = sum_i E[u_i * v_i]. Since u_i and v_i are independent with mean 0, E[u_i * v_i] = E[u_i]*E[v_i].',
      solution: 'For independent zero-mean variables: E[u . v] = sum_{i=1}^d E[u_i] * E[v_i] = sum_{i=1}^d 0 * 0 = 0. Furthermore, the variance of the dot product is d, so normalized vectors have dot products tightly bounded near 0 as d grows large.'
    },
    quiz: [
      {
        question: 'Why did modern models like LLaMA replace standard ReLU with SwiGLU in their feed-forward layers?',
        options: [
          'SwiGLU uses fewer parameters than ReLU.',
          'SwiGLU introduces dynamic bilinear gating with smooth non-linear gradients, empirically improving model perplexity.',
          'SwiGLU eliminates the need for matrix multiplication.',
          'SwiGLU runs natively on CPUs without floating point support.'
        ],
        correctIndex: 1,
        explanation: 'Shazeer (2020) demonstrated that GLU variants, particularly SwiGLU, provide superior gradient flow and lower training perplexity across all model scales compared to standard ReLU or GELU.'
      }
    ]
  },
  {
    id: 5,
    slug: 'transformers-attention',
    level: 3,
    levelName: 'Level 3: Deep Generative Architectures',
    title: '5. Transformers & Attention Mechanisms in Depth',
    duration: '4 Hours',
    description: 'Self-attention, Scaled Dot-Product mathematical proof, Multi-Head vs Multi-Query vs Grouped-Query Attention, and Rotary Position Embeddings (RoPE).',
    learningObjectives: [
      'Derive the Scaled Dot-Product Attention equation and prove why scaling by sqrt(d_k) prevents vanishing gradients',
      'Compare MHA, MQA, and GQA in terms of KV-cache memory footprint and throughput',
      'Derive Rotary Position Embedding (RoPE) and understand how it encodes relative token distances'
    ],
    prerequisites: ['Level 4 Neural Networks & Representation Learning'],
    theory: {
      definition: 'The Transformer architecture (Vaswani et al., 2017) and its core mechanism: Scaled Dot-Product Attention, which dynamically routes information between all tokens in a sequence.',
      intuition: 'Think of Attention as a database lookup: every token broadcasts a Query, holds a Key, and contains a Value. The match between Query and Key determines how much of the Value is passed along.',
      technicalExplanation: 'Given packed sequence matrices Q, K, V, Scaled Dot-Product Attention computes Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V. The division by sqrt(d_k) is mathematically essential to normalize dot product variance back to 1.0, preventing Softmax saturation. In Multi-Head Attention (MHA), each head has independent Q, K, V projections. In Grouped-Query Attention (GQA, used in LLaMA-3), multiple query heads share a single key-value head, cutting KV cache size by 8x with zero quality loss.',
      mathematics: {
        formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M \\right) V',
        variables: [
          { name: 'Q, K, V', desc: 'Query, Key, and Value projected matrices' },
          { name: 'd_k', desc: 'Dimension of each individual attention head' },
          { name: 'M', desc: 'Causal mask (setting upper triangular entries to -inf for autoregressive generation)' }
        ],
        derivation: 'Let q, k be independent d_k-dimensional vectors with Var(q_i) = Var(k_i) = 1. Then Var(q . k) = d_k. Dividing by sqrt(d_k) normalizes variance back to 1.0.'
      },
      example: 'In the sentence "The animal didn\'t cross the street because it was too tired", the token "it" attends heavily to "animal".'
    },
    implementation: {
      language: 'Python / NumPy',
      code: `import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = True) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)
    if causal:
        seq_len = Q.shape[-2]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores += mask
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attention_weights, V)`,
      explanation: 'NumPy implementation of Causal Scaled Dot-Product Attention with numerical stabilization.'
    },
    failureModes: [
      'Attention dilution: in long contexts, Softmax entropy increases, causing the model to miss needle-in-a-haystack facts',
      'Quadratic memory explosion O(N^2) without FlashAttention'
    ],
    engineeringTradeoffs: [
      'MHA vs GQA: GQA reduces KV cache memory by 8x with imperceptible impact on model accuracy.'
    ],
    exercise: {
      prompt: 'If head dimension d_k = 128, calculate the exact scaling factor applied to Q K^T.',
      hint: '1 / sqrt(128).',
      solution: '1 / sqrt(128) = 1 / 11.3137 = 0.088388.'
    },
    quiz: [
      {
        question: 'Why does Scaled Dot-Product Attention divide the query-key dot product by sqrt(d_k)?',
        options: [
          'To ensure the attention weights sum to 1.0.',
          'To prevent dot products from growing excessively large, which causes Softmax gradients to vanish.',
          'To compress the vectors into a smaller dimension.',
          'To enforce causal left-to-right generation order.'
        ],
        correctIndex: 1,
        explanation: 'As d_k increases, the variance of the dot product scales linearly with d_k. Large dot products push the Softmax function into saturation regions with near-zero gradients, paralyzing backpropagation.'
      }
    ]
  },
  {
    id: 6,
    slug: 'llm-engineering-inference',
    level: 3,
    levelName: 'Level 3: Deep Generative Architectures',
    title: '6. LLM Engineering: Training, Serving, KV-Cache & Quantization',
    duration: '4 Hours',
    description: 'Pre-training, Instruction tuning, DPO/RLHF, KV-cache memory arithmetic, continuous batching (PagedAttention), and weight quantization (AWQ, GPTQ).',
    learningObjectives: [
      'Calculate the exact VRAM footprint of the KV-cache across concurrency and context lengths',
      'Understand how PagedAttention and continuous batching eliminate GPU memory fragmentation in vLLM',
      'Compare weight-only quantization (AWQ, GPTQ) vs activation quantization (FP8, SmoothQuant)'
    ],
    prerequisites: ['Level 5 Transformers & Attention Mechanisms in Depth'],
    theory: {
      definition: 'The systems engineering principles that govern high-throughput LLM serving, memory optimization, and parameter quantization.',
      intuition: 'During autoregressive generation, generating each token requires reading past Keys and Values. Storing these in a fast "KV-cache" avoids recalculating past tokens from scratch. PagedAttention eliminates memory fragmentation.',
      technicalExplanation: 'Inference consists of two distinct phases: Prefill (compute-bound) and Decode (memory-bandwidth bound). KV-cache memory formula: Memory_KV = 2 * N_layers * N_kv_heads * d_head * P_bytes * Batch_size * Sequence_length. PagedAttention (Kwon et al., 2023 / vLLM) allocates KV cache in non-contiguous virtual memory pages, slashing memory waste from 60-80% down to under 4%.',
      mathematics: {
        formula: '\\text{Memory}_{\\text{KV}} = 2 \\times N_{\\text{layers}} \\times N_{\\text{kv\\_heads}} \\times d_{\\text{head}} \\times P_{\\text{bytes}} \\times B \\times S',
        variables: [
          { name: 'N_layers', desc: 'Number of transformer decoder layers' },
          { name: 'N_kv_heads', desc: 'Number of Key-Value heads' },
          { name: 'd_head', desc: 'Dimension per attention head' },
          { name: 'P_bytes', desc: 'Precision bytes (2 for FP16/BF16, 1 for FP8)' },
          { name: 'B', desc: 'Batch size' },
          { name: 'S', desc: 'Sequence length in tokens' }
        ]
      },
      example: 'For LLaMA-3-70B: Each token across batch requires 320 KB. Serving 32 concurrent 8k requests requires 81.9 GB of VRAM just for the KV cache.'
    },
    implementation: {
      language: 'Python',
      code: `def calculate_kv_cache_gb(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch_size: int, precision_bytes: int = 2) -> float:
    bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * precision_bytes
    total_bytes = bytes_per_token * seq_len * batch_size
    return total_bytes / (1024 ** 3)

llama3_8b_kv = calculate_kv_cache_gb(32, 8, 128, seq_len=4096, batch_size=16)
print(f"LLaMA-3-8B KV Cache (4k ctx, 16 batch): {llama3_8b_kv:.2f} GB")`,
      explanation: 'Production KV-cache memory calculator for capacity planning and GPU sizing.'
    },
    failureModes: [
      'KV Cache OOM Crash during traffic spikes',
      'Quantization perplexity degradation from aggressive INT4 scaling'
    ],
    engineeringTradeoffs: [
      'FP8 vs INT4: FP8 preserves virtually 100% of FP16 model accuracy while cutting VRAM and doubling memory-bandwidth throughput on Hopper GPUs (H100).'
    ],
    exercise: {
      prompt: 'If you convert the KV cache precision from FP16 (2 bytes) to FP8 (1 byte), what is the percentage reduction in KV cache memory?',
      hint: 'Compare 1 byte vs 2 bytes.',
      solution: 'Exactly 50% reduction in KV-cache memory footprint.'
    },
    quiz: [
      {
        question: 'Why is the decoding phase of LLM inference considered memory-bandwidth bound rather than compute-bound?',
        options: [
          'Because decoding generates multiple tokens simultaneously.',
          'Because generating each token requires loading all model weights and KV-cache tensors from VRAM into registers, performing only 1 FLOP per byte transferred.',
          'Because GPUs do not have tensor cores enabled during generation.',
          'Because Softmax calculations require disk access.'
        ],
        correctIndex: 1,
        explanation: 'In autoregressive decoding (batch=1), the GPU must load gigabytes of weights and cache into on-chip SRAM just to compute a single token, leaving tensor core compute units mostly idle while waiting for memory transfers.'
      }
    ]
  }
];
