# Flexible Tokenization Strategy for Universal Model Support

## Problem Statement

Dynamo's Rust preprocessor uses a HuggingFace tokenizer that only supports models with standard `tokenizer.json` files. This prevents support for models requiring specialized tokenizers (e.g., Ministral-3B with `mistral-common`). Additionally, not all tool call parsers and reasoning parsers implemented by backends are available in Dynamo's frontend postprocessor.

**Key constraints:**
- KV Routing requires token IDs to compute block hashes for prefix matching
- Disaggregated serving needs consistent token representations across workers
- We want to support any model the underlying framework supports

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Frontend (Rust)                                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  OpenAI API │───▶│ Preprocessor │───▶│     KV Router            │   │
│  │   Endpoint  │    │ (Tokenizer)  │    │ (needs token_ids)        │   │
│  └─────────────┘    └──────────────┘    └──────────────────────────┘   │
│                            │                        │                    │
│                            ▼                        ▼                    │
│                    PreprocessedRequest      Routing Decision             │
│                    (token_ids required)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Backend (SGLang/vLLM/TRT-LLM)                       │
│  ┌──────────────┐    ┌─────────────┐    ┌────────────────────────────┐ │
│  │ Tokenizer    │    │ LLM Engine  │    │ Postprocessor              │ │
│  │ (framework)  │    │             │    │ (tool/reasoning parsers)   │ │
│  └──────────────┘    └─────────────┘    └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Current limitation:** If the Rust tokenizer doesn't support a model, we must use `ModelInput.Text` mode which bypasses frontend tokenization—but then KV routing breaks because it has no tokens.

---

## Option 1: Remote Tokenize Endpoint (Backend RPC)

Add a tokenize RPC endpoint on the backend that the frontend calls when it needs tokens but can't tokenize locally.

```
Frontend                              Backend
   │                                     │
   │  ──── TokenizeRequest(text) ────▶   │
   │                                     │ (use framework tokenizer)
   │  ◀─── TokenizeResponse(tokens) ──   │
   │                                     │
   │  (compute KV hashes)                │
   │                                     │
   │  ──── PreprocessedRequest ───────▶  │
   │       (tokens + routing hints)      │
```

**Implementation sketch:**

```rust
// New RPC service on backend
#[async_trait]
pub trait TokenizerService {
    async fn tokenize(&self, text: String) -> Result<Vec<u32>>;
    async fn tokenize_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<u32>>>;
    async fn detokenize(&self, tokens: Vec<u32>) -> Result<String>;
}

// Frontend preprocessor modification
impl Preprocessor {
    async fn get_tokens(&self, text: &str) -> Result<Vec<u32>> {
        match &self.local_tokenizer {
            Some(tok) => tok.encode(text),
            None => self.remote_tokenizer.tokenize(text).await,
        }
    }
}
```

**Pros:**
- Simple to implement
- Reuses backend's existing tokenizer
- No Python in frontend
- Guaranteed tokenizer consistency

**Cons:**
- Added latency (network round-trip per request)
- Requires backend to be running for frontend startup
- Potential bottleneck under high concurrency
- Coupling between frontend and backend lifecycle

---

## Option 2: Frontend Python Tokenizer (Already Partially Implemented)

Instantiate the framework's tokenizer (or custom Python tokenizer) directly in the frontend process.

```rust
// Already exists in tokenizers/python.rs
pub struct PythonTokenizer {
    module_path: String,    // e.g., "dynamo.common.tokenizers.sglang"
    class_name: String,     // e.g., "SGLangTokenizer"
    model_path: String,
    py_tokenizer: Mutex<Option<Py<PyAny>>>,
}

// Configuration via ModelRuntimeConfig
pub struct TokenizerConfig {
    pub backend: TokenizerBackend,  // HuggingFace | Python | SGLang | VLLM
    pub python_module: Option<String>,
    pub python_class: Option<String>,
}
```

**For mistral-common support, add:**

```python
# dynamo/common/tokenizers/mistral.py
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

class MistralCommonTokenizer(BaseTokenizer):
    def __init__(self, model_path: str):
        self._tokenizer = MistralTokenizer.from_file(f"{model_path}/tokenizer.model.v3")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, bos=False, eos=False)
```

**Pros:**
- No network overhead
- Full flexibility—any Python tokenizer works
- Already partially implemented
- Tokenizer lives close to KV routing logic

**Cons:**
- Requires Python runtime in frontend process
- Need to match backend tokenizer exactly (version, config)
- Increased frontend memory footprint
- Must maintain Python tokenizer wrappers

---

## Option 3: Hybrid Input/Output Mode (Tokens In, Text Out)

Tokenize on frontend (using Python tokenizer from Option 2), but let the backend handle all postprocessing (detokenization, tool parsing, reasoning parsing).

```
Frontend                              Backend
   │                                     │
   │  Python tokenize (local)            │
   │  Compute KV hashes                  │
   │                                     │
   │  ──── PreprocessedRequest ───────▶  │
   │       (token_ids, output_mode=Text) │
   │                                     │
   │  ◀─── StreamingResponse ─────────   │
   │       (text chunks, parsed tools)   │
```

**Implementation:**

```rust
// Extend PreprocessedRequest
pub struct PreprocessedRequest {
    pub token_ids: Vec<TokenIdType>,
    pub output_mode: OutputMode,  // NEW: Tokens | Text | Structured
    // ...
}

pub enum OutputMode {
    Tokens,      // Raw token IDs (current behavior)
    Text,        // Detokenized text (let backend handle)
    Structured,  // Backend parses tool calls, reasoning, etc.
}
```

```python
# Backend handler modification
class SGLangHandler:
    async def process_request(self, request: PreprocessedRequest):
        async for token_id in self.engine.generate(request.token_ids):
            if request.output_mode == OutputMode.Structured:
                # Use SGLang's native tool/reasoning parsers
                yield self.postprocessor.process(token_id)
            elif request.output_mode == OutputMode.Text:
                yield self.tokenizer.decode([token_id])
            else:
                yield token_id
```

**Pros:**
- Best of both worlds: frontend tokenization for routing, backend postprocessing
- Leverages backend's native tool/reasoning parsers
- KV routing works normally
- Clean separation of concerns

**Cons:**
- Still requires Python tokenizer in frontend
- Must ensure tokenizer parity between frontend/backend
- New output mode adds complexity

---

## Option 4: Backend-Side KV Routing

Move KV routing decision into the backend. Frontend passes raw text, backend tokenizes and routes.

```
Frontend                              Backend Router           Backend Worker
   │                                       │                        │
   │  ──── RawTextRequest ──────────────▶  │                        │
   │                                       │ tokenize               │
   │                                       │ compute KV hashes      │
   │                                       │ select worker          │
   │                                       │  ────────────────────▶ │
   │                                       │                        │ inference
   │  ◀──── StreamingResponse ───────────────────────────────────── │
```

**Implementation:**

```rust
// New backend component: RouterWorker
pub struct RouterWorker {
    tokenizer: Arc<dyn Tokenizer>,
    kv_router: KvRouter,
    workers: Vec<WorkerClient>,
}

impl RouterWorker {
    async fn route_and_forward(&self, text: &str) -> WorkerClient {
        let tokens = self.tokenizer.encode(text)?;
        let hashes = compute_block_hashes(&tokens);
        let worker = self.kv_router.select_worker(&hashes);
        worker
    }
}
```

**Pros:**
- Frontend stays simple (Rust only)
- Backend has full control over tokenization
- Natural fit for disaggregated architectures

**Cons:**
- Routing latency added to request path
- Complex topology (router → workers)
- If routing to different worker, must forward tokenized input OR re-tokenize
- Harder to debug routing decisions

---

## Option 5: Lazy/On-Demand KV Routing

Default to `ModelInput.Text` without KV routing. Enable KV routing only when a compatible tokenizer is available.

```rust
pub struct PreprocessorConfig {
    pub tokenizer: Option<TokenizerConfig>,
    pub kv_routing_enabled: bool,  // Auto-disabled if no tokenizer
}

impl Preprocessor {
    fn process(&self, request: Request) -> PreprocessedRequest {
        let (tokens, routing) = match &self.tokenizer {
            Some(tok) => {
                let tokens = tok.encode(&request.text)?;
                let hashes = compute_block_hashes(&tokens);
                (Some(tokens), Some(RoutingHints::from_hashes(hashes)))
            }
            None => (None, None),  // Skip routing, pass text through
        };

        PreprocessedRequest {
            input: tokens.map(PromptInput::Tokens)
                        .unwrap_or(PromptInput::Text(request.text)),
            routing,
            // ...
        }
    }
}
```

**Pros:**
- Graceful degradation—models work even without tokenizer
- Simple to implement
- No additional complexity for unsupported models

**Cons:**
- Loses KV routing benefits for unsupported models
- Two different code paths to maintain
- Users may not realize they're missing optimization

---

## Option 6: Universal Python Tokenizer Gateway (NEW)

Create a dedicated tokenizer microservice that can load any tokenizer and serves both frontend and backend.

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│   Frontend   │────▶│  Tokenizer Gateway   │◀────│   Backend    │
│              │     │  (Python service)    │     │              │
└──────────────┘     │  - mistral-common    │     └──────────────┘
                     │  - tiktoken          │
                     │  - sentencepiece     │
                     │  - HF transformers   │
                     └──────────────────────┘
```

**Implementation:**

```python
# tokenizer_gateway/service.py
class TokenizerGateway:
    def __init__(self):
        self.tokenizers: Dict[str, DynamoTokenizer] = {}

    async def get_or_create(self, model_path: str, backend: str) -> DynamoTokenizer:
        key = f"{model_path}:{backend}"
        if key not in self.tokenizers:
            self.tokenizers[key] = self._create_tokenizer(model_path, backend)
        return self.tokenizers[key]

    def _create_tokenizer(self, model_path: str, backend: str):
        if backend == "mistral":
            return MistralCommonTokenizer(model_path)
        elif backend == "sglang":
            return SGLangTokenizer(model_path)
        # ...
```

**Pros:**
- Single source of truth for tokenization
- Can be scaled independently
- Easy to add new tokenizer backends
- Consistent behavior across all components

**Cons:**
- Additional service to deploy and manage
- Network latency for every tokenization
- Single point of failure
- Overkill for most deployments

---

## Option 7: Sidecar Python Tokenizer (NEW)

Embed a lightweight Python sidecar process alongside the Rust frontend, communicating via Unix sockets or shared memory.

```
┌─────────────────────────────────────────────────┐
│              Frontend Pod/Container              │
│  ┌─────────────────┐    ┌────────────────────┐  │
│  │  Rust Frontend  │◀──▶│  Python Sidecar    │  │
│  │  (preprocessor) │    │  (tokenizer)       │  │
│  │                 │    │  - Unix socket IPC │  │
│  └─────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Implementation:**

```rust
// IPC via Unix socket
pub struct SidecarTokenizer {
    socket: UnixStream,
}

impl Tokenizer for SidecarTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.socket.write_all(&encode_request(text))?;
        let response = self.socket.read_response()?;
        Ok(decode_tokens(response))
    }
}
```

```python
# sidecar.py
async def serve(socket_path: str, model_path: str, tokenizer_type: str):
    tokenizer = create_tokenizer(model_path, tokenizer_type)
    async with UnixSocketServer(socket_path) as server:
        async for request in server:
            tokens = tokenizer.encode(request.text)
            await server.respond(tokens)
```

**Pros:**
- Low latency (local IPC)
- Clean process separation
- Can crash/restart independently
- No Python in main Rust process

**Cons:**
- More complex deployment
- IPC serialization overhead
- Process management complexity
- Still need to maintain sidecar

---

## Comparison Matrix

| Option | Latency | Complexity | KV Routing | Flexibility | Maintenance |
|--------|---------|------------|------------|-------------|-------------|
| 1. Remote RPC | High | Low | Yes | High | Low |
| 2. Frontend Python | Low | Medium | Yes | High | Medium |
| 3. Hybrid I/O | Low | Medium | Yes | High | Medium |
| 4. Backend Routing | Medium | High | Yes | High | High |
| 5. Lazy/On-Demand | N/A | Low | Partial | Low | Low |
| 6. Gateway Service | High | High | Yes | Very High | High |
| 7. Sidecar | Low | Medium | Yes | High | Medium |

---

## Recommendation: Option 3 (Hybrid I/O) + Option 2 (Frontend Python Tokenizer)

**Rationale:**

1. **Frontend Python Tokenizer (Option 2)** is already partially implemented and provides the foundation:
   - Zero network latency for tokenization
   - Works with any Python tokenizer (mistral-common, tiktoken, etc.)
   - KV routing gets the tokens it needs
   - Simple configuration via `TokenizerConfig`

2. **Hybrid I/O Mode (Option 3)** solves the postprocessing problem:
   - Frontend tokenizes for routing, backend detokenizes and parses
   - Leverages backend's native tool call and reasoning parsers
   - No need to reimplement parsers in Rust
   - Clean separation: frontend handles routing, backend handles output formatting

**Implementation roadmap:**

```
Phase 1: Complete Frontend Python Tokenizer
├─ Add MistralCommonTokenizer wrapper
├─ Add auto-detection based on model config
├─ Test with Ministral-3B
└─ Benchmark overhead vs native Rust tokenizer

Phase 2: Add Hybrid Output Mode
├─ Extend PreprocessedRequest with output_mode field
├─ Modify backends to respect output_mode
├─ Pass through backend's parsed tool calls
└─ Test tool calling with various models

Phase 3: Polish & Production
├─ Add tokenizer health checks
├─ Implement tokenizer version validation
├─ Add metrics for tokenization latency
└─ Document configuration for new models
```

**Why not the other options?**

- **Option 1 (Remote RPC):** Adds latency and couples frontend/backend lifecycle
- **Option 4 (Backend Routing):** Significant architectural change, complex topology
- **Option 5 (Lazy):** Loses KV routing for models that need it most
- **Option 6 (Gateway):** Overkill for most deployments, operational overhead
- **Option 7 (Sidecar):** Similar benefits to Option 2 but more complex deployment

**The hybrid approach gives us:**
- Universal model support via Python tokenizers
- Full KV routing capability
- Native backend postprocessing (tool parsing, reasoning)
- Minimal architectural changes
- Clear upgrade path for each phase

---

## Appendix: Quick Start for Ministral-3B Support

Once Option 2 is complete, supporting Ministral-3B would look like:

```yaml
# model_config.yaml
model_path: "ministral/Ministral-3b-instruct"
tokenizer:
  backend: python
  python_module: "dynamo.common.tokenizers.mistral"
  python_class: "MistralCommonTokenizer"
```

```python
# dynamo/common/tokenizers/mistral.py
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from .base import BaseTokenizer

class MistralCommonTokenizer(BaseTokenizer):
    def __init__(self, model_path: str):
        self._tokenizer = MistralTokenizer.from_file(
            f"{model_path}/tokenizer.model.v3"
        )

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenizer.encode(text, bos=False, eos=False)
        return [t.id for t in tokens]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids)
```
