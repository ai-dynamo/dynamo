# `max_thinking_tokens` Implementation Analysis in vLLM

## Key Finding: `max_thinking_tokens` is NOT Implemented

vLLM does **not** currently have a `max_thinking_tokens` parameter. However, it has extensive infrastructure for **reasoning/thinking mode** support.

---

## 1. Parameter Definitions (API Layer)

**`vllm/entrypoints/openai/protocol.py`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `reasoning_effort` | `Literal["low", "medium", "high"]` | Controls reasoning intensity (line 561) |
| `include_reasoning` | `bool` | Whether to include reasoning in response (line 562) |
| `reasoning` | `Reasoning` (OpenAI type) | Full Reasoning object for Responses API (line 328) |

The `Reasoning` object is imported from `openai.types.shared` and contains an `effort` field.

---

## 2. Request Flow (API → Engine → Model)

```
API Request
    ↓
ChatCompletionRequest/ResponsesRequest (protocol.py:301-340, 550-562)
    ↓
serving_chat.py / serving_responses.py
    ↓
Chat Template Processing (serving_engine.py:1150-1159)
    ├── Merges chat_template_kwargs with defaults
    └── Passes "thinking"/"enable_thinking" flags
    ↓
Tokenizer (deepseek_v32.py:46-69)
    ├── Reads thinking mode from kwargs
    └── Sets thinking_mode = "thinking" | "chat"
    ↓
Message Encoding (deepseek_v32_encoding.py:313-333)
    └── Wraps content with <think>...</think> tokens
    ↓
Model Generation (standard inference)
    ↓
Reasoning Parser extracts thinking content
```

---

## 3. Model-Level Support Requirements

**Supported Models** (from `docs/features/reasoning_outputs.md`):

| Model | Parser | Default State |
|-------|--------|---------------|
| DeepSeek R1 | `deepseek_r1` | Enabled |
| DeepSeek V3.1 | `deepseek_v3` | **Disabled** (needs `thinking=True`) |
| Qwen3 | `qwen3` | Enabled (disable with `enable_thinking=False`) |
| IBM Granite 3.2 | `granite` | **Disabled** (needs `thinking=True`) |
| Holo2 | `holo2` | Enabled |
| GLM-4.5 | `glm45` | Enabled |
| ERNIE-4.5 | `ernie45` | Enabled |
| MiniMax-M2 | `minimax_m2_append_think` | Enabled |

---

## 4. Reasoning Parser Architecture

**Base Class**: `vllm/reasoning/abs_reasoning_parsers.py:32`

```python
class ReasoningParser:
    def is_reasoning_end(input_ids) -> bool       # Check if thinking ended
    def is_reasoning_end_streaming(...) -> bool   # Streaming version
    def extract_content_ids(input_ids) -> list    # Extract content after </think>
    def extract_reasoning(model_output, request) -> (reasoning, content)  # Non-streaming
    def extract_reasoning_streaming(...) -> DeltaMessage  # Streaming extraction
```

**Common Implementation**: `vllm/reasoning/basic_parsers.py:22`
- `BaseThinkingReasoningParser` handles `<think>...</think>` pattern
- Uses `start_token_id` and `end_token_id` for efficient parsing

---

## 5. Structured Output Integration

**`vllm/v1/structured_output/__init__.py:295-333`**

The structured output manager checks reasoning state to delay FSM advancement:

```python
if request.structured_output_request.reasoning_ended is None:
    request.structured_output_request.reasoning_ended = (
        self.reasoner.is_reasoning_end(request.prompt_token_ids)
    )
```

This ensures structured output constraints (JSON schema, etc.) only apply **after** thinking ends.

---

## 6. How Thinking Token Limits Work (Current State)

Since there's no `max_thinking_tokens`, the thinking length is implicitly bounded by:

1. **`max_tokens`/`max_output_tokens`** in SamplingParams - total output limit
2. **Model's context window** - overall token budget
3. **Model's native behavior** - some models self-limit thinking
4. **`reasoning_effort`** parameter - hints to model via system prompt (via Harmony)

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `vllm/entrypoints/openai/protocol.py` | API parameter definitions |
| `vllm/entrypoints/openai/serving_chat.py:659-1051` | Streaming reasoning extraction |
| `vllm/entrypoints/openai/serving_responses.py:949` | Responses API reasoning handling |
| `vllm/reasoning/abs_reasoning_parsers.py` | Base parser class |
| `vllm/reasoning/basic_parsers.py` | Common `<think>` pattern parser |
| `vllm/reasoning/deepseek_r1_reasoning_parser.py` | DeepSeek R1 implementation |
| `vllm/tokenizers/deepseek_v32_encoding.py` | Thinking mode tokenization |
| `vllm/v1/structured_output/__init__.py` | Reasoning + structured output coordination |

---

## 8. Files Mentioning "Thinking Tokens"

- `vllm/tokenizers/deepseek_v32_encoding.py` - Thinking mode tokenization
- `vllm/reasoning/basic_parsers.py` - Base `<think>` pattern parser
- `vllm/reasoning/olmo3_reasoning_parser.py` - OLMo3 specific parser
- `tests/reasoning/test_base_thinking_reasoning_parser.py` - Test coverage

---

## 9. Implementation Roadmap (If Adding `max_thinking_tokens`)

To implement `max_thinking_tokens`, you would need to:

1. **Add parameter** to `ChatCompletionRequest`/`ResponsesRequest` in `protocol.py`
2. **Pass through** to `SamplingParams`
3. **Implement token counting** in the generation loop (likely in `model_runner` or sampler)
4. **Force-emit `</think>`** when budget is reached
5. **Update reasoning parsers** to handle truncated thinking gracefully
6. **Handle streaming** - emit partial reasoning before truncation

### Challenges:
- Token counting mid-generation requires tracking reasoning state
- Need to differentiate thinking tokens from content tokens during generation
- Structured output integration must account for early thinking termination
