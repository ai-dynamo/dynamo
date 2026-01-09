# `max_thinking_tokens` Implementation in SGLang

## Key Finding

**SGLang does NOT have a dedicated `max_thinking_tokens` field.** Instead, it uses a different architecture centered around custom logit processors and a `thinking_budget` parameter.

---

## 1. Token Budget Control via Custom Logit Processors

The primary mechanism for controlling thinking tokens is through **`thinking_budget`** passed via `custom_params`.

**File**: `python/sglang/srt/sampling/custom_logit_processor.py:60-136`

```python
class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list):
        for i, param_dict in enumerate(custom_param_list):
            thinking_budget = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
                continue

            req = param_dict.get("__req__")
            cur_ids = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if self.THINKING_START_TOKEN_ID not in cur_ids or self.THINKING_END_TOKEN_ID in cur_ids:
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits
```

**Model-specific implementations:**

| Processor Class | Model | Start Token ID | End Token ID | Newline Token ID |
|-----------------|-------|----------------|--------------|------------------|
| `DeepSeekR1ThinkingBudgetLogitProcessor` | DeepSeek-R1 | 128798 | 128799 | 201 |
| `Qwen3ThinkingBudgetLogitProcessor` | Qwen3 | 151667 | 151668 | 198 |
| `Glm4MoeThinkingBudgetLogitProcessor` | GLM-4.5/4.6 | 151350 | 151351 | 198 |

---

## 2. Usage Flow

### Client-Side API Call

```python
from sglang.srt.sampling.custom_logit_processor import DeepSeekR1ThinkingBudgetLogitProcessor

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "Question: Is Paris the Capital of France?"}
    ],
    max_tokens=1024,
    extra_body={
        "custom_logit_processor": DeepSeekR1ThinkingBudgetLogitProcessor().to_str(),
        "custom_params": {
            "thinking_budget": 512,  # Max thinking tokens
        },
    },
)
```

---

## 3. Data Flow

### Complete Pipeline

1. **API Layer** (`protocol.py:252-253, 528-529`):
   - `custom_logit_processor` and `custom_params` are defined as request fields

2. **Request Processing** (`serving_chat.py:250`):
   - Parameters passed to internal `GenerateReqInput`

3. **Scheduler** (`schedule_batch.py:563-567`):
   - `__req__` object injected into `custom_params` for access to request state
   ```python
   if isinstance(sampling_params.custom_params, dict):
       sampling_params.custom_params = sampling_params.custom_params | {"__req__": self}
   ```

4. **Sampling Batch Info** (`sampling_batch_info.py:116-145`):
   - Custom logit processors merged and batch masks created

5. **Sampler** (`sampler.py:544-578`):
   - `apply_custom_logit_processor()` invokes processors with logits and params

6. **Logit Processor**:
   - Checks token count since `<think>` and forces `</think>` when budget exceeded

---

## 4. Related Fields

| Field | Location | Purpose |
|-------|----------|---------|
| `max_completion_tokens` | `protocol.py:476-480` | Total output limit (includes reasoning + visible tokens) |
| `reasoning_effort` | `protocol.py:496-502` | "low"/"medium"/"high" - controls reasoning intensity |
| `reasoning_tokens` | `protocol.py:111` | Output field in `UsageInfo` tracking reasoning token count |
| `separate_reasoning` | `protocol.py:519` | Whether to separate reasoning content from visible output |
| `stream_reasoning` | `protocol.py:520` | Whether to stream reasoning content separately |
| `require_reasoning` | `io_struct.py:229` | Flag indicating request needs reasoning (hybrid models) |

### Protocol Definitions

**File**: `python/sglang/srt/entrypoints/openai/protocol.py`

```python
max_tokens: Optional[int] = Field(
    default=None,
    deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
)

max_completion_tokens: Optional[int] = Field(
    default=None,
    description="The maximum number of completion tokens for a chat completion request, "
    "including visible output tokens and reasoning tokens. Input tokens are not included.",
)

reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
    default="medium",
    description="Constrains effort on reasoning for reasoning models...",
)

separate_reasoning: bool = True
stream_reasoning: bool = True
```

---

## 5. Model-Level Support Requirements

For a model to support thinking budget control:

### 5.1 Chat Template Support
- Must use `<think>`/`</think>` or equivalent tags in its chat template
- Example: DeepSeek-R1 uses `<think>` and `</think>` tokens

### 5.2 Reasoning Parser Configuration
Server requires `--reasoning-parser` flag when launching:
```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --reasoning-parser deepseek-r1
```

Available parsers (defined in `reasoning_parser.py`):
- `deepseek-r1` / `deepseek-v3`
- `qwen3` / `qwen3-thinking`
- `glm45`
- `kimi`
- `nano_v3`
- `interns1`

### 5.3 Token ID Mapping
Custom logit processor needs correct start/end token IDs for the specific model's vocabulary.

### 5.4 Force Reasoning Behavior
Some models are reasoning-by-default:

**File**: `python/sglang/srt/entrypoints/openai/serving_chat.py:1095-1110`

```python
def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
    """Judge whether the request needs reasoning"""
    if not self.reasoning_parser:
        return False
    if self.reasoning_parser in ["deepseek-v3"]:
        return (
            request.chat_template_kwargs is not None
            and request.chat_template_kwargs.get("thinking") is True
        )
    if self.reasoning_parser in ["qwen3", "glm45", "nano_v3", "interns1"]:
        # These models are reasoning by default
        return (
            not request.chat_template_kwargs
            or request.chat_template_kwargs.get("enable_thinking", True) is True
        )
    return True  # default
```

---

## 6. Reasoning Content Parsing

**File**: `python/sglang/srt/parser/reasoning_parser.py`

The `BaseReasoningFormatDetector` class handles:

```python
class BaseReasoningFormatDetector:
    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.stream_reasoning = stream_reasoning
        self._in_reasoning = force_reasoning
```

Key responsibilities:
- Detecting `<think>` start tokens
- Extracting reasoning content between tags
- Separating reasoning from visible output
- Streaming reasoning content incrementally

---

## 7. Reasoning Token Tracking (Output)

**File**: `python/sglang/srt/entrypoints/openai/protocol.py:111`

```python
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[Dict[str, int]] = None
    reasoning_tokens: Optional[int] = 0  # Tracks reasoning tokens separately
```

**File**: `python/sglang/srt/entrypoints/context.py:81`

```python
class HarmonyContext(ConversationContext):
    def __init__(self, messages, tool_sessions):
        self.num_reasoning_tokens = 0  # Tracks reasoning tokens
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
```

---

## 8. Architecture Diagram

```
API Request
    |
    +-- custom_logit_processor (serialized string)
    +-- custom_params: {thinking_budget: 512}
    |
    v
TokenizerManager --> Scheduler
                        |
                        +-- Req.__init__() injects __req__ into custom_params
                        |
                        v
                   SamplingBatchInfo.from_schedule_batch()
                        |
                        +-- Merges custom_logit_processor across batch
                        +-- Creates batch masks for each processor
                        |
                        v
                   Sampler.forward()
                        |
                        +-- apply_custom_logit_processor()
                        |
                        v
                   ThinkingBudgetLogitProcessor.__call__()
                        |
                        +-- Counts tokens since <think>
                        +-- If budget exceeded --> force </think> token
                        |
                        v
                   Output with controlled thinking length
```

---

## 9. Key Files Reference

| File | Purpose |
|------|---------|
| `python/sglang/srt/sampling/custom_logit_processor.py` | Logit processor base class and model-specific implementations |
| `python/sglang/srt/entrypoints/openai/protocol.py` | API request/response definitions |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | Chat completion request handling |
| `python/sglang/srt/parser/reasoning_parser.py` | Reasoning content detection and extraction |
| `python/sglang/srt/managers/io_struct.py` | Internal request structures |
| `python/sglang/srt/managers/schedule_batch.py` | Request scheduling and `__req__` injection |
| `python/sglang/srt/sampling/sampling_batch_info.py` | Batch sampling info with custom processor handling |
| `python/sglang/srt/layers/sampler.py` | Sampling with custom logit processor application |
| `python/sglang/srt/server_args.py:570` | `enable_custom_logit_processor` server flag |

---

## 10. Important Distinction

### SGLang vs OpenAI o1/o3 Approach

**OpenAI's approach** (o1/o3 models):
- Dedicated `max_thinking_tokens` parameter
- Separate limits for thinking and visible output

**SGLang's approach**:
- `max_completion_tokens` covers total output (reasoning + visible)
- `thinking_budget` via custom logit processor for fine-grained control
- `reasoning_tokens` tracked separately in usage info for accounting
- Requires explicit custom logit processor setup per request

This design provides flexibility but requires more client-side configuration compared to a built-in parameter.
