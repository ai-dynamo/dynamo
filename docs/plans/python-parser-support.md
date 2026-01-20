# Python Parser Support for Dynamo Preprocessor

## Overview

Extend the Python tokenizer pattern to support Python-based reasoning parsers and tool call parsers from SGLang/vLLM within Dynamo's preprocessor and postprocessor. This enables using framework-specific parsing logic while keeping Dynamo's orchestration layer.

**Current state**: Dynamo uses `dynamo-parsers` crate for reasoning/tool parsing. Users can specify `--reasoning-parser` and `--tool-call-parser` which map to Rust implementations.

**Goal**: Allow users to optionally delegate parsing to SGLang's or vLLM's Python implementations, similar to how `--tokenizer-backend` works.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dynamo Preprocessor                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │  Tokenizer  │  │  Reasoning  │  │    Tool Call         │    │
│  │   Backend   │  │   Parser    │  │      Parser          │    │
│  ├─────────────┤  ├─────────────┤  ├──────────────────────┤    │
│  │ HuggingFace │  │ Rust Native │  │ Rust Native          │    │
│  │ SGLang (Py) │  │ SGLang (Py) │  │ SGLang (Py)          │    │
│  │ vLLM (Py)   │  │ vLLM (Py)   │  │ vLLM (Py)            │    │
│  │ Custom (Py) │  │ Custom (Py) │  │ Custom (Py)          │    │
│  └─────────────┘  └─────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Core Rust Infrastructure

### 1.1 Define Parser Protocols in Rust

**File**: `lib/llm/src/parsers/traits.rs` (NEW)

```rust
pub trait ReasoningParser: Send + Sync {
    /// Extract reasoning content from model output
    fn parse_reasoning(&self, text: &str) -> Result<Option<ReasoningBlock>>;

    /// Check if text contains reasoning markers
    fn has_reasoning(&self, text: &str) -> bool;
}

pub trait ToolCallParser: Send + Sync {
    /// Extract tool calls from model output
    fn parse_tool_calls(&self, text: &str) -> Result<Vec<ToolCall>>;

    /// Check if text contains tool call markers
    fn has_tool_calls(&self, text: &str) -> bool;

    /// Format tool results for model input
    fn format_tool_result(&self, tool_call_id: &str, result: &str) -> Result<String>;
}
```

### 1.2 Add Parser Config to RuntimeConfig

**File**: `lib/llm/src/local_model/runtime_config.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ParserBackend {
    #[default]
    Native,      // Use Rust dynamo-parsers
    Python,      // Custom Python parser
    SGLang,      // SGLang's parser
    VLLM,        // vLLM's parser
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ReasoningParserConfig {
    pub backend: ParserBackend,
    pub parser_name: Option<String>,  // e.g., "deepseek" for native, class name for Python
    pub python_module: Option<String>,
    pub python_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ToolCallParserConfig {
    pub backend: ParserBackend,
    pub parser_name: Option<String>,  // e.g., "hermes", "llama3_json" for native
    pub python_module: Option<String>,
    pub python_class: Option<String>,
}
```

### 1.3 Create Python Parser Wrappers

**File**: `lib/llm/src/parsers/python_reasoning.rs` (NEW)

```rust
pub struct PythonReasoningParser {
    module_path: String,
    class_name: String,
    py_parser: Mutex<Option<Py<PyAny>>>,
}

impl ReasoningParser for PythonReasoningParser {
    fn parse_reasoning(&self, text: &str) -> Result<Option<ReasoningBlock>> {
        self.with_parser(|parser| {
            let result = parser.call_method1("parse_reasoning", (text,))?;
            // Convert Python result to ReasoningBlock
            ...
        })
    }
}
```

**File**: `lib/llm/src/parsers/python_tool_call.rs` (NEW)

```rust
pub struct PythonToolCallParser {
    module_path: String,
    class_name: String,
    py_parser: Mutex<Option<Py<PyAny>>>,
}

impl ToolCallParser for PythonToolCallParser {
    fn parse_tool_calls(&self, text: &str) -> Result<Vec<ToolCall>> {
        self.with_parser(|parser| {
            let result = parser.call_method1("parse_tool_calls", (text,))?;
            // Convert Python list to Vec<ToolCall>
            ...
        })
    }
}
```

## Phase 2: Python Protocol and Implementations

### 2.1 Define Python Protocols

**File**: `components/src/dynamo/common/parsers/__init__.py` (NEW)

```python
from dynamo.common.parsers.protocol import (
    DynamoReasoningParser,
    DynamoToolCallParser,
    BaseReasoningParser,
    BaseToolCallParser,
    ReasoningBlock,
    ToolCall,
)
```

**File**: `components/src/dynamo/common/parsers/protocol.py` (NEW)

```python
from typing import Protocol, Optional, Sequence
from dataclasses import dataclass

@dataclass
class ReasoningBlock:
    """Extracted reasoning content from model output."""
    content: str
    start_marker: str
    end_marker: str
    start_index: int
    end_index: int

@dataclass
class ToolCall:
    """Extracted tool call from model output."""
    id: str
    name: str
    arguments: dict
    raw_text: str

@runtime_checkable
class DynamoReasoningParser(Protocol):
    def parse_reasoning(self, text: str) -> Optional[ReasoningBlock]: ...
    def has_reasoning(self, text: str) -> bool: ...

@runtime_checkable
class DynamoToolCallParser(Protocol):
    def parse_tool_calls(self, text: str) -> list[ToolCall]: ...
    def has_tool_calls(self, text: str) -> bool: ...
    def format_tool_result(self, tool_call_id: str, result: str) -> str: ...
```

### 2.2 Create SGLang Parser Wrappers

**File**: `components/src/dynamo/common/parsers/sglang.py` (NEW)

```python
class SGLangReasoningParser(BaseReasoningParser):
    """Wrapper around SGLang's reasoning parser."""

    def __init__(self, parser_name: str):
        from sglang.srt.openai_api.parsers.reasoning_parsers import get_reasoning_parser
        self._parser = get_reasoning_parser(parser_name)

    def parse_reasoning(self, text: str) -> Optional[ReasoningBlock]:
        result = self._parser.extract_reasoning(text)
        if result is None:
            return None
        return ReasoningBlock(
            content=result.reasoning_content,
            start_marker=result.start_token,
            end_marker=result.end_token,
            ...
        )

class SGLangToolCallParser(BaseToolCallParser):
    """Wrapper around SGLang's tool call parser."""

    def __init__(self, parser_name: str):
        from sglang.srt.openai_api.parsers.tool_parsers import get_tool_parser
        self._parser = get_tool_parser(parser_name)

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        results = self._parser.extract_tool_calls(text)
        return [
            ToolCall(id=r.id, name=r.function.name, arguments=r.function.arguments, ...)
            for r in results
        ]
```

### 2.3 Create vLLM Parser Wrappers

**File**: `components/src/dynamo/common/parsers/vllm.py` (NEW)

```python
class VLLMReasoningParser(BaseReasoningParser):
    """Wrapper around vLLM's reasoning parser."""

    def __init__(self, parser_name: str):
        from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
        self._parser = ReasoningParserManager.get_reasoning_parser(parser_name)

    def parse_reasoning(self, text: str) -> Optional[ReasoningBlock]:
        # vLLM reasoning parser interface
        ...

class VLLMToolCallParser(BaseToolCallParser):
    """Wrapper around vLLM's tool call parser."""

    def __init__(self, parser_name: str):
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
        self._parser = ToolParserManager.get_tool_parser(parser_name)

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        # vLLM tool parser interface
        ...
```

## Phase 3: CLI Integration

### 3.1 Add CLI Arguments

**File**: `components/src/dynamo/sglang/args.py`

```python
DYNAMO_ARGS = {
    ...
    "reasoning-parser-backend": {
        "flags": ["--reasoning-parser-backend"],
        "type": str,
        "default": "native",
        "choices": ["native", "python", "sglang", "vllm"],
        "help": "Backend for reasoning parsing. 'native' uses Rust parsers.",
    },
    "reasoning-parser-module": {
        "flags": ["--reasoning-parser-module"],
        "type": str,
        "default": None,
        "help": "Python module for custom reasoning parser (requires --reasoning-parser-backend python).",
    },
    "reasoning-parser-class": {
        "flags": ["--reasoning-parser-class"],
        "type": str,
        "default": None,
        "help": "Class name for custom reasoning parser.",
    },
    "tool-call-parser-backend": {
        "flags": ["--tool-call-parser-backend"],
        "type": str,
        "default": "native",
        "choices": ["native", "python", "sglang", "vllm"],
        "help": "Backend for tool call parsing. 'native' uses Rust parsers.",
    },
    "tool-call-parser-module": {
        "flags": ["--tool-call-parser-module"],
        "type": str,
        "default": None,
        "help": "Python module for custom tool call parser.",
    },
    "tool-call-parser-class": {
        "flags": ["--tool-call-parser-class"],
        "type": str,
        "default": None,
        "help": "Class name for custom tool call parser.",
    },
}
```

## Phase 4: Integration Points

### 4.1 Preprocessor Updates

The preprocessor needs to use the configured parser for:
- Detecting tool call format requirements in the prompt
- Adding tool schemas to the prompt in the correct format

### 4.2 Postprocessor/Response Generator Updates

**File**: `lib/llm/src/response_generator.rs`

The response generator needs to:
- Use the configured reasoning parser to extract thinking blocks
- Use the configured tool call parser to extract tool calls from output
- Format responses according to OpenAI API spec

```rust
impl ResponseGenerator {
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        reasoning_parser: Option<Arc<dyn ReasoningParser>>,
        tool_call_parser: Option<Arc<dyn ToolCallParser>>,
    ) -> Self { ... }

    fn process_output(&self, text: &str) -> ProcessedOutput {
        let reasoning = self.reasoning_parser
            .as_ref()
            .and_then(|p| p.parse_reasoning(text).ok().flatten());

        let tool_calls = self.tool_call_parser
            .as_ref()
            .map(|p| p.parse_tool_calls(text).unwrap_or_default())
            .unwrap_or_default();

        ProcessedOutput { reasoning, tool_calls, content: ... }
    }
}
```

### 4.3 Python Bindings

**File**: `lib/bindings/python/rust/llm/local_model.rs`

```rust
impl ModelRuntimeConfig {
    #[pyo3(signature = (backend, parser_name=None, python_module=None, python_class=None))]
    fn set_reasoning_parser_config(
        &mut self,
        backend: &str,
        parser_name: Option<String>,
        python_module: Option<String>,
        python_class: Option<String>,
    ) -> PyResult<()> { ... }

    #[pyo3(signature = (backend, parser_name=None, python_module=None, python_class=None))]
    fn set_tool_call_parser_config(
        &mut self,
        backend: &str,
        parser_name: Option<String>,
        python_module: Option<String>,
        python_class: Option<String>,
    ) -> PyResult<()> { ... }
}
```

## Phase 5: Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `lib/llm/src/parsers/mod.rs` | Create | Parser module root |
| `lib/llm/src/parsers/traits.rs` | Create | Parser trait definitions |
| `lib/llm/src/parsers/python_reasoning.rs` | Create | Python reasoning parser wrapper |
| `lib/llm/src/parsers/python_tool_call.rs` | Create | Python tool call parser wrapper |
| `lib/llm/src/parsers/factory.rs` | Create | Parser factory functions |
| `lib/llm/src/local_model/runtime_config.rs` | Modify | Add parser configs |
| `lib/llm/src/response_generator.rs` | Modify | Use configurable parsers |
| `lib/bindings/python/rust/llm/local_model.rs` | Modify | Add parser config methods |
| `components/src/dynamo/common/parsers/__init__.py` | Create | Package init |
| `components/src/dynamo/common/parsers/protocol.py` | Create | Python protocols |
| `components/src/dynamo/common/parsers/sglang.py` | Create | SGLang wrappers |
| `components/src/dynamo/common/parsers/vllm.py` | Create | vLLM wrappers |
| `components/src/dynamo/sglang/args.py` | Modify | Add CLI args |
| `components/src/dynamo/sglang/register.py` | Modify | Wire parser configs |

## Usage Examples

```bash
# Use SGLang's reasoning parser with native tool call parser
python -m dynamo.sglang \
    --model deepseek-ai/DeepSeek-R1 \
    --reasoning-parser deepseek \
    --reasoning-parser-backend sglang \
    --tool-call-parser hermes

# Use vLLM's tool call parser
python -m dynamo.sglang \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tool-call-parser llama3_json \
    --tool-call-parser-backend vllm

# Use custom Python parsers
python -m dynamo.sglang \
    --model my-model \
    --reasoning-parser-backend python \
    --reasoning-parser-module my.parsers \
    --reasoning-parser-class MyReasoningParser \
    --tool-call-parser-backend python \
    --tool-call-parser-module my.parsers \
    --tool-call-parser-class MyToolCallParser
```

## Design Considerations

1. **Backward Compatibility**: Default to `native` backend; existing deployments unchanged
2. **Performance**: Python parsers run via PyO3 with GIL; consider caching parsed results
3. **Error Handling**: Gracefully handle parser failures without crashing the request
4. **Streaming**: Parsers must support incremental parsing for streaming responses
5. **Thread Safety**: Use `Mutex<Option<Py<PyAny>>>` pattern from tokenizer implementation

## Streaming Considerations

For streaming responses, parsers need to handle partial content:

```rust
pub trait StreamingReasoningParser: ReasoningParser {
    /// Update parser state with new chunk
    fn feed_chunk(&mut self, chunk: &str);

    /// Check if reasoning block is complete
    fn is_reasoning_complete(&self) -> bool;

    /// Get current reasoning state
    fn get_partial_reasoning(&self) -> Option<PartialReasoningBlock>;
}
```

This requires maintaining parser state across chunks, which adds complexity but is essential for proper streaming support of thinking/reasoning blocks.
