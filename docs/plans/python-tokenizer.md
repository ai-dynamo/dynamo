# Python Tokenizer Support for Dynamo Preprocessor

## Overview

This feature adds the ability to use Python-based tokenizers (SGLang, vLLM, or custom) in Dynamo's Rust preprocessor instead of the default HuggingFace Rust tokenizer. This enables support for models that require specialized tokenizers unsupported by HuggingFace's Rust implementation, such as `ministral/Ministral-3b-instruct` which uses the mistral-common tokenizer.

**Key difference from `--use-sglang-tokenizer`**: That flag bypasses Dynamo's preprocessor entirely, delegating all preprocessing to SGLang. This feature keeps Dynamo's preprocessor (chat templates, stop conditions, prompt formatting, etc.) but swaps only the tokenization layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dynamo Preprocessor                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Tokenizer Backend                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  HuggingFace (Rust)  │  Default, uses tokenizer.json    │   │
│  │  SGLang (Python)     │  Uses SGLang's get_tokenizer     │   │
│  │  vLLM (Python)       │  Uses vLLM's get_tokenizer       │   │
│  │  Custom (Python)     │  User-provided module/class      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Chat Template Formatting                    │   │
│  │              Stop Condition Handling                     │   │
│  │              Tool/Reasoning Integration                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Runtime Config

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerBackend {
    #[default]
    HuggingFace,  // Rust tokenizers crate
    Python,       // Custom Python tokenizer
    SGLang,       // SGLang's tokenizer
    VLLM,         // vLLM's tokenizer
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TokenizerConfig {
    pub backend: TokenizerBackend,
    pub python_module: Option<String>,  // Required for Python backend
    pub python_class: Option<String>,   // Required for Python backend
}
```

### CLI Arguments

```
--tokenizer-backend {huggingface,python,sglang,vllm}
    Tokenizer backend to use in Dynamo's preprocessor.
    'huggingface' uses the Rust tokenizer (default).
    'sglang' or 'vllm' use the respective framework's tokenizer.
    'python' uses a custom tokenizer.

--tokenizer-module <module>
    Python module path for custom tokenizer.
    Required when --tokenizer-backend is 'python'.
    Example: 'my_package.tokenizers'

--tokenizer-class <class>
    Class name in the tokenizer module.
    Required when --tokenizer-backend is 'python'.
    Example: 'MyCustomTokenizer'
```

## Python Protocol

All Python tokenizers must implement the `DynamoTokenizer` protocol:

```python
from typing import Protocol, Sequence, runtime_checkable

@runtime_checkable
class DynamoTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode multiple texts to token IDs."""
        ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        ...
```

### Base Class

A `BaseTokenizer` class provides a default `encode_batch` implementation:

```python
class BaseTokenizer:
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        raise NotImplementedError
```

## Rust Implementation

### Encoding Variant

The `Encoding` enum includes a `Py` variant for Python tokenizer output:

```rust
pub enum Encoding {
    Hf(Box<tokenizers::tokenizer::Encoding>),  // HuggingFace
    Sp(Vec<TokenIdType>),                       // SentencePiece
    Py(Vec<TokenIdType>),                       // Python tokenizer
}
```

### Python Tokenizer Wrapper

The `PythonTokenizer` struct wraps Python tokenizers via PyO3:

```rust
pub struct PythonTokenizer {
    module_path: String,
    class_name: String,
    model_path: String,
    py_tokenizer: Mutex<Option<Py<PyAny>>>,  // Lazy-initialized
}

impl PythonTokenizer {
    pub fn sglang(model_path: String) -> Self { ... }
    pub fn vllm(model_path: String) -> Self { ... }
    pub fn new(module: String, class: String, model: String) -> Self { ... }
}

impl Encoder for PythonTokenizer { ... }
impl Decoder for PythonTokenizer { ... }
impl Tokenizer for PythonTokenizer {}
```

### Thread Safety

The `PythonTokenizer` uses `Mutex<Option<Py<PyAny>>>` for thread-safe lazy initialization. All Python calls acquire the GIL via `Python::with_gil()`:

```rust
fn with_tokenizer<F, T>(&self, f: F) -> Result<T>
where
    F: FnOnce(&Bound<'_, PyAny>) -> Result<T>,
{
    let mut guard = self.py_tokenizer.lock()?;

    Python::with_gil(|py| {
        if guard.is_none() {
            // Initialize tokenizer
            let module = py.import(self.module_path.as_str())?;
            let cls = module.getattr(self.class_name.as_str())?;
            let tokenizer = cls.call1((self.model_path.as_str(),))?;
            *guard = Some(tokenizer.unbind());
        }

        let tokenizer = guard.as_ref().unwrap().bind(py);
        f(tokenizer)
    })
}
```

### Feature Flag

Python tokenizer support is gated behind the `python-tokenizer` feature:

```toml
# lib/llm/Cargo.toml
[features]
python-tokenizer = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.23.4", optional = true }
```

The Python bindings enable this by default:

```toml
# lib/bindings/python/Cargo.toml
[features]
default = ["python-tokenizer"]
python-tokenizer = ["dynamo-llm/python-tokenizer"]

[dependencies]
dynamo-llm = { path = "../../llm", features = ["python-tokenizer"] }
```

## Usage Examples

### SGLang Tokenizer

```bash
python -m dynamo.sglang \
    --model Qwen/Qwen3-0.6B \
    --tokenizer-backend sglang
```

### vLLM Tokenizer

```bash
python -m dynamo.sglang \
    --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --tokenizer-backend vllm

# TODO: Verify and see how to handle args like:
# --tokenizer-mode mistral --config-format mistral --load-format mistral
```

### Custom Python Tokenizer

```bash
python -m dynamo.sglang \
    --model my-model \
    --tokenizer-backend python \
    --tokenizer-module my_package.tokenizers \
    --tokenizer-class MyCustomTokenizer
```

### Example Custom Tokenizer

```python
# my_package/tokenizers.py
from dynamo.common.tokenizers import BaseTokenizer

class MyCustomTokenizer(BaseTokenizer):
    def __init__(self, model_path: str):
        # Load your tokenizer
        self._tokenizer = load_my_tokenizer(model_path)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens)
```

## Preprocessor Integration

The `OpenAIPreprocessor` checks `TokenizerConfig` during initialization:

```rust
impl OpenAIPreprocessor {
    pub fn new(mdc: ModelDeploymentCard) -> Result<Arc<Self>> {
        let tokenizer_config = mdc.runtime_config.tokenizer_config.clone();
        let use_python = tokenizer_config
            .as_ref()
            .map(|c| c.is_python_based())
            .unwrap_or(false);

        if use_python {
            let config = tokenizer_config.as_ref().unwrap();
            let tokenizer = create_tokenizer_from_config(
                config,
                mdc.source_path(),
                tokenizer_json_path.as_deref(),
            )?;
            Self::new_with_tokenizer(mdc, formatter, tokenizer)
        } else {
            // Default HuggingFace tokenizer
            let hf_tokenizer = mdc.tokenizer_hf()?;
            Self::new_with_parts(mdc, formatter, hf_tokenizer)
        }
    }
}
```

## Python Bindings

The `ModelRuntimeConfig` Python class exposes tokenizer configuration:

```python
from dynamo._core import ModelRuntimeConfig

config = ModelRuntimeConfig()

# Set SGLang tokenizer
config.set_tokenizer_config("sglang")

# Set custom Python tokenizer
config.set_tokenizer_config(
    "python",
    python_module="my.module",
    python_class="MyTokenizer",
)

# Read back
print(config.tokenizer_backend)        # "sglang" or "python"
print(config.tokenizer_python_module)  # "my.module"
print(config.tokenizer_python_class)   # "MyTokenizer"
```

## Design Considerations

1. **Backward Compatibility**: Default is `HuggingFace`; existing deployments unchanged.

2. **Performance**: Python tokenizer calls go through PyO3 with GIL acquisition. For high-throughput scenarios, consider whether the model actually requires a Python tokenizer.

3. **Lazy Initialization**: The Python tokenizer is only instantiated on first use, not at config time.

4. **Error Handling**: Import and instantiation errors are propagated with descriptive messages.

5. **Validation**: CLI validates that `--tokenizer-module` and `--tokenizer-class` are provided when `--tokenizer-backend python` is specified.

6. **Conflict Detection**: Using `--tokenizer-backend` with `--use-sglang-tokenizer` is an error, as they serve different purposes.

## Limitations

1. **GIL Contention**: All tokenizer calls acquire the Python GIL, which may impact parallelism in heavily concurrent workloads.

2. **Feature Flag Required**: The `python-tokenizer` feature must be enabled at compile time. Without it, Python backends return an error at runtime.

3. **No Streaming Tokenization**: The current implementation tokenizes complete strings. Incremental tokenization would require additional protocol methods.
