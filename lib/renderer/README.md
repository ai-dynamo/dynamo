# Renderer

## Introduction
`dynamo-renderer` turns OpenAI-style chat requests into model-ready prompt strings. It is the *encode* side of inference serving: messages + tools + generation settings in, a fully-rendered prompt out. It is standalone and runtime-free, so an external OpenAI frontend can reuse Dynamo's prompt formatting without pulling in the Dynamo runtime.

It renders HuggingFace `chat_template` jinja2 (via `minijinja` + `minijinja-contrib` pycompat) and also ships native Rust formatters for DeepSeek families whose repos ship no usable template. The crate is a *bridge* between OpenAI request types ([`dynamo-protocols`]) and the template engine; it does **not** depend on tokenizer internals, though it re-exports [`dynamo-tokenizers`] for one-import convenience.

## Features
- **HF chat templates**: faithful `apply_chat_template` rendering, including tool-use and generation-prompt handling.
- **Native DeepSeek formatters**: Rust formatters for V4 / V3.2 families (under [`deepseek`]).
- **Bring-your-own request type**: implement [`OAIChatLikeRequest`] for any request type, or use the ready-made impl for `dynamo-protocols`' OpenAI chat request.
- **Runtime-free**: no async runtime, no networking, no tokenizer dependency on the rendering path.

## Quick Start

```rust
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptFormatter};
use dynamo_protocols::types::CreateChatCompletionRequest;

// `config` is parsed from a model's `tokenizer_config.json`.
let config: ChatTemplate = serde_json::from_str(tokenizer_config_json)?;
let PromptFormatter::OAI(formatter) =
    PromptFormatter::from_parts(config, ContextMixins::default(), /* exclude_tools_when_tool_choice_none */ false)?
else {
    unreachable!("from_parts always builds an OAI formatter")
};

// Any type implementing `OAIChatLikeRequest` can be rendered; the standard
// OpenAI chat request works out of the box.
let request: CreateChatCompletionRequest = serde_json::from_str(request_json)?;
let prompt: String = formatter.render(&request)?;
```

## Relationship to other crates
- [`dynamo-protocols`] — OpenAI/wire request types this crate renders from.
- [`dynamo-tokenizers`] — tokenization (the *next* step after rendering); re-exported here for convenience.
- [`dynamo-parsers`] — the *decode* side (parsing model output back into reasoning / tool calls).
