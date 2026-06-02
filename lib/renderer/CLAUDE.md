# lib/renderer

`dynamo-renderer` is the *encode* side of serving: it renders OpenAI-style chat
requests into prompt strings (HF `chat_template` via minijinja, plus native
DeepSeek formatters). It is standalone and runtime-free.

## Invariants

- **No tokenizer dependency on the rendering path.** This crate is a bridge
  between `dynamo-protocols` request types and the `minijinja` template engine;
  it must not start depending on `dynamo-tokenizers` internals. The
  `pub use dynamo_tokenizers;` re-export is a consumer convenience only.
- **Protocol coupling is intentional.** `dynamo-protocols` is a real dependency:
  this crate owns the `OAIChatLikeRequest` trait and ships the default impl for
  the standard OpenAI chat request, so external consumers need no wrapper.
- **Rendering stays runtime-free** — no async runtime, no networking, no I/O on
  the render path. Keep it that way so external frontends can embed it cheaply.

## Layout

- `lib.rs` — the `OAIChatLikeRequest` rendering trait, `PromptFormatter`,
  input types, and re-exports.
- `template.rs` + `template/` — HF `chat_template` rendering (jinja, OAI request
  adaptation, `tokenizer_config.json` parsing).
- `deepseek/` — native (non-jinja) formatters for DeepSeek families
  (`common`, `v4`, `v32`).

## Consumers

`lib/llm` wires this crate in via `crate::preprocessor::prompt`, which holds the
lib/llm-local glue that can't live here (the `Nv*` request impls, the
`MediaRequestExt` extension trait, and `prompt_formatter_from_mdc`). Everything
else imports `dynamo_renderer` directly.
