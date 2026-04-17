# lib/protocols

OpenAI-compatible request/response types for Dynamo's HTTP surface. Built on top of the `async-openai` crate, with selective Dynamo-owned overrides where we need behaviors upstream won't accept or hasn't merged yet.

If you're extending or debugging types here, read this whole file before editing. The central question every change hinges on is: **do we re-export this upstream, or do we own it?** This document exists so the answer is consistent.

## The core tension

`async-openai` is well-maintained but slow on input-laxity PRs. The maintainer generally wants changes to match the OpenAPI spec exactly, even when OpenAI's *hosted* API accepts more permissive shapes on input than the spec requires. See `64bit/async-openai#535` (optional `ReasoningItem.id`) and prior work on optional `OutputMessage.id`/`status` — both driven by real Agents-SDK / Codex traffic that the spec technically rejects.

We can't block Dynamo on upstream merges. We also can't fork the whole crate — it's enormous and updates often. The rule we settled on is: **re-export upstream by default; own the narrowest type subtree that lets us fix the behavior we need.**

## The ownership rubric

Default to upstream. Own a type only when at least one of these is true:

1. **Upstream rejects a shape real clients send.** The driving case. Example: `OutputMessage.id`/`status`/`annotations` marked required upstream but routinely omitted by Codex / Agents SDK on input.
2. **We need to extend the schema with a Dynamo-specific field** that doesn't belong upstream. Example: `CreateChatCompletionRequest.mm_processor_kwargs` (vLLM multimodal), `ChatCompletionRequestAssistantMessage.reasoning_content` (R1 / QwQ), `ChatCompletionStreamOptions.continuous_usage_stats`.
3. **Upstream's type forces a shape that breaks downstream backends.** Example: `FunctionCall.arguments` is `String` upstream; LangChain and similar send it as an object. We own `FunctionCall` to accept both via a custom deserializer and normalize to `String`.
4. **Upstream has a known bug.** Example: `ChatCompletionMessageToolCall.type` wasn't always serialized; we own it with `#[serde(default = "default_function_type")]` to preserve wire compat.

**Do not own a type just because an adjacent type is owned.** Keep the blast radius small. If owning `OutputMessage` would cascade into owning `Response`, `OutputItem`, streaming events, and half the crate — stop and find a narrower fix (see "Naming: avoiding dual-side collisions" below).

## Layout

- `src/types/chat.rs` — Chat Completions (request, response, stream, messages). Extensively owned: multimodal content, reasoning, continuous usage stats, flexible `arguments`.
- `src/types/responses/mod.rs` — Responses API (Codex, Agents SDK). Input chain owned; output chain fully upstream.
- `src/types/completion.rs` — Legacy completions. Mostly upstream.
- `src/types/anthropic.rs` — Anthropic Messages API. Fully owned (no upstream equivalent in `async-openai`).
- `src/types/embeddings`, `src/types/images` — full upstream re-export (no Dynamo extensions).

## Re-export conventions

Use **explicit re-exports** (`pub use foo::{A, B, C}`), not globs, when you need to selectively shadow. Globs (`pub use foo::*`) are allowed at the top of a module — Rust lets a local `pub struct Foo` shadow a glob-imported `Foo` (the glob just emits `unused_imports` warnings). But explicit lists make the ownership split obvious to readers and catch mistakes at compile time when upstream renames or removes a type.

`src/types/responses/mod.rs` uses glob re-export because the surface is huge (200+ types). `src/types/chat.rs` uses explicit lists because the surface is manageable and Dynamo owns more of it. Either pattern is acceptable; pick based on how many types you'd have to enumerate to exclude the ones you own.

## Naming: avoiding dual-side collisions

**The trap.** Upstream sometimes reuses the same type on both request-input and response-output sides. `OutputMessage` is the canonical example: it appears inside `MessageItem::Output(...)` (input side — a prior assistant turn echoed back) AND inside `OutputItem::Message(...)` (output side — the assistant message we just produced).

If we relax `OutputMessage` (make `id`/`status` optional) and shadow upstream's name, every place that constructs an `OutputItem::Message(OutputMessage { ... })` on the output side breaks: `OutputItem::Message` variant holds upstream's type, not ours, and our relaxed struct doesn't match.

The naive fix is to also own `OutputItem`. But that cascades into owning `Response`, streaming events, and a long tail of their sub-types. The right fix is smaller:

**Rule.** If a type is reused by upstream on both input and output sides, give the Dynamo-owned input-side variant a *different name*. The output side keeps using upstream's name via the glob / explicit re-export.

Current naming in `responses/mod.rs`:

- `InputOutputMessage` — Dynamo-owned, relaxed; used in `MessageItem::Output(...)` on the input side.
- `OutputMessage` — upstream, unchanged; used in `OutputItem::Message(...)` on the output side.
- Same pattern for `InputOutputMessageContent` (input) vs upstream `OutputMessageContent` (output), and `InputOutputTextContent` (input) vs upstream `OutputTextContent` (output).

Input-only types can shadow upstream with the same name — no conflict. Current shadows: `MessageItem`, `Item`, `InputItem`, `InputParam`, `CreateResponse`.

## The Responses input chain, specifically

As of this writing, the owned input chain is:

```
CreateResponse
└── input: InputParam            (shadow)
    └── InputItem                (shadow)
        ├── ItemReference        (upstream)
        ├── EasyInputMessage     (upstream)
        └── Item                 (shadow, mirrors upstream variant-for-variant)
            ├── Message(MessageItem)  (shadow)
            │   ├── Input(InputMessage)  (upstream)
            │   └── Output(InputOutputMessage)  (NEW NAME — relaxed)
            │       └── content: Vec<InputOutputMessageContent>  (NEW NAME)
            │           └── OutputText(InputOutputTextContent)   (NEW NAME — relaxed)
            └── ... 19 other upstream variants (FunctionCall, Reasoning, etc.)
```

`Item` mirrors upstream variant-for-variant because it's a `#[serde(tag = "type")]` enum — we can't inherit variants. If upstream adds a new variant to their `Item`, we must add it here too, or payloads carrying that type will fail to deserialize. This is the one place where upstream drift bites us; accept it as the cost of owning the chain.

The output chain (`Response`, `OutputItem`, `OutputMessage`, streaming events, etc.) is fully upstream. We mint valid id/status on output, so there's no lenience needed and no reason to own it.

## When upstream finally merges a relaxation

If an upstream PR lands that makes a field optional (matching what we relaxed), the checklist is:

1. Bump `async-openai` in `Cargo.toml`.
2. Delete the owned override if it's now identical to upstream, or narrow it if upstream only partially relaxed.
3. Update consumer sites (convert `Option<T>` to `T` if upstream still has the field but non-optional, etc.).
4. Run the full test suite; the serialization-shape tests should catch any regressions.

Don't leave redundant Dynamo-owned types in place "just in case." Dead ownership is tech debt.

## When upstream renames or restructures a type we re-export

Glob re-exports will silently pick up the rename. Explicit re-exports will fail to compile — which is the point. Update the explicit list and any consumer code, confirm no semantic drift, run tests.

## Testing patterns

- Serialization-shape tests (`test_response_wire_format_shape` in `lib/llm`) validate that our serialized JSON matches the API spec. Lean on these when you change owned types.
- Deserialization tests for owned types should cover both the relaxed shape (the reason we own it) and the strict shape (to prove we didn't break spec-conformant clients).
- When you add a new Dynamo field to an owned type, add a test that omits it and asserts the default behavior.

## Things that are explicitly *not* this crate's job

- HTTP transport (request execution, retries, streaming frame parsing) — that's `lib/llm/src/http/`.
- Semantic conversion between API types (Responses → Chat, Anthropic → Chat, etc.) — that lives in `lib/llm/src/protocols/` and uses the types defined here.
- Model-specific tokenization or prompt templating.

Keep this crate declarative: types, serde derives, builders, conversions-by-`From`. Business logic belongs downstream.

## Common mistakes

- Owning a type because it's *nearby* a bug, not because of the bug itself. Narrow the fix.
- Shadowing a dual-side type without checking output-side construction sites. `grep` the workspace for constructor calls before renaming.
- Adding fields to upstream-re-exported types via `#[serde(default)]` on a local wrapper struct. Doesn't work — serde can't inject defaults into a foreign type unless you use `#[serde(remote)]`, which requires field-for-field mirroring and doesn't help with optional-vs-required mismatches.
- Forgetting to update `From` impls when adding variants. The compiler catches exhaustive matches but not variant count on `From<Ours> for Upstream` when the enum is non-exhaustive.
