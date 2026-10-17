<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Endpoint ownership of chat generation controls

`add_generation_prompt` and `continue_final_message` belong to
`NvCreateChatCompletionRequest`, not `CommonExt`. This is a Rust ownership
change, not a new JSON namespace. Chat requests still send both fields at the
root; existing valid root-level requests retain their meaning.

```json
{
  "model": "example",
  "messages": [{"role": "assistant", "content": "The answer is"}],
  "add_generation_prompt": false,
  "continue_final_message": true
}
```

The existing default for `add_generation_prompt` remains true. Continuing the
final message requires explicit `add_generation_prompt=false`. Rust consumers
now read `request.add_generation_prompt` and
`request.continue_final_message`, rather than the corresponding `common`
members. The provider trait still exposes continuation to preprocessing.

Generated request schemas expose these controls on chat requests only.
`CommonExt` is a flattened Rust grouping for nonstandard fields shared by chat
and text completion requests; it is neither a universal protocol base nor a
promise of support by every backend.

Text completions reject either control whenever the key is present, including
`null`, `false` and malformed values. Validation runs before the generic
unsupported-field ignore policy, so that policy cannot silently discard these
chat-only controls. Error messages name the endpoint and fields without echoing
the supplied values. Clients that sent null chat-only keys to text completions
must omit those keys.

This branch does not change logprob types, response projection, native-field
inventory, backend capability admission, error envelopes or Python processor
behavior. Constructor updates elsewhere are required by the Rust field move;
the chat JSON layout and worker protocol are unchanged. Tests cover root-field
roundtrips, schemas, completion rejection and the existing continuation/template
paths; they are not a broad native-server conformance claim.
