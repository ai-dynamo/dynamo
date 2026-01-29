---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# JailedStream Implementation

## Overview

The `JailedStream` is a standalone implementation for handling "jail" detection in token streams. It provides a clean, builder-based API for accumulating tokens when certain sequences are detected, then releasing them as a single chunk when the jail ends.

## Key Features

- **Builder Pattern**: Clean configuration API using the builder pattern
- **Configurable Sequences**: Support for multiple start/end jail sequences
- **Tool Call Parsing**: Integrated tool call detection and parsing
- **Stream Macro**: Uses `async-stream::stream!` for clean async implementation
- **Standalone**: Completely independent of existing code
- **Annotations**: Preserves annotations for observability

## Implementation

### Location
- Main implementation: `lib/llm/src/protocols/openai/chat_completions/jail.rs`
- Examples: `lib/llm/src/protocols/openai/chat_completions/jail_example.rs`

### Usage

```rust
use crate::protocols::openai::chat_completions::jail::JailedStream;
use dynamo_runtime::engine::{AsyncEngineContextProvider, ResponseStream};

// Get your ResponseStream with context
let response_stream: Pin<Box<ResponseStream<_>>> = get_stream_from_engine();

// Extract context BEFORE passing to apply
let context = response_stream.context();

// Apply jail transformation (ResponseStream implements Stream)
let jail = JailedStream::builder()
    .tool_call_parser("nemotron_deci")
    .build();

let jailed_stream = jail.apply(response_stream);

// Re-wrap with context when needed for engine consumption
let final_stream = ResponseStream::new(Box::pin(jailed_stream), context);
```

### Advanced Configuration

```rust
// With custom jail sequences
let jail = JailedStream::builder()
    .jail_start_sequence("<TOOLCALL>")
    .jail_end_sequence("</TOOLCALL>")
    .tool_call_parser("nemotron_deci")
    .build();

// With multiple sequences
let jail = JailedStream::builder()
    .jail_start_sequences(vec!["<TOOLCALL>", "<FUNCTION>"])
    .jail_end_sequences(vec!["</TOOLCALL>", "</FUNCTION>"])
    .tool_call_parser("harmony")
    .build();
```

## How It Works

1. **Detection**: When a jail start sequence (or tool call start) is detected, the stream enters "jail" mode
2. **Accumulation**: While jailed, tokens are accumulated in memory instead of being yielded
3. **Annotations**: Empty chunks with annotations are sent downstream for observability
4. **Release**: When a jail end sequence is detected OR the stream ends:
   - Accumulated content is parsed for tool calls
   - A single chunk with the parsed content is yielded
5. **Pass-through**: Non-jailed content passes through unchanged

## Testing

The implementation includes comprehensive tests:

- `test_jailed_stream_with_start_end_sequences`: Tests explicit jail sequences
- `test_jailed_stream_with_tool_calls`: Tests tool call detection and parsing
- `test_jailed_stream_no_jailing`: Tests normal pass-through behavior

Run tests with:
```bash
cargo test -p dynamo-llm jail --lib
```

## Benefits

1. **Standalone**: No modifications to existing code required
2. **Clean API**: Builder pattern makes configuration intuitive
3. **Flexible**: Supports multiple jail detection strategies
4. **Maintainable**: Uses `stream!` macro for cleaner async code
5. **Testable**: Comprehensive test suite with shared utilities
6. **Efficient**: No unnecessary boxing or context handling in the library
7. **Composable**: Can chain multiple stream transformers before re-adding context

## Performance Optimizations

- **No Boxing in Library**: Returns `impl Stream` instead of `Pin<Box<ResponseStream>>`
- **Stack Pinning**: Uses `tokio::pin!()` instead of `Box::pin()` for better performance
- **No Context Overhead**: JailedStream doesn't manage AsyncEngineContext
- **Lazy Evaluation**: Only processes what's needed
- **Efficient State Management**: Minimal cloning, only when entering jail state

## Integration Options

To replace the existing `apply_tool_calling_jail_internal` function:

```rust
// In preprocessor.rs
pub fn apply_tool_calling_jail_with_parser(
    &self,
    stream: ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
) -> ManyOut<Annotated<NvCreateChatCompletionStreamResponse>> {
    let jail = JailedStream::builder()
        .tool_call_parser(self.tool_call_parser.clone())
        .build();

    jail.apply(stream)
}
```

## Future Enhancements

- Add support for regex patterns for jail sequences
- Add metrics/telemetry for jail detection
- Support for partial sequence matching across chunk boundaries
- Configurable accumulation limits
- Support for nested jails