# Trace Replay for Stateful Responses

This document describes the trace replay capability for testing and debugging stateful responses without a live deployment.

## Overview

The trace replay system allows you to:

1. **Parse** Braintrust-style JSONL trace files
2. **Extract** conversation turns, tool calls, and LLM outputs
3. **Replay** conversations through the storage system
4. **Verify** session isolation and `previous_response_id` chaining

This enables local development and testing of multi-turn conversations using real captured traces.

## Trace Format

We support Braintrust JSONL format with span hierarchy:

```jsonl
{"id": "root-123", "span_id": "root-123", "root_span_id": "root-123", "metadata": {"session_id": "root-123"}, "span_attributes": {"name": "Session", "type": "task"}}
{"id": "turn-1", "span_id": "turn-1", "root_span_id": "root-123", "span_parents": ["root-123"], "input": "Hello", "span_attributes": {"name": "Turn 1", "type": "task"}}
{"id": "llm-1", "span_id": "llm-1", "root_span_id": "root-123", "span_parents": ["turn-1"], "output": {"role": "assistant", "content": "Hi!"}, "span_attributes": {"type": "llm"}}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `span_id` | Unique identifier for this span |
| `root_span_id` | Session root (same for all spans in session) |
| `span_parents` | Parent span IDs (forms hierarchy) |
| `span_attributes.type` | `task` for turns, `llm` for model calls, `tool` for tool calls |
| `metadata.session_id` | Session identifier (extracted from root span) |

## API Reference

### `parse_trace_file(path: &Path) -> Result<ParsedTrace, TraceParseError>`

Parse a JSONL trace file from disk.

```rust
use dynamo_llm::storage::parse_trace_file;
use std::path::Path;

let trace = parse_trace_file(Path::new("my_trace.jsonl"))?;
println!("Session: {}", trace.session_id);
println!("Turns: {}", trace.turns.len());
```

### `parse_trace_content(content: &str) -> Result<ParsedTrace, TraceParseError>`

Parse trace content from a string (useful for testing).

```rust
use dynamo_llm::storage::parse_trace_content;

let trace_content = r#"{"id": "root", "span_id": "root", ...}"#;
let trace = parse_trace_content(trace_content)?;
```

### `replay_trace(storage: &S, trace: &ParsedTrace, tenant_id: &str) -> Result<ReplayResult, StorageError>`

Replay a parsed trace through the storage system.

```rust
use dynamo_llm::storage::{replay_trace, InMemoryResponseStorage};

let storage = InMemoryResponseStorage::new();
let result = replay_trace(&storage, &trace, "my_tenant").await?;

println!("Replayed {} turns", result.turns_replayed);
println!("Response IDs: {:?}", result.response_ids);
```

## Data Structures

### `ParsedTrace`

```rust
pub struct ParsedTrace {
    pub session_id: String,        // Extracted from root span metadata
    pub root_span_id: String,      // Root span identifier
    pub turns: Vec<ConversationTurn>,  // Extracted conversation turns
    pub raw_spans: Vec<TraceSpan>,     // All parsed spans
}
```

### `ConversationTurn`

```rust
pub struct ConversationTurn {
    pub turn_id: String,           // Span ID of this turn
    pub turn_number: usize,        // 1-indexed turn number
    pub user_input: String,        // User's message
    pub assistant_output: Option<String>,  // Assistant's response
    pub tool_calls: Vec<ToolCall>, // Tool invocations in this turn
    pub timestamp: String,         // ISO timestamp
}
```

### `ReplayResult`

```rust
pub struct ReplayResult {
    pub session_id: String,        // Session that was replayed
    pub tenant_id: String,         // Tenant used for replay
    pub turns_replayed: usize,     // Number of turns stored
    pub response_ids: Vec<String>, // Generated response IDs
}
```

## Usage Examples

### Basic Trace Replay

```rust
use dynamo_llm::storage::{
    parse_trace_file, replay_trace,
    ResponseStorage, InMemoryResponseStorage
};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Parse the trace file
    let trace = parse_trace_file(Path::new("traces/session.jsonl"))?;

    // 2. Create storage
    let storage = InMemoryResponseStorage::new();

    // 3. Replay through storage
    let result = replay_trace(&storage, &trace, "test_tenant").await?;

    // 4. Verify the session
    let responses = storage
        .list_responses("test_tenant", &trace.session_id, None)
        .await?;

    for resp in responses {
        println!("Response {}: {:?}", resp.response_id, resp.response);
    }

    Ok(())
}
```

### Testing Session Isolation

```rust
// Replay same trace for different tenants
let result_a = replay_trace(&storage, &trace, "tenant_A").await?;
let result_b = replay_trace(&storage, &trace, "tenant_B").await?;

// Each tenant has isolated data
let a_responses = storage.list_responses("tenant_A", &trace.session_id, None).await?;
let b_responses = storage.list_responses("tenant_B", &trace.session_id, None).await?;

// Tenant A can't access tenant B's data
assert!(storage.get_response("tenant_A", &trace.session_id, &result_b.response_ids[0]).await.is_err());
```

### Verifying `previous_response_id` Chaining

```rust
let result = replay_trace(&storage, &trace, "tenant").await?;

// Each response after the first has previous_response_id set
let responses = storage.list_responses("tenant", &trace.session_id, None).await?;

for (i, resp) in responses.iter().enumerate().skip(1) {
    let prev_id = resp.response["metadata"]["previous_response_id"]
        .as_str()
        .expect("Should have previous_response_id");

    // Should point to previous response
    assert_eq!(prev_id, responses[i - 1].response_id);
}
```

## Integration with Tests

The trace replay module is used in integration tests:

```rust
// lib/llm/tests/responses_stateful_integration_test.rs

#[tokio::test]
async fn test_trace_replay_parse_real_trace() {
    let trace_content = r#"{"id": "4de2bcf7-...", ...}"#;
    let parsed = parse_trace_content(trace_content).expect("Should parse");

    assert_eq!(parsed.turns.len(), 1);
    assert_eq!(parsed.turns[0].user_input, "how many lines in the README.md?");
}

#[tokio::test]
async fn test_trace_replay_through_storage() {
    let storage = InMemoryResponseStorage::new();
    let trace = parse_trace_content(TRACE_CONTENT)?;

    let result = replay_trace(&storage, &trace, "test").await?;

    assert_eq!(result.turns_replayed, 2);
}
```

## File Locations

| File | Description |
|------|-------------|
| `lib/llm/src/storage/trace_replay.rs` | Core implementation |
| `lib/llm/src/storage/mod.rs` | Module exports |
| `lib/llm/tests/responses_stateful_integration_test.rs` | Integration tests |

## Future Enhancements

1. **CLI tool** - `dynamo trace replay <file.jsonl>` command
2. **HTTP endpoint** - POST trace content for live replay
3. **Diff comparison** - Compare replayed responses vs original trace
4. **Streaming support** - Handle streaming response traces
