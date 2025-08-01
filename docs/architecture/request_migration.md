# Request Migration Architecture

This document describes how Dynamo implements request migration to handle worker failures gracefully during LLM text generation. Request migration allows in-progress requests to continue on different workers when the original worker becomes unavailable, providing fault tolerance and improved user experience.

## Overview

Request migration is implemented through a Migration operator that sits in the LLM processing pipeline between the Backend operator and the service backend. When a worker fails during request processing, the migration system preserves the partial generation state and recreates the request on a new worker to continue from where the previous worker left off.

## Architecture Components

### Migrator

The migration system is integrated into the LLM processing pipeline between the frontend preprocessing and the actual service backends. This positioning allows it to intercept all communication flows and manage failure scenarios transparently.

Key responsibilities:
- Intercepts all requests and responses flowing through the pipeline
- Detects worker failure scenarios through error pattern matching
- Manages retry logic with configurable migration limits
- Tracks partial response state for seamless continuation

### Migration Limit Configuration

Each model can be configured with a migration limit parameter that specifies the maximum number of times a request can be migrated to another worker:

- Default behavior: no migration allowed
- Can be set independently for different engine types
- Applicable to LLM worker nodes that perform inference
- Allows engines to override user-specified limits for compatibility

## Token State Tracking and Request Migration

The core of the migration system is the ability to preserve and continue partial generations through token state management. This ensures that when a worker fails mid-generation, the new worker can seamlessly continue from the exact point of failure.

### Token Accumulation Process

When a request is being processed and responses are flowing back from a worker, the migration system tracks every token that has been successfully generated:

1. **Initial Request State**: The system starts with the original preprocessed request containing the initial prompt tokens.

2. **Response Tracking**: As each response arrives from the worker, the migration system extracts the newly generated tokens and appends them to the request's token sequence. This creates accumulates all tokens that have been generated.

3. **Token Count Management**: The system also updates the remaining token budget to reflect the number of tokens already generated, ensuring that the total generation stays within the originally requested limits.

### Migration Trigger Scenarios

The migration system handles two distinct failure scenarios:

#### 1. New Request Migration (Initial Connection Failure)

**Scenario**: Worker is unreachable when creating the initial connection.

**Error Pattern**: Communication system reports chosen worker instance is unavailable.

**Migration Process**:
- Detects connection failure during initial stream setup
- Decrements migration retry count
- Attempts to create a new stream with the original request
- No partial state to preserve since generation hasn't started

#### 2. Ongoing Request Migration (Mid-Stream Disconnection)

**Scenario**: Connection lost during active generation after partial responses have been received.

**Error Pattern**: Stream termination detected before generation completion.

**Migration Process**:

1. **Failure Detection**: The system detects the stream disconnection through error monitoring.

2. **State Preservation**: At this point, the request's token sequence contains both the original prompt tokens and all successfully generated tokens from the failed worker.

3. **New Stream Creation**: A fresh stream is created with the accumulated request state, ensuring the new worker has complete context.

4. **Continuation**: The new worker receives the request with the full token context and continues generation from the exact point where the previous worker left off.

### Seamless Token Flow and Request State Evolution

From the client's perspective, the token stream appears continuous and uninterrupted. The client receives tokens from the first worker until failure occurs, then seamlessly continues receiving tokens from the backup worker without any indication of the underlying migration.

The request state evolves dynamically during processing. Initially, the request contains only the original prompt tokens. As generation proceeds, each successfully generated token is appended to the request's token sequence, creating a growing record of the complete conversation context.

When a migration occurs, this accumulated state is transferred to the new worker, which uses it to reconstruct the complete context. The new worker then continues generation as if it had been processing the request from the beginning, but starting from the current position in the sequence.

The migration is transparent because:
1. No tokens are lost or duplicated during the transition
2. The new worker has complete context via the accumulated token sequence
3. Generation continues from the exact failure point
4. Response streaming maintains consistent format and timing

This token accumulation mechanism ensures that migrations are truly seamless, preserving all computational work and maintaining generation quality across worker transitions.

## Benefits

1. **Fault Tolerance**: System continues operating during individual worker failures
2. **Resource Efficiency**: Partial generations are preserved rather than restarted
3. **Seamless User Experience**: Users experience no interruption during worker failures
4. **Configurable Behavior**: Migration limits allow tuning based on deployment requirements
5. **No Token Loss**: Complete preservation of generation state across migrations

## Design Considerations

The migration system is designed with several important architectural considerations:

**Engine Compatibility**: Different LLM engines may have varying capabilities for handling migrated requests. The system allows engines to override migration settings to ensure compatibility and correctness.

**Multi-Model Support**: Since a frontend may serve multiple models simultaneously, migration limits can be configured at the engine level, providing flexibility for different model types with varying reliability characteristics.

**State Management**: The system carefully tracks not only token sequences but also metadata such as remaining token budgets, stop conditions, and sampling parameters to ensure complete state preservation.

**Error Handling**: The migration system distinguishes between different types of failures and applies appropriate recovery strategies for each scenario.

## Operational Impact

Request migration fundamentally changes how the system handles failures, moving from a "fail-fast" approach to a "graceful degradation" model. This architectural shift enables higher availability and better resource utilization while maintaining the same external API contract for clients.
