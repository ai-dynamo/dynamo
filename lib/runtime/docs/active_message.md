# ActiveMessage System

The ActiveMessage system provides a distributed messaging framework built on top of ZeroMQ, designed for leader-worker architectures with dynamic handler registration, ACK/NACK tracking, and service discovery.

## Architecture Overview

The system uses a **Publisher-Subscriber** pattern where:
- Each service binds a **SUB socket** to receive messages
- Services create ephemeral **PUB sockets** to send messages (per-use pattern)
- Transport is automatically detected (IPC for same-host, TCP for cross-host)
- Built-in system handlers provide core functionality (prefixed with `_`)

## Core Components

### ActiveMessage

The core message structure that flows through the system. Contains:
- `message_id` - Unique message identifier (UUID)
- `handler_name` - Target handler name (String)
- `sender_instance` - Sender's instance ID (UUID)
- `payload` - Message payload as bytes (JSON recommended)
- `metadata` - Additional metadata for routing and processing

Messages are created using `ActiveMessage::new()` and can include metadata for routing and processing.

### ActiveMessageHandler

Trait for implementing message handlers with automatic JSON schema validation. Handlers receive messages, process them asynchronously, and can optionally send responses or ACK/NACK confirmations.

Key methods:
- `handle()` - Process incoming messages
- `name()` - Return handler identifier
- `schema()` - Optional JSON schema for payload validation

### ActiveMessageClient

Interface for sending messages and managing peer connections. Provides methods for:
- Sending messages to specific handlers
- Broadcasting to all connected peers
- Managing peer connections
- Waiting for handler availability

### ActiveMessageManager

Interface for managing handlers and service lifecycle. Handles:
- Handler registration/deregistration
- Service startup/shutdown
- Handler event notifications

## ZMQ Implementation

### ZmqActiveMessageManager

The main entry point for creating an ActiveMessage service. Creates and manages the ZMQ transport layer, handler registry, and background tasks.

### ZmqActiveMessageClient

Provides message sending with per-use PUB socket pattern. Handles peer discovery, connection management, and message delivery.

### LeaderWorkerCohort

Helper for managing groups of workers in leader-worker patterns. Supports broadcasting to worker groups and tracking ACK/NACK responses from multiple workers.

## Built-in System Handlers

The system provides several built-in handlers (all prefixed with `_`):

### `_ack` - Acknowledgment Handler
Handles ACK/NACK responses for tracked messages. Used internally by the ACK system to process confirmations and error responses.

### `_register_service` - Service Registration
Allows workers to register themselves with a leader. Accepts JSON payload with instance_id and endpoint fields.

### `_list_handlers` - Handler Discovery
Returns list of available handlers on a service (useful for debugging and service discovery).

### `_wait_for_handler` - Handler Availability
Blocks until a specific handler becomes available on the target service. Accepts handler_name and optional timeout_ms.

### `_health_check` - Health Check
Simple health check endpoint for service monitoring. Returns status and timestamp.

## Usage Patterns

### Basic Service Setup

1. Create a `ZmqActiveMessageManager` with desired endpoint
2. Register custom handlers using `register_handler()`
3. Connect to peer services using the client
4. Send messages to handlers on connected peers

### Leader-Worker Pattern

**Leader:**
- Binds to a well-known endpoint
- Creates a `LeaderWorkerCohort` to manage worker groups
- Workers register via `_register_service` handler
- Broadcasts tasks using cohort methods

**Worker:**
- Binds to random port or specified endpoint
- Registers task handlers
- Connects to leader and sends registration message
- Processes incoming tasks from leader

### ACK/NACK Tracking

The system supports reliable message delivery with ACK/NACK confirmations:

1. **Automatic ACK/NACK**: Use message builder's `send()` method for automatic confirmation
2. **Manual ACK tracking**: Register ACK expectations and handle responses manually
3. **Payload Validation**: Handlers with schemas automatically send NACK on validation failures
4. **Cohort ACKs**: LeaderWorkerCohort can wait for ACKs from multiple workers

### Handler Configuration

Handlers can be configured with custom TaskTrackers for execution control, scheduling policies, and error handling strategies.

## Error Handling

- All operations return `anyhow::Result<T>`
- Message deserialization errors are handled gracefully
- Network failures are logged and operations can be retried
- Unmatched ACKs/NACKs are logged as errors
- Handler panics are caught by TaskTracker
- NACK responses include error messages for debugging

## Transport Detection

The system automatically optimizes transport:
- **IPC sockets** (`ipc://`) for same-host communication
- **TCP sockets** (`tcp://`) for cross-host communication
- Detection is based on endpoint host comparison

## Lifecycle Management

Proper shutdown sequence:
1. Call `manager.shutdown().await?` to join all handler task trackers
2. Call `cancel_token.cancel()` to stop background tasks
3. All pending operations complete gracefully

## Thread Safety

All components are thread-safe:
- `ActiveMessageHandler` requires `Send + Sync`
- Clients and managers can be shared across async tasks
- Message handling is concurrent per handler via TaskTracker
- ACK/NACK tracking is thread-safe with internal synchronization

This system is designed to be robust, scalable, and easy to use for building distributed applications with dynamic service discovery and reliable message delivery with comprehensive error handling.