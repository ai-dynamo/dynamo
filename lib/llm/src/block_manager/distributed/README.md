# Active Message Handling System

This module provides an async future-based active message handling system with proper error handling, response notifications, and channel-based communication.

## Key Features

- **Async Future-Based**: Handlers are `Arc<dyn Future>` that can capture resources and run asynchronously
- **Concurrency Control**: Configurable concurrency limits with semaphore-based throttling
- **Response Notifications**: Optional response notifications with `:ok` or `:err(<message>)` format
- **Channel-Based Communication**: All communication happens through channels for clean separation
- **Error Handling**: Comprehensive error handling with logging and monitoring
- **Resource Capture**: Handlers can capture and share resources safely

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Communication  │───▶│ ActiveMessage    │───▶│   Handler       │
│     Layer       │    │    Manager       │    │   Futures       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        │                       │
         │                        ▼                       ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         └──────────────│   Response       │◀───│   Async Task    │
                        │  Notifications   │    │    Pool         │
                        └──────────────────┘    └─────────────────┘
```

## Usage

### 1. Initialize the System

```rust
use dynamo_llm::block_manager::distributed::worker::*;

// Create a worker and initialize active message manager
let mut worker = KvBlockManagerWorker::new(config)?;
worker.init_active_message_manager(4)?; // 4 concurrent handlers

// Create handlers
let handlers = create_example_handlers();
worker.register_handlers(handlers)?;

// Get communication channels
let message_sender = worker.get_message_sender()?;
let response_receiver = worker.get_response_receiver()?;
```

### 2. Create Custom Handlers

```rust
#[derive(Clone)]
struct MyHandler {
    name: String,
    shared_resource: Arc<Mutex<SomeResource>>,
}

impl MyHandler {
    async fn handle_message(&self, data: Vec<u8>) -> Result<()> {
        // Process the message asynchronously
        let processed_data = self.process_data(data).await?;

        // Update shared resources
        let mut resource = self.shared_resource.lock().await;
        resource.update(processed_data)?;

        Ok(())
    }
}

// Register the handler
let handler = MyHandler::new("my_handler".to_string(), shared_resource);
let mut handlers = HashMap::new();
handlers.insert("my_message_type".to_string(), create_handler!(handler));
```

### 3. Send Messages

```rust
// Message with response notification
let message = IncomingActiveMessage {
    message_type: "my_message_type".to_string(),
    message_data: b"Hello, World!".to_vec(),
    response_notification: Some("request_123".to_string()),
};

message_sender.send(message)?;
```

### 4. Handle Responses

```rust
// Spawn a task to handle responses
tokio::spawn(async move {
    while let Some(response) = response_receiver.recv().await {
        match response.is_success {
            true => {
                info!("✅ Success: {}", response.notification);
                // response.notification = "request_123:ok"
            }
            false => {
                warn!("❌ Error: {}", response.notification);
                // response.notification = "request_123:err(Error message)"
            }
        }
    }
});
```

## Message Flow

1. **Incoming Message**: Communication layer receives bytes and optional response notification prefix
2. **Channel Send**: Message is sent through the channel to the active message manager
3. **Handler Lookup**: Manager finds the appropriate handler for the message type
4. **Future Creation**: Handler factory creates an async future with captured resources
5. **Async Execution**: Future is spawned in a task with concurrency control
6. **Response Generation**: On completion, response notification is generated (if requested)
7. **Response Send**: Response is sent back through the response channel

## Response Notification Format

- **Success**: `{prefix}:ok`
- **Error**: `{prefix}:err({error_message})`

Example:
- Request with notification prefix: `"user_request_456"`
- Success response: `"user_request_456:ok"`
- Error response: `"user_request_456:err(Invalid data format)"`

## Error Handling

The system provides multiple levels of error handling:

1. **Handler Errors**: Caught and converted to error response notifications
2. **Unknown Message Types**: Generate error responses for unregistered message types
3. **Channel Errors**: Logged and handled gracefully
4. **Concurrency Limits**: Managed with semaphores to prevent resource exhaustion

## Testing

Run the comprehensive test suite:

```bash
cargo test test_active_message_flow
cargo test test_resource_capturing_handler
cargo test test_communication_integration
cargo test test_concurrency_performance
```

## Performance Characteristics

- **Concurrency**: Configurable concurrent handler limit
- **Memory**: Efficient channel-based communication with minimal copying
- **Latency**: Low-latency message dispatch with async processing
- **Throughput**: High throughput with proper backpressure handling

## Best Practices

1. **Handler Design**: Keep handlers lightweight and async-friendly
2. **Resource Management**: Use `Arc<Mutex<T>>` for shared resources
3. **Error Handling**: Always handle errors gracefully in handlers
4. **Concurrency**: Set appropriate concurrency limits based on workload
5. **Monitoring**: Use the response notifications for monitoring and debugging
