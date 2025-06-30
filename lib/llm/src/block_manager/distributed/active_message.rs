//! # Active Message System
//!
//! This module provides an async future-based active message handling system with proper error handling,
//! response notifications, and channel-based communication. The system is split into separate sender and
//! receiver components for better separation of concerns.

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use anyhow::{Context, Result};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, instrument, warn};

/// Type alias for the active message handler future
pub type ActiveMessageFuture = Pin<Box<dyn Future<Output = Result<()>> + Send + 'static>>;

/// Factory function type for creating active message handlers
pub type ActiveMessageHandlerFactory =
    Arc<dyn Fn(Vec<u8>) -> ActiveMessageFuture + Send + Sync + 'static>;

/// Incoming active message with optional response notification
#[derive(Debug, Clone)]
pub struct IncomingActiveMessage {
    /// The message type identifier
    pub message_type: String,
    /// The message payload
    pub message_data: Vec<u8>,
    /// Optional response notification prefix
    pub response_notification: Option<String>,
}

/// Response notification to be sent back
#[derive(Debug, Clone)]
pub struct ResponseNotification {
    /// The notification identifier (includes :ok or :err suffix)
    pub notification: String,
    /// Whether this was a success or error response
    pub is_success: bool,
}

/// Channel types for active message communication
pub type ActiveMessageChannel = mpsc::UnboundedSender<IncomingActiveMessage>;
pub type ResponseChannel = mpsc::UnboundedReceiver<ResponseNotification>;

/// Shared response receiver that can be cloned
#[derive(Clone)]
pub struct SharedResponseReceiver {
    receiver: Arc<tokio::sync::Mutex<ResponseChannel>>,
}

impl SharedResponseReceiver {
    pub fn new(receiver: ResponseChannel) -> Self {
        Self {
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
        }
    }

    pub async fn recv(&self) -> Option<ResponseNotification> {
        let mut receiver = self.receiver.lock().await;
        receiver.recv().await
    }
}

/// Active message sender - handles sending messages to the receiver
#[derive(Clone)]
pub struct ActiveMessageSender {
    /// Channel for sending messages to the receiver
    message_sender: ActiveMessageChannel,
}

impl ActiveMessageSender {
    /// Create a new active message sender
    pub fn new(message_sender: ActiveMessageChannel) -> Self {
        Self { message_sender }
    }

    /// Send an active message
    pub fn send(&self, message: IncomingActiveMessage) -> Result<()> {
        self.message_sender
            .send(message)
            .map_err(|e| anyhow::anyhow!("Failed to send active message: {}", e))
    }

    /// Send a message with automatic response notification generation
    pub fn send_with_response(
        &self,
        message_type: String,
        message_data: Vec<u8>,
        response_id: String,
    ) -> Result<()> {
        let message = IncomingActiveMessage {
            message_type,
            message_data,
            response_notification: Some(response_id),
        };
        self.send(message)
    }

    /// Send a fire-and-forget message (no response notification)
    pub fn send_fire_and_forget(&self, message_type: String, message_data: Vec<u8>) -> Result<()> {
        let message = IncomingActiveMessage {
            message_type,
            message_data,
            response_notification: None,
        };
        self.send(message)
    }
}

/// Active message receiver - handles message processing and response generation
pub struct ActiveMessageReceiver {
    /// Maximum number of concurrent message handlers
    concurrency: usize,
    /// Registered message handlers
    handlers: HashMap<String, ActiveMessageHandlerFactory>,
    /// Receiver for incoming messages
    message_receiver: mpsc::UnboundedReceiver<IncomingActiveMessage>,
    /// Sender for response notifications
    response_sender: mpsc::UnboundedSender<ResponseNotification>,
    /// Shared response receiver for external access
    response_receiver: SharedResponseReceiver,
    /// Cancellation token for graceful shutdown
    cancellation_token: CancellationToken,
    /// Driver task handle
    driver_handle: Option<JoinHandle<Result<()>>>,
}

impl ActiveMessageReceiver {
    /// Create a new active message receiver
    pub fn new(
        concurrency: usize,
        cancellation_token: CancellationToken,
    ) -> Result<(Self, ActiveMessageSender)> {
        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        let (response_sender, response_receiver) = mpsc::unbounded_channel();
        let shared_response_receiver = SharedResponseReceiver::new(response_receiver);

        let receiver = Self {
            concurrency,
            handlers: HashMap::new(),
            message_receiver,
            response_sender,
            response_receiver: shared_response_receiver.clone(),
            cancellation_token,
            driver_handle: None,
        };

        let sender = ActiveMessageSender::new(message_sender);

        Ok((receiver, sender))
    }

    /// Register message handlers
    pub fn register_handlers(
        &mut self,
        handlers: HashMap<String, ActiveMessageHandlerFactory>,
    ) -> Result<()> {
        for (message_type, handler) in handlers {
            self.handlers.insert(message_type, handler);
        }
        Ok(())
    }

    /// Register a single handler
    pub fn register_handler(
        &mut self,
        message_type: String,
        handler: ActiveMessageHandlerFactory,
    ) -> Result<()> {
        self.handlers.insert(message_type, handler);
        Ok(())
    }

    /// Get a shared receiver for response notifications
    pub fn get_response_receiver(&self) -> SharedResponseReceiver {
        self.response_receiver.clone()
    }

    /// Start the receiver's background processing task
    pub fn start(&mut self) -> Result<()> {
        if self.driver_handle.is_some() {
            return Err(anyhow::anyhow!("Receiver is already started"));
        }

        let driver_handle = self.start_driver_task()?;
        self.driver_handle = Some(driver_handle);
        Ok(())
    }

    /// Stop the receiver and wait for graceful shutdown
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.driver_handle.take() {
            self.cancellation_token.cancel();

            // Wait for the driver task to complete
            match tokio::time::timeout(std::time::Duration::from_secs(5), handle).await {
                Ok(result) => {
                    result.map_err(|e| anyhow::anyhow!("Driver task failed: {}", e))?;
                }
                Err(_) => {
                    warn!("Driver task did not complete within timeout, aborting");
                    return Err(anyhow::anyhow!("Driver task shutdown timeout"));
                }
            }
        }
        Ok(())
    }

    /// Start the driver task that processes incoming messages
    fn start_driver_task(&mut self) -> Result<JoinHandle<Result<()>>> {
        // Move the message receiver out of self
        let mut message_receiver = std::mem::replace(
            &mut self.message_receiver,
            mpsc::unbounded_channel().1, // Dummy receiver
        );

        let handlers = self.handlers.clone();
        let response_sender = self.response_sender.clone();
        let cancellation_token = self.cancellation_token.clone();
        let concurrency = self.concurrency;

        let handle = tokio::spawn(async move {
            let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));

            info!(
                "Active message receiver started with concurrency: {}",
                concurrency
            );

            loop {
                tokio::select! {
                    // Handle cancellation
                    _ = cancellation_token.cancelled() => {
                        info!("Active message receiver shutting down due to cancellation");
                        break;
                    }

                    // Process incoming messages
                    message = message_receiver.recv() => {
                        match message {
                            Some(incoming_message) => {
                                let permit = semaphore.clone().acquire_owned().await
                                    .context("Failed to acquire semaphore permit")?;

                                Self::spawn_message_handler(
                                    incoming_message,
                                    handlers.clone(),
                                    response_sender.clone(),
                                    permit,
                                ).await?;
                            }
                            None => {
                                info!("Message channel closed, shutting down receiver");
                                break;
                            }
                        }
                    }
                }
            }

            info!("Active message receiver task completed");
            Ok(())
        });

        Ok(handle)
    }

    /// Spawn a handler for an individual message
    #[instrument(skip(handlers, response_sender, _permit), fields(message_type = %message.message_type))]
    async fn spawn_message_handler(
        message: IncomingActiveMessage,
        handlers: HashMap<String, ActiveMessageHandlerFactory>,
        response_sender: mpsc::UnboundedSender<ResponseNotification>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Result<()> {
        let message_type = message.message_type.clone();
        let response_notification = message.response_notification.clone();

        // Find the appropriate handler
        let handler_factory = match handlers.get(&message_type) {
            Some(factory) => factory.clone(),
            None => {
                warn!("No handler registered for message type: {}", message_type);

                // Send error response if notification is requested
                if let Some(notification_prefix) = response_notification {
                    let error_notification = ResponseNotification {
                        notification: format!(
                            "{}:err(No handler for message type: {})",
                            notification_prefix, message_type
                        ),
                        is_success: false,
                    };
                    let _ = response_sender.send(error_notification);
                }
                return Ok(());
            }
        };

        // Spawn the handler task
        tokio::spawn(async move {
            debug!("Processing message of type: {}", message_type);

            // Create the handler future
            let handler_future = handler_factory(message.message_data);

            // Execute the handler and capture the result
            let result = handler_future.await;

            // Send response notification if requested
            if let Some(notification_prefix) = response_notification {
                let notification = match result {
                    Ok(()) => {
                        debug!(
                            "Message handler completed successfully for type: {}",
                            message_type
                        );
                        ResponseNotification {
                            notification: format!("{}:ok", notification_prefix),
                            is_success: true,
                        }
                    }
                    Err(ref error) => {
                        error!(
                            "Message handler failed for type {}: {}",
                            message_type, error
                        );
                        ResponseNotification {
                            notification: format!("{}:err({})", notification_prefix, error),
                            is_success: false,
                        }
                    }
                };

                if let Err(send_error) = response_sender.send(notification) {
                    error!("Failed to send response notification: {}", send_error);
                }
            } else if let Err(ref error) = result {
                // Log errors even when no response is requested
                error!(
                    "Message handler failed for type {}: {}",
                    message_type, error
                );
            }

            // The permit is automatically dropped here, releasing the semaphore slot
        });

        Ok(())
    }
}

impl Drop for ActiveMessageReceiver {
    fn drop(&mut self) {
        if let Some(handle) = self.driver_handle.take() {
            handle.abort();
        }
    }
}

/// Helper macro for creating active message handlers
#[macro_export]
macro_rules! create_handler {
    ($handler:expr) => {{
        let handler = $handler.clone();
        Arc::new(move |data: Vec<u8>| -> ActiveMessageFuture {
            let handler = handler.clone();
            Box::pin(async move { handler.handle_message(data).await })
        }) as ActiveMessageHandlerFactory
    }};
}

/// Factory for creating active message system components
pub struct ActiveMessageFactory;

impl ActiveMessageFactory {
    /// Create a new active message system with sender and receiver
    pub fn create(
        concurrency: usize,
        cancellation_token: CancellationToken,
    ) -> Result<(ActiveMessageReceiver, ActiveMessageSender)> {
        ActiveMessageReceiver::new(concurrency, cancellation_token)
    }

    /// Create a handler factory from a closure
    pub fn create_handler<F, Fut>(handler: F) -> ActiveMessageHandlerFactory
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Arc::new(move |data: Vec<u8>| -> ActiveMessageFuture { Box::pin(handler(data)) })
    }

    /// Create a handler factory from an object with a handle_message method
    pub fn create_object_handler<T>(handler: T) -> ActiveMessageHandlerFactory
    where
        T: Clone + Send + Sync + 'static,
        T: MessageHandler,
    {
        Arc::new(move |data: Vec<u8>| -> ActiveMessageFuture {
            let handler = handler.clone();
            Box::pin(async move { handler.handle_message(data).await })
        })
    }
}

/// Trait for objects that can handle messages
pub trait MessageHandler: Clone + Send + Sync + 'static {
    /// Handle an incoming message
    fn handle_message(&self, data: Vec<u8>) -> impl Future<Output = Result<()>> + Send;
}

/// Example usage and helper functions
pub mod examples {
    use super::*;
    use std::time::Duration;

    /// Example handler that processes data asynchronously
    #[derive(Clone)]
    pub struct ExampleHandler {
        pub name: String,
    }

    impl ExampleHandler {
        pub fn new(name: String) -> Self {
            Self { name }
        }
    }

    impl MessageHandler for ExampleHandler {
        fn handle_message(&self, data: Vec<u8>) -> impl Future<Output = Result<()>> + Send {
            async move {
                debug!("Handler {} processing {} bytes", self.name, data.len());

                // Simulate some async work
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Process the data (example: just validate it's not empty)
                if data.is_empty() {
                    anyhow::bail!("Empty message data");
                }

                info!("Handler {} completed processing", self.name);
                Ok(())
            }
        }
    }

    /// Example of how to set up handlers
    pub fn create_example_handlers() -> HashMap<String, ActiveMessageHandlerFactory> {
        let mut handlers = HashMap::new();

        // Handler for "ping" messages
        let ping_handler = ExampleHandler::new("ping".to_string());
        handlers.insert(
            "ping".to_string(),
            ActiveMessageFactory::create_object_handler(ping_handler),
        );

        // Handler for "data_transfer" messages
        let data_handler = ExampleHandler::new("data_transfer".to_string());
        handlers.insert(
            "data_transfer".to_string(),
            ActiveMessageFactory::create_object_handler(data_handler),
        );

        handlers
    }

    /// Comprehensive example showing the complete flow
    pub async fn complete_usage_example() -> Result<()> {
        use tokio_util::sync::CancellationToken;

        // Create a cancellation token
        let cancellation_token = CancellationToken::new();

        // Create the active message system
        let (mut receiver, sender) = ActiveMessageFactory::create(4, cancellation_token.clone())?;

        // Register handlers
        let handlers = create_example_handlers();
        receiver.register_handlers(handlers)?;

        // Start the receiver
        receiver.start()?;

        // Get the response receiver
        let response_receiver = receiver.get_response_receiver();

        // Spawn a task to handle responses
        let response_task = tokio::spawn(async move {
            info!("Response handler started");
            while let Some(response) = response_receiver.recv().await {
                if response.is_success {
                    info!("✅ Success: {}", response.notification);
                } else {
                    warn!("❌ Error: {}", response.notification);
                }
            }
            info!("Response handler finished");
        });

        // Send some test messages
        info!("Sending test messages...");

        // Message with response notification
        sender.send_with_response(
            "ping".to_string(),
            b"Hello, World!".to_vec(),
            "test_ping_1".to_string(),
        )?;

        // Fire and forget message
        sender.send_fire_and_forget("data_transfer".to_string(), b"Some data payload".to_vec())?;

        // Message that will cause an error (empty data)
        sender.send_with_response(
            "ping".to_string(),
            Vec::new(),
            "test_ping_error".to_string(),
        )?;

        // Message with unknown handler
        sender.send_with_response(
            "unknown_type".to_string(),
            b"test".to_vec(),
            "test_unknown".to_string(),
        )?;

        // Wait a bit for processing
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Stop the receiver
        receiver.stop().await?;

        // Wait for response task to finish
        let _ = tokio::time::timeout(Duration::from_secs(1), response_task).await;

        info!("Example completed successfully");
        Ok(())
    }

    /// Example of integrating with a communication layer
    pub struct CommunicationLayer {
        sender: ActiveMessageSender,
        response_receiver: SharedResponseReceiver,
    }

    impl CommunicationLayer {
        pub fn new(sender: ActiveMessageSender, response_receiver: SharedResponseReceiver) -> Self {
            Self {
                sender,
                response_receiver,
            }
        }

        /// Simulate receiving a message from the network
        pub async fn handle_incoming_network_message(
            &self,
            message_type: String,
            payload: Vec<u8>,
            response_notification: Option<String>,
        ) -> Result<()> {
            let message = IncomingActiveMessage {
                message_type,
                message_data: payload,
                response_notification,
            };

            self.sender.send(message)?;
            Ok(())
        }

        /// Start a task to handle outgoing response notifications
        pub fn start_response_handler(&self) -> JoinHandle<Result<()>> {
            let response_receiver = self.response_receiver.clone();

            tokio::spawn(async move {
                while let Some(response) = response_receiver.recv().await {
                    // In a real implementation, you'd send this back over the network
                    info!("Sending response notification: {}", response.notification);

                    // Example: Parse the notification to extract the original prefix
                    if let Some((prefix, suffix)) = response.notification.split_once(':') {
                        match suffix {
                            "ok" => {
                                info!("Sending success response for: {}", prefix);
                                // send_network_response(prefix, "ok", None).await?;
                            }
                            s if s.starts_with("err(") && s.ends_with(')') => {
                                let error_msg = &s[4..s.len() - 1];
                                warn!("Sending error response for {}: {}", prefix, error_msg);
                                // send_network_response(prefix, "err", Some(error_msg)).await?;
                            }
                            _ => {
                                warn!("Unknown response format: {}", response.notification);
                            }
                        }
                    }
                }
                Ok(())
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn test_basic_active_message_system() {
        let cancellation_token = CancellationToken::new();
        let (mut receiver, sender) =
            ActiveMessageFactory::create(2, cancellation_token.clone()).unwrap();

        // Register a simple handler
        let handler = ActiveMessageFactory::create_handler(|data: Vec<u8>| async move {
            if data.is_empty() {
                anyhow::bail!("Empty data");
            }
            Ok(())
        });

        receiver
            .register_handler("test".to_string(), handler)
            .unwrap();
        receiver.start().unwrap();

        // Send a message
        sender
            .send_with_response(
                "test".to_string(),
                b"hello".to_vec(),
                "test_response".to_string(),
            )
            .unwrap();

        // Wait a bit for processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the receiver
        receiver.stop().await.unwrap();
    }

    #[test]
    fn test_message_creation() {
        let message = IncomingActiveMessage {
            message_type: "test".to_string(),
            message_data: b"data".to_vec(),
            response_notification: Some("response_id".to_string()),
        };

        assert_eq!(message.message_type, "test");
        assert_eq!(message.message_data, b"data");
        assert_eq!(
            message.response_notification,
            Some("response_id".to_string())
        );
    }

    #[test]
    fn test_response_notification() {
        let response = ResponseNotification {
            notification: "test:ok".to_string(),
            is_success: true,
        };

        assert_eq!(response.notification, "test:ok");
        assert!(response.is_success);
    }
}
