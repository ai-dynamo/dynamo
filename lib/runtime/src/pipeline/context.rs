// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use super::{AsyncEngineContext, AsyncEngineContextProvider, Data};
use crate::engine::AsyncEngineController;
use async_trait::async_trait;

use super::registry::Registry;

pub struct Context<T: Data> {
    current: T,
    controller: Arc<Controller>, //todo: hold this as an arc
    registry: Registry,
    stages: Vec<String>,
    metadata: BTreeMap<String, String>,
}

impl<T: Send + Sync + 'static> Context<T> {
    // Create a new context with initial data
    pub fn new(current: T) -> Self {
        Context {
            current,
            controller: Arc::new(Controller::default()),
            registry: Registry::new(),
            stages: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn rejoin<U: Send + Sync + 'static>(current: T, context: Context<U>) -> Self {
        Context {
            current,
            controller: context.controller,
            registry: context.registry,
            stages: context.stages,
            metadata: context.metadata,
        }
    }

    pub fn with_controller(current: T, controller: Controller) -> Self {
        Context {
            current,
            controller: Arc::new(controller),
            registry: Registry::new(),
            stages: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_id_and_metadata(
        current: T,
        id: String,
        metadata: BTreeMap<String, String>,
    ) -> Self {
        Context {
            current,
            controller: Arc::new(Controller::new(id)),
            registry: Registry::new(),
            stages: Vec::new(),
            metadata,
        }
    }

    /// Get the id of the context
    pub fn id(&self) -> &str {
        self.controller.id()
    }

    /// Get the content of the context
    pub fn content(&self) -> &T {
        &self.current
    }

    pub fn controller(&self) -> &Controller {
        &self.controller
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut BTreeMap<String, String> {
        &mut self.metadata
    }

    pub fn set_metadata(&mut self, metadata: BTreeMap<String, String>) {
        self.metadata = metadata;
    }

    pub fn insert_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Insert an object into the registry with a specific key.
    pub fn insert<K: ToString, U: Send + Sync + 'static>(&mut self, key: K, value: U) {
        self.registry.insert_shared(key, value);
    }

    /// Insert a unique and takable object into the registry with a specific key.
    pub fn insert_unique<K: ToString, U: Send + Sync + 'static>(&mut self, key: K, value: U) {
        self.registry.insert_unique(key, value);
    }

    /// Retrieve an object from the registry by key and type.
    pub fn get<V: Send + Sync + 'static>(&self, key: &str) -> Result<Arc<V>, String> {
        self.registry.get_shared(key)
    }

    /// Retrieve an optional object from the registry by key and type.
    pub fn get_optional<V: Send + Sync + 'static>(
        &self,
        key: &str,
    ) -> Result<Option<Arc<V>>, String> {
        self.registry.get_shared_optional(key)
    }

    /// Clone a unique object from the registry by key and type.
    pub fn clone_unique<V: Clone + Send + Sync + 'static>(&self, key: &str) -> Result<V, String> {
        self.registry.clone_unique(key)
    }

    /// Take a unique object from the registry by key and type.
    pub fn take_unique<V: Send + Sync + 'static>(&mut self, key: &str) -> Result<V, String> {
        self.registry.take_unique(key)
    }

    /// Transfer the Context to a new Object without updating the registry
    /// This returns a tuple of the previous object and the new Context
    pub fn transfer<U: Send + Sync + 'static>(self, new_current: U) -> (T, Context<U>) {
        (
            self.current,
            Context {
                current: new_current,
                controller: self.controller,
                registry: self.registry,
                stages: self.stages,
                metadata: self.metadata,
            },
        )
    }

    /// Separate out the current object and context
    pub fn into_parts(self) -> (T, Context<()>) {
        self.transfer(())
    }

    pub fn stages(&self) -> &Vec<String> {
        &self.stages
    }

    pub fn add_stage(&mut self, stage: &str) {
        self.stages.push(stage.to_string());
    }

    /// Transforms the current context to another type using a provided function.
    pub fn map<U: Send + Sync + 'static, F>(self, f: F) -> Context<U>
    where
        F: FnOnce(T) -> U,
    {
        // Use the transfer method to move the current value out
        let (current, temp_context) = self.transfer(());

        // Apply the transformation function to the current value
        let new_current = f(current);

        // Use transfer again to create the new context with the transformed type
        temp_context.transfer(new_current).1
    }

    pub fn try_map<U, F, E>(self, f: F) -> Result<Context<U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
        U: Send + Sync + 'static,
    {
        // Use the transfer method to move the current value out
        let (current, temp_context) = self.transfer(());

        // Apply the transformation function to the current value
        let new_current = f(current)?;

        // Use transfer again to create the new context with the transformed type
        Ok(temp_context.transfer(new_current).1)
    }
}

impl<T: Data> std::fmt::Debug for Context<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.controller.id())
            .finish()
    }
}

// Implement Deref to allow Context<T> to act like &T
impl<T: Data> Deref for Context<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

// Implement DerefMut to allow Context<T> to act like &mut T
impl<T: Data> DerefMut for Context<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.current
    }
}

// Implement the custom trait for Context<T>
impl<T> From<T> for Context<T>
where
    T: Send + Sync + 'static,
{
    fn from(current: T) -> Self {
        Context::new(current)
    }
}

// Define a custom trait for conversion from Context<T> to Context<U>
pub trait IntoContext<U: Data> {
    fn into_context(self) -> Context<U>;
}

// Implement the custom trait for converting Context<T> to Context<U>
impl<T, U> IntoContext<U> for Context<T>
where
    T: Send + Sync + 'static + Into<U>,
    U: Send + Sync + 'static,
{
    fn into_context(self) -> Context<U> {
        self.map(|current| current.into())
    }
}

impl<T: Data> AsyncEngineContextProvider for Context<T> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.controller.clone()
    }
}

#[derive(Debug, Clone)]
pub struct StreamContext {
    controller: Arc<Controller>,
    registry: Arc<Registry>,
    stages: Vec<String>,
    metadata: BTreeMap<String, String>,
}

impl StreamContext {
    fn new(
        controller: Arc<Controller>,
        registry: Registry,
        metadata: BTreeMap<String, String>,
    ) -> Self {
        StreamContext {
            controller,
            registry: Arc::new(registry),
            stages: Vec::new(),
            metadata,
        }
    }

    /// Retrieve an object from the registry by key and type.
    pub fn get<V: Send + Sync + 'static>(&self, key: &str) -> Result<Arc<V>, String> {
        self.registry.get_shared(key)
    }

    /// Clone a unique object from the registry by key and type.
    pub fn clone_unique<V: Clone + Send + Sync + 'static>(&self, key: &str) -> Result<V, String> {
        self.registry.clone_unique(key)
    }

    pub fn registry(&self) -> Arc<Registry> {
        self.registry.clone()
    }

    pub fn stages(&self) -> &Vec<String> {
        &self.stages
    }

    pub fn add_stage(&mut self, stage: &str) {
        self.stages.push(stage.to_string());
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
}

#[async_trait]
impl AsyncEngineContext for StreamContext {
    fn id(&self) -> &str {
        self.controller.id()
    }

    fn stop(&self) {
        self.controller.stop();
    }

    fn kill(&self) {
        self.controller.kill();
    }

    fn stop_generating(&self) {
        self.controller.stop_generating();
    }

    fn is_stopped(&self) -> bool {
        self.controller.is_stopped()
    }

    fn is_killed(&self) -> bool {
        self.controller.is_killed()
    }

    async fn stopped(&self) {
        self.controller.stopped().await
    }

    async fn killed(&self) {
        self.controller.killed().await
    }

    fn link_child(&self, child: Arc<dyn AsyncEngineContext>) {
        self.controller.link_child(child);
    }
}

impl AsyncEngineContextProvider for StreamContext {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.controller.clone()
    }
}

impl<T: Send + Sync + 'static> From<Context<T>> for StreamContext {
    fn from(value: Context<T>) -> Self {
        StreamContext::new(value.controller, value.registry, value.metadata)
    }
}

// TODO - refactor here - this came from the dynamo.llm-async-engine crate

use tokio::sync::watch::{Receiver, Sender, channel};

#[derive(Debug, Eq, PartialEq)]
enum State {
    Live,
    Stopped,
    Killed,
}

/// A context implementation with cancellation propagation.
#[derive(Debug)]
pub struct Controller {
    id: String,
    tx: Sender<State>,
    rx: Receiver<State>,
    child_context: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
}

impl Controller {
    pub fn new(id: String) -> Self {
        let (tx, rx) = channel(State::Live);
        Self {
            id,
            tx,
            rx,
            child_context: Mutex::new(Vec::new()),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }
}

impl Default for Controller {
    fn default() -> Self {
        Self::new(uuid::Uuid::new_v4().to_string())
    }
}

impl AsyncEngineController for Controller {}

#[async_trait]
impl AsyncEngineContext for Controller {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_stopped(&self) -> bool {
        *self.rx.borrow() != State::Live
    }

    fn is_killed(&self) -> bool {
        *self.rx.borrow() == State::Killed
    }

    async fn stopped(&self) {
        let mut rx = self.rx.clone();
        loop {
            if *rx.borrow_and_update() != State::Live || rx.changed().await.is_err() {
                return;
            }
        }
    }

    async fn killed(&self) {
        let mut rx = self.rx.clone();
        loop {
            if *rx.borrow_and_update() == State::Killed || rx.changed().await.is_err() {
                return;
            }
        }
    }

    fn stop_generating(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.stop_generating();
        }

        let _ = self.tx.send(State::Stopped);
    }

    fn stop(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.stop();
        }

        let _ = self.tx.send(State::Stopped);
    }

    fn kill(&self) {
        // Clone child Arcs to avoid deadlock if parent is accidentally linked under child
        let children = self
            .child_context
            .lock()
            .expect("Failed to lock child context")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for child in children {
            child.kill();
        }

        let _ = self.tx.send(State::Killed);
    }

    fn link_child(&self, child: Arc<dyn AsyncEngineContext>) {
        self.child_context
            .lock()
            .expect("Failed to lock child context")
            .push(child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct Input {
        value: String,
    }

    #[derive(Debug, Clone)]
    struct Processed {
        length: usize,
    }

    #[derive(Debug, Clone)]
    struct Final {
        message: String,
    }

    impl From<Input> for Processed {
        fn from(input: Input) -> Self {
            Processed {
                length: input.value.len(),
            }
        }
    }

    impl From<Processed> for Final {
        fn from(processed: Processed) -> Self {
            Final {
                message: format!("Processed length: {}", processed.length),
            }
        }
    }

    #[test]
    fn test_insert_and_get() {
        let mut ctx = Context::new(Input {
            value: "Hello".to_string(),
        });

        ctx.insert("key1", 42);
        ctx.insert("key2", "some data".to_string());

        assert_eq!(*ctx.get::<i32>("key1").unwrap(), 42);
        assert_eq!(*ctx.get::<String>("key2").unwrap(), "some data");
        assert!(ctx.get::<f64>("key1").is_err()); // Testing a downcast failure
    }

    #[test]
    fn test_metadata_preserved_across_transfers() {
        let mut ctx = Context::new(Input {
            value: "Hello".to_string(),
        });
        ctx.insert_metadata("tenant", "alpha");

        let (_, transferred) = ctx.transfer(Processed { length: 5 });
        assert_eq!(
            transferred.metadata().get("tenant").map(String::as_str),
            Some("alpha")
        );
    }

    #[test]
    fn test_with_id_and_metadata_constructor() {
        let metadata = BTreeMap::from([("tenant".to_string(), "alpha".to_string())]);
        let ctx = Context::with_id_and_metadata(
            Input {
                value: "Hello".to_string(),
            },
            "request-123".to_string(),
            metadata,
        );

        assert_eq!(ctx.id(), "request-123");
        assert_eq!(
            ctx.metadata().get("tenant").map(String::as_str),
            Some("alpha")
        );
    }

    #[test]
    fn test_metadata_preserved_across_rejoin() {
        let mut ctx = Context::new(Input {
            value: "Hello".to_string(),
        });
        ctx.insert_metadata("tenant", "alpha");

        let (input, empty_ctx) = ctx.into_parts();
        let rejoined = Context::rejoin(input, empty_ctx);
        assert_eq!(
            rejoined.metadata().get("tenant").map(String::as_str),
            Some("alpha")
        );
    }

    #[test]
    fn test_metadata_preserved_in_stream_context() {
        let mut ctx = Context::new(Input {
            value: "Hello".to_string(),
        });
        ctx.insert_metadata("tenant", "alpha");

        let stream_ctx = StreamContext::from(ctx);
        assert_eq!(
            stream_ctx.metadata().get("tenant").map(String::as_str),
            Some("alpha")
        );
    }

    #[test]
    fn test_transfer() {
        let ctx = Context::new(Input {
            value: "Hello".to_string(),
        });

        let (input, ctx) = ctx.transfer(Processed { length: 5 });

        assert_eq!(input.value, "Hello");
        assert_eq!(ctx.length, 5);
    }

    #[test]
    fn test_map() {
        let ctx = Context::new(Input {
            value: "Hello".to_string(),
        });

        let ctx: Context<Processed> = ctx.map(|input| input.into());
        let ctx: Context<Final> = ctx.map(|processed| processed.into());

        assert_eq!(ctx.current.message, "Processed length: 5");
    }

    #[test]
    fn test_into_context() {
        let ctx = Context::new(Input {
            value: "Hello".to_string(),
        });

        let ctx: Context<Processed> = ctx.into_context();
        let ctx: Context<Final> = ctx.into_context();

        assert_eq!(ctx.current.message, "Processed length: 5");
    }
}
