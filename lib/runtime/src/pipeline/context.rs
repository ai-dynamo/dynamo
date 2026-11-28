// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::future::poll_fn;
use std::ops::{Deref, DerefMut};
use std::pin::pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use super::{AsyncEngineContext, AsyncEngineContextProvider, Data};
use crate::engine::{AsyncEngineController, EngineContextGuard};
use async_trait::async_trait;
use tokio::sync::Notify;

use super::registry::Registry;

pub struct Context<T: Data> {
    current: T,
    controller: Arc<Controller>,
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

    /// Insert an existing shared object without allocating another Arc.
    pub fn insert_arc<K: ToString, U: Send + Sync + 'static>(&mut self, key: K, value: Arc<U>) {
        self.registry.insert_arc(key, value);
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

    fn retain(&self, guard: EngineContextGuard) {
        self.controller.retain(guard);
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

// Controller states. A controller only moves forward through them.
const LIVE: u8 = 0;
const STOPPED: u8 = 1;
const KILLED: u8 = 2;

/// A context implementation with cancellation propagation.
#[derive(Debug)]
pub struct Controller {
    id: String,
    state: AtomicU8,
    /// Notified after every change of `state`.
    state_changed: Notify,
    child_context: Mutex<Vec<Arc<dyn AsyncEngineContext>>>,
    retained: Mutex<Vec<RetainedGuard>>,
}

struct RetainedGuard {
    _guard: EngineContextGuard,
}

impl std::fmt::Debug for RetainedGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RetainedGuard")
    }
}

impl Controller {
    pub fn new(id: String) -> Self {
        Self {
            id,
            state: AtomicU8::new(LIVE),
            state_changed: Notify::new(),
            child_context: Mutex::new(Vec::new()),
            retained: Mutex::new(Vec::new()),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Moves the state forward to `to` and wakes the waiters. Never moves it back, so a stop
    /// leaves a kill in place.
    ///
    /// Called before the children are told, never after: a child linked
    /// while they are being told is then either among them, or linked after
    /// the state changed, so a check after `link_child` sees it.
    fn advance_to(&self, to: u8) {
        if self.state.fetch_max(to, Ordering::SeqCst) < to {
            self.state_changed.notify_waiters();
        }
    }

    /// Waits until the state is `target` or later.
    ///
    /// Stream loops poll this on every chunk, almost always while the context is live, and a
    /// pending `Notified` locks the waiter list on each poll. A live state means no change has
    /// been notified yet, so the waker an earlier poll registered is still queued and a stop
    /// or kill will wake it: such a poll reads `state` and returns.
    async fn reached(&self, target: u8) {
        let mut notified = pin!(self.state_changed.notified());
        let mut registered: Option<Waker> = None;
        poll_fn(|cx| {
            loop {
                let state = self.state.load(Ordering::SeqCst);
                if state >= target {
                    return Poll::Ready(());
                }
                if state == LIVE
                    && registered
                        .as_ref()
                        .is_some_and(|waker| waker.will_wake(cx.waker()))
                {
                    return Poll::Pending;
                }
                if notified.as_mut().poll(cx).is_pending() {
                    registered = Some(cx.waker().clone());
                    return Poll::Pending;
                }
                // The state changed without reaching `target`. Wait for the next change on a
                // new `Notified`, created before `state` is read again.
                notified.set(self.state_changed.notified());
            }
        })
        .await
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
        self.state.load(Ordering::SeqCst) != LIVE
    }

    fn is_killed(&self) -> bool {
        self.state.load(Ordering::SeqCst) == KILLED
    }

    async fn stopped(&self) {
        self.reached(STOPPED).await
    }

    async fn killed(&self) {
        self.reached(KILLED).await
    }

    fn stop_generating(&self) {
        self.advance_to(STOPPED);
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
    }

    fn stop(&self) {
        self.advance_to(STOPPED);
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
    }

    fn kill(&self) {
        // Published before the children are told (see `advance_to`).
        self.advance_to(KILLED);
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
    }

    fn link_child(&self, child: Arc<dyn AsyncEngineContext>) {
        self.child_context
            .lock()
            .expect("Failed to lock child context")
            .push(child);
    }

    fn retain(&self, guard: EngineContextGuard) {
        self.retained
            .lock()
            .expect("Failed to lock retained engine state")
            .push(RetainedGuard { _guard: guard });
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

    #[tokio::test]
    async fn parent_stop_does_not_downgrade_killed_child() {
        let parent = Controller::default();
        let child = Arc::new(Controller::default());
        parent.link_child(child.clone());

        child.kill();
        parent.stop();

        assert!(child.is_killed());
        tokio::time::timeout(std::time::Duration::from_secs(1), child.killed())
            .await
            .expect("killed child no longer reported its terminal state");

        child.stop_generating();
        assert!(child.is_killed());
    }

    /// A waker that records whether it was woken.
    #[derive(Default)]
    struct WakeFlag(std::sync::atomic::AtomicBool);

    impl std::task::Wake for WakeFlag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    impl WakeFlag {
        fn take(&self) -> bool {
            self.0.swap(false, Ordering::SeqCst)
        }
    }

    // Both tests re-poll the wait with one waker, as a stream loop does on every chunk.

    #[test]
    fn stop_wakes_a_waiter_repolled_while_live() {
        let controller = Controller::default();
        let flag = Arc::new(WakeFlag::default());
        let waker = Waker::from(flag.clone());
        let mut cx = std::task::Context::from_waker(&waker);
        let mut wait = controller.stopped();
        for _ in 0..3 {
            assert!(wait.as_mut().poll(&mut cx).is_pending());
        }

        controller.stop();
        assert!(flag.take(), "stop did not wake the waiter");
        assert!(wait.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn kill_wakes_a_waiter_that_was_woken_by_a_stop() {
        let controller = Controller::default();
        let flag = Arc::new(WakeFlag::default());
        let waker = Waker::from(flag.clone());
        let mut cx = std::task::Context::from_waker(&waker);
        let mut wait = controller.killed();
        assert!(wait.as_mut().poll(&mut cx).is_pending());

        controller.stop();
        assert!(flag.take(), "stop did not wake the waiter");
        for _ in 0..3 {
            assert!(wait.as_mut().poll(&mut cx).is_pending());
        }

        controller.kill();
        assert!(flag.take(), "kill did not wake the waiter");
        assert!(wait.as_mut().poll(&mut cx).is_ready());
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

    /// A child that runs a hook on the first signal its parent sends it.
    struct OnSignal(Mutex<Option<Box<dyn FnOnce() + Send>>>);

    impl OnSignal {
        fn new(hook: impl FnOnce() + Send + 'static) -> Arc<Self> {
            Arc::new(Self(Mutex::new(Some(Box::new(hook)))))
        }

        fn signal(&self) {
            let hook = self.0.lock().unwrap().take();
            if let Some(hook) = hook {
                hook();
            }
        }
    }

    impl std::fmt::Debug for OnSignal {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("OnSignal")
        }
    }

    #[async_trait]
    impl AsyncEngineContext for OnSignal {
        fn id(&self) -> &str {
            "on-signal"
        }

        fn is_stopped(&self) -> bool {
            false
        }

        fn is_killed(&self) -> bool {
            false
        }

        async fn stopped(&self) {
            std::future::pending().await
        }

        async fn killed(&self) {
            std::future::pending().await
        }

        fn stop_generating(&self) {
            self.signal();
        }

        fn stop(&self) {
            self.signal();
        }

        fn kill(&self) {
            self.signal();
        }

        fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {}
    }

    #[derive(Debug, Clone, Copy)]
    enum Signal {
        StopGenerating,
        Stop,
        Kill,
    }

    impl Signal {
        fn send(self, context: &dyn AsyncEngineContext) {
            match self {
                Signal::StopGenerating => context.stop_generating(),
                Signal::Stop => context.stop(),
                Signal::Kill => context.kill(),
            }
        }
    }

    #[tokio::test]
    async fn a_stop_after_a_kill_keeps_the_kill() {
        for stop in [Signal::StopGenerating, Signal::Stop] {
            let controller = Controller::default();
            controller.kill();
            stop.send(&controller);

            assert!(controller.is_stopped(), "{stop:?}");
            assert!(controller.is_killed(), "{stop:?} replaced the kill");
            tokio::time::timeout(std::time::Duration::from_secs(1), controller.killed())
                .await
                .unwrap_or_else(|_| panic!("killed() waits again after {stop:?}"));
        }
    }

    /// A kill that lands while a stop is telling the children wins: the stop
    /// has published its state already, and a kill is final.
    #[test]
    fn a_stop_does_not_replace_a_kill_made_during_it() {
        for stop in [Signal::StopGenerating, Signal::Stop] {
            let controller = Arc::new(Controller::default());
            let target = Arc::downgrade(&controller);
            controller.link_child(OnSignal::new(move || {
                target.upgrade().unwrap().kill();
            }));
            stop.send(controller.as_ref());

            assert!(controller.is_killed(), "{stop:?} replaced the kill");
        }
    }

    /// A child linked while a signal is telling the children, then checked as
    /// Migration does (`link_child`, then the parent's state), is not missed:
    /// it is either told, or it reads the parent's new state.
    #[test]
    fn a_child_linked_during_a_signal_is_not_missed() {
        use std::sync::mpsc;
        use std::time::Duration;

        for signal in [Signal::StopGenerating, Signal::Stop, Signal::Kill] {
            let controller = Arc::new(Controller::default());
            let (entered, on_enter) = mpsc::channel();
            let (release, on_release) = mpsc::channel::<()>();
            controller.link_child(OnSignal::new(move || {
                entered.send(()).unwrap();
                on_release.recv_timeout(Duration::from_secs(5)).unwrap();
            }));

            let sender = controller.clone();
            let signaller = std::thread::spawn(move || signal.send(sender.as_ref()));
            on_enter.recv_timeout(Duration::from_secs(5)).unwrap();

            let late = Arc::new(Controller::default());
            controller.link_child(late.clone());
            let seen = match signal {
                Signal::Kill => controller.is_killed(),
                _ => controller.is_stopped(),
            };
            release.send(()).unwrap();
            signaller.join().unwrap();

            let told = match signal {
                Signal::Kill => late.is_killed(),
                _ => late.is_stopped(),
            };
            assert!(seen || told, "a child linked during {signal:?} missed it");
        }
    }
}
