// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous Engine System with Type Erasure Support
//!
//! This module provides the core asynchronous engine abstraction for Dynamo's runtime system.
//! It defines the `AsyncEngine` trait for streaming engines and provides sophisticated
//! type-erasure capabilities for managing heterogeneous engine collections.
//!
//! ## Type Erasure Overview
//!
//! Type erasure is a critical feature that allows storing different `AsyncEngine` implementations
//! with varying generic type parameters in a single collection (e.g., `HashMap<String, Arc<dyn AnyAsyncEngine>>`).
//! This is essential for:
//!
//! - **Dynamic Engine Management**: Registering and retrieving engines at runtime based on configuration
//! - **Plugin Systems**: Loading different engine implementations without compile-time knowledge
//! - **Service Discovery**: Managing multiple engine types in a unified registry
//!
//! ## Implementation Details
//!
//! The type-erasure system uses several advanced Rust features:
//!
//! - **Trait Objects (`dyn Trait`)**: For runtime polymorphism without compile-time type information
//! - **`std::any::TypeId`**: For runtime type checking during downcasting
//! - **`std::any::Any`**: For type-erased storage and safe downcasting
//! - **`PhantomData`**: For maintaining type relationships in generic wrappers
//! - **Extension Traits**: For ergonomic API design without modifying existing types
//!
//! ## Safety Considerations
//!
//! ⚠️ **IMPORTANT**: The type-erasure system relies on precise type matching at runtime.
//! When modifying these traits or their implementations:
//!
//! - **Never change the type ID logic** in `AnyAsyncEngine` implementations
//! - **Maintain the blanket `Data` implementation** for all `Send + Sync + 'static` types
//! - **Test downcasting thoroughly** when adding new engine types
//! - **Document any changes** that affect the type-erasure behavior
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use std::collections::HashMap;
//! use std::sync::Arc;
//! use crate::engine::{AsyncEngine, AsAnyAsyncEngine, DowncastAnyAsyncEngine};
//!
//! // Create typed engines
//! let string_engine: Arc<dyn AsyncEngine<String, String, ()>> = Arc::new(MyStringEngine::new());
//! let int_engine: Arc<dyn AsyncEngine<i32, i32, ()>> = Arc::new(MyIntEngine::new());
//!
//! // Store in heterogeneous collection
//! let mut engines: HashMap<String, Arc<dyn AnyAsyncEngine>> = HashMap::new();
//! engines.insert("string".to_string(), string_engine.into_any_engine());
//! engines.insert("int".to_string(), int_engine.into_any_engine());
//!
//! // Retrieve and downcast safely
//! if let Some(typed_engine) = engines.get("string").unwrap().downcast::<String, String, ()>() {
//!     let result = typed_engine.generate("hello".to_string()).await;
//! }
//! ```

use std::{
    any::{Any, TypeId},
    fmt::Debug,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
};

pub use async_trait::async_trait;
use futures::stream::Stream;

/// All [`Send`] + [`Sync`] + `'static` types can be used as [`AsyncEngine`] request and response types.
///
/// This is implemented as a blanket implementation for all types that meet the bounds.
/// **Do not manually implement this trait** - the blanket implementation covers all valid types.
pub trait Data: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> Data for T {}

/// [`DataStream`] is a type alias for a stream of [`Data`] items. This can be adapted to a [`ResponseStream`]
/// by associating it with a [`AsyncEngineContext`].
pub type DataUnary<T> = Pin<Box<dyn Future<Output = T> + Send>>;
pub type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

pub type Engine<Req, Resp, E> = Arc<dyn AsyncEngine<Req, Resp, E>>;
pub type EngineUnary<Resp> = Pin<Box<dyn AsyncEngineUnary<Resp>>>;
pub type EngineStream<Resp> = Pin<Box<dyn AsyncEngineStream<Resp>>>;
pub type Context = Arc<dyn AsyncEngineContext>;

impl<T: Data> From<EngineStream<T>> for DataStream<T> {
    fn from(stream: EngineStream<T>) -> Self {
        Box::pin(stream)
    }
}

// The Controller and the Context when https://github.com/rust-lang/rust/issues/65991 becomes stable
pub trait AsyncEngineController: Send + Sync {}

/// The [`AsyncEngineContext`] trait defines the interface to control the resulting stream
/// produced by the engine.
///
/// This trait provides lifecycle management for async operations, including:
/// - Stream identification via unique IDs
/// - Graceful shutdown capabilities (`stop_generating`)
/// - Immediate termination capabilities (`kill`)
/// - Status checking for stopped/killed states
///
/// Implementations should ensure thread-safety and proper state management
/// across concurrent access patterns.
#[async_trait]
pub trait AsyncEngineContext: Send + Sync + Debug {
    /// Unique ID for the Stream
    fn id(&self) -> &str;

    /// Returns true if `stop_generating()` has been called; otherwise, false.
    fn is_stopped(&self) -> bool;

    /// Returns true if `kill()` has been called; otherwise, false.
    /// This can be used with a `.take_while()` stream combinator to immediately terminate
    /// the stream.
    ///
    /// An ideal location for a `[.take_while(!ctx.is_killed())]` stream combinator is on
    /// the most downstream  return stream.
    fn is_killed(&self) -> bool;

    /// Calling this method when [`AsyncEngineContext::is_stopped`] is `true` will return
    /// immediately; otherwise, it will [`AsyncEngineContext::is_stopped`] will return true.
    async fn stopped(&self);

    /// Calling this method when [`AsyncEngineContext::is_killed`] is `true` will return
    /// immediately; otherwise, it will [`AsyncEngineContext::is_killed`] will return true.
    async fn killed(&self);

    // Controller

    /// Informs the [`AsyncEngine`] to stop producing results for this particular stream.
    /// This method is idempotent. This method does not invalidate results current in the
    /// stream. It might take some time for the engine to stop producing results. The caller
    /// can decided to drain the stream or drop the stream.
    fn stop_generating(&self);

    /// See [`AsyncEngineContext::stop_generating`].
    fn stop(&self);

    /// Extends the [`AsyncEngineContext::stop_generating`] also indicates a preference to
    /// terminate without draining the remaining items in the stream. This is implementation
    /// specific and may not be supported by all engines.
    fn kill(&self);

    /// Links child AsyncEngineContext to this AsyncEngineContext. If the `stop_generating`, `stop`
    /// or `kill` on this AsyncEngineContext is called, the same method is called on all linked
    /// child AsyncEngineContext, in the order they are linked, and then the method on this
    /// AsyncEngineContext continues.
    fn link_child(&self, child: Arc<dyn AsyncEngineContext>);
}

/// Provides access to the [`AsyncEngineContext`] associated with an engine operation.
///
/// This trait is implemented by both unary and streaming engine results, allowing
/// uniform access to context information regardless of the operation type.
pub trait AsyncEngineContextProvider: Send + Debug {
    fn context(&self) -> Arc<dyn AsyncEngineContext>;
}

/// A unary (single-response) asynchronous engine operation.
///
/// This trait combines `Future` semantics with context provider capabilities,
/// representing a single async operation that produces one result.
pub trait AsyncEngineUnary<Resp: Data>:
    Future<Output = Resp> + AsyncEngineContextProvider + Send
{
}

/// A streaming asynchronous engine operation.
///
/// This trait combines `Stream` semantics with context provider capabilities,
/// representing a continuous async operation that produces multiple results over time.
pub trait AsyncEngineStream<Resp: Data>:
    Stream<Item = Resp> + AsyncEngineContextProvider + Send
{
}

/// Engine is a trait that defines the interface for a streaming engine.
/// The synchronous Engine version is does not need to be awaited.
///
/// This is the core trait for all async engine implementations. It provides:
/// - Generic type parameters for request, response, and error types
/// - Async generation capabilities with proper error handling
/// - Thread-safe design with `Send + Sync` bounds
///
/// ## Type Parameters
/// - `Req`: The request type that implements `Data`
/// - `Resp`: The response type that implements both `Data` and `AsyncEngineContextProvider`
/// - `E`: The error type that implements `Data`
///
/// ## Implementation Notes
/// Implementations should ensure proper error handling and resource management.
/// The `generate` method should be cancellable via the response's context provider.
#[async_trait]
pub trait AsyncEngine<Req: Send + Sync + 'static, Resp: AsyncEngineContextProvider, E: Data>:
    Send + Sync
{
    /// Generate a stream of completion responses.
    async fn generate(&self, request: Req) -> Result<Resp, E>;
}

/// Adapter for a [`DataStream`] to a [`ResponseStream`].
///
/// A common pattern is to consume the [`ResponseStream`] with standard stream combinators
/// which produces a [`DataStream`] stream, then form a [`ResponseStream`] by propagating the
/// original [`AsyncEngineContext`].
pub struct ResponseStream<R: Data> {
    stream: DataStream<R>,
    ctx: Arc<dyn AsyncEngineContext>,
}

impl<R: Data> ResponseStream<R> {
    pub fn new(stream: DataStream<R>, ctx: Arc<dyn AsyncEngineContext>) -> Pin<Box<Self>> {
        Box::pin(Self { stream, ctx })
    }
}

impl<R: Data> Stream for ResponseStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl<R: Data> AsyncEngineStream<R> for ResponseStream<R> {}

impl<R: Data> AsyncEngineContextProvider for ResponseStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<R: Data> Debug for ResponseStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseStream")
            // todo: add debug for stream - possibly propagate some information about what
            // engine created the stream
            // .field("stream", &self.stream)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: Data> AsyncEngineContextProvider for Pin<Box<dyn AsyncEngineUnary<T>>> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        AsyncEngineContextProvider::context(&**self)
    }
}

impl<T: Data> AsyncEngineContextProvider for Pin<Box<dyn AsyncEngineStream<T>>> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        AsyncEngineContextProvider::context(&**self)
    }
}

/// A type-erased `AsyncEngine`.
///
/// This trait enables storing heterogeneous `AsyncEngine` implementations in collections
/// by erasing their specific generic type parameters. It provides runtime type information
/// and safe downcasting capabilities.
///
/// ## Type Erasure Mechanism
/// The trait uses `std::any::TypeId` to preserve type information at runtime, allowing
/// safe downcasting back to the original `AsyncEngine<Req, Resp, E>` types.
///
/// ## Safety Guarantees
/// - Type IDs are preserved exactly as they were during type erasure
/// - Downcasting is only possible to the original type combination
/// - Incorrect downcasts return `None` rather than panicking
///
/// ## Implementation Notes
/// This trait is implemented by the internal `AnyEngineWrapper` struct. Users should
/// not implement this trait directly - use the `AsAnyAsyncEngine` extension trait instead.
pub trait AnyAsyncEngine: Send + Sync {
    /// Returns the `TypeId` of the request type used by this engine.
    fn request_type_id(&self) -> TypeId;

    /// Returns the `TypeId` of the response type used by this engine.
    fn response_type_id(&self) -> TypeId;

    /// Returns the `TypeId` of the error type used by this engine.
    fn error_type_id(&self) -> TypeId;

    /// Provides access to the underlying engine as a `dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// An internal wrapper to hold a typed `AsyncEngine` behind the `AnyAsyncEngine` trait object.
///
/// This struct uses `PhantomData<fn(Req, Resp, E)>` to maintain the type relationship
/// without storing the types directly, enabling the type-erasure mechanism.
///
/// ## PhantomData Usage
/// The `PhantomData<fn(Req, Resp, E)>` ensures that the compiler knows about the
/// generic type parameters without requiring them to be `'static`, which would
/// prevent storing non-static types in the engine.
struct AnyEngineWrapper<Req, Resp, E>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    engine: Arc<dyn AsyncEngine<Req, Resp, E>>,
    _phantom: PhantomData<fn(Req, Resp, E)>,
}

impl<Req, Resp, E> AnyAsyncEngine for AnyEngineWrapper<Req, Resp, E>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    fn request_type_id(&self) -> TypeId {
        TypeId::of::<Req>()
    }

    fn response_type_id(&self) -> TypeId {
        TypeId::of::<Resp>()
    }

    fn error_type_id(&self) -> TypeId {
        TypeId::of::<E>()
    }

    fn as_any(&self) -> &dyn Any {
        &self.engine
    }
}

/// An extension trait that provides a convenient way to type-erase an `AsyncEngine`.
///
/// This trait provides the `.into_any_engine()` method on any `Arc<dyn AsyncEngine<...>>`,
/// enabling ergonomic type erasure without explicit wrapper construction.
///
/// ## Usage
/// ```rust,ignore
/// use crate::engine::AsAnyAsyncEngine;
///
/// let typed_engine: Arc<dyn AsyncEngine<String, String, ()>> = Arc::new(MyEngine::new());
/// let any_engine = typed_engine.into_any_engine();
/// ```
pub trait AsAnyAsyncEngine {
    /// Converts a typed `AsyncEngine` into a type-erased `AnyAsyncEngine`.
    fn into_any_engine(self) -> Arc<dyn AnyAsyncEngine>;
}

impl<Req, Resp, E> AsAnyAsyncEngine for Arc<dyn AsyncEngine<Req, Resp, E>>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    fn into_any_engine(self) -> Arc<dyn AnyAsyncEngine> {
        Arc::new(AnyEngineWrapper {
            engine: self,
            _phantom: PhantomData,
        })
    }
}

/// An extension trait that provides a convenient method to downcast an `AnyAsyncEngine`.
///
/// This trait provides the `.downcast<Req, Resp, E>()` method on `Arc<dyn AnyAsyncEngine>`,
/// enabling safe downcasting back to the original typed engine.
///
/// ## Safety
/// The downcast method performs runtime type checking using `TypeId` comparison.
/// It will only succeed if the type parameters exactly match the original engine's types.
///
/// ## Usage
/// ```rust,ignore
/// use crate::engine::DowncastAnyAsyncEngine;
///
/// let any_engine: Arc<dyn AnyAsyncEngine> = // ... from collection
/// if let Some(typed_engine) = any_engine.downcast::<String, String, ()>() {
///     // Use the typed engine
///     let result = typed_engine.generate("hello".to_string()).await;
/// }
/// ```
pub trait DowncastAnyAsyncEngine {
    /// Attempts to downcast an `AnyAsyncEngine` to a specific `AsyncEngine` type.
    ///
    /// Returns `Some(engine)` if the type parameters match the original engine,
    /// or `None` if the types don't match.
    fn downcast<Req, Resp, E>(&self) -> Option<Arc<dyn AsyncEngine<Req, Resp, E>>>
    where
        Req: Data,
        Resp: Data + AsyncEngineContextProvider,
        E: Data;
}

impl DowncastAnyAsyncEngine for Arc<dyn AnyAsyncEngine> {
    fn downcast<Req, Resp, E>(&self) -> Option<Arc<dyn AsyncEngine<Req, Resp, E>>>
    where
        Req: Data,
        Resp: Data + AsyncEngineContextProvider,
        E: Data,
    {
        if self.request_type_id() == TypeId::of::<Req>()
            && self.response_type_id() == TypeId::of::<Resp>()
            && self.error_type_id() == TypeId::of::<E>()
        {
            self.as_any()
                .downcast_ref::<Arc<dyn AsyncEngine<Req, Resp, E>>>()
                .cloned()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // 1. Define mock data structures
    #[derive(Debug, PartialEq)]
    struct Req1(String);

    #[derive(Debug, PartialEq)]
    struct Resp1(String);

    // Dummy context provider implementation for the response
    impl AsyncEngineContextProvider for Resp1 {
        fn context(&self) -> Arc<dyn AsyncEngineContext> {
            // For this test, we don't need a real context.
            unimplemented!()
        }
    }

    #[derive(Debug)]
    struct Err1;

    // A different set of types for testing failure cases
    #[derive(Debug)]
    struct Req2;
    #[derive(Debug)]
    struct Resp2;
    impl AsyncEngineContextProvider for Resp2 {
        fn context(&self) -> Arc<dyn AsyncEngineContext> {
            unimplemented!()
        }
    }

    // 2. Define a mock engine
    struct MockEngine;

    #[async_trait]
    impl AsyncEngine<Req1, Resp1, Err1> for MockEngine {
        async fn generate(&self, request: Req1) -> Result<Resp1, Err1> {
            Ok(Resp1(format!("response to {}", request.0)))
        }
    }

    #[tokio::test]
    async fn test_engine_type_erasure_and_downcast() {
        // 3. Create a typed engine
        let typed_engine: Arc<dyn AsyncEngine<Req1, Resp1, Err1>> = Arc::new(MockEngine);

        // 4. Use the extension trait to erase the type
        let any_engine = typed_engine.into_any_engine();

        // Check type IDs are preserved
        assert_eq!(any_engine.request_type_id(), TypeId::of::<Req1>());
        assert_eq!(any_engine.response_type_id(), TypeId::of::<Resp1>());
        assert_eq!(any_engine.error_type_id(), TypeId::of::<Err1>());

        // 5. Use the new downcast method on the Arc
        let downcasted_engine = any_engine.downcast::<Req1, Resp1, Err1>();

        // 6. Assert success
        assert!(downcasted_engine.is_some());

        // We can even use the downcasted engine
        let response = downcasted_engine
            .unwrap()
            .generate(Req1("hello".to_string()))
            .await;
        assert_eq!(response.unwrap(), Resp1("response to hello".to_string()));

        // 7. Assert failure for wrong types
        let failed_downcast = any_engine.downcast::<Req2, Resp2, Err1>();
        assert!(failed_downcast.is_none());

        // 8. HashMap usage test
        let mut engine_map: HashMap<String, Arc<dyn AnyAsyncEngine>> = HashMap::new();
        engine_map.insert("mock".to_string(), any_engine);

        let retrieved_engine = engine_map.get("mock").unwrap();
        let final_engine = retrieved_engine.downcast::<Req1, Resp1, Err1>().unwrap();
        let final_response = final_engine.generate(Req1("world".to_string())).await;
        assert_eq!(
            final_response.unwrap(),
            Resp1("response to world".to_string())
        );
    }
}
