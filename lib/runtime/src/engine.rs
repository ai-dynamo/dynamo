// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
pub trait Data: Send + Sync + 'static {}
impl<T: Send + Sync + 'static> Data for T {}

/// [`DataStream`] is a type alias for a stream of [`Data`] items. This can be adapted to a [`ResponseStream`]
/// by associating it with a [`AsyncEngineContext`].
pub type DataUnary<T> = Pin<Box<dyn Future<Output = T> + Send + Sync>>;
pub type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

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
}

pub trait AsyncEngineContextProvider: Send + Sync + Debug {
    fn context(&self) -> Arc<dyn AsyncEngineContext>;
}

pub trait AsyncEngineUnary<Resp: Data>:
    Future<Output = Resp> + AsyncEngineContextProvider + Send + Sync
{
}

pub trait AsyncEngineStream<Resp: Data>:
    Stream<Item = Resp> + AsyncEngineContextProvider + Send + Sync
{
}

/// Engine is a trait that defines the interface for a steaming LLM completion engine.
/// The synchronous Engine version is does not need to be awaited.
#[async_trait]
pub trait AsyncEngine<Req: Data, Resp: Data + AsyncEngineContextProvider, E: Data>:
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
/// This trait is used to store heterogenous `AsyncEngine` implementations in a collection.
/// It provides a mechanism to safely downcast back to a specific `Arc<dyn AsyncEngine<...>>`.
pub trait AnyAsyncEngine: Send + Sync {
    fn request_type_id(&self) -> TypeId;
    fn response_type_id(&self) -> TypeId;
    fn error_type_id(&self) -> TypeId;
    fn as_any(&self) -> &dyn Any;
}

/// An internal wrapper to hold a typed `AsyncEngine` behind the `AnyAsyncEngine` trait object.
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
pub trait AsAnyAsyncEngine {
    /// Converts a typed `AsyncEngine` into a type-erased `AnyAsyncEngine`.
    fn as_any_engine(self) -> Arc<dyn AnyAsyncEngine>;
}

impl<Req, Resp, E> AsAnyAsyncEngine for Arc<dyn AsyncEngine<Req, Resp, E>>
where
    Req: Data,
    Resp: Data + AsyncEngineContextProvider,
    E: Data,
{
    fn as_any_engine(self) -> Arc<dyn AnyAsyncEngine> {
        Arc::new(AnyEngineWrapper {
            engine: self,
            _phantom: PhantomData,
        })
    }
}

/// An extension trait that provides a convenient method to downcast an `AnyAsyncEngine`.
pub trait DowncastAnyAsyncEngine {
    /// Attempts to downcast an `AnyAsyncEngine` to a specific `AsyncEngine` type.
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
        let any_engine = typed_engine.as_any_engine();

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
