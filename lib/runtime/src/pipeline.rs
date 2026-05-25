// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// In a Pipeline, the [`AsyncEngine`] is constrained to take a [`Context`] as input and return
/// a [`super::engine::ResponseStream`] as output.
use serde::{Deserialize, Serialize};

mod nodes;
pub use nodes::{
    Operator, PipelineNode, PipelineOperator, SegmentSink, SegmentSource, Service, ServiceBackend,
    ServiceFrontend, Sink, Source,
};

pub mod context;
pub mod error;
pub mod network;
pub use network::egress::addressed_router::{AddressedPushRouter, AddressedRequest};
pub use network::egress::push_router::{PushRouter, RouterMode, WorkerLoadMonitor};
pub mod registry;

pub use crate::engine::{
    self as engine, AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream,
    Data, DataStream, Engine, EngineStream, EngineUnary, RequestStream, ResponseStream,
    async_trait,
};
pub use anyhow::Error;
pub use context::Context;
pub use error::{PipelineError, PipelineErrorExt, TwoPartCodecError};

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type SingleIn<T> = Context<T>;

/// Pipeline input for streaming-request engines.
///
/// Symmetric to [`SingleIn<T>`] (which is `Context<T>`): both inputs carry a
/// typed payload alongside the full pipeline `Context` sidecar (metadata,
/// registry, stages, controller). For the streaming side the payload is a
/// stream of `T` rather than a single value, so this is a two-field struct
/// rather than a `Context<T>` directly.
///
/// Earlier definitions wrapped the stream inside `Context<DataStream<T>>`;
/// that shape was uninstantiable because `DataStream<T>` is `!Sync` while
/// `Context<T: Data>` requires `Sync`. Sibling-field composition sidesteps
/// the bound: `Context<()>` is fine and the stream sits next to it.
///
/// Implements [`Stream<Item = T>`] (delegating to the inner stream) and
/// [`AsyncEngineContextProvider`] (delegating to the inner `Context`), so
/// engines that only need the cancellation slice keep their existing usage
/// (`input.context()`, `.next().await`) without change. Engines that want
/// the full typed sidecar reach for [`Self::context_ref`].
pub struct ManyIn<T: Data> {
    stream: DataStream<T>,
    context: Context<()>,
}

impl<T: Data> ManyIn<T> {
    /// Wrap a stream + context into a streaming input. The stream is the
    /// per-frame payload; the context carries lifecycle + metadata + registry.
    pub fn new(stream: DataStream<T>, context: Context<()>) -> Self {
        Self { stream, context }
    }

    /// Consume `self`, returning the inner stream and context. Useful when an
    /// engine wants to spawn the stream into a forwarder while keeping
    /// access to metadata separately.
    pub fn into_parts(self) -> (DataStream<T>, Context<()>) {
        (self.stream, self.context)
    }

    /// Borrow the inner `Context<()>` for access to metadata / registry /
    /// stages. The dyn-safe cancellation handle is also available via the
    /// [`AsyncEngineContextProvider`] impl as `self.context()`.
    pub fn context_ref(&self) -> &Context<()> {
        &self.context
    }
}

impl<T: Data> futures::Stream for ManyIn<T> {
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.get_mut().stream.as_mut().poll_next(cx)
    }
}

impl<T: Data> AsyncEngineContextProvider for ManyIn<T> {
    fn context(&self) -> std::sync::Arc<dyn AsyncEngineContext> {
        self.context.context()
    }
}

impl<T: Data> AsyncEngineStream<T> for ManyIn<T> {}

impl<T: Data> std::fmt::Debug for ManyIn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManyIn")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

/// Type alias for the output of pipeline that returns a single value
pub type SingleOut<T> = EngineUnary<T>;

/// Type alias for the output of pipeline that returns multiple values
pub type ManyOut<T> = EngineStream<T>;

pub type ServiceEngine<T, U> = Engine<T, U, Error>;

/// Unary Engine is a pipeline that takes a single input and returns a single output
pub type UnaryEngine<T, U> = ServiceEngine<SingleIn<T>, SingleOut<U>>;

/// `ClientStreaming` Engine is a pipeline that takes multiple inputs and returns a single output
/// Typically the engine will consume the entire input stream; however, it can also decided to exit
/// early and emit a response without consuming the entire input stream.
pub type ClientStreamingEngine<T, U> = ServiceEngine<ManyIn<T>, SingleOut<U>>;

/// `ServerStreaming` takes a single input and returns multiple outputs.
pub type ServerStreamingEngine<T, U> = ServiceEngine<SingleIn<T>, ManyOut<U>>;

/// `BidirectionalStreaming` takes multiple inputs and returns multiple outputs. Input and output values
/// are considered independent of each other; however, they could be constrained to be related.
pub type BidirectionalStreamingEngine<T, U> = ServiceEngine<ManyIn<T>, ManyOut<U>>;

pub trait AsyncTransportEngine<T: Data + PipelineIO, U: Data + PipelineIO>:
    AsyncEngine<T, U, Error> + Send + Sync + 'static
{
}

// pub type TransportEngine<T, U> = Arc<dyn AsyncTransportEngine<T, U>>;

mod sealed {
    use super::*;

    #[allow(dead_code)]
    pub struct Token;

    pub trait Connectable {
        type DataType: Data;
    }

    impl<T: Data> Connectable for Context<T> {
        type DataType = T;
    }
    impl<T: Data> Connectable for EngineUnary<T> {
        type DataType = T;
    }
    impl<T: Data> Connectable for EngineStream<T> {
        type DataType = T;
    }
    impl<T: Data> Connectable for ManyIn<T> {
        type DataType = T;
    }
}

pub trait PipelineIO: sealed::Connectable + AsyncEngineContextProvider + 'static {
    fn id(&self) -> String;
}

impl<T: Data> PipelineIO for Context<T> {
    fn id(&self) -> String {
        self.id().to_string()
    }
}
impl<T: Data> PipelineIO for EngineUnary<T> {
    fn id(&self) -> String {
        self.context().id().to_string()
    }
}
impl<T: Data> PipelineIO for EngineStream<T> {
    fn id(&self) -> String {
        self.context().id().to_string()
    }
}
impl<T: Data> PipelineIO for ManyIn<T> {
    fn id(&self) -> String {
        self.context.id().to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Event {
    pub id: String,
}
