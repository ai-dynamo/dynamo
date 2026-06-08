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
    Data, DataStream, Engine, EngineStream, EngineUnary, ResponseStream, async_trait,
};
pub use anyhow::Error;
pub use context::Context;
pub use error::{PipelineError, PipelineErrorExt, TwoPartCodecError};

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type SingleIn<T> = Context<T>;

/// `Sync` ownership cell for a non-`Sync` [`DataStream<T>`]. [`Self::take`]
/// moves the inner stream out; the mutex serialises concurrent attempts so
/// the first caller observes `Some(stream)` and all later callers see
/// `None`. Iteration happens on the returned `DataStream<T>`.
pub struct RequestStream<T: Data> {
    inner: std::sync::Mutex<Option<DataStream<T>>>,
}

impl<T: Data> RequestStream<T> {
    /// Wrap a [`DataStream<T>`] in a `Sync` ownership cell.
    pub fn new(stream: DataStream<T>) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(stream)),
        }
    }

    /// Atomically move the inner stream out. Returns `Some(stream)` exactly
    /// once across all threads racing on the same `RequestStream`; every
    /// subsequent call (on any thread) returns `None`. The returned stream
    /// is the unique owner; the wrapper retains nothing.
    pub fn take(&self) -> Option<DataStream<T>> {
        self.inner.lock().unwrap().take()
    }
}

impl<T: Data> std::fmt::Debug for RequestStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let taken = self.inner.lock().map(|g| g.is_none()).unwrap_or(true);
        f.debug_struct("RequestStream")
            .field("taken", &taken)
            .finish()
    }
}

/// Pipeline input for streaming-request engines: a [`RequestStream<T>`]
/// payload wrapped in a [`Context`], symmetric to [`SingleIn<T>`] for unary
/// inputs.
pub type ManyIn<T> = Context<RequestStream<T>>;

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Event {
    pub id: String,
}
