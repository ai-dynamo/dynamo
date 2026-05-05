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
#[cfg(feature = "velo-transport")]
pub use network::bidi::{
    BIDI_INIT_HANDLER, BIDI_INIT_KEY, BIDI_UNATTACHED_TIMEOUT, BidiFrame, BidiInitRequest,
    BidiInitResponse,
};
#[cfg(feature = "velo-transport")]
pub use network::ingress::bidi_handler::{BidiIngress, BidiPushWorkHandler};
pub mod registry;

pub use crate::engine::{
    self as engine, AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Data, DataStream,
    Engine, EngineStream, EngineUnary, ResponseStream, async_trait,
};
pub use anyhow::Error;
pub use context::Context;
pub use error::{PipelineError, PipelineErrorExt, TwoPartCodecError};

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type SingleIn<T> = Context<T>;

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type ManyIn<T> = Context<AsyncRequestStream<T>>;

/// `Send + Sync` wrapper around a [`DataStream<T>`] so it can ride inside a
/// `Context<...>` (and therefore inside the `AsyncEngine<Req, ..>` slot that
/// requires `Req: Send + Sync + 'static`).
///
/// The underlying [`DataStream<T>`] is `Pin<Box<dyn Stream<Item = T> + Send>>` —
/// `Send`-only, since most concrete `Stream` impls are not `Sync`. Wrapping
/// in `Mutex<Option<...>>` gives us interior mutability that *is* `Sync`,
/// while the `Option` lets the user `take()` ownership of the stream once
/// (typical pattern: take the stream and iterate it inside the handler).
pub struct AsyncRequestStream<T: Send + 'static> {
    inner: std::sync::Mutex<Option<DataStream<T>>>,
}

impl<T: Send + 'static> std::fmt::Debug for AsyncRequestStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let present = self.is_present();
        f.debug_struct("AsyncRequestStream")
            .field("present", &present)
            .finish()
    }
}

impl<T: Send + 'static> AsyncRequestStream<T> {
    pub fn new(stream: DataStream<T>) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(stream)),
        }
    }

    /// Take the underlying stream out of the holder. Returns `None` on
    /// subsequent calls (the holder is intended for a single take).
    pub fn take(&self) -> Option<DataStream<T>> {
        self.inner.lock().ok().and_then(|mut g| g.take())
    }

    /// Returns `true` if the stream has not yet been taken.
    pub fn is_present(&self) -> bool {
        self.inner.lock().map(|g| g.is_some()).unwrap_or(false)
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
