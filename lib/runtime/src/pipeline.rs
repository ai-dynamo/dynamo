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
pub use network::egress::push_router::{PushRouter, RouterMode};
pub mod registry;

pub use crate::engine::{
    self as engine, async_trait, AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Data,
    DataStream, Engine, EngineStream, EngineUnary, ResponseStream,
};
pub use anyhow::Error;
pub use context::Context;
pub use error::{PipelineError, PipelineErrorExt, TwoPartCodecError};

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type SingleIn<T> = Context<T>;

/// Pipeline inputs carry a [`Context`] which can be used to carry metadata or additional information
/// about the request. This information propagates through the stages, both local and distributed.
pub type ManyIn<T> = Context<DataStream<T>>;

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

pub trait AsyncTransportEngine<T: PipelineIO, U: PipelineIO>:
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

pub trait PipelineIO: Data + sealed::Connectable + AsyncEngineContextProvider {
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
