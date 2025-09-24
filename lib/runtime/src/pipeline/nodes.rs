// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline Nodes
//!
//! A `ServicePipeline` is a directed graph of nodes where each node defines a behavior for both
//! forward/request path and the backward/response path. The allowed behaviors in each direction
//! are is either a `Source`, or a `Sink`.
//!
//! A `Frontend` is a the start of a graph and is a [`Source`] for the forward path and a [`Sink`] for the
//! backward path.
//!
//! A `Backend` is the end of a graph and is a [`Sink`] for the forward path and a [`Source`] for the
//! backward path.
//!
//! An [`PipelineOperator`] is a node that can transform both the forward and backward paths using the
//! logic supplied by the implementation of an [`Operator`] trait. Because the [`PipelineOperator`] is
//! both a [`Source`] and a [`Sink`] of the forward request path and the backward response path respectively,
//! i.e. it is two sources and two sinks. We can differentiate the two by using the [`PipelineOperator::forward_edge`]
//! and [`PipelineOperator::backward_edge`] methods.
//!
//! - The [`PipelineOperator::forward_edge`] returns a [`PipelineOperatorForwardEdge`] which is a [`Sink`]
//!   for incoming/upstream request and a [`Source`] for the downstream request.
//! - The [`PipelineOperator::backward_edge`] returns a [`PipelineOperatorBackwardEdge`] which is a [`Sink`]
//!   for the downstream response and a [`Source`] for the upstream response.
//!
//! An `EdgeOperator` currently named [`PipelineNode`] is a node in the graph can transform only a forward
//! or a backward path, but does not transform both.
//!
//! This makes the [`Operator`] a more powerful trait as it can propagate information from the forward
//! path to the backward path. An `EdgeOperator` on the forward path has no visibility into the backward
//! path and therefore, cannot directly influence the backward path.
//!
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use super::AsyncEngine;
use async_trait::async_trait;
use tokio::sync::oneshot;

use super::{Data, Error, PipelineError, PipelineIO};

mod sinks;
mod sources;

pub use sinks::{SegmentSink, ServiceBackend};
pub use sources::{SegmentSource, ServiceFrontend};

pub type Service<In, Out> = Arc<ServiceFrontend<In, Out>>;

mod private {
    pub struct Token;
}

// todo rename `ServicePipelineExt`
/// A [`Source`] trait defines how data is emitted from a source to a downstream sink.
#[async_trait]
pub trait Source<T: PipelineIO>: Data {
    async fn on_next(&self, data: T, _: private::Token) -> Result<(), Error>;

    fn set_edge(&self, edge: Edge<T>, _: private::Token) -> Result<(), PipelineError>;

    fn link<S: Sink<T> + 'static>(&self, sink: Arc<S>) -> Result<Arc<S>, PipelineError> {
        let edge = Edge::new(sink.clone());
        self.set_edge(edge, private::Token)?;
        Ok(sink)
    }
}

/// A [`Sink`] trait defines how data is received from a source and processed.
#[async_trait]
pub trait Sink<T: PipelineIO>: Data {
    async fn on_data(&self, data: T, _: private::Token) -> Result<(), Error>;
}

/// An [`Edge`] is a connection between a [`Source`] and a [`Sink`].
pub struct Edge<T: PipelineIO> {
    downstream: Arc<dyn Sink<T>>,
}

impl<T: PipelineIO> Edge<T> {
    fn new(downstream: Arc<dyn Sink<T>>) -> Self {
        Edge { downstream }
    }

    async fn write(&self, data: T) -> Result<(), Error> {
        self.downstream.on_data(data, private::Token).await
    }
}

type NodeFn<In, Out> = Box<dyn Fn(In) -> Result<Out, Error> + Send + Sync>;

/// An [`Operator`] is a trait that defines the behavior of how two [`AsyncEngine`] can be chained together.
/// An [`Operator`] is not quite an [`AsyncEngine`] because its generate method requires both the upstream
/// request, but also the downstream [`AsyncEngine`] to which it will pass the transformed request.
/// The [`Operator`] logic must transform the upstream request `UpIn` to the downstream request `DownIn`,
/// then transform the downstream response `DownOut` to the upstream response `UpOut`.
///
/// A [`PipelineOperator`] accepts an [`Operator`] and presents itself as an [`AsyncEngine`] for the upstream
/// [`AsyncEngine<UpIn, UpOut, Error>`].
///
/// ### Example of type transformation and data flow
/// ```text
/// ... --> <UpIn> ---> [Operator] --> <DownIn> ---> ...
/// ... <-- <UpOut> --> [Operator] <-- <DownOut> <-- ...
/// ```
#[async_trait]
pub trait Operator<UpIn: PipelineIO, UpOut: PipelineIO, DownIn: PipelineIO, DownOut: PipelineIO>:
    Data
{
    /// This method is expected to transform the upstream request `UpIn` to the downstream request `DownIn`,
    /// call the next [`AsyncEngine`] with the transformed request, then transform the downstream response
    /// `DownOut` to the upstream response `UpOut`.
    async fn generate(
        &self,
        req: UpIn,
        next: Arc<dyn AsyncEngine<DownIn, DownOut, Error>>,
    ) -> Result<UpOut, Error>;

    fn into_operator(self: &Arc<Self>) -> Arc<PipelineOperator<UpIn, UpOut, DownIn, DownOut>>
    where
        Self: Sized,
    {
        PipelineOperator::new(self.clone())
    }
}

/// A [`PipelineOperatorForwardEdge`] is [`Sink`] for the upstream request type `UpIn` and a [`Source`] for the
/// downstream request type `DownIn`.
pub struct PipelineOperatorForwardEdge<
    UpIn: PipelineIO,
    UpOut: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
> {
    parent: Arc<PipelineOperator<UpIn, UpOut, DownIn, DownOut>>,
}

/// A [`PipelineOperatorBackwardEdge`] is [`Sink`] for the downstream response type `DownOut` and a [`Source`] for the
/// upstream response type `UpOut`.
pub struct PipelineOperatorBackwardEdge<
    UpIn: PipelineIO,
    UpOut: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
> {
    parent: Arc<PipelineOperator<UpIn, UpOut, DownIn, DownOut>>,
}

/// A [`PipelineOperator`] is a node that can transform both the forward and backward paths using the logic defined
/// by the implementation of an [`Operator`] trait.
pub struct PipelineOperator<
    UpIn: PipelineIO,
    UpOut: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
> {
    // core business logic of this object
    operator: Arc<dyn Operator<UpIn, UpOut, DownIn, DownOut>>,

    // this hold the downstream connections via the generic frontend
    // frontends provide both a source and a sink interfaces
    downstream: Arc<sources::Frontend<DownIn, DownOut>>,

    // this hold the connection to the previous/upstream response sink
    // we are a source to that upstream's response sink
    upstream: sinks::SinkEdge<UpOut>,
}

impl<UpIn, UpOut, DownIn, DownOut> PipelineOperator<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO,
    UpOut: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
{
    /// Create a new [`PipelineOperator`] with the given [`Operator`] implementation.
    pub fn new(operator: Arc<dyn Operator<UpIn, UpOut, DownIn, DownOut>>) -> Arc<Self> {
        Arc::new(PipelineOperator {
            operator,
            downstream: Arc::new(sources::Frontend::default()),
            upstream: sinks::SinkEdge::default(),
        })
    }

    /// Access the forward edge of the [`PipelineOperator`] allowing the forward/requests paths to be linked.
    pub fn forward_edge(
        self: &Arc<Self>,
    ) -> Arc<PipelineOperatorForwardEdge<UpIn, UpOut, DownIn, DownOut>> {
        Arc::new(PipelineOperatorForwardEdge {
            parent: self.clone(),
        })
    }

    /// Access the backward edge of the [`PipelineOperator`] allowing the backward/responses paths to be linked.
    pub fn backward_edge(
        self: &Arc<Self>,
    ) -> Arc<PipelineOperatorBackwardEdge<UpIn, UpOut, DownIn, DownOut>> {
        Arc::new(PipelineOperatorBackwardEdge {
            parent: self.clone(),
        })
    }
}

/// A [`PipelineOperator`] is an [`AsyncEngine`] for the upstream [`AsyncEngine<UpIn, UpOut, Error>`].
#[async_trait]
impl<UpIn, UpOut, DownIn, DownOut> AsyncEngine<UpIn, UpOut, Error>
    for PipelineOperator<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO + Sync,
    DownIn: PipelineIO + Sync,
    DownOut: PipelineIO,
    UpOut: PipelineIO,
{
    async fn generate(&self, req: UpIn) -> Result<UpOut, Error> {
        self.operator.generate(req, self.downstream.clone()).await
    }
}

#[async_trait]
impl<UpIn, UpOut, DownIn, DownOut> Sink<UpIn>
    for PipelineOperatorForwardEdge<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO + Sync,
    DownIn: PipelineIO + Sync,
    DownOut: PipelineIO,
    UpOut: PipelineIO,
{
    async fn on_data(&self, data: UpIn, _token: private::Token) -> Result<(), Error> {
        let stream = self.parent.generate(data).await?;
        self.parent.upstream.on_next(stream, private::Token).await
    }
}

#[async_trait]
impl<UpIn, UpOut, DownIn, DownOut> Source<DownIn>
    for PipelineOperatorForwardEdge<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
    UpOut: PipelineIO,
{
    async fn on_next(&self, data: DownIn, token: private::Token) -> Result<(), Error> {
        self.parent.downstream.on_next(data, token).await
    }

    fn set_edge(&self, edge: Edge<DownIn>, token: private::Token) -> Result<(), PipelineError> {
        self.parent.downstream.set_edge(edge, token)
    }
}

#[async_trait]
impl<UpIn, UpOut, DownIn, DownOut> Sink<DownOut>
    for PipelineOperatorBackwardEdge<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
    UpOut: PipelineIO,
{
    async fn on_data(&self, data: DownOut, token: private::Token) -> Result<(), Error> {
        self.parent.downstream.on_data(data, token).await
    }
}

#[async_trait]
impl<UpIn, UpOut, DownIn, DownOut> Source<UpOut>
    for PipelineOperatorBackwardEdge<UpIn, UpOut, DownIn, DownOut>
where
    UpIn: PipelineIO,
    DownIn: PipelineIO,
    DownOut: PipelineIO,
    UpOut: PipelineIO,
{
    async fn on_next(&self, data: UpOut, token: private::Token) -> Result<(), Error> {
        self.parent.upstream.on_next(data, token).await
    }

    fn set_edge(&self, edge: Edge<UpOut>, token: private::Token) -> Result<(), PipelineError> {
        self.parent.upstream.set_edge(edge, token)
    }
}

pub struct PipelineNode<In: PipelineIO, Out: PipelineIO> {
    edge: OnceLock<Edge<Out>>,
    map_fn: NodeFn<In, Out>,
}

impl<In: PipelineIO, Out: PipelineIO> PipelineNode<In, Out> {
    pub fn new(map_fn: NodeFn<In, Out>) -> Arc<Self> {
        Arc::new(PipelineNode::<In, Out> {
            edge: OnceLock::new(),
            map_fn,
        })
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> Source<Out> for PipelineNode<In, Out> {
    async fn on_next(&self, data: Out, _: private::Token) -> Result<(), Error> {
        self.edge
            .get()
            .ok_or(PipelineError::NoEdge)?
            .write(data)
            .await
    }

    fn set_edge(&self, edge: Edge<Out>, _: private::Token) -> Result<(), PipelineError> {
        self.edge
            .set(edge)
            .map_err(|_| PipelineError::EdgeAlreadySet)?;

        Ok(())
    }
}

#[async_trait]
impl<In: PipelineIO, Out: PipelineIO> Sink<In> for PipelineNode<In, Out> {
    async fn on_data(&self, data: In, _: private::Token) -> Result<(), Error> {
        self.on_next((self.map_fn)(data)?, private::Token).await
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::pipeline::*;

    #[tokio::test]
    async fn test_pipeline_source_no_edge() {
        let source = ServiceFrontend::<SingleIn<()>, ManyOut<()>>::new();
        let stream = source.generate(().into()).await;
        assert!(stream.is_err());
    }
}
