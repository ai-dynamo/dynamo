// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pipeline Error
//
use async_nats::error::Error as NatsError;

pub use anyhow::{Context, Error, Result, anyhow, anyhow as error, bail, ensure};

pub trait PipelineErrorExt {
    /// Downcast the [`Error`] to a [`PipelineError`]
    fn try_into_pipeline_error(self) -> Result<PipelineError, Error>;

    /// If the [`Error`] can be downcast to a [`PipelineError`], then the left variant is returned,
    /// otherwise the right variant is returned.
    fn either_pipeline_error(self) -> either::Either<PipelineError, Error>;
}

impl PipelineErrorExt for Error {
    fn try_into_pipeline_error(self) -> Result<PipelineError, Error> {
        self.downcast::<PipelineError>()
    }

    fn either_pipeline_error(self) -> either::Either<PipelineError, Error> {
        match self.downcast::<PipelineError>() {
            Ok(err) => either::Left(err),
            Err(err) => either::Right(err),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// For starter, to remove as code matures.
    #[error("Generic error: {0}")]
    Generic(String),

    /// Edges can only be set once. This error is thrown on subsequent attempts to set an edge.s
    #[error("Link failed: Edge already set")]
    EdgeAlreadySet,

    /// The source node is not connected to an edge.
    #[error("Disconnected source; no edge on which to send data")]
    NoEdge,

    #[error("SegmentSink is not connected to an EgressPort")]
    NoNetworkEdge,

    /// In the interim between when a request was made and when the stream was received, the
    /// requesting task was dropped. This maybe a logic error in the pipeline; and become a
    /// panic/fatal error in the future. This error is thrown when the `on_data` method of a
    /// terminating sink either cannot find the `oneshot` channel sender or the corresponding
    /// receiver was dropped
    #[error("Unlinked request; initiating request task was dropped or cancelled")]
    DetachedStreamReceiver,

    // In the interim between when a response was made and when the stream was received, the
    // Sender for the stream was dropped. This maybe a logic error in the pipeline; and become a
    // panic/fatal error in the future.
    #[error("Unlinked response; response task was dropped or cancelled")]
    DetachedStreamSender,

    #[error("Serialzation Error: {0}")]
    SerializationError(String),

    #[error("Deserialization Error: {0}")]
    DeserializationError(String),

    #[error("Failed to issue request to the control plane: {0}")]
    ControlPlaneRequestError(String),

    #[error("Failed to establish a streaming connection: {0}")]
    ConnectionFailed(String),

    #[error("Generate Error: {0}")]
    GenerateError(Error),

    #[error("An endpoint URL must have the format: namespace/component/endpoint")]
    InvalidEndpointFormat,

    #[error("NATS Request Error: {0}")]
    NatsRequestError(#[from] NatsError<async_nats::jetstream::context::RequestErrorKind>),

    #[error("NATS Get Stream Error: {0}")]
    NatsGetStreamError(#[from] NatsError<async_nats::jetstream::context::GetStreamErrorKind>),

    #[error("NATS Create Stream Error: {0}")]
    NatsCreateStreamError(#[from] NatsError<async_nats::jetstream::context::CreateStreamErrorKind>),

    #[error("NATS Consumer Error: {0}")]
    NatsConsumerError(#[from] NatsError<async_nats::jetstream::stream::ConsumerErrorKind>),

    #[error("NATS Batch Error: {0}")]
    NatsBatchError(#[from] NatsError<async_nats::jetstream::consumer::pull::BatchErrorKind>),

    #[error("NATS Publish Error: {0}")]
    NatsPublishError(#[from] NatsError<async_nats::client::PublishErrorKind>),

    #[error("NATS Connect Error: {0}")]
    NatsConnectError(#[from] NatsError<async_nats::ConnectErrorKind>),

    #[error("NATS Subscriber Error: {0}")]
    NatsSubscriberError(#[from] async_nats::SubscribeError),

    #[error("Local IP Address Error: {0}")]
    LocalIpAddressError(#[from] local_ip_address::Error),

    #[error("Prometheus Error: {0}")]
    PrometheusError(#[from] prometheus::Error),

    #[error("Other NATS Error: {0}")]
    NatsError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Two Part Codec Error: {0}")]
    TwoPartCodec(#[from] TwoPartCodecError),

    #[error("Serde Json Error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("NATS KV Err: {0} for bucket '{1}")]
    KeyValueError(String, String),

    /// All instances are busy and cannot handle new requests
    #[error("Service temporarily unavailable: {0}")]
    ServiceOverloaded(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TwoPartCodecError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Message size {0} exceeds the maximum allowed size of {1} bytes")]
    MessageTooLarge(usize, usize),

    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,
}
