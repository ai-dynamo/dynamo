// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};

#[derive(Debug)]
pub enum SidecarStartupError {
    Cli(clap::Error),
    Dynamo(DynamoError),
}

impl SidecarStartupError {
    pub fn into_dynamo(self) -> DynamoError {
        match self {
            Self::Cli(error) => invalid_argument(error.to_string()),
            Self::Dynamo(error) => error,
        }
    }
}

impl fmt::Display for SidecarStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Dynamo(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SidecarStartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cli(error) => Some(error),
            Self::Dynamo(error) => Some(error),
        }
    }
}

impl From<clap::Error> for SidecarStartupError {
    fn from(error: clap::Error) -> Self {
        Self::Cli(error)
    }
}

impl From<DynamoError> for SidecarStartupError {
    fn from(error: DynamoError) -> Self {
        Self::Dynamo(error)
    }
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_argument(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub fn protocol_error(peer: &str, message: impl Into<String>) -> DynamoError {
    backend(
        BackendError::Unknown,
        format!("invalid {peer} gRPC response: {}", message.into()),
    )
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn stream_incomplete(message: impl Into<String>) -> DynamoError {
    backend(BackendError::StreamIncomplete, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

pub fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
}

fn typed(kind: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(kind)
        .message(message)
        .build()
}

pub fn cancelled(message: impl Into<String>) -> DynamoError {
    typed(ErrorType::Cancelled, message)
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    status_to_dynamo_parts(rpc, status.message(), status.code())
}

#[cfg(feature = "tonic-v14")]
pub fn status_to_dynamo_v14(rpc: &str, status: tonic_v14::Status) -> DynamoError {
    status_to_dynamo_parts(
        rpc,
        status.message(),
        tonic::Code::from_i32(status.code() as i32),
    )
}

/// Whether tonic built `status` from a transport failure (the engine died or
/// dropped the connection) rather than receiving it from the engine: only the
/// former carries a source error. UNAVAILABLE is left out because it already
/// maps to the migratable `CannotConnect`.
pub fn is_broken_stream(status: &tonic::Status) -> bool {
    std::error::Error::source(status).is_some() && status.code() != tonic::Code::Unavailable
}

#[cfg(feature = "tonic-v14")]
pub fn is_broken_stream_v14(status: &tonic_v14::Status) -> bool {
    std::error::Error::source(status).is_some() && status.code() != tonic_v14::Code::Unavailable
}

/// Maps a status from opening or reading a generate stream. A broken stream
/// becomes `Disconnected`, which the frontend migrates as it does when an
/// in-process worker dies.
pub fn stream_status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    if is_broken_stream(&status) {
        return disconnected(rpc, status.message(), status.code());
    }
    status_to_dynamo(rpc, status)
}

/// Tonic 0.14 reports a stream that ended without trailers, the engine
/// having closed it without a final status, only through this message.
#[cfg(feature = "tonic-v14")]
const MISSING_TRAILER: &str = "protocol error: missing grpc-status trailer";

#[cfg(feature = "tonic-v14")]
pub fn stream_status_to_dynamo_v14(rpc: &str, status: tonic_v14::Status) -> DynamoError {
    let code = tonic::Code::from_i32(status.code() as i32);
    if is_broken_stream_v14(&status) {
        return disconnected(rpc, status.message(), code);
    }
    if status.code() == tonic_v14::Code::Unknown && status.message().starts_with(MISSING_TRAILER) {
        return stream_incomplete(format!("{rpc}: {} ({code:?})", status.message()));
    }
    status_to_dynamo_parts(rpc, status.message(), code)
}

fn disconnected(rpc: &str, message: &str, code: tonic::Code) -> DynamoError {
    backend(
        BackendError::Disconnected,
        format!("{rpc}: stream broke: {message} ({code:?})"),
    )
}

fn status_to_dynamo_parts(rpc: &str, message: &str, code: tonic::Code) -> DynamoError {
    let kind = match code {
        tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::OutOfRange
        | tonic::Code::FailedPrecondition
        | tonic::Code::AlreadyExists => BackendError::InvalidArgument,
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(kind, format!("{rpc}: {message} ({code:?})"))
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, ErrorType};

    use super::{status_to_dynamo, stream_status_to_dynamo};

    #[test]
    fn maps_transport_statuses_to_backend_errors() {
        for (code, expected) in [
            (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
            (tonic::Code::NotFound, BackendError::InvalidArgument),
            (tonic::Code::OutOfRange, BackendError::InvalidArgument),
            (
                tonic::Code::FailedPrecondition,
                BackendError::InvalidArgument,
            ),
            (tonic::Code::AlreadyExists, BackendError::InvalidArgument),
            (tonic::Code::Unknown, BackendError::Unknown),
            (tonic::Code::Unimplemented, BackendError::Unknown),
            (tonic::Code::ResourceExhausted, BackendError::Unknown),
            (tonic::Code::PermissionDenied, BackendError::Unknown),
            (tonic::Code::Unauthenticated, BackendError::Unknown),
            (tonic::Code::Aborted, BackendError::Unknown),
            (tonic::Code::DataLoss, BackendError::Unknown),
            (tonic::Code::Ok, BackendError::Unknown),
            (tonic::Code::Unavailable, BackendError::CannotConnect),
            (tonic::Code::Cancelled, BackendError::Cancelled),
            (
                tonic::Code::DeadlineExceeded,
                BackendError::ConnectionTimeout,
            ),
            (tonic::Code::Internal, BackendError::Unknown),
            // Not WorkerOverloaded: that type is migratable, and a shared
            // helper cannot tell an overloaded engine from a per-request
            // rejection such as an oversized message, which every worker would
            // reject identically. A backend that knows its server means
            // backpressure maps this itself.
            (tonic::Code::ResourceExhausted, BackendError::Unknown),
        ] {
            let error = status_to_dynamo("Test", tonic::Status::new(code, "failure"));
            assert_eq!(error.error_type(), ErrorType::Backend(expected));
            assert!(error.to_string().contains("Test: failure"));
            assert!(error.to_string().contains(&format!("{code:?}")));
            #[cfg(feature = "tonic-v14")]
            {
                let v14 = crate::error::status_to_dynamo_v14(
                    "Test",
                    tonic_v14::Status::new(tonic_v14::Code::from_i32(code as i32), "failure"),
                );
                assert_eq!(v14.error_type(), error.error_type());
                assert_eq!(v14.to_string(), error.to_string());
            }
        }
    }

    #[test]
    fn broken_stream_is_disconnected_but_engine_status_is_not() {
        let broken = tonic::Status::from_error(Box::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "error reading a body from connection",
        )));
        let error = stream_status_to_dynamo("Test", broken);
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Disconnected)
        );
        assert!(error.to_string().contains("error reading a body"));

        // Sent by the engine: no transport source, so the regular mapping.
        let sent = stream_status_to_dynamo("Test", tonic::Status::unknown("engine failure"));
        assert_eq!(sent.error_type(), ErrorType::Backend(BackendError::Unknown));
    }

    /// Produced by tonic's own decoder, so a tonic upgrade that changes the
    /// status fails here rather than silently disabling migration.
    #[cfg(feature = "tonic-v14")]
    #[tokio::test]
    async fn stream_closed_without_trailers_is_incomplete_on_tonic_v14() {
        struct NoMessages;
        impl tonic_v14::codec::Decoder for NoMessages {
            type Item = ();
            type Error = tonic_v14::Status;
            fn decode(
                &mut self,
                _: &mut tonic_v14::codec::DecodeBuf<'_>,
            ) -> Result<Option<()>, Self::Error> {
                Ok(None)
            }
        }
        let mut stream = tonic_v14::codec::Streaming::new_response(
            NoMessages,
            tonic_v14::body::Body::empty(),
            tonic_v14::codegen::http::StatusCode::OK,
            None,
            None,
        );
        let status = stream
            .message()
            .await
            .expect_err("a 200 stream without trailers fails");
        assert_eq!(
            crate::error::stream_status_to_dynamo_v14("Test", status).error_type(),
            ErrorType::Backend(BackendError::StreamIncomplete)
        );
    }

    #[cfg(feature = "tonic-v14")]
    #[test]
    fn broken_stream_is_disconnected_on_tonic_v14() {
        let broken = tonic_v14::Status::from_error(Box::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "error reading a body from connection",
        )));
        assert_eq!(
            crate::error::stream_status_to_dynamo_v14("Test", broken).error_type(),
            ErrorType::Backend(BackendError::Disconnected)
        );
        let sent = tonic_v14::Status::unknown("engine failure");
        assert_eq!(
            crate::error::stream_status_to_dynamo_v14("Test", sent).error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }
}
