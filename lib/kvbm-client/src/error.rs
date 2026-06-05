// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the kvbm-client.

/// Errors that can arise from connecting or registering with `kvbm-service`.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("uds connect: {0}")]
    Connect(#[from] tonic::transport::Error),
    #[error("rpc: {0}")]
    Rpc(#[from] tonic::Status),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("registration rejected before first event")]
    NoAcceptedEvent,
    #[error("unexpected first event: {0}")]
    UnexpectedFirstEvent(String),
}

pub type ClientResult<T> = Result<T, ClientError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_converts_to_rpc_variant() {
        let status = tonic::Status::internal("boom");
        let err = ClientError::from(status);
        assert!(matches!(err, ClientError::Rpc(_)));
        assert!(err.to_string().contains("rpc"));
    }

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "socket gone");
        let err = ClientError::from(io_err);
        assert!(matches!(err, ClientError::Io(_)));
    }

    #[test]
    fn no_accepted_event_display() {
        let err = ClientError::NoAcceptedEvent;
        assert_eq!(err.to_string(), "registration rejected before first event");
    }

    #[test]
    fn unexpected_first_event_display() {
        let err = ClientError::UnexpectedFirstEvent("heartbeat".into());
        assert!(err.to_string().contains("heartbeat"));
    }
}
