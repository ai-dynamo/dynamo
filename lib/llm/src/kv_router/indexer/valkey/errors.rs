// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retry and module-error classification.

use super::*;

pub(super) fn retryable_write_error(error: &RespError) -> bool {
    matches!(error, RespError::Io(_) | RespError::Timeout)
        || matches!(error, RespError::Server(message) if retryable_server_error(message))
        || matches!(error, RespError::Protocol(message) if message.starts_with("Valkey replication quorum not met:"))
}

pub(super) fn replication_quorum_error(error: &RespError) -> bool {
    matches!(
        error,
        RespError::Protocol(message)
            if message.starts_with("Valkey replication quorum not met:")
    )
}

pub(super) fn retryable_server_error(message: &str) -> bool {
    topology_server_error(message)
        || matches!(
            message.split_ascii_whitespace().next(),
            Some("NOREPLICAS" | "LOADING" | "TRYAGAIN" | "CLUSTERDOWN")
        )
}

pub(super) fn topology_server_error(message: &str) -> bool {
    matches!(
        message.split_ascii_whitespace().next(),
        Some("READONLY" | "MASTERDOWN" | "DYNKV_NOT_PRIMARY")
    ) || message.contains("WAIT cannot be used with replica")
}

pub(super) fn topology_error(error: &RespError) -> bool {
    matches!(error, RespError::Server(message) if topology_server_error(message))
}

pub(super) fn sentinel_refresh_error(error: &RespError) -> bool {
    matches!(error, RespError::Io(_) | RespError::Timeout)
        || topology_error(error)
        || matches!(
            error,
            RespError::Server(message)
                if message.split_ascii_whitespace().next() == Some("NOREPLICAS")
        )
        || matches!(
            error,
            RespError::Protocol(message)
                if message.starts_with("Valkey replication quorum not met:")
        )
}

pub(super) fn retryable_primary_read_error(error: &RespError) -> bool {
    matches!(error, RespError::Io(_) | RespError::Timeout) || topology_error(error)
}

pub(super) fn stale_rank_generation_error(error: &RespError) -> bool {
    matches!(
        error,
        RespError::Server(message)
            if message.split_ascii_whitespace().next() == Some("DYNKV_STALE_GENERATION")
    )
}

pub(super) fn reservation_expired_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<RespError>(),
        Some(RespError::Server(message))
            if message.split_ascii_whitespace().next() == Some("DYNKV_RESERVATION_EXPIRED")
    )
}

pub(super) fn stale_registration_generation_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<RespError>(),
        Some(RespError::Server(message))
            if message.split_ascii_whitespace().next()
                == Some("DYNKV_STALE_REGISTRATION_GENERATION")
    )
}

pub(super) fn worker_cleanup_pending_error(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<RespError>(),
        Some(RespError::Server(message))
            if message.split_ascii_whitespace().next()
                == Some("DYNKV_WORKER_CLEANUP_PENDING")
    )
}
