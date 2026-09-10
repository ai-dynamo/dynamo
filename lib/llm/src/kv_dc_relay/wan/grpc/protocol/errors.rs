// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic::{Code, Status, metadata::MetadataValue};

use super::RelayErrorReason;

pub const RELAY_ERROR_REASON_HEADER: &str = "kv-relay-error-reason";

impl RelayErrorReason {
    pub(crate) fn status(self, message: impl Into<String>) -> Status {
        let code = match self {
            Self::ContractMismatch
            | Self::ProducerChanged
            | Self::InvalidPublication
            | Self::UnsupportedFeature => Code::FailedPrecondition,
            Self::InvalidRequest => Code::InvalidArgument,
            Self::PoolNotFound => Code::NotFound,
            Self::ResourceLimit | Self::SubscriberLagged | Self::SnapshotProgressTimeout => {
                Code::ResourceExhausted
            }
            Self::PublicationUnavailable => Code::Unavailable,
            Self::Internal | Self::Unspecified => Code::Internal,
        };
        let mut status = Status::new(code, message);
        // Protobuf enum names contain only ASCII identifier characters.
        status.metadata_mut().insert(
            RELAY_ERROR_REASON_HEADER,
            MetadataValue::from_static(self.as_str_name()),
        );
        status
    }
}

/// Decode a known reason; absent, unknown and unspecified values require code-based fallback.
pub fn relay_error_reason(status: &Status) -> Option<RelayErrorReason> {
    let value = status
        .metadata()
        .get(RELAY_ERROR_REASON_HEADER)?
        .to_str()
        .ok()?;
    RelayErrorReason::from_str_name(value).filter(|reason| *reason != RelayErrorReason::Unspecified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_survive_metadata_without_parsing_diagnostic_text() {
        for (reason, code) in [
            (RelayErrorReason::ContractMismatch, Code::FailedPrecondition),
            (RelayErrorReason::ProducerChanged, Code::FailedPrecondition),
            (RelayErrorReason::ResourceLimit, Code::ResourceExhausted),
            (RelayErrorReason::SubscriberLagged, Code::ResourceExhausted),
            (
                RelayErrorReason::SnapshotProgressTimeout,
                Code::ResourceExhausted,
            ),
        ] {
            let status = reason.status("arbitrary diagnostic text");
            assert_eq!(status.code(), code);
            assert_eq!(relay_error_reason(&status), Some(reason));
        }
        let mut status = Status::unavailable("transport error");
        assert_eq!(relay_error_reason(&status), None);
        for value in [
            "RELAY_ERROR_REASON_FUTURE",
            "RELAY_ERROR_REASON_UNSPECIFIED",
        ] {
            status
                .metadata_mut()
                .insert(RELAY_ERROR_REASON_HEADER, MetadataValue::from_static(value));
            assert_eq!(relay_error_reason(&status), None);
        }
    }
}
