// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-safe context for frontend-local compatibility failures. This annotation
//! does not change the canonical error class or the N-2 worker error wire format.

use serde::Serialize;

use super::profile::CompatibilityProfile;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RejectionKind {
    InvalidValue,
    UnsafeCombination,
    UnsupportedField,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RejectionStage {
    RequestValidation,
    Admission,
    BackendCapability,
}

/// Only bounded identities and reviewed static guidance may enter public details.
/// In particular, never derive these strings from a request or worker diagnostic.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct CompatibilityRejection {
    schema_version: u32,
    field: &'static str,
    kind: RejectionKind,
    stage: RejectionStage,
    profile: Option<CompatibilityProfile>,
    alternatives: &'static [&'static str],
}

#[derive(Debug, thiserror::Error)]
#[error("{source}")]
pub(crate) struct CompatibilityFailure {
    pub details: CompatibilityRejection,
    #[source]
    source: anyhow::Error,
}

impl CompatibilityRejection {
    /// Request validation precedes pipeline selection; do not invent a profile.
    pub(crate) fn invalid_prompt_logprobs() -> Self {
        Self::request_field(
            "prompt_logprobs",
            RejectionKind::InvalidValue,
            &["Use -1 or a non-negative count below 4294967295"],
        )
    }

    /// Before pipeline selection the profile must remain unknown.
    pub(crate) fn request_field(
        field: &'static str,
        kind: RejectionKind,
        alternatives: &'static [&'static str],
    ) -> Self {
        Self {
            schema_version: 1,
            field,
            kind,
            stage: RejectionStage::RequestValidation,
            profile: None,
            alternatives,
        }
    }

    pub(crate) fn unsupported_sampling_field(
        profile: CompatibilityProfile,
        field: &'static str,
    ) -> Self {
        Self {
            schema_version: 1,
            field,
            kind: RejectionKind::UnsupportedField,
            stage: RejectionStage::BackendCapability,
            profile: Some(profile),
            alternatives: &[
                "Omit the field",
                "Use a worker that advertises support for the field",
            ],
        }
    }

    pub(crate) fn attach(self, source: anyhow::Error) -> anyhow::Error {
        CompatibilityFailure {
            details: self,
            source,
        }
        .into()
    }

    pub(crate) fn prompt_logprobs(
        profile: CompatibilityProfile,
        kind: RejectionKind,
        stage: RejectionStage,
        alternatives: &'static [&'static str],
    ) -> Self {
        Self {
            schema_version: 1,
            field: "prompt_logprobs",
            kind,
            stage,
            profile: Some(profile),
            alternatives,
        }
    }
}
