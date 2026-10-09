// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental enrollment protocol. Transcript validation is not EK trust verification.

#[cfg(all(target_os = "linux", feature = "enrollment-client"))]
mod certify_creation;
#[cfg(all(target_os = "linux", feature = "enrollment-client"))]
pub mod client;
#[cfg(feature = "enrollment-authority")]
pub mod credential;
pub mod format;
#[cfg(all(target_os = "linux", feature = "enrollment-client"))]
pub mod provision;
#[cfg(feature = "enrollment-authority")]
pub mod registry;
#[cfg(feature = "enrollment-authority")]
pub mod trust;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EnrollmentError {
    #[error("ENROLLMENT_INPUT_INVALID")]
    Input,
    #[error("ENROLLMENT_SIGNATURE_INVALID")]
    Signature,
    #[error("ENROLLMENT_BINDING_INVALID")]
    Binding,
    #[error("ENROLLMENT_CHALLENGE_EXPIRED")]
    Expired,
    #[error("ENROLLMENT_PROOF_INVALID")]
    Proof,
    #[error("ENROLLMENT_REGISTRY_INVALID")]
    Registry,
    #[error("ENROLLMENT_EK_TRUST_INVALID")]
    Trust,
    #[error("ENROLLMENT_REPLAY_OR_CONFLICT")]
    Conflict,
}

pub type Result<T> = std::result::Result<T, EnrollmentError>;
