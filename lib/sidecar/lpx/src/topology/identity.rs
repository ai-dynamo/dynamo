// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The identity a snapshot belongs to and the discovery scope derived from it.

use super::proto::{GetSnapshotRequest, Identity};

/// The most bytes an identity field may hold.
pub(crate) const MAX_IDENTITY_BYTES: usize = 512;

fn fields(identity: &Identity) -> [(&'static str, &str); 3] {
    [
        ("deployment", &identity.deployment),
        ("allocation", &identity.allocation),
        ("build", &identity.build),
    ]
}

fn check_field(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("identity {name} is empty"));
    }
    if value.len() > MAX_IDENTITY_BYTES {
        return Err(format!(
            "identity {name} is {} bytes, more than {MAX_IDENTITY_BYTES}",
            value.len()
        ));
    }
    if value.chars().any(|character| character.is_ascii_control()) {
        return Err(format!(
            "identity {name} `{}` contains an ASCII control character",
            value.escape_debug()
        ));
    }
    Ok(())
}

/// Checks that every field of `identity` is non-empty, at most
/// [`MAX_IDENTITY_BYTES`] long and free of ASCII control characters.
pub(crate) fn check_identity(identity: &Identity) -> Result<(), String> {
    fields(identity)
        .into_iter()
        .try_for_each(|(name, value)| check_field(name, value))
}

/// The identity a conductor-side request expects, which must be complete
/// because the server finds the authority by it.
pub(crate) fn require_expected(request: &GetSnapshotRequest) -> Result<&Identity, String> {
    let expected = request
        .expected
        .as_ref()
        .ok_or("the request has no expected identity")?;
    check_identity(expected).map_err(|error| format!("expected {error}"))?;
    Ok(expected)
}

/// Checks that every non-empty field of `expected` equals `actual`'s.
pub(crate) fn check_expected(actual: &Identity, expected: Option<&Identity>) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    for ((name, actual), (_, expected)) in fields(actual).into_iter().zip(fields(expected)) {
        if !expected.is_empty() && expected != actual {
            return Err(format!(
                "the snapshot's {name} `{}` differs from the expected `{}`",
                actual.escape_debug(),
                expected.escape_debug()
            ));
        }
    }
    Ok(())
}
