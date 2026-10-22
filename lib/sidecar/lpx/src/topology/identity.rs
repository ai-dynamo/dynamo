// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The identity a snapshot belongs to and the discovery scope derived from it.

use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::proto::{GetSnapshotRequest, Identity};

/// The most bytes an identity field may hold.
pub(crate) const MAX_IDENTITY_BYTES: usize = 512;

/// The domain of the scope hash: the protocol's package.
const SCOPE_DOMAIN: &str = "lpx.topology.v1";

/// Lowercase hex digits of the scope.
const SCOPE_DIGITS: usize = 16;

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

/// The first 16 lowercase hex digits of the SHA-256 of `"lpx.topology.v1"`
/// followed by each of `values` after a NUL byte.
fn scope_of<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut hash = Sha256::new();
    hash.update(SCOPE_DOMAIN);
    for value in values {
        hash.update([0]);
        hash.update(value);
    }
    let mut scope = String::with_capacity(SCOPE_DIGITS);
    for byte in &hash.finalize()[..SCOPE_DIGITS / 2] {
        let _ = write!(scope, "{byte:02x}");
    }
    scope
}

/// The start of the endpoint name of every authority of `identity`'s
/// allocation, whatever build it serves: `snapshot-<allocation scope>-`,
/// where the allocation scope hashes the deployment and the allocation.
pub(crate) fn allocation_prefix(identity: &Identity) -> String {
    format!(
        "snapshot-{}-",
        scope_of([identity.deployment.as_str(), &identity.allocation])
    )
}

/// The Dynamo endpoint on which the authority for `identity` is registered:
/// the [allocation prefix](allocation_prefix) and the build scope, which
/// hashes all three identity fields.
pub(crate) fn endpoint_name(identity: &Identity) -> String {
    let build_scope = scope_of(fields(identity).map(|(_, value)| value));
    format!("{}{build_scope}", allocation_prefix(identity))
}
