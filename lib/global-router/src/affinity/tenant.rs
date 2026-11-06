// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// The tenant of a request that carried no credential. Such requests share one
/// affinity namespace, which is correct only where the deployment itself is
/// single-tenant.
pub const ANONYMOUS_TENANT: &str = "anonymous";

/// An opaque, stable tenant identifier for `credential`, which is the caller's
/// `Authorization` header verbatim.
///
/// The credential is hashed here, at the edge, so neither the in-memory nor the
/// persistent store ever holds it: the digest is what travels in
/// [`AffinityKey::tenant`](super::AffinityKey). Two callers presenting the same
/// credential share a binding; different credentials never collide on one
/// session ID.
///
/// This separates tenants' bindings. It is not an admission decision: the
/// Global Frontend does not validate the credential, and the pool it forwards
/// to still authorizes the request.
pub fn tenant_id(credential: Option<&str>) -> String {
    let Some(credential) = credential.map(str::trim).filter(|value| !value.is_empty()) else {
        return ANONYMOUS_TENANT.to_owned();
    };
    // Derived, not a bare hash, so the digest is useless against any other
    // context. The derivation context is part of the v1 storage schema.
    let digest = blake3::derive_key(
        "dynamo.global_router.affinity.tenant.v1",
        credential.as_bytes(),
    );
    // Half the digest: 128 bits is far past collision risk for a key field.
    digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_credentials_never_share_a_tenant() {
        assert_ne!(tenant_id(Some("Bearer a")), tenant_id(Some("Bearer b")));
        assert_eq!(tenant_id(Some("Bearer a")), tenant_id(Some("Bearer a")));
    }

    #[test]
    fn a_missing_or_blank_credential_is_anonymous() {
        assert_eq!(tenant_id(None), ANONYMOUS_TENANT);
        assert_eq!(tenant_id(Some("   ")), ANONYMOUS_TENANT);
        assert_ne!(tenant_id(Some("Bearer a")), ANONYMOUS_TENANT);
    }

    #[test]
    fn the_credential_does_not_survive_in_the_identifier() {
        let id = tenant_id(Some("Bearer secret-token"));
        assert!(!id.contains("secret-token"));
        assert_eq!(id.len(), 32);
    }
}
