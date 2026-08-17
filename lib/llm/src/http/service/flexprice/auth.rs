// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JWT authentication helpers.
//!
//! Validates a Bearer JWT signed with HMAC-SHA256/384/512, extracts the
//! `org_uuid` claim, and optionally enforces an org allowlist. Implemented
//! with the small RustCrypto `hmac`/`sha2` primitives already used elsewhere
//! in this crate — no `jsonwebtoken` dependency needed.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Sha256, Sha384, Sha512};

/// Raised when a request fails authentication or authorization.
#[derive(Debug)]
pub struct AuthError {
    pub status: StatusCode,
    pub message: String,
}

impl AuthError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.into(),
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "statusCode": self.status.as_u16(),
                "message": self.message,
            })),
        )
            .into_response()
    }
}

/// Decoded claims from a validated JWT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCtx {
    pub user_uuid: String,
    pub org_uuid: String,
    pub token_uuid: String,
}

fn b64url_decode(segment: &str) -> Result<Vec<u8>, AuthError> {
    URL_SAFE_NO_PAD
        .decode(segment)
        .map_err(|_| AuthError::new("malformed token segment"))
}

/// Verify `signing_input` against `sig` for the given JWT `alg`, trying `secret`.
fn verify_signature(alg: &str, secret: &[u8], signing_input: &[u8], sig: &[u8]) -> bool {
    match alg {
        "HS256" => {
            let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret) else {
                return false;
            };
            mac.update(signing_input);
            mac.verify_slice(sig).is_ok()
        }
        "HS384" => {
            let Ok(mut mac) = Hmac::<Sha384>::new_from_slice(secret) else {
                return false;
            };
            mac.update(signing_input);
            mac.verify_slice(sig).is_ok()
        }
        "HS512" => {
            let Ok(mut mac) = Hmac::<Sha512>::new_from_slice(secret) else {
                return false;
            };
            mac.update(signing_input);
            mac.verify_slice(sig).is_ok()
        }
        _ => false,
    }
}

fn verify_jwt(token: &str, secret_keys: &[String]) -> Result<Value, AuthError> {
    let parts: Vec<&str> = token.split('.').collect();
    let [header_b64, payload_b64, sig_b64] = parts[..] else {
        return Err(AuthError::new("invalid token format"));
    };

    let header: Value = serde_json::from_slice(&b64url_decode(header_b64)?)
        .map_err(|_| AuthError::new("malformed token header"))?;
    let alg = header.get("alg").and_then(Value::as_str).unwrap_or("");
    if !matches!(alg, "HS256" | "HS384" | "HS512") {
        return Err(AuthError::new(format!(
            "unsupported signing algorithm: {alg:?}"
        )));
    }

    let sig = b64url_decode(sig_b64)?;
    let signing_input = format!("{header_b64}.{payload_b64}");
    let verified = secret_keys
        .iter()
        .any(|secret| verify_signature(alg, secret.as_bytes(), signing_input.as_bytes(), &sig));
    if !verified {
        return Err(AuthError::new("invalid token signature"));
    }

    let claims: Value = serde_json::from_slice(&b64url_decode(payload_b64)?)
        .map_err(|_| AuthError::new("malformed token payload"))?;

    if let Some(exp) = claims.get("exp").and_then(Value::as_f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        if exp < now {
            return Err(AuthError::new("token expired"));
        }
    }

    Ok(claims)
}

/// Authenticate an `Authorization: Bearer <jwt>` header.
///
/// `secret_keys` are tried in order (key rotation support). `valid_orgs`, if
/// non-empty, restricts access to a specific org allowlist.
pub fn authenticate(
    auth_header: &str,
    secret_keys: &[String],
    valid_orgs: &[String],
) -> Result<AuthCtx, AuthError> {
    if auth_header.is_empty() {
        return Err(AuthError::new("Authorization header is required"));
    }

    let mut parts = auth_header.splitn(2, ' ');
    let scheme = parts.next().unwrap_or("");
    let token = parts.next().unwrap_or("").trim();
    if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() {
        return Err(AuthError::new(
            "Authorization header must be: Bearer <token>",
        ));
    }

    let claims = verify_jwt(token, secret_keys)?;

    let user_uuid = claims
        .get("uuid")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let org_uuid = claims
        .get("org_uuid")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let token_uuid = claims
        .get("token_uuid")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    if user_uuid.is_empty() {
        return Err(AuthError::new("token is missing the 'uuid' claim"));
    }
    if org_uuid.is_empty() {
        return Err(AuthError::new("token is missing the 'org_uuid' claim"));
    }
    if !valid_orgs.is_empty() && !valid_orgs.iter().any(|o| o == &org_uuid) {
        return Err(AuthError::new(format!(
            "organization {org_uuid:?} is not permitted to access this endpoint"
        )));
    }

    Ok(AuthCtx {
        user_uuid,
        org_uuid,
        token_uuid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(alg: &str, secret: &str, header: &Value, claims: &Value) -> String {
        let header_b64 = URL_SAFE_NO_PAD.encode(header.to_string());
        let payload_b64 = URL_SAFE_NO_PAD.encode(claims.to_string());
        let signing_input = format!("{header_b64}.{payload_b64}");
        let sig = match alg {
            "HS256" => {
                let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
                mac.update(signing_input.as_bytes());
                mac.finalize().into_bytes().to_vec()
            }
            "HS384" => {
                let mut mac = Hmac::<Sha384>::new_from_slice(secret.as_bytes()).unwrap();
                mac.update(signing_input.as_bytes());
                mac.finalize().into_bytes().to_vec()
            }
            "HS512" => {
                let mut mac = Hmac::<Sha512>::new_from_slice(secret.as_bytes()).unwrap();
                mac.update(signing_input.as_bytes());
                mac.finalize().into_bytes().to_vec()
            }
            _ => panic!("unsupported alg in test helper"),
        };
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(sig))
    }

    fn token(alg: &str, secret: &str, org_uuid: &str, exp_offset_secs: i64) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let claims = serde_json::json!({
            "uuid": "user-1",
            "org_uuid": org_uuid,
            "token_uuid": "token-1",
            "exp": now + exp_offset_secs,
        });
        let header = serde_json::json!({"alg": alg, "typ": "JWT"});
        sign(alg, secret, &header, &claims)
    }

    #[test]
    fn accepts_valid_hs256_token() {
        let t = token("HS256", "secret", "org-1", 300);
        let ctx = authenticate(&format!("Bearer {t}"), &["secret".to_string()], &[]).unwrap();
        assert_eq!(ctx.org_uuid, "org-1");
        assert_eq!(ctx.user_uuid, "user-1");
    }

    #[test]
    fn accepts_hs384_and_hs512() {
        for alg in ["HS384", "HS512"] {
            let t = token(alg, "secret", "org-1", 300);
            assert!(authenticate(&format!("Bearer {t}"), &["secret".to_string()], &[]).is_ok());
        }
    }

    #[test]
    fn supports_key_rotation() {
        let t = token("HS256", "new-secret", "org-1", 300);
        let secrets = vec!["old-secret".to_string(), "new-secret".to_string()];
        assert!(authenticate(&format!("Bearer {t}"), &secrets, &[]).is_ok());
    }

    #[test]
    fn rejects_invalid_signature() {
        let t = token("HS256", "wrong-secret", "org-1", 300);
        let err = authenticate(&format!("Bearer {t}"), &["secret".to_string()], &[]).unwrap_err();
        assert_eq!(err.status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn rejects_expired_token() {
        let t = token("HS256", "secret", "org-1", -300);
        let err = authenticate(&format!("Bearer {t}"), &["secret".to_string()], &[]).unwrap_err();
        assert!(err.message.contains("expired"));
    }

    #[test]
    fn rejects_missing_org_uuid_claim() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = serde_json::json!({"uuid": "user-1", "exp": now + 300});
        let header = serde_json::json!({"alg": "HS256", "typ": "JWT"});
        let t = sign("HS256", "secret", &header, &claims);
        let err = authenticate(&format!("Bearer {t}"), &["secret".to_string()], &[]).unwrap_err();
        assert!(err.message.contains("org_uuid"));
    }

    #[test]
    fn enforces_org_allowlist() {
        let t = token("HS256", "secret", "org-1", 300);
        let secrets = vec!["secret".to_string()];
        assert!(authenticate(&format!("Bearer {t}"), &secrets, &["org-2".to_string()]).is_err());
        assert!(authenticate(&format!("Bearer {t}"), &secrets, &["org-1".to_string()]).is_ok());
    }

    #[test]
    fn rejects_missing_or_malformed_header() {
        assert!(authenticate("", &["secret".to_string()], &[]).is_err());
        assert!(authenticate("Basic abc123", &["secret".to_string()], &[]).is_err());
        assert!(authenticate("Bearer not-a-jwt", &["secret".to_string()], &[]).is_err());
    }
}
