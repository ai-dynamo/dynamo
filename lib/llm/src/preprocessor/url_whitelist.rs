// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! URL whitelist for multimodal media URLs (SSRF prevention).
//!
//! When the [`DYN_ALLOWED_URL_PATTERNS`] environment variable is set, only URLs
//! matching the configured patterns are permitted. This prevents Server-Side
//! Request Forgery (SSRF) attacks where a user could probe internal services
//! (e.g. `localhost`, Envoy sidecars, cloud metadata endpoints) through
//! `image_url` / `video_url` / `audio_url` fields in chat completion requests.
//!
//! # Configuration
//!
//! Set `DYN_ALLOWED_URL_PATTERNS` to a comma-separated list of URL patterns.
//! A trailing `*` acts as a prefix wildcard; without it the match is exact.
//!
//! ```text
//! DYN_ALLOWED_URL_PATTERNS=https://i.pinimg.com/*,https://cdn.example.com/*
//! ```
//!
//! When unset or empty, **all** URLs are allowed (backward-compatible default).
//! When set, private / loopback / link-local addresses are always blocked,
//! regardless of whether a pattern would match them.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use dynamo_runtime::config::environment_names::llm::DYN_ALLOWED_URL_PATTERNS;

/// Global, lazily-initialized whitelist. `None` means "no restriction".
static URL_WHITELIST: OnceLock<Option<UrlWhitelist>> = OnceLock::new();

// ---------------------------------------------------------------------------
// UrlWhitelist
// ---------------------------------------------------------------------------

/// Compiled set of allowed URL patterns.
pub(crate) struct UrlWhitelist {
    patterns: Vec<UrlPattern>,
}

enum UrlPattern {
    /// Matches any URL whose string representation starts with `prefix`.
    /// Derived from patterns ending with `*`, e.g. `https://cdn.example.com/*`
    /// becomes `Prefix("https://cdn.example.com/")`.
    Prefix(String),
    /// Matches a single exact URL string.
    Exact(String),
}

impl UrlWhitelist {
    /// Parse a comma-separated patterns string.
    /// Returns `None` when the input is empty or contains no valid patterns.
    pub(crate) fn from_patterns(patterns_str: &str) -> Option<Self> {
        let trimmed = patterns_str.trim();
        if trimmed.is_empty() {
            return None;
        }

        let patterns: Vec<UrlPattern> = trimmed
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Some(prefix) = s.strip_suffix('*') {
                    UrlPattern::Prefix(prefix.to_string())
                } else {
                    UrlPattern::Exact(s.to_string())
                }
            })
            .collect();

        if patterns.is_empty() {
            None
        } else {
            Some(UrlWhitelist { patterns })
        }
    }

    /// Load from the `DYN_ALLOWED_URL_PATTERNS` environment variable.
    fn from_env() -> Option<Self> {
        let val = std::env::var(DYN_ALLOWED_URL_PATTERNS).ok()?;
        let wl = Self::from_patterns(&val);
        if let Some(ref wl) = wl {
            tracing::info!(
                patterns = wl.patterns.len(),
                "URL whitelist active ({})",
                DYN_ALLOWED_URL_PATTERNS,
            );
        }
        wl
    }

    /// Return `true` if `url_str` matches any pattern.
    pub(crate) fn matches(&self, url_str: &str) -> bool {
        self.patterns.iter().any(|p| match p {
            UrlPattern::Prefix(prefix) => url_str.starts_with(prefix.as_str()),
            UrlPattern::Exact(exact) => url_str == exact.as_str(),
        })
    }
}

// ---------------------------------------------------------------------------
// SSRF helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the URL host is a loopback, private, link-local, or
/// cloud-metadata address.
pub(crate) fn is_private_or_loopback(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(d)) => {
            d == "localhost" || d.ends_with(".local") || d == "metadata.google.internal"
        }
        Some(url::Host::Ipv4(ip)) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                // AWS / GCP / Azure metadata endpoint
                || ip == std::net::Ipv4Addr::new(169, 254, 169, 254)
        }
        Some(url::Host::Ipv6(ip)) => ip.is_loopback() || ip.is_unspecified(),
        None => true, // no host → block
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate a URL against a concrete whitelist (testable without the global).
pub(crate) fn validate_url(url: &url::Url, whitelist: &UrlWhitelist) -> Result<()> {
    if is_private_or_loopback(url) {
        bail!(
            "URL '{}' targets a private or loopback address which is blocked for security",
            url
        );
    }
    if !whitelist.matches(url.as_str()) {
        bail!(
            "URL '{}' does not match any allowed pattern (configured via {})",
            url,
            DYN_ALLOWED_URL_PATTERNS
        );
    }
    Ok(())
}

/// Check whether a multimodal media URL is allowed.
///
/// * If `DYN_ALLOWED_URL_PATTERNS` is **not** set → all URLs pass.
/// * If set → the URL must match a pattern **and** must not be private/loopback.
/// * `data:` URLs are always allowed (inline content, no network request).
pub fn check_url_allowed(url: &url::Url) -> Result<()> {
    // data: URLs are inline content — never a network fetch.
    if url.scheme() == "data" {
        return Ok(());
    }

    let whitelist = URL_WHITELIST.get_or_init(UrlWhitelist::from_env);

    if let Some(wl) = whitelist {
        validate_url(url, wl)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> url::Url {
        url::Url::parse(s).unwrap()
    }

    // -- UrlWhitelist::from_patterns ----------------------------------------

    #[test]
    fn empty_string_returns_none() {
        assert!(UrlWhitelist::from_patterns("").is_none());
        assert!(UrlWhitelist::from_patterns("   ").is_none());
    }

    #[test]
    fn single_prefix_pattern() {
        let wl = UrlWhitelist::from_patterns("https://cdn.example.com/*").unwrap();
        assert!(wl.matches("https://cdn.example.com/image.jpg"));
        assert!(wl.matches("https://cdn.example.com/a/b/c.png"));
        assert!(!wl.matches("https://other.com/image.jpg"));
    }

    #[test]
    fn single_exact_pattern() {
        let wl = UrlWhitelist::from_patterns("https://cdn.example.com/logo.png").unwrap();
        assert!(wl.matches("https://cdn.example.com/logo.png"));
        assert!(!wl.matches("https://cdn.example.com/other.png"));
    }

    #[test]
    fn multiple_patterns() {
        let wl = UrlWhitelist::from_patterns(
            "https://i.pinimg.com/*, https://cdn.example.com/*, https://specific.url/image.png",
        )
        .unwrap();
        assert!(wl.matches("https://i.pinimg.com/originals/abc.jpg"));
        assert!(wl.matches("https://cdn.example.com/media/photo.webp"));
        assert!(wl.matches("https://specific.url/image.png"));
        assert!(!wl.matches("https://evil.com/malware.jpg"));
        // http ≠ https
        assert!(!wl.matches("http://i.pinimg.com/originals/abc.jpg"));
    }

    #[test]
    fn whitespace_is_trimmed() {
        let wl = UrlWhitelist::from_patterns("  https://a.com/* , https://b.com/*  ").unwrap();
        assert!(wl.matches("https://a.com/foo"));
        assert!(wl.matches("https://b.com/bar"));
    }

    // -- is_private_or_loopback ---------------------------------------------

    #[test]
    fn blocks_localhost() {
        assert!(is_private_or_loopback(&u("http://localhost/image.jpg")));
        assert!(is_private_or_loopback(&u("http://localhost:8080/img")));
    }

    #[test]
    fn blocks_loopback_ip() {
        assert!(is_private_or_loopback(&u("http://127.0.0.1/image.jpg")));
        assert!(is_private_or_loopback(&u("http://127.0.0.1:19193/secret")));
    }

    #[test]
    fn blocks_private_ranges() {
        assert!(is_private_or_loopback(&u("http://10.0.0.1/img")));
        assert!(is_private_or_loopback(&u("http://172.16.0.1/img")));
        assert!(is_private_or_loopback(&u("http://192.168.1.1/img")));
    }

    #[test]
    fn blocks_link_local() {
        assert!(is_private_or_loopback(&u("http://169.254.1.1/img")));
    }

    #[test]
    fn blocks_cloud_metadata() {
        assert!(is_private_or_loopback(&u(
            "http://169.254.169.254/latest/meta-data/"
        )));
        assert!(is_private_or_loopback(&u(
            "http://metadata.google.internal/computeMetadata/"
        )));
    }

    #[test]
    fn blocks_dot_local_domains() {
        assert!(is_private_or_loopback(&u("http://myservice.local/img")));
    }

    #[test]
    fn blocks_ipv6_loopback() {
        assert!(is_private_or_loopback(&u("http://[::1]/image.jpg")));
    }

    #[test]
    fn allows_public_domains() {
        assert!(!is_private_or_loopback(&u(
            "https://i.pinimg.com/image.jpg"
        )));
        assert!(!is_private_or_loopback(&u(
            "https://cdn.example.com/photo.png"
        )));
    }

    #[test]
    fn allows_public_ips() {
        assert!(!is_private_or_loopback(&u("http://93.184.216.34/img")));
    }

    // -- validate_url -------------------------------------------------------

    #[test]
    fn allowed_url_passes() {
        let wl = UrlWhitelist::from_patterns("https://i.pinimg.com/*").unwrap();
        assert!(validate_url(&u("https://i.pinimg.com/originals/abc.jpg"), &wl).is_ok());
    }

    #[test]
    fn non_matching_url_rejected() {
        let wl = UrlWhitelist::from_patterns("https://i.pinimg.com/*").unwrap();
        let err = validate_url(&u("https://evil.com/img.jpg"), &wl).unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn localhost_blocked_even_if_pattern_matches() {
        // Pattern technically matches, but private-address check takes priority.
        let wl = UrlWhitelist::from_patterns("http://localhost/*").unwrap();
        let err = validate_url(&u("http://localhost/secret"), &wl).unwrap_err();
        assert!(err.to_string().contains("private or loopback"));
    }

    #[test]
    fn loopback_ip_blocked_even_if_pattern_matches() {
        let wl = UrlWhitelist::from_patterns("http://127.0.0.1:19193/*").unwrap();
        let err = validate_url(&u("http://127.0.0.1:19193/api"), &wl).unwrap_err();
        assert!(err.to_string().contains("private or loopback"));
    }

    #[test]
    fn private_ip_blocked_even_if_pattern_matches() {
        let wl = UrlWhitelist::from_patterns("http://10.0.0.5/*").unwrap();
        let err = validate_url(&u("http://10.0.0.5/internal"), &wl).unwrap_err();
        assert!(err.to_string().contains("private or loopback"));
    }

    // -- data: URL scheme ---------------------------------------------------

    #[test]
    fn data_url_scheme_detected() {
        // Verify that data: URLs have scheme "data" — check_url_allowed
        // short-circuits before reaching the whitelist.
        let url = u("data:image/png;base64,iVBORw0KGgo=");
        assert_eq!(url.scheme(), "data");
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn query_params_matched_by_prefix() {
        let wl = UrlWhitelist::from_patterns("https://cdn.example.com/*").unwrap();
        assert!(wl.matches("https://cdn.example.com/img.jpg?w=800&h=600"));
    }

    #[test]
    fn fragment_matched_by_prefix() {
        let wl = UrlWhitelist::from_patterns("https://cdn.example.com/*").unwrap();
        assert!(wl.matches("https://cdn.example.com/img.jpg#section"));
    }

    #[test]
    fn scheme_mismatch_rejected() {
        let wl = UrlWhitelist::from_patterns("https://cdn.example.com/*").unwrap();
        // http ≠ https
        assert!(!wl.matches("http://cdn.example.com/img.jpg"));
    }
}
