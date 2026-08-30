// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve `oci://` model references to a local path.
//!
//! Models published as [CNCF ModelPack](https://github.com/modelpack/model-spec)
//! OCI artifacts live in ordinary container registries, so they reuse the
//! registry, credentials, mirroring and air-gap tooling a deployment already
//! has for container images. This module makes such a reference usable
//! anywhere Dynamo accepts a Hugging Face repo id.
//!
//! Acquisition is delegated to a running [`llmman serve`](https://github.com/llmmanorg/llmman),
//! which already implements the ModelPack media types, registry auth,
//! resumable blob download and a content-addressed local store. Two pieces are
//! needed, because the daemon deliberately exposes no local path:
//!
//! - `POST /api/pull` streams newline-delimited JSON status objects, ending in
//!   either `{"status":"success"}` or `{"error":"..."}`. An error can arrive
//!   in-band at HTTP 200, and a stream that simply ends without success is
//!   also a failure.
//! - `llmman resolve --no-pull <reference>` then reports where the bytes
//!   landed, printing one line of JSON on stdout:
//!
//!   ```json
//!   {"reference":"ghcr.io/org/model:tag","path":"/abs/path","format":"safetensors"}
//!   ```
//!
//!   `--no-pull` guarantees it only reports on bytes `/api/pull` already
//!   fetched, so the daemon stays the only thing that touches the network.
//!
//! An `oci://` pull therefore needs both the daemon reachable *and* the binary
//! on `PATH`; each missing piece has its own actionable error.
//!
//! An explicit `oci://` scheme is required rather than sniffing a bare
//! `registry/name:tag`: that shape is indistinguishable from a Hugging Face
//! repo id (`org/model`), and guessing would silently hijack existing
//! HF-backed deployments.

use std::path::PathBuf;
use std::process::Stdio;

use anyhow::Context;
use futures::StreamExt;
use serde::Deserialize;

use dynamo_runtime::config::environment_names::model as env_model;

/// URI scheme identifying a model reference this module resolves.
pub const SCHEME: &str = "oci://";

/// Binary consulted to report where a pulled reference landed.
const DEFAULT_BIN: &str = "llmman";

/// `llmman serve`'s own default bind address.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 17434;

/// True if `value` uses the `oci://` scheme.
///
/// Case-insensitive: a URI scheme is case-insensitive per RFC 3986, and
/// `OCI://` is a plausible thing for a user to type.
pub fn is_oci_ref(value: &str) -> bool {
    value.len() > SCHEME.len() && value[..SCHEME.len()].eq_ignore_ascii_case(SCHEME)
}

/// Drop the `oci://` prefix, leaving the bare reference `llmman` understands.
///
/// Returns `value` unchanged if it isn't prefixed with `oci://`.
pub fn strip_scheme(value: &str) -> &str {
    if is_oci_ref(value) {
        &value[SCHEME.len()..]
    } else {
        value
    }
}

/// Name of the `llmman` binary, overridable via `DYN_LLMMAN_BIN`.
fn llmman_bin() -> String {
    std::env::var(env_model::oci::DYN_LLMMAN_BIN)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BIN.to_string())
}

/// The `http://host:port` origin of the llmman daemon.
///
/// `LLMMAN_HOST` is parsed as `[scheme://]host[:port][/path]`, matching
/// llmman's own client-side resolution, and a wildcard bind host is rewritten
/// to loopback since a client cannot connect to "every interface".
fn endpoint() -> String {
    endpoint_from(std::env::var(env_model::oci::LLMMAN_HOST).ok().as_deref())
}

/// Split out from [`endpoint`] so the parsing can be tested without the
/// environment.
fn endpoint_from(value: Option<&str>) -> String {
    let raw = value.unwrap_or("").trim().trim_matches(['"', '\'']);
    if raw.is_empty() {
        return format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}");
    }

    let after_scheme = raw.split_once("://").map(|(_, rest)| rest).unwrap_or(raw);
    let hostport = after_scheme.split('/').next().unwrap_or(after_scheme);

    let (host, port) = if let Some(rest) = hostport.strip_prefix('[') {
        // Bracketed IPv6, optionally followed by :port.
        match rest.split_once(']') {
            Some((inner, tail)) => (
                format!("[{inner}]"),
                tail.strip_prefix(':')
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(DEFAULT_PORT),
            ),
            None => (hostport.to_string(), DEFAULT_PORT),
        }
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) if !h.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
                (h.to_string(), p.parse().unwrap_or(DEFAULT_PORT))
            }
            _ => (hostport.to_string(), DEFAULT_PORT),
        }
    };

    let host = if host.is_empty() {
        DEFAULT_HOST.to_string()
    } else {
        connectable_host(&host)
    };
    format!("http://{host}:{port}")
}

/// Rewrite a wildcard bind host (`0.0.0.0`, `[::]`) to its loopback
/// equivalent, by value rather than spelling so expanded IPv6 forms are caught.
fn connectable_host(host: &str) -> String {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => {
            if ip.is_ipv4() {
                "127.0.0.1".to_string()
            } else {
                "[::1]".to_string()
            }
        }
        Ok(ip) if ip.is_ipv6() => format!("[{bare}]"),
        _ => host.to_string(),
    }
}

/// One newline-delimited JSON object from `/api/pull`, mirroring Ollama's
/// `ProgressResponse`.
#[derive(Deserialize, Default)]
struct PullLine {
    #[serde(default)]
    status: String,
    #[serde(default)]
    error: String,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    completed: u64,
}

/// The subset of `llmman resolve`'s output contract we depend on.
///
/// Deliberately ignores unknown fields: `format` and `mmproj` are part of the
/// documented output but are not needed here, and the contract is allowed to
/// grow.
#[derive(Deserialize)]
struct ResolveOutput {
    path: String,
}

/// Confirm a llmman daemon is listening and answering.
///
/// `/api/version` is llmman's own identity endpoint; a response without a
/// `version` field means something else is bound to the port, which is worth
/// distinguishing from nothing listening at all.
async fn check_daemon(client: &reqwest::Client, base: &str) -> anyhow::Result<()> {
    let resp = client
        .get(format!("{base}/api/version"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .with_context(|| {
            format!(
                "no llmman daemon reachable at {base}. Start one with `llmman serve`, \
                 or point {} at an existing daemon.",
                env_model::oci::LLMMAN_HOST
            )
        })?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "llmman daemon at {base} answered /api/version with {}",
            resp.status()
        );
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .with_context(|| format!("the server at {base} is not an llmman daemon"))?;
    if body.get("version").and_then(|v| v.as_str()).is_none() {
        anyhow::bail!("the server at {base} is not an llmman daemon (no version in /api/version)");
    }
    Ok(())
}

/// Stream `POST /api/pull` until the daemon reports success.
async fn pull(client: &reqwest::Client, base: &str, reference: &str) -> anyhow::Result<()> {
    let resp = client
        .post(format!("{base}/api/pull"))
        .json(&serde_json::json!({ "model": reference }))
        .send()
        .await
        .with_context(|| format!("llmman pull of '{reference}' failed"))?;

    if !resp.status().is_success() {
        anyhow::bail!("llmman pull of '{reference}' failed: {}", resp.status());
    }

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut succeeded = false;

    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.with_context(|| format!("reading llmman pull stream for '{reference}'"))?;
        buf.extend_from_slice(&chunk);

        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            if handle_pull_line(&line[..line.len() - 1], reference, &mut succeeded)? {
                // `success` seen; keep draining so the connection closes cleanly.
            }
        }
    }
    if !buf.is_empty() {
        handle_pull_line(&buf, reference, &mut succeeded)?;
    }

    if !succeeded {
        anyhow::bail!("llmman pull of '{reference}' ended without reporting success");
    }
    Ok(())
}

/// Interpret one NDJSON line. Returns whether the line was `success`.
///
/// A non-JSON line is tolerated rather than aborting a pull that may still be
/// progressing.
fn handle_pull_line(line: &[u8], reference: &str, succeeded: &mut bool) -> anyhow::Result<bool> {
    let text = String::from_utf8_lossy(line);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let Ok(parsed) = serde_json::from_str::<PullLine>(trimmed) else {
        return Ok(false);
    };

    if !parsed.error.is_empty() {
        anyhow::bail!("llmman pull of '{reference}' failed: {}", parsed.error);
    }
    if parsed.status == "success" {
        *succeeded = true;
        return Ok(true);
    }
    if !parsed.status.is_empty() {
        if parsed.total > 0 {
            tracing::info!(
                "llmman: {} ({}/{} bytes)",
                parsed.status,
                parsed.completed,
                parsed.total
            );
        } else {
            tracing::info!("llmman: {}", parsed.status);
        }
    }
    Ok(false)
}

/// Ask the CLI where the daemon's pull left the model on disk.
async fn resolve(reference: &str) -> anyhow::Result<PathBuf> {
    let bin = llmman_bin();
    let output = tokio::process::Command::new(&bin)
        .arg("resolve")
        .arg("--no-pull")
        .arg(reference)
        .stderr(Stdio::inherit())
        .stdout(Stdio::piped())
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| {
            format!(
                "failed to run '{bin} resolve --no-pull {reference}'. Install llmman \
                 (https://github.com/llmmanorg/llmman) and put it on PATH, or point \
                 {} at it.",
                env_model::oci::DYN_LLMMAN_BIN
            )
        })?;

    if !output.status.success() {
        anyhow::bail!(
            "'{bin} resolve --no-pull {reference}' failed with {}. See the error above.",
            output.status
        );
    }

    parse_resolve_output(&output.stdout, reference)
}

/// Parse `llmman resolve`'s stdout into the resolved local path.
///
/// Split out from [`resolve`] so the contract can be tested without a
/// subprocess.
fn parse_resolve_output(stdout: &[u8], reference: &str) -> anyhow::Result<PathBuf> {
    let text = std::str::from_utf8(stdout)
        .with_context(|| format!("llmman resolve '{reference}': stdout was not valid UTF-8"))?;

    // The contract is a single line, but be tolerant of a trailing newline or
    // of a diagnostic that leaked onto stdout before it: take the last
    // non-empty line.
    let line = text
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .with_context(|| format!("llmman resolve '{reference}': no output on stdout"))?;

    let parsed: ResolveOutput = serde_json::from_str(line).with_context(|| {
        format!("llmman resolve '{reference}': could not parse output as JSON: {line}")
    })?;

    if parsed.path.trim().is_empty() {
        anyhow::bail!("llmman resolve '{reference}': returned an empty path");
    }

    let path = PathBuf::from(parsed.path);
    if !path.exists() {
        anyhow::bail!(
            "llmman resolve '{reference}': reported path '{}' does not exist",
            path.display()
        );
    }

    tracing::debug!("Resolved '{reference}' to '{}'", path.display());
    Ok(path)
}

/// Pull (if necessary) an `oci://` reference through a running `llmman serve`,
/// returning the local path to the model files.
///
/// `reference` may be given with or without the `oci://` prefix.
pub async fn from_oci(reference: &str) -> anyhow::Result<PathBuf> {
    let bare = strip_scheme(reference);
    if bare.trim().is_empty() {
        anyhow::bail!("empty OCI model reference: '{reference}'");
    }
    let bare = bare.trim();

    let base = endpoint();
    let client = reqwest::Client::new();
    check_daemon(&client, &base).await?;

    tracing::info!("Pulling OCI model '{bare}' via llmman daemon at {base}");
    pull(&client, &base, bare).await?;

    resolve(bare).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_the_oci_scheme() {
        assert!(is_oci_ref("oci://ghcr.io/org/model:tag"));
        assert!(is_oci_ref("OCI://ghcr.io/org/model:tag"));
        assert!(is_oci_ref("Oci://ghcr.io/org/model:tag"));
    }

    #[test]
    fn leaves_every_other_reference_shape_alone() {
        // A bare HF repo id must never be claimed: that is the whole reason
        // the scheme is explicit.
        assert!(!is_oci_ref("Qwen/Qwen3-0.6B"));
        assert!(!is_oci_ref("ghcr.io/org/model:tag"));
        assert!(!is_oci_ref("/local/path/to/model"));
        assert!(!is_oci_ref("s3://bucket/key"));
        assert!(!is_oci_ref("hf://org/model"));
        assert!(!is_oci_ref(""));
        // Scheme with nothing after it is not a usable reference.
        assert!(!is_oci_ref("oci://"));
    }

    #[test]
    fn strips_the_scheme_only_when_present() {
        assert_eq!(
            strip_scheme("oci://ghcr.io/org/model:tag"),
            "ghcr.io/org/model:tag"
        );
        assert_eq!(
            strip_scheme("OCI://ghcr.io/org/model:tag"),
            "ghcr.io/org/model:tag"
        );
        assert_eq!(strip_scheme("Qwen/Qwen3-0.6B"), "Qwen/Qwen3-0.6B");
        assert_eq!(strip_scheme("oci://"), "oci://");
    }

    #[test]
    fn endpoint_defaults_to_llmman_serve_default() {
        assert_eq!(endpoint_from(None), "http://127.0.0.1:17434");
        assert_eq!(endpoint_from(Some("")), "http://127.0.0.1:17434");
        assert_eq!(endpoint_from(Some("   ")), "http://127.0.0.1:17434");
    }

    #[test]
    fn endpoint_parses_every_llmman_host_form() {
        assert_eq!(endpoint_from(Some("1.2.3.4:9999")), "http://1.2.3.4:9999");
        assert_eq!(endpoint_from(Some("1.2.3.4")), "http://1.2.3.4:17434");
        assert_eq!(
            endpoint_from(Some("http://1.2.3.4:9999")),
            "http://1.2.3.4:9999"
        );
        assert_eq!(
            endpoint_from(Some("http://1.2.3.4:9999/ignored")),
            "http://1.2.3.4:9999"
        );
        assert_eq!(
            endpoint_from(Some("\"1.2.3.4:9999\"")),
            "http://1.2.3.4:9999"
        );
    }

    #[test]
    fn endpoint_rewrites_a_wildcard_bind_to_loopback() {
        // A client cannot connect to "every interface".
        assert_eq!(endpoint_from(Some("0.0.0.0:9999")), "http://127.0.0.1:9999");
        assert_eq!(endpoint_from(Some("[::]:9999")), "http://[::1]:9999");
        // By value, not spelling: an expanded IPv6 form is caught too.
        assert_eq!(
            endpoint_from(Some("[0:0:0:0:0:0:0:0]:9999")),
            "http://[::1]:9999"
        );
    }

    #[test]
    fn parses_the_documented_resolve_contract() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let line = format!(
            r#"{{"reference":"ghcr.io/org/model:tag","path":"{path}","format":"safetensors"}}"#
        );
        assert_eq!(
            parse_resolve_output(line.as_bytes(), "ghcr.io/org/model:tag").unwrap(),
            dir.path()
        );
    }

    #[test]
    fn tolerates_a_trailing_newline_and_leaked_diagnostics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let out = format!("pulling blobs...\n{{\"reference\":\"r\",\"path\":\"{path}\"}}\n");
        assert_eq!(
            parse_resolve_output(out.as_bytes(), "r").unwrap(),
            dir.path()
        );
    }

    #[test]
    fn ignores_unknown_fields_so_the_contract_can_grow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let line = format!(
            r#"{{"reference":"r","path":"{path}","format":"gguf","mmproj":"/x","future":1}}"#
        );
        assert!(parse_resolve_output(line.as_bytes(), "r").is_ok());
    }

    #[test]
    fn rejects_malformed_resolve_output() {
        assert!(parse_resolve_output(b"", "r").is_err());
        assert!(parse_resolve_output(b"   \n\n", "r").is_err());
        assert!(parse_resolve_output(b"not json", "r").is_err());
        assert!(parse_resolve_output(b"{\"no_path\":1}", "r").is_err());
        assert!(parse_resolve_output(br#"{"path":""}"#, "r").is_err());
        assert!(parse_resolve_output(br#"{"path":"/nonexistent/xyzzy"}"#, "r").is_err());
        assert!(parse_resolve_output(&[0xff, 0xfe], "r").is_err());
    }

    #[test]
    fn pull_line_reports_an_in_band_error() {
        // The daemon streams errors in-band, so HTTP 200 does not mean success.
        let mut ok = false;
        let err = handle_pull_line(br#"{"error":"unauthorized"}"#, "r", &mut ok).unwrap_err();
        assert!(err.to_string().contains("unauthorized"));
        assert!(!ok);
    }

    #[test]
    fn pull_line_marks_success() {
        let mut ok = false;
        assert!(handle_pull_line(br#"{"status":"success"}"#, "r", &mut ok).unwrap());
        assert!(ok);
    }

    #[test]
    fn pull_line_tolerates_noise_and_progress() {
        let mut ok = false;
        // A non-JSON diagnostic must not abort a pull still in progress.
        assert!(!handle_pull_line(b"not json", "r", &mut ok).unwrap());
        assert!(!handle_pull_line(b"", "r", &mut ok).unwrap());
        assert!(
            !handle_pull_line(
                br#"{"status":"pulling blobs","completed":5,"total":10}"#,
                "r",
                &mut ok
            )
            .unwrap()
        );
        assert!(!ok);
    }

    #[serial_test::serial]
    #[test]
    fn binary_name_defaults_and_is_overridable() {
        temp_env::with_var(env_model::oci::DYN_LLMMAN_BIN, None::<&str>, || {
            assert_eq!(llmman_bin(), "llmman");
        });
        temp_env::with_var(
            env_model::oci::DYN_LLMMAN_BIN,
            Some("/opt/bin/llmman"),
            || {
                assert_eq!(llmman_bin(), "/opt/bin/llmman");
            },
        );
        // An empty override is a configuration mistake, not a request to run
        // the empty string.
        temp_env::with_var(env_model::oci::DYN_LLMMAN_BIN, Some("  "), || {
            assert_eq!(llmman_bin(), "llmman");
        });
    }

    #[tokio::test]
    async fn empty_reference_is_rejected_without_contacting_the_daemon() {
        assert!(from_oci("oci://").await.is_err());
        assert!(from_oci("oci://   ").await.is_err());
    }
}
