// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbmctl p2p` — drive the hub-mediated P2P block-transfer lifecycle from the
//! command line, no inference required.
//!
//! Unlike the read-only [`FeatureCli`](crate::features::cli::FeatureCli)
//! surface under `kvbmctl get`, these are **per-instance actions**: each targets
//! a leader through the hub's `/v1/instances/{id}/control/...` routes. The
//! route templates are the [`crate::protocol::paths`] consts (shared with the
//! server-side routers) — this CLI never hand-rolls a path string.
//!
//! Four verbs map the transfer lifecycle:
//!
//! - `pin` — `open_session` (sync/prefix) on the **holder**; returns the
//!   `{session_id, endpoint, committed}` triple. The session keeps the matched
//!   G2 blocks pinned for the puller. The holder's blocks must already be
//!   present (warmed by real inference); this CLI does not create them.
//! - `pull` — make the **puller** learn the holder as a peer
//!   (`core/register_leader`), then `pull_from_session` the holder's session
//!   into the puller's G2. `--endpoint` is **required** in v1 (the leader
//!   rejects a pull without it — there is no hub-registry lookup yet).
//! - `unpin` — `close_session` on the holder. Idempotent.
//! - `xfer` — `pin` → `pull` → `unpin`, threading the session endpoint
//!   internally. The one-shot copy.
//!
//! `--hashes` is a comma-separated list of decimal `u128`s; each encodes to the
//! 16-big-endian-byte wire shape the leader expects (same as
//! `kvbmctl get indexer query`). Instance ids pass through to the URL path
//! verbatim. There is no `--ttl`: the session watchdog is fixed at the
//! `SessionManager` default (per-session override is unwired in v1).

use anyhow::{Context, Result, anyhow, bail};
use clap::{Arg, ArgMatches, Command, value_parser};
use serde_json::{Value, json};

use crate::client::HubClient;
use crate::protocol::paths::{
    CONTROL_CORE_REGISTER_LEADER, CONTROL_TRANSFER_CLOSE_SESSION, CONTROL_TRANSFER_OPEN_SESSION,
    CONTROL_TRANSFER_PULL_FROM_SESSION,
};

/// The `p2p` command group. `kvbmctl` injects a global `--hub`.
pub fn p2p_command() -> Command {
    Command::new("p2p")
        .about("Drive hub-mediated P2P block transfers (pin/pull/unpin/xfer)")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("pin")
                .about("Open a transfer session over a holder's matching blocks (open_session)")
                .arg(instance_arg("instance-id", "Holder instance id"))
                .arg(hashes_arg(true)),
        )
        .subcommand(
            Command::new("pull")
                .about("Pull a pinned session's blocks into the puller's G2 (register + pull)")
                .arg(instance_arg("from", "Source (holder) instance id"))
                .arg(instance_arg("to", "Destination (puller) instance id"))
                .arg(session_arg())
                .arg(endpoint_arg())
                .arg(hashes_arg(false)),
        )
        .subcommand(
            Command::new("unpin")
                .about("Close a transfer session on the holder (close_session)")
                .arg(instance_arg("instance-id", "Holder instance id"))
                .arg(session_arg()),
        )
        .subcommand(
            Command::new("xfer")
                .about("One-shot copy holder→puller: pin + pull + unpin")
                .arg(instance_arg("from", "Source (holder) instance id"))
                .arg(instance_arg("to", "Destination (puller) instance id"))
                .arg(hashes_arg(true)),
        )
}

fn instance_arg(name: &'static str, help: &'static str) -> Arg {
    Arg::new(name).long(name).required(true).help(help)
}

fn session_arg() -> Arg {
    Arg::new("session-id")
        .long("session-id")
        .required(true)
        .help("Session id from `pin`")
}

fn endpoint_arg() -> Arg {
    Arg::new("endpoint")
        .long("endpoint")
        .required(true)
        .help("Session endpoint JSON from `pin` (required in v1)")
}

/// `--hashes <a,b,c>` — comma-separated decimal `u128`s.
fn hashes_arg(required: bool) -> Arg {
    Arg::new("hashes")
        .long("hashes")
        .required(required)
        .value_delimiter(',')
        .value_parser(value_parser!(u128))
        .help("Comma-separated block hashes as decimal u128")
}

/// Build the `sequence_hashes` wire value (array of 16-big-endian-byte arrays)
/// from `--hashes`. `None` when the arg was absent (only valid where optional).
fn sequence_hashes(m: &ArgMatches) -> Option<Vec<Vec<u8>>> {
    m.get_many::<u128>("hashes")
        .map(|vals| vals.map(|h| h.to_be_bytes().to_vec()).collect())
}

fn instance(m: &ArgMatches, name: &str) -> String {
    m.get_one::<String>(name)
        .expect("instance arg is required")
        .clone()
}

/// Fill a `protocol::paths` control-route template (which carries the literal
/// `{instance_id}` segment) with a concrete instance id. The CLI and the
/// server-side router share the same const; only the substitution differs.
fn fill_instance(template: &str, instance_id: &str) -> String {
    template.replace("{instance_id}", instance_id)
}

/// Dispatch a `p2p` subcommand against `hub`. Returns the JSON to print.
pub async fn run_p2p(hub: &HubClient, matches: &ArgMatches) -> Result<Value> {
    match matches.subcommand() {
        Some(("pin", m)) => {
            let id = instance(m, "instance-id");
            let hashes = sequence_hashes(m).expect("--hashes is required for pin");
            pin(hub, &id, hashes).await
        }
        Some(("pull", m)) => {
            let from = instance(m, "from");
            let to = instance(m, "to");
            let session_id = m.get_one::<String>("session-id").expect("required").clone();
            let endpoint = parse_endpoint(m)?;
            let selector = sequence_hashes(m);
            pull(hub, &from, &to, &session_id, endpoint, selector).await
        }
        Some(("unpin", m)) => {
            let id = instance(m, "instance-id");
            let session_id = m.get_one::<String>("session-id").expect("required");
            unpin(hub, &id, session_id).await
        }
        Some(("xfer", m)) => {
            let from = instance(m, "from");
            let to = instance(m, "to");
            let hashes = sequence_hashes(m).expect("--hashes is required for xfer");
            xfer(hub, &from, &to, hashes).await
        }
        _ => unreachable!("subcommand_required(true) guarantees a subcommand"),
    }
}

/// Parse the `--endpoint` JSON the user copied from `pin`'s output.
fn parse_endpoint(m: &ArgMatches) -> Result<Value> {
    let ep = m.get_one::<String>("endpoint").expect("required");
    serde_json::from_str(ep).with_context(|| format!("--endpoint must be JSON, got {ep:?}"))
}

/// Holder-side `open_session` (sync/prefix). Returns the typed
/// `(session_id, endpoint, committed)` triple — `committed` as `u128`s. Shared
/// by the `pin` verb (formats JSON) and `xfer` (consumes directly, so it never
/// re-parses `pin`'s own JSON).
async fn open_session(
    hub: &HubClient,
    holder: &str,
    hashes: Vec<Vec<u8>>,
) -> Result<(String, Value, Vec<u128>)> {
    let open: Value = hub
        .post_json(
            &fill_instance(CONTROL_TRANSFER_OPEN_SESSION, holder),
            &json!({ "sequence_hashes": hashes, "search_mode": "prefix", "find_mode": "sync" }),
        )
        .await
        .context("open_session")?;
    extract_capability(holder, &open)
}

/// Holder-side `open_session` (sync/prefix). Returns the `{session_id,
/// endpoint, committed}` the puller needs.
async fn pin(hub: &HubClient, holder: &str, hashes: Vec<Vec<u8>>) -> Result<Value> {
    let requested = hashes.len();
    let (session_id, endpoint, committed) = open_session(hub, holder, hashes).await?;
    // Echo committed hashes as decimal strings (u128 overflows JSON numbers) so
    // the user can paste a subset straight into `pull --hashes`.
    let committed_decimal: Vec<String> = committed.iter().map(u128::to_string).collect();
    Ok(json!({
        "session_id": session_id,
        "endpoint": endpoint,
        "requested": requested,
        "committed_count": committed.len(),
        "committed": committed_decimal,
    }))
}

/// Puller-side: register the holder as a peer, then `pull_from_session`.
async fn pull(
    hub: &HubClient,
    from: &str,
    to: &str,
    session_id: &str,
    endpoint: Value,
    selector: Option<Vec<Vec<u8>>>,
) -> Result<Value> {
    register_peer(hub, to, from)
        .await
        .context("register source as peer on destination")?;
    let mut body = json!({
        "session_id": session_id,
        "source_instance_id": from,
        "endpoint": endpoint,
    });
    if let Some(sel) = selector {
        body["selector"] = json!(sel);
    }
    hub.post_json(
        &fill_instance(CONTROL_TRANSFER_PULL_FROM_SESSION, to),
        &body,
    )
    .await
    .context("pull_from_session")
}

/// Holder-side `close_session`. Idempotent.
async fn unpin(hub: &HubClient, holder: &str, session_id: &str) -> Result<Value> {
    hub.post_json(
        &fill_instance(CONTROL_TRANSFER_CLOSE_SESSION, holder),
        &json!({ "session_id": session_id }),
    )
    .await
    .context("close_session")
}

/// `open_session` → `pull` → `close_session`, threading the session endpoint
/// internally.
///
/// Once `open_session` succeeds the holder has an open session, so a failed
/// `pull` closes it best-effort before propagating — otherwise the session
/// would stay pinned on the holder until its 30s watchdog fires. (`open_session`
/// returns typed values, so there is no fragile re-parse of `pin`'s JSON.)
async fn xfer(hub: &HubClient, from: &str, to: &str, hashes: Vec<Vec<u8>>) -> Result<Value> {
    let requested = hashes.len();
    let (session_id, endpoint, committed) = open_session(hub, from, hashes).await?;

    let pulled = match pull(hub, from, to, &session_id, endpoint, None).await {
        Ok(pull_res) => pull_res
            .get("pulled")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0),
        Err(e) => {
            // Best-effort teardown so the holder doesn't keep the session
            // pinned until the watchdog; surface the original pull error.
            if let Err(close_err) = unpin(hub, from, &session_id).await {
                tracing::warn!(
                    session = %session_id, error = %close_err,
                    "xfer: cleanup close_session failed after pull error"
                );
            }
            return Err(e);
        }
    };

    let close = unpin(hub, from, &session_id).await.context("unpin")?;

    Ok(json!({
        "from": from,
        "to": to,
        "session_id": session_id,
        "requested": requested,
        "committed_count": committed.len(),
        "pulled": pulled,
        "close": close,
    }))
}

/// Make `instance` learn `peer` as a velo peer via the hub's typed
/// `core/register_leader` route.
async fn register_peer(hub: &HubClient, instance: &str, peer: &str) -> Result<Value> {
    hub.post_json(
        &fill_instance(CONTROL_CORE_REGISTER_LEADER, instance),
        &json!({ "instance_id": peer }),
    )
    .await
}

/// Pull `(session_id, endpoint, committed)` out of an
/// `OpenTransferSessionResponse` JSON. `committed` is the decoded list of
/// committed sequence hashes as `u128` — symmetric with `--hashes` input, so a
/// user can feed a subset straight back into `pull --hashes`. Errors on
/// `no_blocks_found` or a malformed capability.
fn extract_capability(holder: &str, open: &Value) -> Result<(String, Value, Vec<u128>)> {
    if open.get("result").and_then(Value::as_str) == Some("no_blocks_found") {
        bail!(
            "open_session on {holder} found no matching blocks \
             (is the holder's G2 warmed for these hashes?)"
        );
    }
    let capability = open
        .get("capability")
        .ok_or_else(|| anyhow!("open_session response missing capability: {open}"))?;
    let session_id = capability
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("capability missing session_id: {capability}"))?
        .to_string();
    // The endpoint is session-specific (not derivable from the holder's
    // instance id), so it must be threaded into the pull.
    let endpoint = capability
        .get("endpoint")
        .cloned()
        .ok_or_else(|| anyhow!("capability missing endpoint: {capability}"))?;
    // `committed` is a `Vec<SequenceHash>`; each hash serializes as 16
    // big-endian bytes (a JSON array of u8), the inverse of the `--hashes`
    // encoding. Decode back to `u128` so `pin` can echo decimal hashes.
    let committed = open
        .get("committed")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(decode_be_u128).collect())
        .unwrap_or_default();
    Ok((session_id, endpoint, committed))
}

/// Decode a JSON 16-byte big-endian array (how `serde_bytes_u128` renders a
/// `SequenceHash`) back into a `u128`. `None` on a wrong-length / non-numeric
/// entry.
fn decode_be_u128(v: &Value) -> Option<u128> {
    let arr = v.as_array()?;
    if arr.len() != 16 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (slot, byte) in bytes.iter_mut().zip(arr) {
        *slot = u8::try_from(byte.as_u64()?).ok()?;
    }
    Some(u128::from_be_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Parse argv through the real command tree and return the action subcommand
    // matches (e.g. the matches below `p2p pin`).
    fn sub_matches(argv: &[&str]) -> ArgMatches {
        let m = p2p_command()
            .try_get_matches_from(std::iter::once("p2p").chain(argv.iter().copied()))
            .expect("parse");
        let (_, sm) = m.subcommand().expect("has subcommand");
        sm.clone()
    }

    #[test]
    fn hashes_encode_to_be_bytes() {
        let m = sub_matches(&["pin", "--instance-id", "A", "--hashes", "0,1,2"]);
        let hashes = sequence_hashes(&m).expect("required");
        assert_eq!(hashes.len(), 3);
        assert!(hashes.iter().all(|h| h.len() == 16));
        assert_eq!(hashes[0], 0u128.to_be_bytes().to_vec());
        assert_eq!(hashes[2], 2u128.to_be_bytes().to_vec());
    }

    #[test]
    fn explicit_decimal_hashes() {
        let m = sub_matches(&["pin", "--instance-id", "A", "--hashes", "5,7"]);
        let hashes = sequence_hashes(&m).expect("required");
        assert_eq!(
            hashes,
            vec![5u128.to_be_bytes().to_vec(), 7u128.to_be_bytes().to_vec()]
        );
    }

    #[test]
    fn pin_requires_hashes() {
        let res = p2p_command().try_get_matches_from(["p2p", "pin", "--instance-id", "A"]);
        assert!(res.is_err(), "pin must require --hashes");
    }

    #[test]
    fn pull_selector_is_optional() {
        let m = sub_matches(&[
            "pull",
            "--from",
            "A",
            "--to",
            "B",
            "--session-id",
            "s1",
            "--endpoint",
            "{}",
        ]);
        assert!(sequence_hashes(&m).is_none());
    }

    #[test]
    fn fill_instance_substitutes_the_path_template() {
        assert_eq!(
            fill_instance(CONTROL_TRANSFER_OPEN_SESSION, "inst-1"),
            "/v1/instances/inst-1/control/transfer/open_session"
        );
        assert_eq!(
            fill_instance(CONTROL_CORE_REGISTER_LEADER, "inst-2"),
            "/v1/instances/inst-2/control/core/register_leader"
        );
    }

    #[test]
    fn extract_capability_reads_session_endpoint_and_decodes_hashes() {
        // `committed` is a Vec<SequenceHash>; each renders as 16 big-endian
        // bytes (JSON array of u8). 5 and 7 as the inverse of `--hashes`.
        let five = 5u128.to_be_bytes().to_vec();
        let seven = 7u128.to_be_bytes().to_vec();
        let open = json!({
            "result": "sync",
            "capability": {
                "session_id": "abc",
                "instance_id": "holder",
                "endpoint": { "kind": "kvbm_cd_session_v2", "payload": null },
            },
            "committed": [five, seven],
        });
        let (sid, ep, committed) = extract_capability("holder", &open).unwrap();
        assert_eq!(sid, "abc");
        assert_eq!(ep["kind"], "kvbm_cd_session_v2");
        assert_eq!(committed, vec![5u128, 7u128]);
    }

    #[test]
    fn extract_capability_errors_on_no_blocks() {
        let open = json!({ "result": "no_blocks_found" });
        assert!(extract_capability("holder", &open).is_err());
    }
}
