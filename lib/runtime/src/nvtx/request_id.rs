// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The request-identity format shared by every NVTX producer: the [`rid_key`] payload and
//! the `dynamo.request rid=<rid>[ key=value]...` identity message. Pure string and integer
//! code with no FFI; the parent module re-exports everything here, so callers use
//! `nvtx::rid_key` and friends.
//!
//! Everything goes to the default NVTX domain:
//!
//! - **Request-scoped range or mark**: ASCII message = the stage name only
//!   (`http-request`, `kv_router.route_request:prefill`, ...), event payload = unsigned
//!   64-bit [`rid_key`]`(rid)`. In the nsys SQLite export these are `NVTX_EVENTS.text` and
//!   `NVTX_EVENTS.uint64Value`.
//! - **Request identity mark** (`eventType` 34): `dynamo.request rid=<rid>[ key=value]...`
//!   with the same payload, mapping the key back to the id (see [`format_request_message`]
//!   for ordering and sanitizing). Several may exist per request; readers merge them.
//!
//! [`rid_key`] is FNV-1a-64 of the UTF-8 request id with the top bit cleared so it stays
//! positive in SQLite's signed INTEGER. Any other implementation (such as a Python one) must
//! reproduce the same values; the test vectors in this module pin them.

/// Longest value (in bytes) emitted for any `rid=`/`key=` token.
pub const MAX_VALUE_BYTES: usize = 128;

/// Keys that always come first after `rid=`, in this order.
const LEADING_KEYS: [&str; 3] = ["trace", "xrid", "engine_rid"];

/// Append `value` sanitized for a space-separated token: whitespace becomes `_`,
/// control characters are dropped, and the result is cut to [`MAX_VALUE_BYTES`] on a
/// char boundary. Returns whether anything was written.
fn push_sanitized(out: &mut String, value: &str) -> bool {
    let start = out.len();
    for ch in value.chars() {
        let ch = if ch.is_whitespace() {
            '_'
        } else if ch.is_control() {
            continue;
        } else {
            ch
        };
        if out.len() - start + ch.len_utf8() > MAX_VALUE_BYTES {
            break;
        }
        out.push(ch);
    }
    out.len() > start
}

fn push_attr(out: &mut String, key: &str, value: &str) {
    let rollback = out.len();
    out.push(' ');
    let key_start = out.len();
    out.extend(key.chars().map(|c| {
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
            c
        } else {
            '_'
        }
    }));
    if out.len() == key_start {
        out.truncate(rollback);
        return;
    }
    out.push('=');
    if !push_sanitized(out, value) {
        out.truncate(rollback);
    }
}

/// Stage name of the request identity mark.
pub const IDENTITY_STAGE: &str = "dynamo.request";

/// The 64-bit NVTX payload that identifies a request: FNV-1a-64 of the UTF-8 request id
/// with the top bit cleared, so the value stays positive in SQLite's signed INTEGER.
///
/// Stable across processes and languages (the test vectors below pin it), which is
/// what lets the frontend's and the workers' events for one request be joined.
#[must_use]
pub const fn rid_key(rid: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let bytes = rid.as_bytes();
    let mut hash = OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    hash & 0x7FFF_FFFF_FFFF_FFFF
}

/// Render the request identity mark, `dynamo.request rid=<rid>[ key=value]...` (always
/// [`IDENTITY_STAGE`] first). `trace`, `xrid` and `engine_rid` come first (in that order),
/// then the remaining `attrs` sorted by key.
/// Values are sanitized and bounded (see [`MAX_VALUE_BYTES`]) and attributes whose value
/// ends up empty are omitted. This allocates; callers must test [`enabled()`](super::enabled) first (the
/// `dynamo_nvtx_request_*` macros do).
pub fn format_request_message(rid: &str, attrs: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(IDENTITY_STAGE.len() + rid.len() + 32 + attrs.len() * 24);
    out.push_str(IDENTITY_STAGE);
    out.push_str(" rid=");
    push_sanitized(&mut out, rid);

    for key in LEADING_KEYS {
        if let Some((_, value)) = attrs.iter().find(|(k, _)| *k == key) {
            push_attr(&mut out, key, value);
        }
    }
    let mut rest: Vec<&(&str, &str)> = attrs
        .iter()
        .filter(|(k, _)| !LEADING_KEYS.contains(k))
        .collect();
    rest.sort_by_key(|(k, _)| *k);
    for (key, value) in rest {
        push_attr(&mut out, key, value);
    }
    out
}

/// Pinned [`rid_key`] vectors, shared with the parent module's keyed-range test.
#[cfg(test)]
pub(super) const PIN_A: u64 = 3_414_815_163_700_866_188;
#[cfg(test)]
const PIN_UUID: u64 = 1_346_822_055_457_142_067;

#[cfg(test)]
mod tests {
    use super::{PIN_A, PIN_UUID};

    // ── Request message formatter ────────────────────────────────────────────

    #[test]
    fn format_minimal_message() {
        assert_eq!(
            super::format_request_message("abc", &[]),
            "dynamo.request rid=abc"
        );
    }

    #[test]
    fn format_orders_leading_keys_then_sorted_rest() {
        let msg = super::format_request_message(
            "r1",
            &[
                ("zeta", "z"),
                ("engine_rid", "e"),
                ("alpha", "a"),
                ("xrid", "x"),
                ("trace", "t"),
            ],
        );
        assert_eq!(
            msg,
            "dynamo.request rid=r1 trace=t xrid=x engine_rid=e alpha=a zeta=z"
        );
    }

    #[test]
    fn format_omits_empty_and_fully_dropped_values() {
        let msg = super::format_request_message(
            "r",
            &[("xrid", ""), ("a", "\u{1}\u{7f}"), ("b", "ok"), ("", "v")],
        );
        assert_eq!(msg, "dynamo.request rid=r b=ok");
    }

    #[test]
    fn format_sanitizes_whitespace_and_control_chars() {
        let msg = super::format_request_message(
            "a b\tc\nd",
            &[("xrid", "x y\u{0}z\r\n"), ("k=ey v", "v")],
        );
        assert_eq!(msg, "dynamo.request rid=a_b_c_d xrid=x_yz__ k_ey_v=v");
        assert_eq!(msg.matches(' ').count(), 3);
    }

    #[test]
    fn format_truncates_on_char_boundary() {
        let long = "a".repeat(500);
        let msg = super::format_request_message(&long, &[]);
        assert_eq!(
            msg,
            format!("dynamo.request rid={}", "a".repeat(super::MAX_VALUE_BYTES))
        );

        // 3-byte chars: 128 / 3 = 42 whole chars (126 bytes); never split one.
        let wide = "\u{20ac}".repeat(100);
        let msg = super::format_request_message("r", &[("xrid", &wide)]);
        let value = msg
            .strip_prefix("dynamo.request rid=r xrid=")
            .expect("prefix");
        assert_eq!(value.chars().count(), 42);
        assert_eq!(value.len(), 126);

        // Exactly at the limit is kept whole.
        let exact = "b".repeat(super::MAX_VALUE_BYTES);
        assert!(super::format_request_message(&exact, &[]).ends_with(&exact));
    }

    #[test]
    fn format_matches_the_parser_regex_shape() {
        let msg = super::format_request_message(
            "123e4567-e89b-12d3-a456-426614174000",
            &[
                ("trace", "0af7651916cd43dd8448eb211c80319c"),
                ("xrid", "my id"),
            ],
        );
        let mut tokens = msg.split(' ');
        assert_eq!(tokens.next(), Some(super::IDENTITY_STAGE));
        assert!(tokens.next().expect("rid").starts_with("rid="));
        for rest in tokens {
            assert!(rest.contains('='), "{rest}");
        }
    }

    #[test]
    fn rid_key_is_fnv1a_with_top_bit_cleared() {
        // Pinned: other implementations must reproduce these vectors.
        assert_eq!(super::rid_key(""), 5_472_609_002_491_880_229);
        assert_eq!(super::rid_key("a"), PIN_A);
        assert_eq!(
            super::rid_key("6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42"),
            PIN_UUID
        );
        for rid in ["", "a", "zz", "6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42"] {
            assert!(super::rid_key(rid) < 1 << 63, "{rid:?}");
        }
    }
}
