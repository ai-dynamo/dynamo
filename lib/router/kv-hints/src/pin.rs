// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pinning a session's pages in the Mooncake store while it waits on a tool.

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// A worker's Mooncake objects as SGLang publishes them (`hicache_object_layout`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MooncakeObjectLayout {
    pub pools: Vec<MooncakeObjectPool>,
    #[serde(default)]
    pub key_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MooncakeObjectPool {
    pub pool: String,
    pub suffixes: Vec<String>,
    /// `all_pages`, or `trailing_pages` for state kept only on the last pages.
    pub hit_policy: String,
    #[serde(default)]
    pub trailing_pages: Option<u32>,
}

impl MooncakeObjectLayout {
    /// Every object key of the pages named by `page_hashes`, in page order.
    pub fn object_keys(&self, page_hashes: &[String]) -> Vec<String> {
        let mut keys = Vec::new();
        for (index, page) in page_hashes.iter().enumerate() {
            let base = match &self.key_prefix {
                Some(prefix) => format!("{prefix}_{page}"),
                None => page.clone(),
            };
            for pool in &self.pools {
                let trailing = pool.trailing_pages.unwrap_or(1) as usize;
                if pool.hit_policy != "trailing_pages" || index + trailing >= page_hashes.len() {
                    keys.extend(pool.suffixes.iter().map(|suffix| format!("{base}{suffix}")));
                }
            }
        }
        keys
    }
}

/// The Mooncake master takes at most this many keys in one `/api/v1/soft_pin` request.
pub const MAX_SOFT_PIN_KEYS: usize = 4096;

/// A Mooncake master `POST /api/v1/soft_pin` body for a tool-call pin: extend-only, so it never
/// shortens another session's pin on a shared page, and held for keys still being written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SoftPinRequest<'a> {
    pub tenant_id: &'a str,
    pub keys: &'a [String],
    pub ttl_ms: u64,
    pub priority: u32,
    pub extend_only: bool,
    pub pending: bool,
}

/// The requests that pin `keys` for `ttl` at `priority`, at most [`MAX_SOFT_PIN_KEYS`] keys each.
pub fn soft_pin_requests<'a>(
    tenant_id: &'a str,
    keys: &'a [String],
    ttl: Duration,
    priority: u32,
) -> impl Iterator<Item = SoftPinRequest<'a>> {
    let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX).max(1);
    keys.chunks(MAX_SOFT_PIN_KEYS)
        .map(move |keys| SoftPinRequest {
            tenant_id,
            keys,
            ttl_ms,
            priority,
            extend_only: true,
            pending: true,
        })
}

/// How long to pin and at what priority: the expected tool time with headroom (two minutes
/// when unknown), and the log2 of the tokens a miss would recompute.
pub fn pin_ttl_and_priority(expected: Option<Duration>, tokens: usize) -> (Duration, u32) {
    let ttl = expected.map_or(Duration::from_secs(120), |expected| {
        expected.mul_f64(1.5) + Duration::from_secs(5)
    });
    (
        ttl.min(Duration::from_secs(1800)),
        usize::BITS - tokens.leading_zeros(),
    )
}

type ArmedPin = Box<dyn Fn(&[u32], Option<Duration>) + Send + Sync>;

/// One request's pin: the router arms it with the worker it picked, the response records its
/// tokens and tool calls, and the response's end fires it if a tool call ended the response.
#[derive(Default)]
pub struct ToolCallPinSlot {
    pin: OnceLock<ArmedPin>,
    called: Mutex<bool>,
    output: Mutex<Vec<u32>>,
}

impl ToolCallPinSlot {
    /// `pin` receives the response's tokens and the expected tool time.
    pub fn arm(&self, pin: impl Fn(&[u32], Option<Duration>) + Send + Sync + 'static) {
        let _ = self.pin.set(Box::new(pin));
    }

    pub fn push_output(&self, tokens: &[u32]) {
        self.output.lock().unwrap().extend_from_slice(tokens);
    }

    pub fn tool_called(&self) {
        *self.called.lock().unwrap() = true;
    }

    /// Pin if the response called a tool or `announced` (the client's
    /// `x-dynamo-experimental-sim-tool-call`, with its expected time) says one follows.
    pub fn finish(&self, announced: Option<Option<Duration>>) {
        let called = *self.called.lock().unwrap();
        if let (Some(pin), true) = (self.pin.get(), called || announced.is_some()) {
            pin(&self.output.lock().unwrap(), announced.flatten());
        }
    }
}

/// `x-dynamo-experimental-sim-tool-call: <tool>[; expected_ms=<ms>]`: the expected tool time, if given.
pub fn parse_sim_tool_call(value: &str) -> Option<Duration> {
    value.split(';').skip(1).find_map(|part| {
        let (name, ms) = part.split_once('=')?;
        if name.trim() != "expected_ms" {
            return None;
        }
        ms.trim().parse().ok().map(Duration::from_millis)
    })
}
