// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A model of SGLang's incremental `/generate` stream.
//!
//! The simulator answers the request body. Recorded fixtures check its shapes.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

const DEFAULT_MAX_NEW_TOKENS: u32 = 128;
/// Generated when a request asks for more tokens than the ground truth holds,
/// so an over-long budget shows up as extra output.
pub const FILLER_TOKEN: u32 = 0;

/// What an aggregated run generates.
#[derive(Debug, Clone)]
pub struct Truth {
    pub prompt: Vec<u32>,
    /// Logprob of each prompt token; the first is unused.
    pub prompt_logprobs: Vec<f64>,
    pub output: Vec<(u32, f64)>,
    /// Whether generation ended at an EOS token, rather than at the budget.
    pub stop: bool,
    /// Routed-expert bytes per token row.
    pub row_bytes: usize,
    /// Rows for every prompt and output position, sized for overshoot.
    pub experts: Vec<u8>,
    /// Chunk ends as output positions; the last is `output.len()`.
    pub cuts: Vec<usize>,
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Aggregated,
    Prefill,
    Decode,
}

impl Truth {
    fn token(&self, position: usize) -> (u32, f64) {
        self.output
            .get(position)
            .copied()
            .unwrap_or((FILLER_TOKEN, -1.0))
    }

    /// Token and logprob at a sequence position, prompt then output.
    fn sequence(&self, position: usize) -> (u32, Option<f64>) {
        match self.prompt.get(position) {
            Some(&id) => (id, (position > 0).then(|| self.prompt_logprobs[position])),
            None => {
                let (id, lp) = self.token(position - self.prompt.len());
                (id, Some(lp))
            }
        }
    }

    fn rows(&self, start: usize, end: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((end - start) * self.row_bytes);
        for row in start..end {
            let at = row * self.row_bytes;
            match self.experts.get(at..at + self.row_bytes) {
                Some(slice) => bytes.extend_from_slice(slice),
                None => bytes.extend(std::iter::repeat_n(0xEE, self.row_bytes)),
            }
        }
        bytes
    }
}

fn int(body: &Map<String, Value>, key: &str) -> Option<i64> {
    body.get(key).and_then(Value::as_i64)
}

fn flag(body: &Map<String, Value>, key: &str) -> bool {
    body.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn top_logprobs(id: u32, lp: f64, k: usize) -> Value {
    Value::Array(
        (0..k)
            .map(|j| json!([lp - j as f64, id.wrapping_add(j as u32), null]))
            .collect(),
    )
}

/// SGLang's response stream for `body` sent with `prompt`.
pub fn stream(
    truth: &Truth,
    body: &Map<String, Value>,
    prompt: &[u32],
    role: Role,
    engine_info: &Value,
) -> Vec<Value> {
    assert_eq!(
        prompt,
        truth.prompt.as_slice(),
        "the worker receives the client prompt"
    );
    let sampling = body.get("sampling_params").and_then(Value::as_object);
    // `--preferred-sampling-params` is a JSON string on the command line.
    let preferred = match engine_info.get("preferred_sampling_params") {
        Some(Value::String(json)) => serde_json::from_str(json).ok(),
        other => other.cloned(),
    }
    .and_then(|p: Value| p.get("max_new_tokens").and_then(Value::as_u64));
    let budget = sampling
        .and_then(|s| s.get("max_new_tokens"))
        .and_then(Value::as_u64)
        .or(preferred)
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS.into()) as usize;
    let rid = body.get("rid").cloned().unwrap_or(Value::Null);
    let return_logprob = flag(body, "return_logprob");
    let start = int(body, "logprob_start_len").unwrap_or(-1);
    let top_k = int(body, "top_logprobs_num").unwrap_or(0).max(0) as usize;
    let experts = flag(body, "return_routed_experts");
    let prompt_len = prompt.len();

    let input_logprobs = || {
        let from = usize::try_from(start).ok().filter(|&s| s < prompt_len);
        let positions = from.map_or(0..0, |from| from..prompt_len);
        let tokens: Vec<Value> = positions
            .clone()
            .map(|p| {
                let (id, lp) = truth.sequence(p);
                json!([lp, id, null])
            })
            .collect();
        let tops: Vec<Value> = positions
            .map(|p| match truth.sequence(p) {
                (_, None) => Value::Null,
                (id, Some(lp)) => top_logprobs(id, lp, top_k),
            })
            .collect();
        (tokens, tops)
    };

    if role == Role::Prefill {
        let mut meta = json!({
            "id": rid,
            "finish_reason": {"type": "length", "length": 1},
            "prompt_tokens": prompt_len,
            "completion_tokens": 1,
            "cached_tokens": 0,
        });
        if return_logprob {
            let (tokens, tops) = input_logprobs();
            meta["input_token_logprobs"] = json!(tokens);
            if top_k > 0 {
                meta["input_top_logprobs"] = json!(tops);
            }
        }
        if experts {
            meta["routed_experts"] = json!(STANDARD.encode(truth.rows(0, prompt_len)));
        }
        return vec![json!({"output_ids": [truth.token(0).0], "meta_info": meta})];
    }

    // Generate until the budget or the truth's EOS.
    let remaining = truth.output.len();
    let (generated, stop) = if truth.stop && remaining <= budget {
        (remaining, true)
    } else {
        (budget, false)
    };
    let end = generated;
    let mut ends: Vec<usize> = truth
        .cuts
        .iter()
        .copied()
        .filter(|&cut| cut > 0 && cut < end)
        .collect();
    ends.push(end);

    let mut chunks = Vec::with_capacity(ends.len());
    let mut from = 0;
    let (prompt_logprobs, prompt_tops) = if return_logprob && role == Role::Aggregated {
        input_logprobs()
    } else {
        (Vec::new(), Vec::new())
    };
    for to in ends {
        let last = to == end;
        let tokens: Vec<(u32, f64)> = (from..to).map(|p| truth.token(p)).collect();
        let mut meta = json!({
            "id": rid,
            "finish_reason": Value::Null,
            "prompt_tokens": prompt_len,
            "completion_tokens": to,
            "cached_tokens": truth.cached_tokens,
            "weight_version": "default",
        });
        if last {
            meta["finish_reason"] = if stop {
                json!({"type": "stop", "matched": tokens.last().map_or(0, |t| t.0)})
            } else {
                json!({"type": "length", "length": generated})
            };
        }
        if return_logprob {
            meta["input_token_logprobs"] = json!(prompt_logprobs);
            meta["output_token_logprobs_length"] = json!(to);
            meta["output_token_logprobs"] = Value::Array(
                tokens
                    .iter()
                    .map(|(id, lp)| json!([lp, id, null]))
                    .collect(),
            );
            if top_k > 0 {
                meta["input_top_logprobs"] = json!(prompt_tops);
                meta["output_top_logprobs"] = Value::Array(
                    tokens
                        .iter()
                        .map(|&(id, lp)| top_logprobs(id, lp, top_k))
                        .collect(),
                );
            }
        }
        if last && experts {
            // One row per position that produced a token: the prompt and every
            // output token but the last.
            let rows = prompt_len + generated - 1;
            let mut bytes = truth.rows(0, rows);
            if role == Role::Decode {
                // Decode never ran its prompt: those rows are stale.
                bytes[..prompt_len.min(rows) * truth.row_bytes].fill(0xFF);
            }
            meta["routed_experts"] = json!(STANDARD.encode(bytes));
        }
        chunks.push(json!({
            "output_ids": tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
            "meta_info": meta,
        }));
        from = to;
    }
    chunks
}
