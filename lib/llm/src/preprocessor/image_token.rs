// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve a model's image-placeholder token id (e.g. `<|image_pad|>`,
//! `<image>`, `[IMG]`) for approx MM routing.
//!
//! When the `image-token-pyo3` feature is enabled, we query HF transformers
//! through PyO3 with a defensive lookup chain — different VLM families expose
//! the placeholder in different places, so we check several known locations
//! before giving up:
//!
//!   1. `processor.tokenizer.image_token_id`   (numeric — LLaVA, Idefics, ...)
//!   2. `model.config.image_token_index`       (numeric — PaliGemma, Qwen-VL, ...)
//!   3. `model.config.image_token_id`          (numeric — alt naming)
//!   4. `processor.image_token`                (string  — Qwen3-VL, LLaVA-NeXT, ...)
//!   5. vocab probe of common literal strings
//!
//! The first hit wins. When the feature is disabled (or PyO3 fails entirely),
//! we fall back to a tiny vocab-probe list so builds without the Python dep
//! still work for the common VLM families.

use crate::tokenizers::TokenIdType;
use crate::tokenizers::traits::Tokenizer;

/// Vocab-probe list. Used both as the inner Tier 5 inside the PyO3 path and
/// as the standalone fallback when PyO3 isn't available. First match wins.
const COMMON_PLACEHOLDERS: &[&str] = &[
    "<|image_pad|>", // Qwen2-VL / Qwen3-VL
    "<image>",       // LLaVA / LLaVA-NeXT / Idefics2/3 / Mllama
    "[IMG]",         // Pixtral
    "<IMG_CONTEXT>", // InternVL
    "<|image|>",     // Some custom VLMs
];

/// Resolve the image-placeholder token id for `model_name_or_path`.
///
/// `model_name_or_path` is a string passable to `AutoProcessor.from_pretrained`
/// (HF repo id or local path). Only consulted on the PyO3 code path; the
/// no-pyo3 fallback uses the tokenizer alone.
///
/// Returns `None` when no placeholder can be resolved. Callers should treat
/// this as "MM-aware approx routing disabled for this model".
pub fn resolve_image_token_id(
    model_name_or_path: Option<&str>,
    tokenizer: &dyn Tokenizer,
) -> Option<TokenIdType> {
    #[cfg(feature = "image-token-pyo3")]
    {
        if let Some(name) = model_name_or_path {
            match resolve_via_pyo3(name) {
                Ok(Some(Resolved::Id(id))) => {
                    let id = id as TokenIdType;
                    tracing::info!(
                        image_token_id = id,
                        "[mm-approx] resolved image-placeholder token id via HF transformers (pyo3)"
                    );
                    return Some(id);
                }
                Ok(Some(Resolved::Str(s))) => {
                    if let Some(id) = tokenizer.token_to_id(&s) {
                        tracing::info!(
                            image_token = %s,
                            image_token_id = id,
                            "[mm-approx] resolved image-placeholder token via HF transformers (pyo3)"
                        );
                        return Some(id);
                    }
                    tracing::warn!(
                        image_token = %s,
                        "[mm-approx] HF transformers reported image_token but the tokenizer \
                         doesn't map it to a single id; falling back to vocab probe"
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        "[mm-approx] HF transformers has no image-token attribute on \
                         processor/tokenizer/config; falling back to vocab probe"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "[mm-approx] could not query HF transformers; falling back to vocab probe"
                    );
                }
            }
        } else {
            tracing::debug!(
                "[mm-approx] no model name available for HF transformers lookup; \
                 falling back to vocab probe"
            );
        }
    }
    #[cfg(not(feature = "image-token-pyo3"))]
    {
        let _ = model_name_or_path; // silence unused-arg warning
    }

    resolve_via_vocab_probe(tokenizer)
}

fn resolve_via_vocab_probe(tokenizer: &dyn Tokenizer) -> Option<TokenIdType> {
    for tok in COMMON_PLACEHOLDERS {
        if let Some(id) = tokenizer.token_to_id(tok) {
            tracing::info!(
                image_token = tok,
                image_token_id = id,
                "[mm-approx] resolved image-placeholder token via vocab probe"
            );
            return Some(id);
        }
    }
    tracing::warn!(
        "[mm-approx] cannot find the image token in the tokenizer; \
         MM-aware approx routing disabled for this model"
    );
    None
}

/// Result of a successful PyO3 lookup. Some HF locations expose the token as
/// a numeric id directly; others expose only the string and need a tokenizer
/// round-trip on the Rust side.
#[cfg(feature = "image-token-pyo3")]
enum Resolved {
    Id(u64),
    Str(String),
}

#[cfg(feature = "image-token-pyo3")]
fn resolve_via_pyo3(model_name_or_path: &str) -> anyhow::Result<Option<Resolved>> {
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    // The pyo3 dep is built without `auto-initialize` (multi-arch builds can't
    // ship libpython for embedding), so `Python::with_gil` panics when no
    // Python interpreter is running in the host process. That's the case for
    // standalone Rust test/bench binaries — return None and let the caller
    // fall back to the vocab-probe path. The actual deployment (`python -m
    // dynamo.frontend`) always has Python initialized by the time we get here.
    // SAFETY: `Py_IsInitialized` is a thread-safe read of a global flag.
    if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
        return Ok(None);
    }

    Python::with_gil(|py| -> PyResult<Option<Resolved>> {
        let transformers = py.import("transformers")?;
        let auto_processor = transformers.getattr("AutoProcessor")?;
        let auto_config = transformers.getattr("AutoConfig")?;

        // `trust_remote_code=True` lets us load custom processor/config
        // classes (research-lab models, custom forks). Disable by setting
        // `DYN_PYO3_TRUST_REMOTE_CODE=0`.
        let trust_remote_code = std::env::var("DYN_PYO3_TRUST_REMOTE_CODE")
            .map(|v| v != "0")
            .unwrap_or(true);

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("trust_remote_code", trust_remote_code)?;

        let processor =
            auto_processor.call_method("from_pretrained", (model_name_or_path,), Some(&kwargs))?;
        // AutoConfig is cheap (reads config.json only, no weights).
        let config =
            auto_config.call_method("from_pretrained", (model_name_or_path,), Some(&kwargs))?;
        let tokenizer = processor.getattr("tokenizer").ok();

        // Tier 1-3: numeric ids from common attribute locations.
        let int_candidates: &[(Option<&Bound<PyAny>>, &str)] = &[
            (tokenizer.as_ref(), "image_token_id"), // LLaVA, Idefics, ...
            (Some(&config), "image_token_index"),   // PaliGemma, Qwen-VL, ...
            (Some(&config), "image_token_id"),      // alt naming
        ];
        for (src, attr) in int_candidates {
            let Some(src) = src else { continue };
            if let Ok(val) = src.getattr(*attr)
                && !val.is_none()
                && let Ok(id) = val.extract::<i64>()
                && id >= 0
            {
                return Ok(Some(Resolved::Id(id as u64)));
            }
        }

        // Tier 4: string from processor (covers Qwen3-VL, LLaVA-NeXT, ...).
        if let Ok(val) = processor.getattr("image_token")
            && !val.is_none()
            && let Ok(s) = val.extract::<String>()
        {
            return Ok(Some(Resolved::Str(s)));
        }

        // Tier 5: vocab probe with common literal strings (catches custom
        // forks that expose the token only via the tokenizer vocab).
        if let Some(tok) = tokenizer.as_ref()
            && let Ok(vocab) = tok.call_method0("get_vocab")
        {
            for placeholder in COMMON_PLACEHOLDERS {
                if let Ok(true) = vocab.contains(*placeholder) {
                    return Ok(Some(Resolved::Str((*placeholder).to_string())));
                }
            }
        }

        Ok(None)
    })
    .map_err(|e| anyhow::anyhow!("pyo3: {}", e))
}

#[cfg(test)]
mod tests {
    // The vocab-probe path is exercised by the existing integration suite via
    // OpenAIPreprocessor::new_with_parts. The PyO3 path is verified end-to-end
    // by tests/serve/test_vllm.py::test_multimodal_b64_approx_routing, which
    // drives the actual launch script and looks for the resolver log line.
}
