// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified parser family names a model deployment card may select through
//! `tool_call_parser` and `reasoning_parser`, the Dynamo aliases for them, and the rule
//! that pairs the two selectors into one family. The parsers themselves live in
//! `dynamo-parsers-v2`; routing a request to one stays with the protocol layer.

use std::sync::LazyLock;

/// The `dynamo-parsers-v2` unified family that serves Qwen3.
///
/// `REGISTERED_UNIFIED_FAMILIES` accepts both `qwen3` and the `qwen3_coder` alias for
/// the same XML grammar; `qwen3` is the canonical registry name and the one the
/// conformance corpus uses, so it is what this module passes and logs.
pub const QWEN3_UNIFIED_FAMILY: &str = "qwen3";
pub const KIMI_K2_UNIFIED_FAMILY: &str = "kimi_k2";
pub const KIMI_K3_UNIFIED_FAMILY: &str = "kimi_k3";
pub const DEEPSEEK_V41_UNIFIED_FAMILY: &str = "deepseek_v41";
pub const DEEPSEEK_V4_UNIFIED_FAMILY: &str = "deepseek_v4";
pub const GEMMA4_UNIFIED_FAMILY: &str = "gemma4";
pub const GLM47_UNIFIED_FAMILY: &str = "glm47";
pub const MUSE_UNIFIED_FAMILY: &str = "muse_glimmer";
/// The `dynamo-parsers-v2` unified family that serves Beam. Dynamo has no v1 Beam
/// parser, so the `beam` + `beam` pair routes to the unified parser without
/// `DYN_PARSER_VERSION=2`, like DeepSeek V4.1.
pub const BEAM_UNIFIED_FAMILY: &str = "beam";
pub const MUSE_SPARK_UNIFIED_FAMILY: &str = "muse_spark";
/// Nemotron 3 (Ultra, Super, 3.5 Super): `nemotron_v3` forced reasoning over the
/// Qwen3-Coder XML tool grammar. Model cards pair `--dyn-tool-call-parser qwen3_coder`
/// with `--dyn-reasoning-parser nemotron_v3`.
pub const NEMOTRON_V3_UNIFIED_FAMILY: &str = "nemotron_v3";
/// The `dynamo-parsers-v2` unified family for Mistral v11+ (`[TOOL_CALLS]NAME[ARGS]{json}`).
/// The `mistral` parser name maps to it through [`DYNAMO_FAMILY_ALIASES`], and the unified
/// parser rejects pre-v11 tokenizers at model setup.
pub const MISTRAL_UNIFIED_FAMILY: &str = "mistral_4";

/// A family that borrows another family's tool grammar names the owner through its
/// reasoning half: `qwen3_coder` + `nemotron_v3` is Nemotron, not Qwen3, because the
/// forced-reasoning default lives in the reasoning parser and the XML grammar is shared.
fn shared_grammar_family(tool_family: &str, reasoning_family: &str) -> Option<&'static str> {
    match (tool_family, reasoning_family) {
        (QWEN3_UNIFIED_FAMILY, NEMOTRON_V3_UNIFIED_FAMILY) => Some(NEMOTRON_V3_UNIFIED_FAMILY),
        _ => None,
    }
}

const DYNAMO_FAMILY_ALIASES: &[(&str, &str)] = &[
    ("deepseek-v4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("deepseekv4", DEEPSEEK_V4_UNIFIED_FAMILY),
    ("muse", MUSE_UNIFIED_FAMILY),
    ("gemma-4", GEMMA4_UNIFIED_FAMILY),
    ("glm45", GLM47_UNIFIED_FAMILY),
    ("kimi_k25", KIMI_K2_UNIFIED_FAMILY),
    ("mistral", MISTRAL_UNIFIED_FAMILY),
];

/// Canonical family names and aliases share one route for registration, stream, and batch.
pub fn canonical_family(parser: &str) -> Option<&'static str> {
    let parser = DYNAMO_FAMILY_ALIASES
        .iter()
        .find_map(|(alias, family)| (*alias == parser).then_some(*family))
        .unwrap_or(parser);
    dynamo_parsers_v2::canonical_unified_family(parser)
}

pub static FAMILY_NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut names = dynamo_parsers_v2::REGISTERED_UNIFIED_FAMILIES.to_vec();
    names.extend(DYNAMO_FAMILY_ALIASES.iter().map(|(alias, _)| *alias));
    names.sort_unstable();
    names.dedup();
    names
});

/// A configured half may select a unified family, but two configured halves must agree.
pub fn configured_family(
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> Option<&'static str> {
    match (tool_call_parser, reasoning_parser) {
        (Some(tool), Some(reasoning)) => {
            let tool_family = canonical_family(tool)?;
            let reasoning_family = canonical_family(reasoning)?;
            if tool_family == reasoning_family {
                return Some(tool_family);
            }
            shared_grammar_family(tool_family, reasoning_family)
        }
        (Some(parser), None) | (None, Some(parser)) => canonical_family(parser),
        (None, None) => None,
    }
}

/// Every selectable unified family name, aliases included.
pub fn unified_family_names() -> &'static [&'static str] {
    &FAMILY_NAMES
}
