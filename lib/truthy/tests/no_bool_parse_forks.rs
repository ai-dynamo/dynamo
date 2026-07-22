// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Grep-guard: no hand-rolled truthy/bool string parsing outside this crate.
//!
//! Dynamo once had 8+ divergent copies of `matches!(v, "1" | "true" | ...)`
//! with different accepted sets, so `SOMEFLAG=on` worked for some flags and
//! silently not others. All bool parsing of user-supplied strings
//! must go through `dynamo-truthy` (re-exported as
//! `dynamo_runtime::config::{is_truthy, is_falsey, parse_bool, parse_bool_opt,
//! env_is_truthy, env_is_falsey}`). Use strict `parse_bool` when an invalid
//! value should be an error, and `parse_bool_opt` when the call site preserves
//! its own default (tri-state: empty/unrecognized yield `None`) — do not swap
//! one for the other, they have different default/error semantics. If this
//! test flags your new code, call those helpers instead; if the match is a
//! false positive (e.g. a string that genuinely is not a boolean flag), add it
//! to `ALLOWED` with a justification.

use std::path::{Path, PathBuf};

/// Known non-fork matches, as (path suffix, line substring) pairs. A multiline
/// (file-level) match is allowed when any entry's path suffix matches the file.
const ALLOWED: &[(&str, &str)] = &[];

/// Substring patterns that indicate a hand-rolled bool parser. Matched against
/// lowercased text, so `"TRUE"` / `"False"` variants are caught too.
const FORK_PATTERNS: &[&str] = &[
    r#"eq_ignore_ascii_case("true")"#,
    r#"eq_ignore_ascii_case("false")"#,
    r#"== "true""#,
    r#"== "false""#,
    r#"!= "true""#,
    r#"!= "false""#,
    r#""true" |"#,
    r#"| "true""#,
    r#""false" |"#,
    r#"| "false""#,
];

fn text_is_fork(text: &str) -> bool {
    let text = text.to_lowercase();
    FORK_PATTERNS.iter().any(|p| text.contains(p))
        || (text.contains("matches!") && text.contains(r#""true""#))
}

/// Whole-file check with whitespace collapsed, so a parser split across lines
/// (e.g. a match arm ending in `"true"` with `| "1"` on the next line) cannot
/// slip past the per-line scan. The `matches!` heuristic is limited to a
/// proximity window here — at file scope, `matches!` and `"true"` merely
/// coexisting is not evidence of a bool parser.
fn normalized_content_is_fork(content: &str) -> bool {
    let squashed = content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if FORK_PATTERNS.iter().any(|p| squashed.contains(p)) {
        return true;
    }
    squashed.match_indices("matches!").any(|(i, _)| {
        let mut end = squashed.len().min(i + 160);
        while !squashed.is_char_boundary(end) {
            end += 1;
        }
        squashed[i..end].contains(r#""true""#)
    })
}

fn scan_dir(dir: &Path, offenders: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            // Skip build output and the canonical implementation itself.
            if name == "target" || name == "node_modules" || path.ends_with("lib/truthy") {
                continue;
            }
            scan_dir(&path, offenders);
        } else if name.ends_with(".rs") {
            scan_file(&path, offenders);
        }
    }
}

fn scan_file(path: &Path, offenders: &mut Vec<String>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let mut line_hit = false;
    for (idx, line) in content.lines().enumerate() {
        if !text_is_fork(line) {
            continue;
        }
        line_hit = true;
        let allowed = ALLOWED.iter().any(|(suffix, substr)| {
            path.to_string_lossy().ends_with(suffix) && line.contains(substr)
        });
        if !allowed {
            offenders.push(format!("{}:{}: {}", path.display(), idx + 1, line.trim()));
        }
    }
    // Backstop: catch forks formatted across multiple lines. Only report at
    // file level when no line-level hit already pinpointed the offender.
    if !line_hit && normalized_content_is_fork(&content) {
        let allowed = ALLOWED
            .iter()
            .any(|(suffix, _)| path.to_string_lossy().ends_with(suffix));
        if !allowed {
            offenders.push(format!(
                "{}: multiline or case-variant bool-parse fork (whitespace-normalized match)",
                path.display()
            ));
        }
    }
}

#[test]
fn no_bool_parse_forks_outside_canonical_crate() {
    // lib/truthy/ -> repo root
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf();
    // Only meaningful inside the repo checkout (not a published crate).
    if !repo_root.join("Cargo.toml").is_file() || !repo_root.join("lib").is_dir() {
        eprintln!("skipping: not running inside the dynamo repo");
        return;
    }

    let mut offenders = Vec::new();
    for dir in ["lib", "deploy/inference-gateway"] {
        let dir = repo_root.join(dir);
        if dir.is_dir() {
            scan_dir(&dir, &mut offenders);
        }
    }

    assert!(
        offenders.is_empty(),
        "Hand-rolled truthy/bool parsing found; use dynamo_runtime::config::{{is_truthy, \
         parse_bool, parse_bool_opt, env_is_truthy}} (or dynamo-truthy directly if the crate \
         cannot depend on dynamo-runtime):\n{}",
        offenders.join("\n")
    );
}

#[test]
fn detector_catches_case_variants() {
    assert!(text_is_fork(r#"if v == "TRUE" {"#));
    assert!(text_is_fork(r#"v.eq_ignore_ascii_case("True")"#));
    assert!(text_is_fork(r#"matches!(v, "TRUE" | "1")"#));
}

#[test]
fn detector_catches_multiline_forks() {
    let src = "match v {\n    \"true\"\n        | \"1\" => true,\n    _ => false,\n}";
    // The per-line pass alone misses this shape; the normalized pass must not.
    assert!(!src.lines().any(text_is_fork));
    assert!(normalized_content_is_fork(src));
}

#[test]
fn detector_ignores_canonical_usage() {
    for src in [
        "let enabled = is_truthy(&v);",
        "let enabled = parse_bool(&v)?;",
        "let enabled = parse_bool_opt(&v).unwrap_or(default);",
        "tracing::info!(\"flag is true\");",
    ] {
        assert!(!text_is_fork(src), "false positive on: {src}");
        assert!(!normalized_content_is_fork(src), "false positive on: {src}");
    }
}
