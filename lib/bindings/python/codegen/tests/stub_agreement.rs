// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `.pyi` stub is handwritten, and until now nothing checked it.
//!
//! `entrypoint.rs` says so itself: "The `.pyi` stub is the leg nothing guards
//! — it must agree by name, order and default value, and no test checks it."
//! A typed caller that trusts a stale stub type-checks and then fails at
//! runtime with an unexpected keyword. That is how a deleted knob survived a
//! removal which touched config, env, serde, PyO3 and the Python CLI.
//!
//! This test lives here rather than beside the binding because the binding
//! crate is outside the workspace and does not build on every developer
//! machine. Reading two files needs neither.

const ENTRYPOINT: &str = include_str!("../../rust/llm/entrypoint.rs");
const STUB: &str = include_str!("../../src/dynamo/_core.pyi");

/// Argument names in the `KvRouterConfig` PyO3 signature.
fn pyo3_arguments() -> Vec<String> {
    // There are several signatures in this file. Take the one that configures
    // the router, identified by a keyword only it has.
    let signature = ENTRYPOINT
        .split("#[pyo3(signature = (")
        .filter_map(|rest| rest.split("))]").next())
        .find(|block| block.contains("prefill_continue_enabled"))
        .expect("the KvRouterConfig pyo3 signature");
    signature
        .split(',')
        .filter_map(|argument| {
            let name = argument.trim().trim_start_matches('*').trim();
            let name = name.split('=').next()?.trim();
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}

/// Argument names in the stub's matching `__new__`.
fn stub_arguments() -> Vec<String> {
    let block = STUB
        .split("class KvRouterConfig:")
        .nth(1)
        .and_then(|rest| rest.split("def __init__(").nth(1))
        .expect("the KvRouterConfig stub argument list");
    let mut names = Vec::new();
    for line in block.lines() {
        let line = line.trim();
        if line.starts_with(')') {
            break;
        }
        // `self` carries no keyword, and a wrapped default line has no colon.
        if let Some(name) = line.split(':').next() {
            let name = name.trim();
            if name == "self" || !line.contains(':') {
                continue;
            }
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                names.push(name.to_string());
            }
        }
    }
    names
}

#[test]
fn every_pyo3_keyword_appears_in_the_stub() {
    let stub = stub_arguments();
    for name in pyo3_arguments() {
        assert!(
            stub.contains(&name),
            "PyO3 accepts `{name}`, which _core.pyi does not declare"
        );
    }
}

#[test]
fn the_stub_advertises_no_keyword_pyo3_rejects() {
    let declared = pyo3_arguments();
    for name in stub_arguments() {
        assert!(
            declared.contains(&name),
            "_core.pyi advertises `{name}`, which PyO3 does not accept"
        );
    }
}
