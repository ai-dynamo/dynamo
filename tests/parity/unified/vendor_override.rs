// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A separate test process owns the registry override so native parser checks
//! cannot accidentally observe this deliberately incorrect implementation.

use std::collections::BTreeSet;

use anyhow::Result;
use dynamo_conformance_fixtures_v2::{RunOptions, run_unified_family};
use dynamo_parsers_v2::{Tool, UnifiedParser, UnifiedParserOutput};

struct VisibleTextParser;

impl UnifiedParser for VisibleTextParser {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        output.push_text(delta);
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }
}

fn visible_text_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(VisibleTextParser))
}

struct RestoreRegistry;

impl Drop for RestoreRegistry {
    fn drop(&mut self) {
        dynamo_parsers_v2::unregister_unified_parser("qwen3");
    }
}

#[test]
fn runner_uses_the_downstream_factory_and_rejects_its_wrong_output() -> Result<()> {
    dynamo_parsers_v2::register_unified_parser("qwen3", visible_text_factory);
    let _restore = RestoreRegistry;
    let report = run_unified_family(
        "qwen3",
        &RunOptions {
            case_ids: Some(BTreeSet::from(["UNIFIED.1-tool_only".to_owned()])),
            ..RunOptions::default()
        },
    )?;

    assert_eq!(report.executed, 1, "{report:#?}");
    assert!(
        report
            .failures
            .iter()
            .any(|failure| failure.check == "golden"),
        "the runner must use this crate's registry override: {report:#?}"
    );
    assert_eq!(report.exit_code(false), 1, "{report:#?}");
    Ok(())
}
