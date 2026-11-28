// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::process::ExitCode;

use anyhow::{Result, bail, ensure};
use dynamo_conformance_fixtures_v2::{RunOptions, RunReport, run_unified_family};
use dynamo_parsers_v2::{builtin_unified_families, canonical_unified_family};

const USAGE: &str = "Usage: dynamo-conformance <all|FAMILY|list-v2-parsers> [--fail-on-unsupported]

Run every active authored case for each selected Unified v2 family.
Examples: all, qwen3, glm (alias for glm47), qwen3_coder (alias for qwen3).
list-v2-parsers lists canonical Unified families and their registry aliases.
Exit: 0 = pass/list/help, 1 = conformance/coverage failure, 2 = invalid arguments.
Unsupported cases are a separate verdict; --fail-on-unsupported makes them blocking.
This command does not test legacy tool-only or reasoning-only parsers.";

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    List,
    Run {
        families: BTreeSet<&'static str>,
        fail_on_unsupported: bool,
    },
}

fn families() -> BTreeSet<&'static str> {
    builtin_unified_families()
        .iter()
        .filter_map(|family| canonical_unified_family(family))
        .collect()
}

fn parse(args: &[String]) -> Result<Command> {
    let Some(selector) = args.first() else {
        bail!("a family or command is required");
    };
    let policy = args.get(1).map(String::as_str);
    ensure!(
        args.len() == 1
            || (args.len() == 2
                && matches!(
                    policy,
                    Some("--allow-unsupported" | "--fail-on-unsupported")
                )),
        "unexpected arguments"
    );
    match selector.as_str() {
        "--help" | "-h" | "list-v2-parsers" => {
            ensure!(args.len() == 1, "this command takes no options");
            Ok(if selector == "list-v2-parsers" {
                Command::List
            } else {
                Command::Help
            })
        }
        "all" => Ok(Command::Run {
            families: families(),
            fail_on_unsupported: policy == Some("--fail-on-unsupported"),
        }),
        family => {
            let family = if family == "glm" { "glm47" } else { family };
            let Some(canonical) = canonical_unified_family(family) else {
                bail!("unknown Unified v2 family `{family}`; use list-v2-parsers");
            };
            Ok(Command::Run {
                families: BTreeSet::from([canonical]),
                fail_on_unsupported: policy == Some("--fail-on-unsupported"),
            })
        }
    }
}

fn run_families(
    families: &BTreeSet<&str>,
    fail_on_unsupported: bool,
    mut run: impl FnMut(&str) -> Result<RunReport>,
) -> u8 {
    let mut failed = 0;
    for family in families {
        println!("{family}: running all active authored cases");
        // Each family is an independent unit: retain its error and run the rest.
        match run(family) {
            Ok(report) => {
                let status = report.exit_code(fail_on_unsupported);
                println!(
                    "{family}: selected={} executed={} passed={} unsupported={} failures={} isolation_pairs={} isolation_failures={} status={status}",
                    report.selected,
                    report.executed,
                    report.passed,
                    report.unsupported.len(),
                    report.failures.len(),
                    report.isolation_pairs,
                    report.isolation_failures.len(),
                );
                for unsupported in &report.unsupported {
                    println!(
                        "  UNSUPPORTED {}: {}",
                        unsupported.case_id, unsupported.reason
                    );
                }
                for failure in report.failures.iter().chain(&report.isolation_failures) {
                    println!(
                        "  FAIL {} [{}]: {}",
                        failure.case_id, failure.check, failure.message
                    );
                }
                failed += usize::from(status != 0);
            }
            Err(error) => {
                println!("{family}: ERROR {error:#}");
                failed += 1;
            }
        }
    }
    println!(
        "families={} passed={} failed={failed}",
        families.len(),
        families.len() - failed
    );
    u8::from(failed != 0)
}

fn main() -> ExitCode {
    match parse(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(Command::Help) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Command::List) => {
            for family in families() {
                let mut aliases: Vec<_> = builtin_unified_families()
                    .iter()
                    .copied()
                    .filter(|alias| {
                        *alias != family && canonical_unified_family(alias) == Some(family)
                    })
                    .collect();
                if family == "glm47" {
                    aliases.push("glm");
                }
                println!("{family}\taliases: {}", aliases.join(", "));
            }
            ExitCode::SUCCESS
        }
        Ok(Command::Run {
            families,
            fail_on_unsupported,
        }) => ExitCode::from(run_families(&families, fail_on_unsupported, |family| {
            run_unified_family(family, &RunOptions::default())
        })),
        Err(error) => {
            eprintln!("{error:#}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn aliases_select_the_same_corpus() -> Result<()> {
        for alias in builtin_unified_families() {
            let canonical = canonical_unified_family(alias).unwrap();
            assert_eq!(parse(&args(&[alias]))?, parse(&args(&[canonical]))?);
        }
        assert_eq!(parse(&args(&["glm"]))?, parse(&args(&["glm47"]))?);
        Ok(())
    }

    #[test]
    fn all_runs_every_canonical_family_once_and_continues_after_errors() -> Result<()> {
        let Command::Run {
            families,
            fail_on_unsupported,
        } = parse(&args(&["all"]))?
        else {
            bail!("expected run command");
        };
        assert!(!fail_on_unsupported);
        let mut seen = Vec::new();
        let status = run_families(&families, true, |family| {
            seen.push(family.to_owned());
            bail!("missing corpus")
        });
        assert_eq!(status, 1);
        assert_eq!(
            seen,
            families
                .iter()
                .map(|family| (*family).to_owned())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            families.len(),
            builtin_unified_families()
                .iter()
                .filter(|family| canonical_unified_family(family) == Some(**family))
                .count()
        );
        Ok(())
    }

    #[test]
    fn invalid_arguments_fail_before_running() {
        for input in [
            &[][..],
            &["qwen33"],
            &["all", "qwen3"],
            &["qwen3", "--typo"],
            &["list-v2-parsers", "--allow-unsupported"],
        ] {
            assert!(parse(&args(input)).is_err(), "{input:?}");
        }
    }

    #[test]
    fn allowing_unsupported_does_not_hide_other_failures() -> Result<()> {
        let Command::Run {
            fail_on_unsupported,
            ..
        } = parse(&args(&["qwen3", "--allow-unsupported"]))?
        else {
            bail!("expected run command");
        };
        assert!(!fail_on_unsupported);
        let mut report = RunReport {
            family: "qwen3".to_owned(),
            selected: 1,
            executed: 0,
            passed: 0,
            unsupported: vec![dynamo_conformance_fixtures_v2::CaseUnsupported {
                case_id: "generation".to_owned(),
                reason: "structural-tag".to_owned(),
            }],
            failures: Vec::new(),
            isolation_pairs: 0,
            isolation_failures: Vec::new(),
        };
        let families = BTreeSet::from(["qwen3"]);
        assert_eq!(run_families(&families, true, |_| Ok(report.clone())), 1);
        assert_eq!(run_families(&families, false, |_| Ok(report.clone())), 0);
        report
            .failures
            .push(dynamo_conformance_fixtures_v2::CaseFailure {
                case_id: "tool_call".to_owned(),
                check: "golden".to_owned(),
                message: "mismatch".to_owned(),
            });
        assert_eq!(run_families(&families, false, |_| Ok(report.clone())), 1);
        assert_eq!(
            run_families(&families, false, |_| anyhow::bail!("unreadable corpus")),
            1
        );
        Ok(())
    }
}
