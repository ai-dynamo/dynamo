// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identifying the server and mapping it onto Dynamo's registration.

use clap::Parser;

use super::*;
use crate::args::Args;
use crate::pb;

/// Applies `mutate` to a healthy server in `mode` and expects a rejection
/// whose message mentions every one of `expect`.
fn assert_rejected(
    mode: DisaggregationMode,
    mutate: impl FnOnce(&mut pb::ServerInfo, &mut pb::ModelInfo),
    expect: &[&str],
) {
    let mut server = fake::server_info(mode);
    let mut model = fake::model_info();
    mutate(&mut server, &mut model);
    let error =
        DiscoveredEngine::check(server, model, mode).expect_err("the server must be rejected");
    assert_backend(&error, BackendError::InvalidArgument);
    for needle in expect {
        assert!(
            error.to_string().contains(needle),
            "error {error} should mention {needle:?}"
        );
    }
}

fn parse(extra: &[&str]) -> Args {
    Args::try_parse_from(argv("127.0.0.1:50051", extra)).expect("arguments parse")
}

/// Any engine name is served as reported, trimmed, and names the engine in the
/// sidecar's messages; an engine that reports none gets a neutral name.
#[test]
fn any_reported_engine_name_is_accepted() {
    for (reported, named) in [(" Some Engine ", "Some Engine"), (" ", "LPX OpenEngine")] {
        let mut server = fake::server_info(AGG);
        server.engine_name = reported.to_string();
        let discovered = DiscoveredEngine::check(server, fake::model_info(), AGG)
            .unwrap_or_else(|error| panic!("{reported:?} must be accepted: {error}"));
        assert_eq!(&*discovered.engine_name, reported.trim());
        let error = discovered
            .check_served_name(Some("other"))
            .expect_err("the names differ");
        assert!(
            error
                .to_string()
                .contains(&format!("{named} reports served_model_name")),
            "{error}"
        );
    }
}

#[test]
fn identification_fails_closed() {
    assert_rejected(
        AGG,
        |server, _| server.minimum_client_revision = 2,
        &["client revision 2"],
    );
    // A decode sidecar in front of an aggregated engine would misroute.
    assert_rejected(
        DECODE,
        |server, _| server.engine_role = pb::EngineRole::Aggregated as i32,
        &["lpx-test reports engine_role AGGREGATED, but --disaggregation-mode decode needs DECODE"],
    );
    assert_rejected(
        AGG,
        |_, model| model.served_model_name = "  ".to_string(),
        &["lpx-test reports an empty served_model_name"],
    );
    assert_rejected(
        AGG,
        |_, model| model.max_context_length = Some(0),
        &["max_context_length"],
    );
}

/// `--served-model-name` only checks the name the engine reports, which is the
/// one registered. The bootstrap test owns a mismatch.
#[test]
fn a_served_model_name_must_be_the_one_the_engine_reports() {
    let identified = discovered(AGG);
    identified
        .check_served_name(Some(" llama32-1b "))
        .expect("surrounding spaces");
    // The worker registers the engine's name, not the flag's spaced one.
    let config = identified.worker_config(parse(&["--served-model-name", " llama32-1b "]));
    assert_eq!(config.served_model_name, None);

    // An alias is served too, but it is not the name the worker registers.
    let mut model = fake::model_info();
    model.served_model_aliases = vec!["llama".to_string()];
    DiscoveredEngine::check(fake::server_info(AGG), model, AGG)
        .expect("accepted")
        .check_served_name(Some("llama"))
        .expect_err("an alias is not the served name");
}

#[test]
fn registration_comes_from_what_the_engine_reports() {
    let mut server = fake::server_info(AGG);
    let capacity = pb::DeploymentCapacity {
        kv_block_size: Some(64),
        total_kv_blocks: Some(1024),
        max_running_requests: Some(3),
        max_batched_tokens: Some(4096),
        ..Default::default()
    };
    server.capacity = Some(capacity);
    let mut model = fake::model_info();
    model.served_model_aliases = vec!["llama".to_string(), String::new()];
    let config = DiscoveredEngine::check(server.clone(), model, AGG)
        .expect("accepted")
        .engine_config();

    assert_eq!(config.model, SERVED);
    assert_eq!(config.served_model_name.as_deref(), Some(SERVED));
    assert_eq!(config.model_aliases, ["llama"]);
    // WorkerSet admission compares card checksums, so per-process identity
    // would split one deployment into many.
    let rendered = format!("{config:?}");
    assert!(!rendered.contains("lpx-test-instance-7"), "{rendered}");
    assert!(!rendered.contains("1.13.1-test"), "{rendered}");
    let llm = config.llm.expect("LLM registration");
    assert_eq!(llm.context_length, Some(8192));
    assert_eq!(llm.kv_cache_block_size, Some(64));
    assert_eq!(llm.total_kv_blocks, Some(1024));
    assert_eq!(llm.max_num_seqs, Some(3));
    assert_eq!(llm.max_num_batched_tokens, Some(4096));
    assert_eq!(llm.data_parallel_size, Some(1));
    assert!(llm.bootstrap_host.is_none() && llm.bootstrap_port.is_none());

    // The router counts blocks in the card's block size; without one, the
    // engine's block count would be read in the wrong unit.
    server.capacity = Some(pb::DeploymentCapacity {
        kv_block_size: None,
        max_batched_tokens: None,
        ..capacity
    });
    let llm = DiscoveredEngine::check(server, fake::model_info(), AGG)
        .expect("accepted")
        .engine_config()
        .llm
        .expect("LLM registration");
    assert_eq!(llm.kv_cache_block_size, None);
    assert_eq!(llm.total_kv_blocks, None);
    assert_eq!(llm.max_num_batched_tokens, None);
}

#[test]
fn an_explicit_dynamo_parser_wins_over_an_advertised_one() {
    let mut model = fake::model_info();
    model.tool_call_parser = "llama3_json".to_string();
    model.reasoning_parser = " ".to_string();
    let discovered = DiscoveredEngine::check(fake::server_info(AGG), model, AGG).expect("accepted");

    let config = discovered.worker_config(parse(&[]));
    assert_eq!(config.tool_call_parser.as_deref(), Some("llama3_json"));
    assert_eq!(
        config.reasoning_parser, None,
        "an empty parser is no parser"
    );

    let config = discovered.worker_config(parse(&[
        "--dyn-tool-call-parser",
        "hermes",
        "--dyn-reasoning-parser",
        "deepseek_r1",
    ]));
    assert_eq!(config.tool_call_parser.as_deref(), Some("hermes"));
    assert_eq!(config.reasoning_parser.as_deref(), Some("deepseek_r1"));
}

/// A decode worker that registers under the operator's component instead of
/// the decode role's is invisible to the frontend's prefill router.
#[test]
fn the_role_decides_where_the_worker_registers() {
    let config = discovered(AGG).worker_config(parse(&["--component", "operator-chosen"]));
    assert_eq!(config.component, "operator-chosen");
    assert_eq!(config.disaggregation_mode, AGG);
    assert_eq!(config.model_name, MODEL_PATH);
    assert_eq!(
        config.metrics_labels,
        [("model", SERVED), ("model_name", SERVED)]
            .map(|(name, value)| (name.to_string(), value.to_string()))
    );
    assert!(!config.enable_kv_routing, "LPX publishes no KV events");
    // The router would otherwise expect this worker to answer KV recovery.
    assert!(!config.enable_local_indexer, "LPX hosts no KV indexer");

    let config = discovered(DECODE).worker_config(parse(&[
        "--component",
        "operator-chosen",
        "--disaggregation-mode",
        "decode",
    ]));
    assert_eq!(config.component, "backend");
    assert_eq!(config.disaggregation_mode, DECODE);
}
