// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo request-trace schema loading and lowering.

use std::path::PathBuf;

use aisimulate_core::replay::loadgen::{
    AGENTIC_MOONCAKE_SCHEMA, AGENTIC_MOONCAKE_VERSION, AgenticDependency,
    AgenticDependencyRelation, AgenticDependencyTrigger, AgenticGraphBuilder, AgenticHashIdScope,
    AgenticMooncakeHeader, AgenticMooncakeRow, AgenticSourceProvenance, AgenticTrace, MooncakeRow,
    Trace, TraceFileFormat, validate_trace_files,
};
use anyhow::{Result, bail};
use dynamo_data_gen::WekaImporter;
use dynamo_data_gen::request_trace::{
    agentic::lower_agentic_mooncake_rows,
    load::{RequestTraceMode, load_request_trace_records},
    mooncake::lower_mooncake_rows,
};

/// Request traces exported from Dynamo's serving instrumentation.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamoRequestTrace {
    Standard(Trace),
    Agentic(AgenticTrace),
}

pub fn load_weka_trace(path: &std::path::Path) -> Result<AgenticTrace> {
    let importer = WekaImporter::open(path)?;
    let header = importer.header();
    let mut builder = AgenticGraphBuilder::new(AgenticMooncakeHeader {
        schema: header.schema.clone(),
        version: header.version,
        block_size: header.block_size,
        hash_id_scope: AgenticHashIdScope::Local,
        source: AgenticSourceProvenance {
            format: header.source.format.clone(),
            digest: header.source.digest.clone(),
        },
    })?;
    importer.for_each_row(|row| builder.push(agentic_mooncake_row(row)))?;
    builder.finish()
}

impl DynamoRequestTrace {
    pub fn from_request_trace_files(
        paths: &[PathBuf],
        expected_block_size: Option<usize>,
    ) -> Result<Self> {
        validate_trace_files(TraceFileFormat::Dynamo, paths)?;

        let loaded = load_request_trace_records(paths)?;
        match loaded.mode()? {
            RequestTraceMode::Standard => {
                let mut rows = Vec::new();
                let block_size = lower_mooncake_rows(loaded.requests, |_, row| {
                    rows.push(mooncake_row(row));
                    Ok(())
                })?;
                validate_dynamo_trace_block_size(expected_block_size, block_size)?;
                Ok(Self::Standard(Trace::from_mooncake_rows(rows, block_size)?))
            }
            RequestTraceMode::Agentic => {
                let mut rows = Vec::new();
                let block_size = lower_agentic_mooncake_rows(loaded, |_, row| {
                    rows.push(agentic_mooncake_row(row));
                    Ok(())
                })?;
                validate_dynamo_trace_block_size(expected_block_size, block_size)?;
                let digest = blake3::hash(&serde_json::to_vec(&rows)?)
                    .to_hex()
                    .to_string();
                let header = AgenticMooncakeHeader {
                    schema: AGENTIC_MOONCAKE_SCHEMA.to_string(),
                    version: AGENTIC_MOONCAKE_VERSION,
                    block_size,
                    hash_id_scope: AgenticHashIdScope::Local,
                    source: AgenticSourceProvenance {
                        format: "dynamo_request_trace".to_string(),
                        digest,
                    },
                };
                Ok(Self::Agentic(AgenticTrace::from_agentic_mooncake_rows(
                    header, rows,
                )?))
            }
        }
    }
}

fn validate_dynamo_trace_block_size(expected: Option<usize>, embedded: usize) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if expected != embedded {
        bail!(
            "trace_block_size {expected} does not match embedded Dynamo request trace block size {embedded}"
        );
    }
    Ok(())
}

fn mooncake_row(row: dynamo_data_gen::MooncakeRow) -> MooncakeRow {
    MooncakeRow {
        request_id: row.request_id,
        session_id: row.session_id,
        input_length: row.input_length,
        output_length: row.output_length,
        output_token_ids: row.output_token_ids,
        hash_ids: row.hash_ids,
        timestamp: row.timestamp,
        delay: row.delay,
        priority: row.priority,
        strict_priority: row.strict_priority,
        policy_class: row.policy_class,
    }
}

fn agentic_mooncake_row(row: dynamo_data_gen::AgenticMooncakeRow) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: row.request_id,
        play_id: row.play_id,
        session_id: row.session_id,
        model: row.model,
        input_length: row.input_length,
        output_length: row.output_length,
        output_token_ids: row.output_token_ids,
        hash_ids: row.hash_ids,
        not_before_ms: row.not_before_ms,
        priority: row.priority,
        strict_priority: row.strict_priority,
        policy_class: row.policy_class,
        dependencies: row
            .dependencies
            .into_iter()
            .map(|dependency| AgenticDependency {
                request_id: dependency.request_id,
                trigger: match dependency.trigger {
                    dynamo_data_gen::AgenticDependencyTrigger::Dispatch => {
                        AgenticDependencyTrigger::Dispatch
                    }
                    dynamo_data_gen::AgenticDependencyTrigger::Completion => {
                        AgenticDependencyTrigger::Completion
                    }
                },
                delay_ms: dependency.delay_ms,
                relation: match dependency.relation {
                    dynamo_data_gen::AgenticDependencyRelation::Sequence => {
                        AgenticDependencyRelation::Sequence
                    }
                    dynamo_data_gen::AgenticDependencyRelation::Spawn => {
                        AgenticDependencyRelation::Spawn
                    }
                    dynamo_data_gen::AgenticDependencyRelation::Join => {
                        AgenticDependencyRelation::Join
                    }
                },
            })
            .collect(),
    }
}
