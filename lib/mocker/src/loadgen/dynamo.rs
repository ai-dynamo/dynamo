// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo request-trace schema loading and lowering.

use std::path::PathBuf;

use aisimulate_core::replay::loadgen::{
    AgenticMooncakeRow, AgenticToolEvent, AgenticTrace, MooncakeRow, Trace, TraceFileFormat,
    validate_trace_files,
};
use anyhow::{Result, bail};
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
                Ok(Self::Agentic(AgenticTrace::from_agentic_mooncake_rows(
                    rows, block_size,
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
        request_kind: row.request_kind,
        wait_for: row.wait_for,
        branches: row.branches,
        prefix_reset: row.prefix_reset,
        tool_wait_ms: row.tool_wait_ms,
        tool_events: row
            .tool_events
            .into_iter()
            .map(|event| AgenticToolEvent {
                tool_call_id: event.tool_call_id,
                tool_class: event.tool_class,
                started_at_unix_ms: event.started_at_unix_ms,
                ended_at_unix_ms: event.ended_at_unix_ms,
                duration_ms: event.duration_ms,
                status: event.status,
                output_bytes: event.output_bytes,
                output_tokens: event.output_tokens,
                error_type: event.error_type,
            })
            .collect(),
    }
}
