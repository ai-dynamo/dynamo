// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Weka ingestion adapter for the AISimulate agentic graph.

use aisimulate_core::replay::loadgen::{
    AgenticDependency, AgenticDependencyRelation, AgenticDependencyTrigger, AgenticGraphBuilder,
    AgenticHashIdScope, AgenticMooncakeHeader, AgenticMooncakeRow, AgenticSourceProvenance,
    AgenticTrace,
};
use anyhow::Result;
use dynamo_data_gen::WekaImporter;

pub fn load_weka_trace(path: &std::path::Path) -> Result<AgenticTrace> {
    let importer = WekaImporter::open(path)?;
    let header = importer.header();
    let mut builder = AgenticGraphBuilder::new(AgenticMooncakeHeader {
        schema: header.schema.clone(),
        version: header.version,
        block_size: header.block_size,
        hash_id_scope: match header.hash_id_scope {
            dynamo_data_gen::AgenticHashIdScope::Local => AgenticHashIdScope::Local,
        },
        source: AgenticSourceProvenance {
            format: header.source.format.clone(),
            digest: header.source.digest.clone(),
        },
    })?;
    importer.for_each_row(|row| builder.push(agentic_mooncake_row(row)))?;
    builder.finish()
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
                    dynamo_data_gen::AgenticDependencyRelation::ReplayBarrier => {
                        AgenticDependencyRelation::ReplayBarrier
                    }
                },
            })
            .collect(),
    }
}
