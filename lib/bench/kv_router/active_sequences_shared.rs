// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use dynamo_mocker::loadgen::Trace;
use dynamo_tokens::SequenceHash;

use dynamo_bench::kv_router_common::replay::generate_replay_artifacts;

/// A single timestamped entry in a worker's sequence trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceTraceEntry {
    Add {
        request_id: String,
        block_hashes: Vec<SequenceHash>,
        isl: usize,
        output_length: u64,
    },
    PrefillComplete {
        request_id: String,
    },
    Free {
        request_id: String,
    },
}

/// A timestamped sequence trace entry for benchmark replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceTrace {
    pub entry: SequenceTraceEntry,
    pub timestamp_us: u64,
}

/// Pre-computed metadata for a request, stored before submission so the
/// output signal can look it up by UUID.
struct RequestMetadata {
    block_hashes: Vec<SequenceHash>,
    isl: usize,
    output_length: u64,
}

/// Run requests through the mocker to produce sequence lifecycle events
/// (add / prefill_complete / free) with realistic timing.
pub async fn generate_sequence_events(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Vec<Vec<SequenceTrace>>> {
    println!("Generating sequence events...");
    let artifacts = generate_replay_artifacts(
        traces,
        num_gpu_blocks,
        block_size,
        trace_simulation_duration_ms,
    )
    .await?;
    let mut all_traces = Vec::with_capacity(artifacts.len());

    for artifact in artifacts {
        let mut metadata = artifact
            .requests
            .into_iter()
            .map(|request| {
                Ok((
                    request.uuid,
                    RequestMetadata {
                        block_hashes: request.replay_hashes.sequence_hashes,
                        isl: request.input_length,
                        output_length: u64::try_from(request.output_length)?,
                    },
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let mut entries = Vec::new();
        let mut seen = HashMap::new();

        for timed_signal in artifact.output_signals {
            let signal = timed_signal.signal;
            let request_id = signal.uuid.to_string();

            if let std::collections::hash_map::Entry::Vacant(entry) = seen.entry(signal.uuid) {
                entry.insert(());
                let meta = metadata.remove(&signal.uuid).ok_or_else(|| {
                    anyhow::anyhow!("output signal references unknown request {}", signal.uuid)
                })?;
                entries.push(SequenceTrace {
                    entry: SequenceTraceEntry::Add {
                        request_id: request_id.clone(),
                        block_hashes: meta.block_hashes,
                        isl: meta.isl,
                        output_length: meta.output_length,
                    },
                    timestamp_us: timed_signal.timestamp_us,
                });
                entries.push(SequenceTrace {
                    entry: SequenceTraceEntry::PrefillComplete {
                        request_id: request_id.clone(),
                    },
                    timestamp_us: timed_signal.timestamp_us,
                });
            }

            if signal.completed {
                entries.push(SequenceTrace {
                    entry: SequenceTraceEntry::Free { request_id },
                    timestamp_us: timed_signal.timestamp_us,
                });
            }
        }

        all_traces.push(entries);
    }

    let total_adds = all_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Add { .. }))
        .count();
    let total_frees = all_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Free { .. }))
        .count();

    println!("Add events: {}, Free events: {}", total_adds, total_frees);

    Ok(all_traces)
}
