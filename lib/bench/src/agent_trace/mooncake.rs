// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, anyhow, bail};
use dynamo_data_gen::{MooncakeRow, RollingHashIdMapper};

use super::load::RequestEntry;

/// Each request becomes one independent row whose `timestamp` is its offset
/// from the earliest request. See [`super::agentic`] for the DAG-aware lowering.
pub fn build_mooncake_rows(mut requests: Vec<RequestEntry>) -> Result<(usize, Vec<MooncakeRow>)> {
    let global_start_ms = requests
        .iter()
        .map(|request| request.start_ms)
        .min()
        .ok_or_else(|| anyhow!("no request records to convert"))?;
    let trace_block_size = requests[0].replay.trace_block_size;
    for request in &requests {
        if request.replay.trace_block_size != trace_block_size {
            bail!(
                "mixed replay trace_block_size values are not supported: {} and {}",
                trace_block_size,
                request.replay.trace_block_size
            );
        }
    }

    requests.sort_by(|left, right| {
        (left.start_ms, left.end_ms, &left.request.request_id).cmp(&(
            right.start_ms,
            right.end_ms,
            &right.request.request_id,
        ))
    });

    let mut mapper = RollingHashIdMapper::new(trace_block_size);
    let mut rows = Vec::new();
    for request in requests {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        rows.push(MooncakeRow {
            session_id: None,
            input_length: Some(request.replay.input_length),
            output_length: Some(
                usize::try_from(output_length).context("output length does not fit in usize")?,
            ),
            hash_ids: Some(hash_ids),
            timestamp: Some((request.start_ms - global_start_ms) as f64),
            delay: None,
        });
    }

    Ok((trace_block_size, rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_trace::load::{AgentReplayMetrics, AgentRequestMetrics, RequestEntry};

    fn request(
        request_id: &str,
        start_ms: i64,
        end_ms: i64,
        sequence_hashes: Vec<u64>,
    ) -> RequestEntry {
        RequestEntry {
            start_ms,
            end_ms,
            agent_context: None,
            request: AgentRequestMetrics {
                request_id: request_id.to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: AgentReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
            },
        }
    }

    #[test]
    fn converter_preserves_absolute_timestamps_per_request() {
        let requests = vec![
            request("req-a", 1_000, 1_100, vec![11, 22]),
            request("req-b", 1_500, 1_600, vec![11, 33]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0.0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].timestamp, Some(500.0));
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
        assert_eq!(
            entries[0].hash_ids.as_ref().unwrap()[0],
            entries[1].hash_ids.as_ref().unwrap()[0]
        );
    }

    #[test]
    fn converter_preserves_parallel_start_times_as_independent_rows() {
        let requests = vec![
            request("req-a", 1_000, 1_500, vec![11]),
            request("req-b", 1_000, 1_700, vec![22]),
        ];

        let (_, entries) = build_mooncake_rows(requests).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].timestamp, Some(0.0));
        assert_eq!(entries[1].timestamp, Some(0.0));
        assert_eq!(entries[0].delay, None);
        assert_eq!(entries[1].delay, None);
        assert_eq!(entries[0].session_id, None);
        assert_eq!(entries[1].session_id, None);
    }
}
