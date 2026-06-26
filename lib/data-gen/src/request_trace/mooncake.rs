// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MooncakeRow, RollingHashIdMapper};
use anyhow::{Context, Result, anyhow, bail};

use super::load::RequestEntry;

/// Emits each request as an independent Mooncake row in replay order.
pub fn lower_mooncake_rows<F>(mut requests: Vec<RequestEntry>, mut emit: F) -> Result<usize>
where
    F: FnMut(usize, MooncakeRow) -> Result<()>,
{
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
    for request in requests {
        let hash_ids = mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        emit(
            trace_block_size,
            MooncakeRow {
                session_id: None,
                input_length: Some(request.replay.input_length),
                output_length: Some(
                    usize::try_from(output_length)
                        .context("output length does not fit in usize")?,
                ),
                hash_ids: Some(hash_ids),
                timestamp: Some((request.start_ms - global_start_ms) as f64),
                delay: None,
                ..Default::default()
            },
        )?;
    }

    Ok(trace_block_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_trace::load::{
        RequestEntry, RequestTraceReplayMetrics, RequestTraceRequestMetrics,
    };

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
            request: RequestTraceRequestMetrics {
                request_id: request_id.to_string(),
                output_tokens: Some(5),
                request_received_ms: Some(start_ms as u64),
                total_time_ms: Some((end_ms - start_ms) as f64),
                replay: None,
            },
            replay: RequestTraceReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
            },
        }
    }

    #[test]
    fn lowering_preserves_timestamp_offsets_and_parallel_requests() {
        let requests = vec![
            request("req-a", 1_000, 1_100, vec![11, 22]),
            request("req-b", 1_000, 1_700, vec![22]),
            request("req-c", 1_500, 1_600, vec![11, 33]),
        ];

        let mut entries = Vec::new();
        lower_mooncake_rows(requests, |_, row| {
            entries.push(row);
            Ok(())
        })
        .unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].timestamp, Some(0.0));
        assert_eq!(entries[1].timestamp, Some(0.0));
        assert_eq!(entries[2].timestamp, Some(500.0));
        assert!(entries.iter().all(|entry| entry.delay.is_none()));
        assert!(entries.iter().all(|entry| entry.session_id.is_none()));
        assert_eq!(
            entries[0].hash_ids.as_ref().unwrap()[0],
            entries[2].hash_ids.as_ref().unwrap()[0]
        );
    }
}
