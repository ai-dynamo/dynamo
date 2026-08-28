// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{MooncakeRow, RollingHashIdMapper};
use anyhow::{Context, Result, anyhow, bail};

use super::load::RequestEntry;

pub(crate) fn replay_hash_ids_and_output_tokens(
    mapper: &mut RollingHashIdMapper,
    replay: &super::load::RequestTraceReplayMetrics,
    output_length: usize,
) -> Result<(Vec<u64>, Option<Vec<u32>>)> {
    let mut input_hash_ids = mapper.ids_for_sequence_hashes(&replay.input_sequence_hashes);
    let Some(completion_sequence_hashes) = replay.completion_sequence_hashes.as_ref() else {
        return Ok((input_hash_ids, None));
    };

    let completion_hash_ids = mapper.ids_for_sequence_hashes(completion_sequence_hashes);
    let partial_input_block = !replay.input_length.is_multiple_of(replay.trace_block_size);
    let first_output_block = replay.input_length / replay.trace_block_size;
    if partial_input_block && output_length > 0 {
        input_hash_ids[first_output_block] = completion_hash_ids[first_output_block];
    }

    let output_token_ids = (0..output_length)
        .map(|offset| {
            let sequence_position = replay
                .input_length
                .checked_add(offset)
                .context("request replay output position overflow")?;
            let block_index = sequence_position / replay.trace_block_size;
            u32::try_from(completion_hash_ids[block_index])
                .context("request replay hash ID does not fit in u32")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((input_hash_ids, Some(output_token_ids)))
}

/// Streams each request through a Mooncake-compatible row into the replay builder.
///
/// This is an in-memory compatibility layer; it does not write a Mooncake trace.
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
    // Seed only authored input hashes first. The downstream Mooncake interner
    // sees these rows in the same order, so the compact IDs remain stable when
    // completion hashes are lowered into synthetic output token IDs.
    for request in &requests {
        mapper.ids_for_sequence_hashes(&request.replay.input_sequence_hashes);
    }
    for request in requests {
        let output_length = request.request.output_tokens.ok_or_else(|| {
            anyhow!(
                "request {} is missing output length",
                request.request.request_id
            )
        })?;
        let output_length =
            usize::try_from(output_length).context("output length does not fit in usize")?;
        let (hash_ids, output_token_ids) =
            replay_hash_ids_and_output_tokens(&mut mapper, &request.replay, output_length)?;
        emit(
            trace_block_size,
            MooncakeRow {
                session_id: None,
                input_length: Some(request.replay.input_length),
                output_length: Some(output_length),
                output_token_ids,
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
                ..Default::default()
            },
            replay: RequestTraceReplayMetrics {
                trace_block_size: 2,
                input_length: sequence_hashes.len() * 2,
                input_sequence_hashes: sequence_hashes,
                completion_sequence_hashes: None,
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

    #[test]
    fn lowering_makes_an_exact_generated_continuation_reusable() {
        let mut first = request("req-parent", 1_000, 1_100, vec![11]);
        first.request.output_tokens = Some(2);
        first.replay.completion_sequence_hashes = Some(vec![11, 22]);

        let mut second = request("req-child", 1_200, 1_300, vec![11, 22]);
        second.request.output_tokens = Some(1);
        second.replay.completion_sequence_hashes = Some(vec![11, 22, 33]);

        let mut rows = Vec::new();
        lower_mooncake_rows(vec![first, second], |_, row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();

        let parent_prompt = rows[0]
            .hash_ids
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|hash_id| [*hash_id as u32; 2])
            .collect::<Vec<_>>();
        let mut completed_parent = parent_prompt;
        completed_parent.extend(rows[0].output_token_ids.as_ref().unwrap());
        let child_prompt = rows[1]
            .hash_ids
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|hash_id| [*hash_id as u32; 2])
            .collect::<Vec<_>>();

        assert_eq!(completed_parent, child_prompt);
    }

    #[test]
    fn lowering_completes_a_partial_input_block_with_generated_tokens() {
        let mut first = request("req-parent", 1_000, 1_100, vec![11, 12]);
        first.replay.input_length = 3;
        first.request.output_tokens = Some(1);
        first.replay.completion_sequence_hashes = Some(vec![11, 22]);

        let mut second = request("req-child", 1_200, 1_300, vec![11, 22]);
        second.request.output_tokens = Some(1);
        second.replay.completion_sequence_hashes = Some(vec![11, 22, 33]);

        let mut rows = Vec::new();
        lower_mooncake_rows(vec![first, second], |_, row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();

        let parent_hash_ids = rows[0].hash_ids.as_ref().unwrap();
        let parent_output_ids = rows[0].output_token_ids.as_ref().unwrap();
        let child_hash_ids = rows[1].hash_ids.as_ref().unwrap();
        assert_eq!(parent_hash_ids[1], child_hash_ids[1]);
        assert_eq!(u64::from(parent_output_ids[0]), child_hash_ids[1]);
    }

    #[test]
    fn lowering_preserves_legacy_synthetic_output_behavior() {
        let mut rows = Vec::new();
        lower_mooncake_rows(vec![request("legacy", 1_000, 1_100, vec![11])], |_, row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();

        assert!(rows[0].output_token_ids.is_none());
    }
}
