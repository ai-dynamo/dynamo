// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod arrival;
mod driver;
mod trace;
mod types;

use rand::Rng;
use rand::rngs::StdRng;

pub use driver::WorkloadDriver;
pub use trace::validate_trace_files;
pub use types::{
    AgenticTrace, AgenticTurnTrace, ArrivalSpec, DelaySpec, DynamoRequestTrace, LengthSpec,
    OUTPUT_REPLAY_CONSUMER_RUNTIME_KEY, OUTPUT_REPLAY_ID_ANNOTATION_KEY, ReadyTurn,
    ReplayRequestHashes, RouterSequence, SequenceHashMode, SessionPartitionSpec, SessionTrace,
    SyntheticTraceSpec, Trace, TraceFileFormat, TurnTrace, effective_replay_key,
    output_replay_id_annotation,
};

pub(super) const SYNTHETIC_OUTPUT_SEED: u64 = 0xD37A_0A7E_5EED;

pub(super) fn planned_output_token_ids(
    authored: Option<Vec<u32>>,
    max_output_tokens: usize,
    output_rng: &mut StdRng,
) -> Vec<u32> {
    authored.unwrap_or_else(|| {
        (0..max_output_tokens)
            .map(|_| output_rng.random::<u32>())
            .collect()
    })
}

#[cfg(test)]
mod tests;
