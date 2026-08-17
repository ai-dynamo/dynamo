// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engine::common::protocols::{KvTransferTimingMode, WorkerType as CoreWorkerType};
use crate::engine::{
    HandoffTransferTiming, TransferTimingMode, WorkerType,
    prefill_handoff_delay_ms as native_prefill_handoff_delay_ms,
};

pub fn prefill_handoff_transfer_timing(
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
    mode: KvTransferTimingMode,
) -> HandoffTransferTiming {
    HandoffTransferTiming {
        mode: match mode {
            KvTransferTimingMode::FullPrompt => TransferTimingMode::FullPrompt,
            KvTransferTimingMode::DestinationMissing => TransferTimingMode::DestinationMissing,
        },
        full_prompt_tokens: num_input_tokens,
        kv_bytes_per_token,
        bandwidth_gb_s: kv_transfer_bandwidth,
    }
}

/// Compute the modeled handoff delay after a prefill worker emits its terminal token.
///
/// NOTE: this intentionally does not model the internal prefill TTFT itself accurately, and the
/// exact prefill/decode boundary is backend dependent. For now we only care about decode-visible
/// TTFT, which is what the client observes, so modeling the delay as prefill-to-decode handoff is
/// good enough.
pub fn compute_prefill_handoff_delay_ms(
    worker_type: CoreWorkerType,
    completed: bool,
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    let delay_ms = native_prefill_handoff_delay_ms(
        match worker_type {
            CoreWorkerType::Aggregated => WorkerType::Aggregated,
            CoreWorkerType::Prefill => WorkerType::Prefill,
            CoreWorkerType::Decode => WorkerType::Decode,
        },
        completed,
        num_input_tokens,
        kv_transfer_bandwidth,
        kv_bytes_per_token,
    );
    match delay_ms {
        Some(delay_ms) => {
            tracing::debug!(
                num_input_tokens,
                bandwidth_gb_s = kv_transfer_bandwidth,
                delay_ms = format!("{delay_ms:.2}"),
                "KV handoff delay for prefill completion"
            );
            Some(delay_ms)
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_handoff_delay_only_applies_to_completed_prefill() {
        let delay_ms = compute_prefill_handoff_delay_ms(
            CoreWorkerType::Prefill,
            true,
            128,
            Some(1.0),
            Some(1_000_000),
        )
        .expect("prefill completion should produce a handoff delay");
        assert!((delay_ms - 128.0).abs() < 1e-9);

        assert!(
            compute_prefill_handoff_delay_ms(
                CoreWorkerType::Prefill,
                false,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
        assert!(
            compute_prefill_handoff_delay_ms(
                CoreWorkerType::Decode,
                true,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
    }
}
