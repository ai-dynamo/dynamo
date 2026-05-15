// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use crate::common::protocols::{MockEngineArgs, WorkerType};

/// Compute the modeled handoff delay after a prefill worker emits its terminal token.
///
/// NOTE: this intentionally does not model the internal prefill TTFT itself accurately, and the
/// exact prefill/decode boundary is backend dependent. For now we only care about decode-visible
/// TTFT, which is what the client observes, so modeling the delay as prefill-to-decode handoff is
/// good enough.
pub fn compute_prefill_handoff_delay_ms(
    worker_type: WorkerType,
    completed: bool,
    num_input_tokens: usize,
    kv_transfer_bandwidth: Option<f64>,
    kv_bytes_per_token: Option<usize>,
) -> Option<f64> {
    if worker_type != WorkerType::Prefill || !completed {
        return None;
    }

    match (kv_transfer_bandwidth, kv_bytes_per_token) {
        (Some(bw), Some(bpt)) if bw > 0.0 => {
            let kv_bytes = num_input_tokens as f64 * bpt as f64;
            let delay_ms = kv_bytes / (bw * 1e9) * 1000.0;
            tracing::debug!(
                num_input_tokens,
                kv_bytes,
                bandwidth_gb_s = bw,
                delay_ms = format!("{delay_ms:.2}"),
                "KV handoff delay for prefill completion"
            );
            Some(delay_ms)
        }
        _ => None,
    }
}

/// Compute the KV transfer delay duration for a given number of input tokens.
///
/// Returns `None` if KV transfer simulation is disabled (bandwidth is 0 or not configured).
pub fn compute_kv_transfer_delay(
    args: &MockEngineArgs,
    num_input_tokens: usize,
) -> Option<Duration> {
    compute_prefill_handoff_delay_ms(
        args.worker_type,
        true,
        num_input_tokens,
        args.kv_transfer_bandwidth,
        args.kv_bytes_per_token,
    )
    .map(|delay_ms| Duration::from_secs_f64(delay_ms / 1000.0))
}

/// Sleep for the specified duration using timerfd on Linux for precision.
pub async fn sleep_precise(duration: Duration) {
    sleep_until_precise(Instant::now() + duration).await;
}

/// Sleep until the specified deadline using timerfd on Linux for precision.
///
/// Unlike `sleep_precise`, this accounts for time already elapsed since the
/// deadline's reference point, making it suitable for simulation loops where
/// computation time should be subtracted from the sleep.
pub async fn sleep_until_precise(deadline: Instant) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(delay) = tokio_timerfd::Delay::new(deadline) {
            let _ = delay.await;
        } else {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_handoff_delay_only_applies_to_completed_prefill() {
        let delay_ms = compute_prefill_handoff_delay_ms(
            WorkerType::Prefill,
            true,
            128,
            Some(1.0),
            Some(1_000_000),
        )
        .expect("prefill completion should produce a handoff delay");
        assert!((delay_ms - 128.0).abs() < 1e-9);

        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Prefill,
                false,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
        assert!(
            compute_prefill_handoff_delay_ms(
                WorkerType::Decode,
                true,
                128,
                Some(1.0),
                Some(1_000_000),
            )
            .is_none()
        );
    }
}
