// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use crate::common::protocols::MockEngineArgs;

/// Compute the KV transfer delay duration for a given number of input tokens.
///
/// Returns `None` if KV transfer simulation is disabled (bandwidth is 0 or not configured).
pub fn compute_kv_transfer_delay(
    args: &MockEngineArgs,
    num_input_tokens: usize,
) -> Option<Duration> {
    match (args.kv_transfer_bandwidth, args.kv_bytes_per_token) {
        (Some(bw), Some(bpt)) if bw > 0.0 => {
            let kv_bytes = num_input_tokens as f64 * bpt as f64;
            let delay = Duration::from_secs_f64(kv_bytes / (bw * 1e9));
            tracing::debug!(
                num_input_tokens,
                kv_bytes,
                bandwidth_gb_s = bw,
                delay_ms = format!("{:.2}", delay.as_secs_f64() * 1000.0),
                "KV transfer delay for prefill"
            );
            Some(delay)
        }
        _ => None,
    }
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
