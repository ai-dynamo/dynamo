// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent boundary for backend admission.
//!
//! Every request plane reaches this boundary through
//! `Ingress::handle_payload_shared` immediately before backend generation.
//! This initial implementation is deliberately a direct pass-through; admission
//! policies can be added here without changing the shared ingress interface.

use std::future::Future;
use std::sync::{Arc, LazyLock};

use crate::engine::{AsyncEngineContext, Data, EngineStream};

static GATE: LazyLock<Arc<BackendAdmissionGate>> = LazyLock::new(|| Arc::new(BackendAdmissionGate));

pub(crate) fn global() -> &'static Arc<BackendAdmissionGate> {
    &GATE
}

/// Receive engine-capacity metadata through the final runtime hook.
///
/// The pass-through gate intentionally records no admission policy yet.
pub fn record_engine_capacity(_max_num_seqs: Option<u64>, _data_parallel_size: Option<u32>) {}

/// Register the gate's metric family through the final runtime hook.
///
/// The pass-through gate has no policy state or admission metrics to expose.
pub(crate) fn register_metrics(_registry: &crate::MetricsRegistry) {}

#[derive(Debug)]
pub(crate) struct BackendAdmissionGate;

impl BackendAdmissionGate {
    /// Run `generate` through the shared admission boundary.
    ///
    /// The pass-through gate preserves the engine's stream and error exactly.
    /// The context parameter is part of the stable interface for admission
    /// policies that need to observe cancellation while a request is waiting.
    pub(crate) async fn admit<R, F>(
        self: &Arc<Self>,
        _context: Option<&dyn AsyncEngineContext>,
        generate: F,
    ) -> anyhow::Result<EngineStream<R>>
    where
        R: Data,
        F: Future<Output = anyhow::Result<EngineStream<R>>>,
    {
        generate.await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::engine::{AsyncEngineContextProvider, ResponseStream};
    use futures::StreamExt;

    async fn generate(context: Arc<dyn AsyncEngineContext>) -> anyhow::Result<EngineStream<usize>> {
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([1usize, 2])),
            context,
        ))
    }

    #[tokio::test]
    async fn pass_through_preserves_the_engine_stream() {
        let context: Arc<dyn AsyncEngineContext> =
            Arc::new(crate::pipeline::context::Controller::default());

        let mut stream = global()
            .admit(None, generate(Arc::clone(&context)))
            .await
            .expect("engine stream passes through");

        assert!(Arc::ptr_eq(&stream.context(), &context));
        assert_eq!(stream.next().await, Some(1));
        assert_eq!(stream.next().await, Some(2));
        assert_eq!(stream.next().await, None);
    }

    #[tokio::test]
    async fn pass_through_preserves_the_engine_error() {
        let error = global()
            .admit(None, async {
                Err::<EngineStream<usize>, _>(anyhow::anyhow!("engine failed to start"))
            })
            .await
            .expect_err("engine error passes through");

        assert_eq!(error.to_string(), "engine failed to start");
    }
}
