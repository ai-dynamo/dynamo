// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Propagating client cancellation into a remote prefill request.
//!
//! How long a prefill stays cancellable is a per-worker property, because
//! engines differ in what happens to KV that was committed for a decode leg
//! which never collects it: some release it, others hold it until a transfer
//! timeout expires, which costs far more than letting the prefill finish. The
//! worker declares its window through
//! [`PrefillCancelUntil`](crate::local_model::runtime_config::PrefillCancelUntil);
//! this module owns the machinery that honours it.

use std::sync::Arc;

use dynamo_runtime::engine::AsyncEngineContext;

/// Propagates client cancellation to a remote prefill request, revocably.
///
/// `AsyncEngineContext::link_child` is permanent, but the safe window for
/// cancelling a prefill ends at different points per engine, so this has to be
/// revocable. `stopped()`/`killed()` are level-triggered, so a client that
/// disappeared while the request was being constructed still fires here rather
/// than being missed.
pub(super) struct PrefillCancelLink {
    task: tokio::task::JoinHandle<()>,
}

impl PrefillCancelLink {
    pub(super) fn new(
        parent: Arc<dyn AsyncEngineContext>,
        child: Arc<dyn AsyncEngineContext>,
    ) -> Self {
        let task = tokio::spawn(async move {
            parent.stopped().await;
            child.stop();
        });
        Self { task }
    }

    /// Stop propagating. Used once a worker's safe window has closed: past that
    /// point the KV is committed and aborting the prefill orphans it, which
    /// costs more than letting the prefill run to completion.
    pub(super) fn revoke(self) {
        self.task.abort();
    }
}

impl Drop for PrefillCancelLink {
    fn drop(&mut self) {
        // The task holds an Arc of the client context, which owns the watch
        // sender that `stopped()` waits on, so it cannot observe that sender
        // being dropped -- it would only ever exit on a real cancellation. A
        // request that completes normally is never cancelled (the HTTP layer
        // only kills the context on an unexpected close), so without this the
        // task and both contexts leak once per successful request.
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};

    #[tokio::test]
    async fn propagates_client_cancellation_to_prefill() {
        let parent = Context::new(()).context();
        let child = Context::new(()).context();
        let _link = PrefillCancelLink::new(parent.clone(), child.clone());

        parent.stop_generating();
        tokio::time::timeout(std::time::Duration::from_secs(1), child.stopped())
            .await
            .expect("prefill context did not observe client cancellation");
        assert!(child.is_stopped());
    }

    #[tokio::test]
    async fn fires_for_a_client_that_left_before_linking() {
        // stopped()/killed() are level-triggered, so a client that disconnected
        // while the prefill request was still being constructed must not be
        // missed. This is the race an atomic link_child call would lose.
        let parent = Context::new(()).context();
        parent.stop_generating();

        let child = Context::new(()).context();
        let _link = PrefillCancelLink::new(parent, child.clone());

        tokio::time::timeout(std::time::Duration::from_secs(1), child.stopped())
            .await
            .expect("prefill context did not observe an already-cancelled client");
        assert!(child.is_stopped());
    }

    #[tokio::test]
    async fn revoked_link_leaves_prefill_running() {
        // Past the handoff commitment a PreCommit worker must be left alone:
        // aborting there orphans KV that the decode leg still needs to collect.
        let parent = Context::new(()).context();
        let child = Context::new(()).context();
        PrefillCancelLink::new(parent.clone(), child.clone()).revoke();

        parent.stop_generating();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            !child.is_stopped(),
            "revoked link still cancelled the prefill request"
        );
    }

    #[tokio::test]
    async fn task_exits_when_the_request_finishes_normally() {
        // A request that completes normally is never stopped or killed: the
        // HTTP layer only kills the context on an unexpected close
        // (http/service/disconnect.rs). The propagation task must not outlive
        // the request anyway, or every successful disaggregated request leaks a
        // task plus the contexts it holds.
        let parent_ctx = Context::new(());
        let parent = parent_ctx.context();
        let child = Context::new(()).context();
        let link = PrefillCancelLink::new(parent.clone(), child.clone());

        // The request finishes: the router drops its handle to the link.
        drop(link);
        drop(parent_ctx);
        drop(parent);

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            !child.is_stopped(),
            "a normally-finished request must not cancel its prefill"
        );
        assert_eq!(
            Arc::strong_count(&child),
            1,
            "propagation task still holds the prefill context, so it never exited"
        );
    }
}
