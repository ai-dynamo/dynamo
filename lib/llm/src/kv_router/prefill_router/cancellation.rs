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

use crate::local_model::runtime_config::PrefillCancelUntil;

/// Link client cancellation to a prefill request, if the worker allows it.
///
/// Returns `None` for a worker whose window is [`PrefillCancelUntil::Never`],
/// which includes every worker that declares nothing. That case must not be
/// expressed by linking and revoking later: the link fires the moment it
/// exists, so a client that disconnects during prefill would already have
/// cancelled a worker that never opted in. The policy therefore has to be known
/// before anything is linked, which means after worker selection.
pub(super) fn arm_for(
    policy: PrefillCancelUntil,
    client: Arc<dyn AsyncEngineContext>,
    prefill: Arc<dyn AsyncEngineContext>,
) -> Option<Arc<PrefillCancelLink>> {
    (policy != PrefillCancelUntil::Never).then(|| Arc::new(PrefillCancelLink::new(client, prefill)))
}

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
    ///
    /// Takes `&self` because the link is shared: on the bootstrap path a drain
    /// task holds it so cancellation still reaches the worker while the stream
    /// is consumed, and revoking has to work from the routing side regardless
    /// of who else is holding a reference.
    pub(super) fn revoke(&self) {
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
    async fn an_undeclared_worker_is_never_linked() {
        // A worker that declares nothing is Never, and Never has to mean "no
        // link at all" rather than "link, then revoke": the link propagates as
        // soon as it exists, so revoking after the handoff would be far too
        // late for a client that disconnected during prefill. Getting this
        // wrong silently makes every legacy worker cancellable.
        let client = Context::new(()).context();
        let prefill = Context::new(()).context();

        let link = arm_for(PrefillCancelUntil::Never, client.clone(), prefill.clone());
        assert!(link.is_none(), "a Never worker must not be linked");

        client.stop_generating();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(
            !prefill.is_stopped(),
            "an undeclared worker was cancelled during prefill"
        );
    }

    #[tokio::test]
    async fn a_declared_window_is_linked() {
        for policy in [PrefillCancelUntil::Anytime, PrefillCancelUntil::PreHandoff] {
            let client = Context::new(()).context();
            let prefill = Context::new(()).context();

            let _link = arm_for(policy, client.clone(), prefill.clone())
                .unwrap_or_else(|| panic!("{policy:?} declares a window and must be linked"));

            client.stop_generating();
            tokio::time::timeout(std::time::Duration::from_secs(1), prefill.stopped())
                .await
                .unwrap_or_else(|_| panic!("{policy:?} did not propagate cancellation"));
        }
    }

    #[tokio::test]
    async fn revoked_link_leaves_prefill_running() {
        // Past the handoff commitment a PreHandoff worker must be left alone:
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
