// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Weak};

use dynamo_runtime::engine::{AsyncEngineContext, EngineContextGuard};
use tokio_util::task::AbortOnDropHandle;

pub(super) fn arm_for(
    prefill_supports_cancellation: bool,
    decode_supports_cancellation: bool,
    client: Arc<dyn AsyncEngineContext>,
    prefill: Weak<dyn AsyncEngineContext>,
) -> Option<EngineContextGuard> {
    (prefill_supports_cancellation && decode_supports_cancellation).then(|| {
        let task = tokio::spawn(async move {
            client.stopped().await;
            if client.is_killed() {
                if let Some(prefill) = prefill.upgrade() {
                    prefill.kill();
                }
                return;
            }
            if let Some(prefill) = prefill.upgrade() {
                prefill.stop();
            } else {
                return;
            }

            client.killed().await;
            if let Some(prefill) = prefill.upgrade() {
                prefill.kill();
            }
        });
        Arc::new(AbortOnDropHandle::new(task)) as EngineContextGuard
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};

    #[tokio::test]
    async fn propagates_cancellation_observed_before_arming() {
        let parent = Context::new(()).context();
        parent.stop_generating();
        let child = Context::new(()).context();

        let _guard = arm_for(true, true, parent, Arc::downgrade(&child)).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(1), child.stopped())
            .await
            .expect("prefill context missed an already-cancelled client");
    }

    #[tokio::test]
    async fn propagates_kill_after_arming() {
        let parent = Context::new(()).context();
        let child = Context::new(()).context();
        let _guard = arm_for(true, true, parent.clone(), Arc::downgrade(&child)).unwrap();

        parent.kill();

        tokio::time::timeout(std::time::Duration::from_secs(1), child.killed())
            .await
            .expect("prefill context did not observe client kill");
    }

    #[tokio::test]
    async fn propagates_kill_after_stop() {
        let parent = Context::new(()).context();
        let child = Context::new(()).context();
        let _guard = arm_for(true, true, parent.clone(), Arc::downgrade(&child)).unwrap();

        parent.stop_generating();
        tokio::time::timeout(std::time::Duration::from_secs(1), child.stopped())
            .await
            .expect("prefill context did not observe client stop");

        parent.kill();
        tokio::time::timeout(std::time::Duration::from_secs(1), child.killed())
            .await
            .expect("prefill context did not observe later client kill");
    }

    #[tokio::test]
    async fn requires_both_worker_sets_to_advertise_support() {
        for (prefill_supports, decode_supports) in [(false, true), (true, false)] {
            let parent = Context::new(()).context();
            let child = Context::new(()).context();

            let link = arm_for(
                prefill_supports,
                decode_supports,
                parent.clone(),
                Arc::downgrade(&child),
            );
            assert!(link.is_none());

            parent.stop_generating();
            tokio::task::yield_now().await;
            assert!(
                !child.is_stopped(),
                "cancellation reached a worker set without paired support"
            );
        }
    }

    #[tokio::test]
    async fn dropping_guard_releases_contexts_without_cancelling_prefill() {
        let parent_context = Context::new(());
        let parent = parent_context.context();
        let parent_weak = Arc::downgrade(&parent);
        let child = Context::new(()).context();
        let guard = arm_for(true, true, parent.clone(), Arc::downgrade(&child)).unwrap();

        drop(guard);
        drop(parent_context);
        drop(parent);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while parent_weak.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted watcher retained the client context");

        assert!(!child.is_stopped());
    }
}
