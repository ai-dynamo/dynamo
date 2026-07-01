// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use super::{PolicyQueue, QueueRejection, QueueSnapshot};

#[derive(Default)]
pub(super) struct SessionTable {
    // ponytail: router-lifetime state; add eviction when session completion reaches scheduling.
    attained_service: HashMap<String, usize>,
}

impl<T> PolicyQueue<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_with_attained_service(
        &mut self,
        class_index: usize,
        worker_count: usize,
        snapshot: QueueSnapshot,
        arrival_offset_secs: f64,
        priority_jump: f64,
        strict_priority: u32,
        attained_service: usize,
        payload: T,
    ) -> Result<(), (QueueRejection, T)> {
        self.enqueue_with_score(
            class_index,
            worker_count,
            snapshot,
            arrival_offset_secs,
            priority_jump,
            strict_priority,
            Some(-(attained_service as f64)),
            payload,
        )
    }

    pub fn session_attained_service(&self, session_id: &str) -> usize {
        self.sessions
            .attained_service
            .get(session_id)
            .copied()
            .unwrap_or_default()
    }

    pub fn record_session_dispatch(&mut self, session_id: &str, service: usize) {
        let attained = self
            .sessions
            .attained_service
            .entry(session_id.to_string())
            .or_default();
        *attained = attained.saturating_add(service);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterQueuePolicy;
    use crate::scheduling::RouterPolicyConfig;

    fn queue() -> PolicyQueue<&'static str> {
        let profile = RouterPolicyConfig::from_yaml(
            r#"
default_policy_family: session
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: session
    policy_family: session
    cache_bucket: all
    queue_policy: session_las
    quantum: 1
"#,
        )
        .unwrap()
        .resolve_profile(None, None, RouterQueuePolicy::Fcfs);
        PolicyQueue::new(profile)
    }

    #[test]
    fn prefers_less_attained_service_within_strict_priority() {
        let mut queue = queue();
        queue.record_session_dispatch("heavy", 100);
        for (attained_service, strict_priority, payload) in
            [(100, 0, "heavy"), (0, 0, "fresh"), (1_000, 1, "high")]
        {
            queue
                .enqueue_with_attained_service(
                    0,
                    1,
                    QueueSnapshot::new(1, 0),
                    0.0,
                    0.0,
                    strict_priority,
                    attained_service,
                    payload,
                )
                .unwrap();
        }

        assert_eq!(
            queue.pop_next(|_, _, _| true).unwrap().into_payload(),
            "high"
        );
        assert_eq!(
            queue.pop_next(|_, _, _| true).unwrap().into_payload(),
            "fresh"
        );
        assert_eq!(
            queue.pop_next(|_, _, _| true).unwrap().into_payload(),
            "heavy"
        );
        assert_eq!(queue.session_attained_service("heavy"), 100);
    }
}
