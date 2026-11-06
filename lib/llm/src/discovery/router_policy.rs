// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use dynamo_kv_router::scheduling::queue::RuntimePolicyControl;
use dynamo_runtime::protocols::EndpointId;
use parking_lot::Mutex;
use serde::Serialize;
use uuid::Uuid;

use crate::worker_type::WorkerType;

/// One concrete process-local router. Rebuilding the same endpoint creates a new ID.
#[derive(Clone, Serialize)]
pub(crate) struct RouterPolicyTarget {
    pub router_id: Uuid,
    pub model: String,
    pub namespace: String,
    pub component: String,
    pub endpoint: String,
    pub worker_type: WorkerType,
}

struct Entry {
    target: RouterPolicyTarget,
    control: Weak<RuntimePolicyControl>,
}

#[derive(Default)]
pub(crate) struct RouterPolicyRegistry {
    entries: Mutex<HashMap<Uuid, Entry>>,
}

impl RouterPolicyRegistry {
    pub(crate) fn register(
        &self,
        model: String,
        endpoint: EndpointId,
        worker_type: WorkerType,
        control: &Arc<RuntimePolicyControl>,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let mut entries = self.entries.lock();
        // Weak registrations must not retain retired schedulers or accumulate across rollouts.
        entries.retain(|_, entry| entry.control.strong_count() != 0);
        entries.insert(
            id,
            Entry {
                target: RouterPolicyTarget {
                    router_id: id,
                    model,
                    namespace: endpoint.namespace,
                    component: endpoint.component,
                    endpoint: endpoint.name,
                    worker_type,
                },
                control: Arc::downgrade(control),
            },
        );
        id
    }

    pub(crate) fn get(&self, id: Uuid) -> Option<(RouterPolicyTarget, Arc<RuntimePolicyControl>)> {
        let entries = self.entries.lock();
        let entry = entries.get(&id)?;
        Some((entry.target.clone(), entry.control.upgrade()?))
    }

    pub(crate) fn list(
        &self,
        model: Option<&str>,
    ) -> Vec<(RouterPolicyTarget, Arc<RuntimePolicyControl>)> {
        let mut entries = self.entries.lock();
        entries.retain(|_, entry| entry.control.strong_count() != 0);
        let mut live: Vec<_> = entries
            .values()
            .filter_map(|entry| {
                if model.is_some_and(|model| model != entry.target.model) {
                    return None;
                }
                Some((entry.target.clone(), entry.control.upgrade()?))
            })
            .collect();
        live.sort_unstable_by_key(|(target, _)| target.router_id);
        live
    }
}
