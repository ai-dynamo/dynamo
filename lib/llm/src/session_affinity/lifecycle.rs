// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    pin::Pin,
    sync::{Arc, Weak, atomic::Ordering},
    task::{Context, Poll},
};

use dynamo_runtime::pipeline::{Error, ManyOut, ResponseStream};
use futures::Stream;
use tokio::{sync::Notify, time::Instant};

use super::{
    AffinityTarget, LlmResponse,
    coordinator::{
        AffinityCoordinatorInner, AffinityEntry, invalid_argument, validate_bound_target,
    },
    state::AffinityRevision,
};

pub(super) trait VacantEntryExt {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
    ) -> AffinityInitialization;
}

impl<'a> VacantEntryExt for dashmap::mapref::entry::VacantEntry<'a, String, AffinityEntry> {
    fn insert_initializing(
        self,
        inner: &Arc<AffinityCoordinatorInner>,
        session_id: String,
        requested_target: Option<AffinityTarget>,
    ) -> AffinityInitialization {
        let revision = inner.next_revision();
        let generation = inner.next_generation();
        let notify = Arc::new(Notify::new());
        self.insert(AffinityEntry::Initializing {
            revision,
            generation,
            notify: notify.clone(),
            pending_replica: None,
        });
        AffinityInitialization {
            coordinator: Arc::downgrade(inner),
            session_id,
            revision,
            generation,
            notify,
            requested_target,
            active: true,
        }
    }
}

pub(crate) enum AffinityAcquire {
    Initialize(AffinityInitialization),
    Bound {
        target: AffinityTarget,
        lease: AffinityLease,
    },
}

impl AffinityAcquire {
    pub(crate) fn target(&self) -> Option<AffinityTarget> {
        match self {
            Self::Initialize(_) => None,
            Self::Bound { target, .. } => Some(*target),
        }
    }

    pub(crate) fn into_stream(
        self,
        selected_target: AffinityTarget,
        stream: ManyOut<LlmResponse>,
    ) -> Result<ManyOut<LlmResponse>, Error> {
        match self {
            Self::Initialize(initialization) => {
                let lease = initialization.commit(selected_target)?;
                lease.publish_current();
                Ok(lease.into_stream(stream))
            }
            Self::Bound { target, mut lease } => {
                if target == selected_target || lease.rebind(selected_target) {
                    lease.publish_current();
                }
                Ok(lease.into_stream(stream))
            }
        }
    }

    pub(crate) fn invalidate_selected(self, selected_target: AffinityTarget) {
        if let Self::Bound {
            target, mut lease, ..
        } = self
            && target == selected_target
        {
            lease.invalidate(target);
        }
    }

    pub(crate) fn invalidate(self) {
        if let Self::Bound {
            target, mut lease, ..
        } = self
        {
            lease.invalidate(target);
        }
    }
}

pub(crate) struct AffinityInitialization {
    pub(super) coordinator: Weak<AffinityCoordinatorInner>,
    pub(super) session_id: String,
    pub(super) revision: AffinityRevision,
    pub(super) generation: u64,
    pub(super) notify: Arc<Notify>,
    pub(super) requested_target: Option<AffinityTarget>,
    pub(super) active: bool,
}

impl AffinityInitialization {
    pub(crate) fn commit(mut self, target: AffinityTarget) -> Result<AffinityLease, Error> {
        validate_bound_target(&self.session_id, target, self.requested_target)?;
        let Some(inner) = self.coordinator.upgrade() else {
            return Err(anyhow::anyhow!("session affinity coordinator dropped"));
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            return Err(invalid_argument(
                "session affinity initialization was cancelled",
            ));
        };
        let AffinityEntry::Initializing {
            revision,
            generation,
            ..
        } = entry.value()
        else {
            return Err(invalid_argument("session affinity initialization changed"));
        };
        if *revision != self.revision || *generation != self.generation {
            return Err(invalid_argument("session affinity initialization changed"));
        }
        let local_revision = inner.next_revision();
        let now = Instant::now();
        *entry = AffinityEntry::Bound {
            target,
            revision: local_revision,
            generation: self.generation,
            active_leases: 1,
            idle_deadline: now + inner.ttl,
        };
        drop(entry);
        self.active = false;
        self.notify.notify_waiters();
        Ok(AffinityLease {
            coordinator: Arc::downgrade(&inner),
            session_id: self.session_id.clone(),
            revision: local_revision,
            generation: self.generation,
            active: true,
        })
    }
}

impl Drop for AffinityInitialization {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let mut retained = false;
        if let Some(mut entry) = inner.entries.get_mut(&self.session_id)
            && let AffinityEntry::Initializing {
                revision,
                generation,
                pending_replica,
                ..
            } = entry.value_mut()
            && *revision == self.revision
            && *generation == self.generation
            && let Some(binding) = pending_replica.take()
        {
            *entry = AffinityEntry::Bound {
                target: binding.target,
                revision: binding.revision,
                generation: self.generation,
                active_leases: 0,
                idle_deadline: Instant::now() + inner.ttl,
            };
            retained = true;
        }
        if !retained {
            let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
                matches!(
                    entry,
                    AffinityEntry::Initializing {
                        revision,
                        generation,
                        ..
                    } if *revision == self.revision && *generation == self.generation
                )
            });
            if removed.is_some() {
                inner.entry_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
        self.notify.notify_waiters();
    }
}

pub(crate) struct AffinityLease {
    pub(super) coordinator: Weak<AffinityCoordinatorInner>,
    pub(super) session_id: String,
    pub(super) revision: AffinityRevision,
    pub(super) generation: u64,
    pub(super) active: bool,
}

impl AffinityLease {
    fn publish_current(&self) {
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let update = {
            let Some(entry) = inner.entries.get(&self.session_id) else {
                return;
            };
            let AffinityEntry::Bound {
                target,
                revision,
                generation,
                ..
            } = entry.value()
            else {
                return;
            };
            (*generation == self.generation).then_some((*target, *revision))
        };
        if let Some((target, revision)) = update {
            inner.publish_replica_update(&self.session_id, target, revision);
        }
    }

    fn rebind(&mut self, target: AffinityTarget) -> bool {
        if !self.active {
            return false;
        }
        let Some(inner) = self.coordinator.upgrade() else {
            self.active = false;
            return false;
        };
        let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
            self.active = false;
            return false;
        };
        let AffinityEntry::Bound {
            target: existing_target,
            revision: existing_revision,
            generation,
            ..
        } = entry.value_mut()
        else {
            self.active = false;
            return false;
        };
        if *generation != self.generation {
            self.active = false;
            return false;
        }
        if *existing_revision != self.revision {
            return false;
        }
        let revision = inner.next_revision();
        *existing_target = target;
        *existing_revision = revision;
        self.revision = revision;
        true
    }

    pub(crate) fn into_stream(self, stream: ManyOut<LlmResponse>) -> ManyOut<LlmResponse> {
        let context = stream.context();
        ResponseStream::new(
            Box::pin(AffinityTrackedStream {
                stream,
                lease: Some(self),
            }),
            context,
        )
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let Some(inner) = self.coordinator.upgrade() else {
            return;
        };
        let (target, revision) = {
            let Some(mut entry) = inner.entries.get_mut(&self.session_id) else {
                return;
            };
            let AffinityEntry::Bound {
                target,
                revision,
                generation,
                active_leases,
                idle_deadline,
                ..
            } = entry.value_mut()
            else {
                return;
            };
            if *generation != self.generation || *active_leases == 0 {
                return;
            }
            *active_leases -= 1;
            *idle_deadline = Instant::now() + inner.ttl;
            (*target, *revision)
        };
        inner.publish_replica_update(&self.session_id, target, revision);
    }

    fn invalidate(&mut self, expected_target: AffinityTarget) {
        if !self.active {
            return;
        }
        let Some(inner) = self.coordinator.upgrade() else {
            self.active = false;
            return;
        };
        let removed = inner.entries.remove_if(&self.session_id, |_, entry| {
            matches!(
                entry,
                AffinityEntry::Bound {
                    target,
                    revision,
                    generation,
                    ..
                } if *target == expected_target
                    && *revision == self.revision
                    && *generation == self.generation
            )
        });
        if removed.is_some() {
            self.active = false;
            inner.entry_count.fetch_sub(1, Ordering::Relaxed);
        } else {
            self.release();
        }
    }
}

impl Drop for AffinityLease {
    fn drop(&mut self) {
        self.release();
    }
}

struct AffinityTrackedStream {
    stream: ManyOut<LlmResponse>,
    lease: Option<AffinityLease>,
}

impl Stream for AffinityTrackedStream {
    type Item = LlmResponse;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(None) => {
                drop(self.lease.take());
                Poll::Ready(None)
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            poll => poll,
        }
    }
}
