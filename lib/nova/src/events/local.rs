// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tokio_util::task::TaskTracker;

use super::*;

/// Local event handle with ability to trigger exactly once.
#[derive(Clone)]
pub struct LocalEvent {
    inner: Arc<LocalEventInner>,
}

struct LocalEventInner {
    system: Arc<LocalEventSystem>,
    entry: Arc<EventEntry>,
    handle: EventHandle,
    triggered: AtomicBool,
}

impl LocalEvent {
    fn new(system: Arc<LocalEventSystem>, entry: Arc<EventEntry>, handle: EventHandle) -> Self {
        Self {
            inner: Arc::new(LocalEventInner {
                system,
                entry,
                handle,
                triggered: AtomicBool::new(false),
            }),
        }
    }

    pub fn handle(&self) -> EventHandle {
        self.inner.handle
    }

    pub fn trigger(&self) -> Result<()> {
        if self
            .inner
            .triggered
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            bail!("Event {} already triggered", self.inner.handle);
        }

        self.inner
            .system
            .trigger_local_entry(self.inner.entry.clone(), self.inner.handle)
    }

    pub fn poison(&self, reason: impl Into<String>) -> Result<()> {
        self.inner.system.poison_local_entry(
            self.inner.entry.clone(),
            self.inner.handle,
            Arc::new(EventPoison::new(self.inner.handle, reason)),
        )
    }
}

/// Local-only event system with reusable event entries.
pub struct LocalEventSystem {
    worker_id: u64,
    events: DashMap<EventKey, Arc<EventEntry>>,
    free_lists: ParkingMutex<VecDeque<Arc<EventEntry>>>,
    next_local_index: AtomicU32,
    tasks: TaskTracker,
    shutdown: AtomicBool,
}

impl LocalEventSystem {
    /// Create a new LocalEventSystem with the specified worker_id.
    ///
    /// The worker_id should be derived from an InstanceId using `instance_id.worker_id().as_u64()`.
    /// For Nova-managed systems, get the event system from `nova.events().local()` instead.
    pub fn new(worker_id: u64) -> Arc<Self> {
        Arc::new(Self {
            worker_id,
            events: DashMap::new(),
            free_lists: ParkingMutex::new(VecDeque::new()),
            next_local_index: AtomicU32::new(0),
            tasks: TaskTracker::new(),
            shutdown: AtomicBool::new(false),
        })
    }

    pub fn new_local_only() -> Arc<Self> {
        Arc::new(Self {
            worker_id: 0,
            events: DashMap::new(),
            free_lists: ParkingMutex::new(VecDeque::new()),
            next_local_index: AtomicU32::new(0),
            tasks: TaskTracker::new(),
            shutdown: AtomicBool::new(false),
        })
    }

    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    pub fn task_tracker(&self) -> &TaskTracker {
        &self.tasks
    }

    pub fn new_event(self: &Arc<Self>) -> Result<LocalEvent> {
        if self.is_shutdown() {
            bail!("Event system shutdown in progress");
        }
        loop {
            let entry = self.allocate_entry()?;
            match entry.begin_generation() {
                Ok(generation) => {
                    if self.is_shutdown() {
                        self.recycle_entry(entry);
                        bail!("Event system shutdown in progress");
                    }
                    let handle = entry.key().handle(generation)?;
                    return Ok(LocalEvent::new(self.clone(), entry, handle));
                }
                Err(EventEntryError::GenerationOverflow { key }) => {
                    trace!(
                        ?key,
                        "retiring event entry after exhausting generation space"
                    );
                    self.retire_entry(entry);
                    continue;
                }
                Err(err) => {
                    self.recycle_entry(entry);
                    return Err(err.into());
                }
            }
        }
    }

    pub fn create_user_event(self: &Arc<Self>) -> Result<LocalEvent> {
        self.new_event()
    }

    /// Returns a future that waits for the event to complete (triggered or poisoned).
    ///
    /// # Cancellation Safety
    ///
    /// The returned `LocalEventWaiter` is **cancellation safe** and can be used in
    /// `tokio::select!` statements. If the future is dropped before completion, the internal
    /// waiter count is properly decremented via the `Drop` implementation.
    ///
    /// # Performance Considerations
    ///
    /// This method performs a `DashMap` lookup and acquires a mutex lock to register a waiter.
    /// The returned future is efficient for use in `tokio::select!` loops - it can be polled
    /// multiple times without overhead:
    ///
    /// ```rust,ignore
    /// // Efficient: Lookup happens once, future can be polled repeatedly
    /// let mut wait_fut = system.awaiter(handle)?;
    ///
    /// loop {
    ///     tokio::select! {
    ///         result = &mut wait_fut => break result,
    ///         _ = other_event.tick() => continue,
    ///     }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The handle does not belong to this worker
    /// - The event is unknown (invalid handle)
    ///
    /// When awaited, the future returns an error if the event was poisoned.
    pub fn awaiter(self: &Arc<Self>, handle: EventHandle) -> Result<LocalEventWaiter> {
        self.ensure_local_handle(handle)?;
        self.wait_local(handle)
    }

    pub fn poll(self: &Arc<Self>, handle: EventHandle) -> Result<EventStatus> {
        self.ensure_local_handle(handle)?;
        self.poll_local(handle)
    }

    pub fn trigger(&self, handle: EventHandle) -> Result<()> {
        self.ensure_local_handle(handle)?;

        let entry = self
            .events
            .get(&handle.key())
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown event {}", handle))?;

        self.trigger_local_entry(entry, handle)
    }

    pub fn poison(&self, handle: EventHandle, reason: impl Into<String>) -> Result<()> {
        self.ensure_local_handle(handle)?;
        let reason = reason.into();

        let entry = self
            .events
            .get(&handle.key())
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown event {}", handle))?;

        match entry.status_for(handle.generation()) {
            EventStatus::Poisoned => return Ok(()),
            EventStatus::Ready => {
                bail!("Event {} already completed successfully", handle);
            }
            EventStatus::Pending => {}
        }

        let poison = Arc::new(EventPoison::new(handle, reason));
        self.poison_local_entry(entry, handle, poison)
    }

    pub fn merge_events(self: &Arc<Self>, inputs: Vec<EventHandle>) -> Result<EventHandle> {
        if inputs.is_empty() {
            bail!("Cannot merge empty event list");
        }

        for handle in &inputs {
            self.ensure_local_handle(*handle)?;
        }

        let merged = self.new_event()?;
        let handle = merged.handle();
        let trigger = merged.clone();

        let system = Arc::clone(self);
        self.tasks.spawn(async move {
            let mut failure_reasons: Option<Vec<Arc<str>>> = None;

            for dependency in inputs {
                let wait_result = match system.awaiter(dependency) {
                    Ok(waiter) => waiter.await,
                    Err(err) => Err(err),
                };

                match wait_result {
                    Ok(()) => {}
                    Err(err) => {
                        let reason = match err.downcast::<EventPoison>() {
                            Ok(poison) => format!(
                                "Merge dependency {} poisoned: {}",
                                dependency,
                                poison.reason()
                            ),
                            Err(other) => {
                                format!("Merge dependency {} failed: {}", dependency, other)
                            }
                        };
                        let reason_arc: Arc<str> = Arc::from(reason);
                        error!("{}", &*reason_arc);
                        failure_reasons
                            .get_or_insert_with(Vec::new)
                            .push(reason_arc);
                    }
                }
            }

            let result = match failure_reasons {
                None => trigger.trigger(),
                Some(reasons) => {
                    if reasons.len() == 1 {
                        system.poison(handle, reasons[0].as_ref())
                    } else {
                        let mut message = String::from("Multiple merge dependencies failed:\n");
                        for (idx, reason) in reasons.iter().enumerate() {
                            if idx > 0 {
                                message.push('\n');
                            }
                            message.push_str(reason.as_ref());
                        }
                        system.poison(handle, message)
                    }
                }
            };

            if let Err(e) = result {
                error!("Failed to complete merged event {}: {}", handle, e);
            }
        });

        Ok(handle)
    }

    /// Return the poison reason for a completed generation, if any.
    pub(crate) fn poison_reason(&self, handle: EventHandle) -> Option<Arc<str>> {
        let entry = self.events.get(&handle.key())?;
        entry.poison_reason(handle.generation())
    }

    fn ensure_local_handle(&self, handle: EventHandle) -> Result<()> {
        if handle.owner_worker().as_u64() != self.worker_id {
            bail!(
                "Event {} does not belong to worker {}",
                handle,
                self.worker_id
            );
        }
        Ok(())
    }

    fn wait_local(&self, handle: EventHandle) -> Result<LocalEventWaiter> {
        let entry = self
            .events
            .get(&handle.key())
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown local event {}", handle))?;

        match entry.register_local_waiter(handle.generation())? {
            WaitRegistration::Ready => Ok(LocalEventWaiter::immediate(Arc::new(
                CompletionKind::Triggered,
            ))),
            WaitRegistration::Poisoned(poison) => Ok(LocalEventWaiter::immediate(Arc::new(
                CompletionKind::Poisoned(poison),
            ))),
            WaitRegistration::Pending(waiter) => Ok(waiter),
        }
    }

    fn poll_local(&self, handle: EventHandle) -> Result<EventStatus> {
        let entry = self
            .events
            .get(&handle.key())
            .map(|guard| guard.clone())
            .ok_or_else(|| anyhow!("Unknown local event {}", handle))?;

        Ok(entry.status_for(handle.generation()))
    }

    fn trigger_local_entry(&self, entry: Arc<EventEntry>, handle: EventHandle) -> Result<()> {
        self.complete_local_entry(entry, handle, CompletionKind::Triggered)
    }

    fn poison_local_entry(
        &self,
        entry: Arc<EventEntry>,
        handle: EventHandle,
        poison: PoisonArc,
    ) -> Result<()> {
        self.complete_local_entry(entry, handle, CompletionKind::Poisoned(poison))
    }

    fn complete_local_entry(
        &self,
        entry: Arc<EventEntry>,
        handle: EventHandle,
        completion: CompletionKind,
    ) -> Result<()> {
        entry
            .finalize_completion(handle.generation(), completion)
            .map_err(anyhow::Error::new)?;
        self.recycle_entry(entry);
        Ok(())
    }

    fn allocate_entry(self: &Arc<Self>) -> Result<Arc<EventEntry>> {
        if let Some(entry) = self.try_reuse_entry() {
            return Ok(entry);
        }

        let local_index = self
            .next_local_index
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < MAX_LOCAL_INDEX).then_some(current + 1)
            })
            .map_err(|_| {
                anyhow!(
                    "Local event index space exhausted ({} entries)",
                    (MAX_LOCAL_INDEX as u64) + 1
                )
            })?;
        let key = EventKey::new(self.worker_id, local_index);
        let entry = Arc::new(EventEntry::new(key));
        self.events.insert(key, entry.clone());
        Ok(entry)
    }

    fn try_reuse_entry(&self) -> Option<Arc<EventEntry>> {
        let mut free_lists = self.free_lists.lock();
        free_lists.pop_front()
    }

    fn recycle_entry(&self, entry: Arc<EventEntry>) {
        if entry.is_retired() {
            return;
        }
        let mut free_lists = self.free_lists.lock();
        free_lists.push_back(entry);
    }

    fn retire_entry(&self, entry: Arc<EventEntry>) {
        entry.retire();
    }

    pub fn force_shutdown(&self, reason: impl Into<String>) {
        let was_shutdown = self.shutdown.swap(true, Ordering::SeqCst);
        if was_shutdown {
            return;
        }

        let reason: Arc<str> = Arc::from(reason.into());

        let mut pending = Vec::new();
        for entry in self.events.iter() {
            if let Some(handle) = entry.value().active_handle() {
                pending.push((entry.value().clone(), handle));
            }
        }

        for (entry, handle) in pending {
            let poison = Arc::new(EventPoison::from_shared(handle, Arc::clone(&reason)));
            if let Err(err) = self.poison_local_entry(entry, handle, poison) {
                error!("force_shutdown: failed to poison {}: {}", handle, err);
            }
        }

        self.free_lists.lock().clear();
    }

    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}
