// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test harness utilities for exercising the slot lifecycle with console hooks.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use dashmap::DashMap;
use dynamo_nova::events::{EventHandle, EventManager, LocalEventSystem};
use dynamo_tokens::{TokenBlock, TokenBlockSequence, Tokens};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::v2::integrations::connector::slot::{
    OperationInfo, Slot, SlotCore, SlotState, TransferDirection, TransferEngineCommand,
    TransferSlotHandle,
};
use crate::v2::logical::blocks::BlockId;

use super::TransferEngineEnvelope;
use super::console::{ConnectorConsoleHook, ConsoleHookRef, NoopConsoleHook};

/// Event log produced by [`RecordingConsoleHook`].
#[derive(Debug, Clone)]
pub enum ConsoleEvent {
    SlotCreated {
        request_id: String,
        state: SlotState,
    },
    StateTransition {
        request_id: String,
        from: SlotState,
        to: SlotState,
    },
    OperationRegistered {
        request_id: String,
        operation_id: Uuid,
        info: OperationInfo,
    },
    OperationCompleted {
        request_id: String,
        operation_id: Uuid,
    },
    FinishStarted {
        request_id: String,
        outstanding: usize,
    },
    SlotFinished {
        request_id: String,
    },
}

/// Console hook that records events for later assertions.
#[derive(Default)]
pub struct RecordingConsoleHook {
    events: Mutex<Vec<ConsoleEvent>>,
}

impl RecordingConsoleHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn as_hook(self: &Arc<Self>) -> ConsoleHookRef {
        Arc::clone(self) as ConsoleHookRef
    }

    pub fn events(&self) -> Vec<ConsoleEvent> {
        self.events.lock().unwrap().clone()
    }

    fn push(&self, event: ConsoleEvent) {
        self.events.lock().unwrap().push(event);
    }
}

impl ConnectorConsoleHook for RecordingConsoleHook {
    fn on_slot_created(&self, request_id: &str, state: SlotState) {
        self.push(ConsoleEvent::SlotCreated {
            request_id: request_id.to_string(),
            state,
        });
    }

    fn on_state_transition(&self, request_id: &str, prev: SlotState, next: SlotState) {
        self.push(ConsoleEvent::StateTransition {
            request_id: request_id.to_string(),
            from: prev,
            to: next,
        });
    }

    fn on_operation_registered(&self, request_id: &str, operation_id: Uuid, info: &OperationInfo) {
        self.push(ConsoleEvent::OperationRegistered {
            request_id: request_id.to_string(),
            operation_id,
            info: info.clone(),
        });
    }

    fn on_operation_completed(&self, request_id: &str, operation_id: Uuid) {
        self.push(ConsoleEvent::OperationCompleted {
            request_id: request_id.to_string(),
            operation_id,
        });
    }

    fn on_request_finish_started(&self, request_id: &str, outstanding: usize) {
        self.push(ConsoleEvent::FinishStarted {
            request_id: request_id.to_string(),
            outstanding,
        });
    }

    fn on_slot_finished(&self, request_id: &str) {
        self.push(ConsoleEvent::SlotFinished {
            request_id: request_id.to_string(),
        });
    }
}

/// Result of invoking [`MockLeader::request_finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotFinishStatus {
    Ready,
    Pending {
        outstanding: Vec<Uuid>,
        finish_event: Option<EventHandle>,
    },
}

/// Multi-slot leader facade for tests.
pub struct MockLeader {
    hook: ConsoleHookRef,
    slots: DashMap<String, Arc<MockSlot>>,
}

impl MockLeader {
    pub fn new(hook: Option<ConsoleHookRef>) -> Arc<Self> {
        Arc::new(Self {
            hook: hook.unwrap_or_else(NoopConsoleHook::shared),
            slots: DashMap::new(),
        })
    }

    pub fn create_slot(&self, request_id: impl Into<String>, block_size: usize) -> Arc<MockSlot> {
        let request_id = request_id.into();
        let slot = Arc::new(MockSlot::new(
            request_id.clone(),
            block_size,
            Some(Arc::clone(&self.hook)),
        ));
        self.slots.insert(request_id, Arc::clone(&slot));
        slot
    }

    pub fn get_slot(&self, request_id: &str) -> Option<Arc<MockSlot>> {
        self.slots
            .get(request_id)
            .map(|entry| Arc::clone(entry.value()))
    }

    pub fn slots(&self) -> Vec<Arc<MockSlot>> {
        self.slots
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    pub fn acknowledge_operation(&self, request_id: &str, operation_id: Uuid) -> bool {
        self.get_slot(request_id)
            .map(|slot| slot.acknowledge_operation(operation_id))
            .unwrap_or(false)
    }
}

/// Individual slot handle owned by [`MockLeader`].
pub struct MockSlot {
    request_id: String,
    slot: Arc<Mutex<Slot>>,
    hook: ConsoleHookRef,
    finished: AtomicBool,
    next_event_id: AtomicU64,
    event_manager: EventManager,
}

impl MockSlot {
    pub fn new(
        request_id: impl Into<String>,
        block_size: usize,
        hook: Option<ConsoleHookRef>,
    ) -> Self {
        let request_id = request_id.into();
        let sequence = empty_sequence(block_size);
        let core = SlotCore::new(request_id.clone(), sequence, block_size);
        let slot = Arc::new(Mutex::new(Slot::new(core)));
        let event_manager = LocalEventSystem::new(42);

        let hook = hook.unwrap_or_else(NoopConsoleHook::shared);

        let slot_handle = Self {
            request_id: request_id.clone(),
            slot,
            hook: Arc::clone(&hook),
            finished: AtomicBool::new(false),
            next_event_id: AtomicU64::new(1),
            event_manager,
        };

        {
            let slot = slot_handle.slot.lock().unwrap();
            slot_handle
                .hook
                .on_slot_created(&slot_handle.request_id, *slot.state());
        }

        slot_handle
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn state(&self) -> SlotState {
        let slot = self.slot.lock().unwrap();
        *slot.state()
    }

    pub fn slot_handle(&self) -> Arc<Mutex<Slot>> {
        Arc::clone(&self.slot)
    }

    pub fn slot_handle_dyn(&self) -> Arc<dyn TransferSlotHandle> {
        Arc::clone(&self.slot) as Arc<dyn TransferSlotHandle>
    }

    pub fn with_slot_mut<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut Slot) -> T,
    {
        let mut slot = self.slot.lock().unwrap();
        f(&mut slot)
    }

    pub fn set_state(&self, next: SlotState) {
        let mut slot = self.slot.lock().unwrap();
        let prev = *slot.state();
        if prev != next {
            slot.set_state(next);
            drop(slot);
            self.hook.on_state_transition(&self.request_id, prev, next);
        }
    }

    fn next_event_handle(&self) -> EventHandle {
        self.event_manager.create_user_event().unwrap().handle()
    }

    pub fn plan_operation(
        &self,
        worker_id: usize,
        direction: TransferDirection,
        num_blocks: usize,
    ) -> Uuid {
        {
            let slot = self.slot.lock().unwrap();
            assert!(
                slot.allow_new_operations(),
                "slot {} disallows new operations",
                self.request_id
            );
        }
        let operation_id = Uuid::new_v4();
        let info = OperationInfo {
            direction,
            num_blocks,
        };
        let event = self.next_event_handle();
        {
            let mut slot = self.slot.lock().unwrap();
            slot.record_operation(operation_id, info.clone(), Some(event));
        }
        self.hook
            .on_operation_registered(&self.request_id, operation_id, &info);
        operation_id
    }

    pub fn request_finish(&self) -> SlotFinishStatus {
        let outstanding;
        let transition = {
            let mut slot = self.slot.lock().unwrap();
            slot.disallow_new_operations();

            outstanding = slot.outstanding_operations();

            if outstanding.is_empty() {
                let prev = *slot.state();
                if prev != SlotState::Finished {
                    slot.set_state(SlotState::Finished);
                }
                drop(slot);
                if !self.finished.swap(true, Ordering::SeqCst) {
                    if prev != SlotState::Finished {
                        self.hook
                            .on_state_transition(&self.request_id, prev, SlotState::Finished);
                    }
                    self.hook.on_slot_finished(&self.request_id);
                }
                return SlotFinishStatus::Ready;
            }
            let prev = *slot.state();
            if prev != SlotState::Finishing {
                slot.set_state(SlotState::Finishing);
                Some(prev)
            } else {
                None
            }
        };

        if let Some(prev) = transition {
            self.hook
                .on_state_transition(&self.request_id, prev, SlotState::Finishing);
        }

        self.hook
            .on_request_finish_started(&self.request_id, outstanding.len());

        SlotFinishStatus::Pending {
            outstanding,
            finish_event: None, // Mock doesn't need a real event handle
        }
    }

    pub fn acknowledge_operation(&self, operation_id: Uuid) -> bool {
        {
            let mut slot = self.slot.lock().unwrap();
            slot.complete_operation(operation_id);
        }
        self.hook
            .on_operation_completed(&self.request_id, operation_id);

        let mut slot = self.slot.lock().unwrap();
        let outstanding = slot.outstanding_operations();
        let allow_new = slot.allow_new_operations();
        let prev = *slot.state();

        if outstanding.is_empty() && !allow_new {
            if prev != SlotState::Finished {
                slot.set_state(SlotState::Finished);
            }
            drop(slot);
            if !self.finished.swap(true, Ordering::SeqCst) {
                if prev != SlotState::Finished {
                    self.hook
                        .on_state_transition(&self.request_id, prev, SlotState::Finished);
                }
                self.hook.on_slot_finished(&self.request_id);
            }
            true
        } else {
            drop(slot);
            false
        }
    }
}

/// Lightweight worker wrapper mirroring the console hook interface.
pub struct MockWorker {
    id: usize,
    direction: TransferDirection,
}

impl MockWorker {
    pub fn new(id: usize, direction: TransferDirection) -> Self {
        Self { id, direction }
    }

    pub fn start_operation(&self, slot: &MockSlot, num_blocks: usize) -> Uuid {
        slot.plan_operation(self.id, self.direction, num_blocks)
    }

    pub fn finish_operation(&self, slot: &MockSlot, operation_id: Uuid) -> bool {
        slot.acknowledge_operation(operation_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineCommandKind {
    Offload { num_blocks: usize },
    Onboard { num_blocks: usize },
    Noop,
}

#[derive(Debug, Clone)]
pub struct EngineEvent {
    pub request_id: String,
    pub uuid: Uuid,
    pub kind: EngineCommandKind,
}

pub struct MockTransferEngine {
    events: Arc<Mutex<Vec<EngineEvent>>>,
    handle: Option<JoinHandle<()>>,
}

impl MockTransferEngine {
    pub fn spawn(
        mut engine_rx: mpsc::UnboundedReceiver<TransferEngineEnvelope>,
        leader: Arc<MockLeader>,
    ) -> Self {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let handle = tokio::spawn(async move {
            while let Some(envelope) = engine_rx.recv().await {
                let TransferEngineEnvelope {
                    request_id,
                    command,
                } = envelope;
                let (uuid, kind) = match command {
                    TransferEngineCommand::Offload {
                        uuid, block_ids, ..
                    } => {
                        let num_blocks = block_ids.len();
                        (uuid, EngineCommandKind::Offload { num_blocks })
                    }
                    TransferEngineCommand::Onboard {
                        uuid, src_blocks, ..
                    } => {
                        let num_blocks = src_blocks.len();
                        (uuid, EngineCommandKind::Onboard { num_blocks })
                    }
                    TransferEngineCommand::Noop { uuid } => (uuid, EngineCommandKind::Noop),
                };

                events_clone.lock().unwrap().push(EngineEvent {
                    request_id: request_id.clone(),
                    uuid,
                    kind,
                });

                leader.acknowledge_operation(&request_id, uuid);
            }
        });

        Self {
            events,
            handle: Some(handle),
        }
    }

    pub fn events(&self) -> Vec<EngineEvent> {
        self.events.lock().unwrap().clone()
    }

    pub async fn shutdown(mut self) -> Vec<EngineEvent> {
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
        self.events.lock().unwrap().clone()
    }
}

impl Drop for MockTransferEngine {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

pub fn sample_block_ids(start: BlockId, count: usize) -> Vec<BlockId> {
    (0..count).map(|idx| start + idx as BlockId).collect()
}

pub fn sample_token_blocks(block_size: usize, num_blocks: usize) -> Vec<TokenBlock> {
    let total_tokens = block_size * num_blocks;
    let tokens = Tokens::from((0..total_tokens).map(|idx| idx as i32).collect::<Vec<_>>());
    let sequence = TokenBlockSequence::new(tokens, block_size as u32, None);
    let (blocks, _) = sequence.into_parts();
    blocks
}

fn empty_sequence(block_size: usize) -> TokenBlockSequence {
    Tokens::from(Vec::<i32>::new()).into_sequence(block_size as u32, None)
}
