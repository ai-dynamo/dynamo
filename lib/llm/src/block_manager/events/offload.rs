use std::sync::Arc;
use tokio::sync::broadcast;

use crate::block_manager::events::{
    EventManager, EventPublisher, EventReleaseManager, EventType, RegistrationHandle,
};
use crate::tokens::SequenceHash;

pub struct OffloadEventManager {
    tx: broadcast::Sender<Vec<SequenceHash>>,
}

impl Default for OffloadEventManager {
    fn default() -> Self {
        Self::new()
    }
}

impl OffloadEventManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self { tx }
    }

    pub fn receiver(&self) -> broadcast::Receiver<Vec<SequenceHash>> {
        self.tx.subscribe()
    }
}

impl EventPublisher for OffloadEventManager {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>, event_type: EventType) {
        if event_type != EventType::CacheHit {
            return;
        }

        let _ = self.tx.send(
            handles
                .iter()
                .map(|handle| handle.sequence_hash())
                .collect(),
        );
    }
}

impl EventReleaseManager for OffloadEventManager {
    fn block_release(&self, _registration_handle: &RegistrationHandle) {}
}

impl EventManager for OffloadEventManager {}
