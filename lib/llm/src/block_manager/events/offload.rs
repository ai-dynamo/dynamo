use std::sync::{Arc, Weak};
use tokio::sync::mpsc;

use crate::block_manager::events::{
    EventManager, EventPublisher, EventReleaseManager, EventType, RegistrationHandle,
};

pub struct OffloadEventManager {
    tx: mpsc::UnboundedSender<Vec<Weak<RegistrationHandle>>>,
}

impl OffloadEventManager {
    pub fn new(tx: mpsc::UnboundedSender<Vec<Weak<RegistrationHandle>>>) -> Self {
        Self { tx }
    }
}

impl EventPublisher for OffloadEventManager {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>, event_type: EventType) {
        if event_type != EventType::CacheHit {
            return;
        }
        self.tx
            .send(handles.iter().map(Arc::downgrade).collect())
            .unwrap();
    }
}

impl EventReleaseManager for OffloadEventManager {
    fn block_release(&self, _registration_handle: &RegistrationHandle) {}
}

impl EventManager for OffloadEventManager {}
