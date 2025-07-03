use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct EtcdEntry {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Default, Clone)]
pub struct AppState {
    pub entries: BTreeMap<String, EtcdEntry>,
}

pub type SharedState = Arc<RwLock<AppState>>;
