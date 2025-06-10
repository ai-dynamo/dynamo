use crate::app_state::{EtcdEntry, SharedState};
use etcd_client::{Client, EventType, GetOptions, WatchOptions};
use tokio::sync::mpsc;
use tracing::info;

pub async fn watch_etcd(state: SharedState) -> anyhow::Result<()> {
    let mut client = Client::connect(["http://localhost:2379"], None).await?;
    let (_watcher, mut stream) = client
        .watch("/dynamo/", Some(WatchOptions::new().with_prefix()))
        .await?;

    info!("ETCD watcher started");

    while let Some(resp) = stream.message().await? {
        for event in resp.events() {
            let key = String::from_utf8_lossy(event.kv().unwrap().key()).to_string();

            match event.event_type() {
                EventType::Put => {
                    let value = event.kv().unwrap().value_str().unwrap_or("").to_string();
                    let mut guard = state.write().await;
                    guard.entries.insert(key.clone(), EtcdEntry { key, value });
                }
                EventType::Delete => {
                    let mut guard = state.write().await;
                    guard.entries.remove(&key);
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub enum EtcdEvent {
    Put { key: String, value: String },
    Delete { key: String },
}

pub struct EtcdWatcher {
    etcd_url: String,
    prefix: String,
}

impl EtcdWatcher {
    pub fn new<S: Into<String>>(etcd_url: S, prefix: S) -> Self {
        Self {
            etcd_url: etcd_url.into(),
            prefix: prefix.into(),
        }
    }

    pub async fn run(self, sender: mpsc::Sender<EtcdEvent>) -> anyhow::Result<()> {
        let mut client = Client::connect([&self.etcd_url], None).await?;
        let (_watcher, mut stream) = client
            .watch(self.prefix.clone(), Some(WatchOptions::new().with_prefix()))
            .await?;

        info!("Started watching prefix: {}", self.prefix);

        while let Some(resp) = stream.message().await? {
            for event in resp.events() {
                match event.event_type() {
                    EventType::Put => {
                        if let (Some(kv), Some(value)) = (
                            event.kv(),
                            event
                                .kv()
                                .and_then(|kv| Some(kv.value_str().ok()?.to_string())),
                        ) {
                            let key = String::from_utf8_lossy(kv.key()).to_string();
                            sender.send(EtcdEvent::Put { key, value }).await?;
                        }
                    }
                    EventType::Delete => {
                        if let Some(kv) = event.kv() {
                            let key = String::from_utf8_lossy(kv.key()).to_string();
                            sender.send(EtcdEvent::Delete { key }).await?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn list_all(&self) -> anyhow::Result<Vec<(String, String)>> {
        let mut client = Client::connect([&self.etcd_url], None).await?;
        let resp = client
            .get(self.prefix.clone(), Some(GetOptions::new().with_prefix()))
            .await?;
        let kvs = resp.kvs();

        Ok(kvs
            .iter()
            .map(|kv| {
                (
                    String::from_utf8_lossy(kv.key()).to_string(),
                    kv.value_str().unwrap_or("").to_string(),
                )
            })
            .collect())
    }
}
