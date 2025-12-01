// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interface to a traditional key-value store such as etcd.
//! "key_value_store" spelt out because in AI land "KV" means something else.

use std::borrow::Cow;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, path::PathBuf};
use std::{env, fmt};

use crate::CancellationToken;
use crate::transports::etcd as etcd_transport;
use async_trait::async_trait;
use futures::StreamExt;
use percent_encoding::{NON_ALPHANUMERIC, percent_decode_str, percent_encode};
use serde::{Deserialize, Serialize};

mod mem;
pub use mem::MemoryStore;
mod nats;
pub use nats::NATSStore;
mod etcd;
pub use etcd::EtcdStore;
mod file;
pub use file::FileStore;

const WATCH_SEND_TIMEOUT: Duration = Duration::from_millis(100);

/// String we use as the Key in a key-value storage operation. Simple String wrapper
/// that can encode / decode a string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key(String);

impl Key {
    pub fn new(s: String) -> Key {
        Key(s)
    }

    /// Takes a URL-safe percent-encoded string and creates a Key from it by decoding first.
    /// dynamo%2Fbackend%2Fgenerate%2F17216e63492ef21f becomes dynamo/backend/generate/17216e63492ef21f
    pub fn from_url_safe(s: &str) -> Key {
        Key(percent_decode_str(s).decode_utf8_lossy().to_string())
    }

    /// A URL-safe percent-encoded representation of this key.
    /// e.g.  dynamo/backend/generate/17216e63492ef21f becomes dynamo%2Fbackend%2Fgenerate%2F17216e63492ef21f
    pub fn url_safe(&self) -> Cow<'_, str> {
        percent_encode(self.0.as_bytes(), NON_ALPHANUMERIC).into()
    }
}

impl From<&str> for Key {
    fn from(s: &str) -> Key {
        Key::new(s.to_string())
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Key {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl From<&Key> for String {
    fn from(k: &Key) -> String {
        k.0.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyValue {
    key: Key,
    value: bytes::Bytes,
}

impl KeyValue {
    pub fn new(key: Key, value: bytes::Bytes) -> Self {
        KeyValue { key, value }
    }

    pub fn key(&self) -> String {
        self.key.clone().to_string()
    }

    pub fn key_str(&self) -> &str {
        self.key.as_ref()
    }

    pub fn value(&self) -> &[u8] {
        &self.value
    }

    pub fn value_str(&self) -> anyhow::Result<&str> {
        std::str::from_utf8(self.value()).map_err(From::from)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WatchEvent {
    Put(KeyValue),
    Delete(Key),
}

#[async_trait]
pub trait Store: Send + Sync {
    type Bucket: Bucket + Send + Sync + 'static;

    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // auto-delete items older than this
        ttl: Option<Duration>,
    ) -> Result<Self::Bucket, StoreError>;

    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError>;

    fn connection_id(&self) -> u64;

    fn shutdown(&self);
}

#[derive(Clone, Debug, Default)]
pub enum Selector {
    // Box it because it is significantly bigger than the other variants
    Etcd(Box<etcd_transport::ClientOptions>),
    File(PathBuf),
    #[default]
    Memory,
    // Nats not listed because likely we want to remove that impl. It is not currently used and not well tested.
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Selector::Etcd(opts) => {
                let urls = opts.etcd_url.join(",");
                write!(f, "Etcd({urls})")
            }
            Selector::File(path) => write!(f, "File({})", path.display()),
            Selector::Memory => write!(f, "Memory"),
        }
    }
}

impl FromStr for Selector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Selector> {
        match s {
            "etcd" => Ok(Self::Etcd(Box::default())),
            "file" => {
                let root = env::var("DYN_FILE_KV")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| env::temp_dir().join("dynamo_store_kv"));
                Ok(Self::File(root))
            }
            "mem" => Ok(Self::Memory),
            x => anyhow::bail!("Unknown key-value store type '{x}'"),
        }
    }
}

impl TryFrom<String> for Selector {
    type Error = anyhow::Error;

    fn try_from(s: String) -> anyhow::Result<Selector> {
        s.parse()
    }
}

#[allow(clippy::large_enum_variant)]
enum KeyValueStoreEnum {
    Memory(MemoryStore),
    Nats(NATSStore),
    Etcd(EtcdStore),
    File(FileStore),
}

impl KeyValueStoreEnum {
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // auto-delete items older than this
        ttl: Option<Duration>,
    ) -> Result<Box<dyn Bucket>, StoreError> {
        use KeyValueStoreEnum::*;
        Ok(match self {
            Memory(x) => Box::new(x.get_or_create_bucket(bucket_name, ttl).await?),
            Nats(x) => Box::new(x.get_or_create_bucket(bucket_name, ttl).await?),
            Etcd(x) => Box::new(x.get_or_create_bucket(bucket_name, ttl).await?),
            File(x) => Box::new(x.get_or_create_bucket(bucket_name, ttl).await?),
        })
    }

    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Box<dyn Bucket>>, StoreError> {
        use KeyValueStoreEnum::*;
        let maybe_bucket: Option<Box<dyn Bucket>> = match self {
            Memory(x) => x
                .get_bucket(bucket_name)
                .await?
                .map(|b| Box::new(b) as Box<dyn Bucket>),
            Nats(x) => x
                .get_bucket(bucket_name)
                .await?
                .map(|b| Box::new(b) as Box<dyn Bucket>),
            Etcd(x) => x
                .get_bucket(bucket_name)
                .await?
                .map(|b| Box::new(b) as Box<dyn Bucket>),
            File(x) => x
                .get_bucket(bucket_name)
                .await?
                .map(|b| Box::new(b) as Box<dyn Bucket>),
        };
        Ok(maybe_bucket)
    }

    fn connection_id(&self) -> u64 {
        use KeyValueStoreEnum::*;
        match self {
            Memory(x) => x.connection_id(),
            Etcd(x) => x.connection_id(),
            Nats(x) => x.connection_id(),
            File(x) => x.connection_id(),
        }
    }

    fn shutdown(&self) {
        use KeyValueStoreEnum::*;
        match self {
            Memory(x) => x.shutdown(),
            Etcd(x) => x.shutdown(),
            Nats(x) => x.shutdown(),
            File(x) => x.shutdown(),
        }
    }
}

#[derive(Clone)]
pub struct Manager(Arc<KeyValueStoreEnum>);

impl Default for Manager {
    fn default() -> Self {
        Manager::memory()
    }
}

impl Manager {
    /// In-memory KeyValueStoreManager for testing
    pub fn memory() -> Self {
        Self::new(KeyValueStoreEnum::Memory(MemoryStore::new()))
    }

    pub fn etcd(etcd_client: crate::transports::etcd::Client) -> Self {
        Self::new(KeyValueStoreEnum::Etcd(EtcdStore::new(etcd_client)))
    }

    pub fn file<P: Into<PathBuf>>(root: P) -> Self {
        Self::new(KeyValueStoreEnum::File(FileStore::new(root)))
    }

    fn new(s: KeyValueStoreEnum) -> Manager {
        Manager(Arc::new(s))
    }

    pub async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        // auto-delete items older than this
        ttl: Option<Duration>,
    ) -> Result<Box<dyn Bucket>, StoreError> {
        self.0.get_or_create_bucket(bucket_name, ttl).await
    }

    pub async fn get_bucket(
        &self,
        bucket_name: &str,
    ) -> Result<Option<Box<dyn Bucket>>, StoreError> {
        self.0.get_bucket(bucket_name).await
    }

    pub fn connection_id(&self) -> u64 {
        self.0.connection_id()
    }

    pub async fn load<T: for<'a> Deserialize<'a>>(
        &self,
        bucket: &str,
        key: &Key,
    ) -> Result<Option<T>, StoreError> {
        let Some(bucket) = self.0.get_bucket(bucket).await? else {
            // No bucket means no cards
            return Ok(None);
        };
        Ok(match bucket.get(key).await? {
            Some(card_bytes) => {
                let card: T = serde_json::from_slice(card_bytes.as_ref())?;
                Some(card)
            }
            None => None,
        })
    }

    /// Returns a receiver that will receive all the existing keys, and
    /// then block and receive new keys as they are created.
    /// Starts a task that runs forever, watches the store.
    pub fn watch(
        self: Arc<Self>,
        bucket_name: &str,
        bucket_ttl: Option<Duration>,
        cancel_token: CancellationToken,
    ) -> (
        tokio::task::JoinHandle<Result<(), StoreError>>,
        tokio::sync::mpsc::Receiver<WatchEvent>,
    ) {
        let bucket_name = bucket_name.to_string();
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let watch_task = tokio::spawn(async move {
            // Start listening for changes but don't poll this yet
            let bucket = self
                .0
                .get_or_create_bucket(&bucket_name, bucket_ttl)
                .await?;
            let mut stream = bucket.watch().await?;

            // Send all the existing keys
            for (key, bytes) in bucket.entries().await? {
                if let Err(err) = tx
                    .send_timeout(
                        WatchEvent::Put(KeyValue::new(key, bytes)),
                        WATCH_SEND_TIMEOUT,
                    )
                    .await
                {
                    tracing::error!(bucket_name, %err, "KeyValueStoreManager.watch failed adding existing key to channel");
                }
            }

            // Now block waiting for new entries
            loop {
                let event = tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    result = stream.next() => match result {
                        Some(event) => event,
                        None => break,
                    }
                };
                if let Err(err) = tx.send_timeout(event, WATCH_SEND_TIMEOUT).await {
                    tracing::error!(bucket_name, %err, "KeyValueStoreManager.watch failed adding new key to channel");
                }
            }

            Ok::<(), StoreError>(())
        });
        (watch_task, rx)
    }

    pub async fn publish<T: Serialize + Versioned + Send + Sync>(
        &self,
        bucket_name: &str,
        bucket_ttl: Option<Duration>,
        key: &Key,
        obj: &mut T,
    ) -> anyhow::Result<StoreOutcome> {
        let obj_json = serde_json::to_vec(obj)?;
        let bucket = self.0.get_or_create_bucket(bucket_name, bucket_ttl).await?;

        let outcome = bucket.insert(key, obj_json.into(), obj.revision()).await?;

        match outcome {
            StoreOutcome::Created(revision) | StoreOutcome::Exists(revision) => {
                obj.set_revision(revision);
            }
        }
        Ok(outcome)
    }

    /// Cleanup any temporary state.
    /// TODO: Should this be async? Take &mut self?
    pub fn shutdown(&self) {
        self.0.shutdown()
    }
}

/// An online storage for key-value config values.
#[async_trait]
pub trait Bucket: Send + Sync {
    /// A bucket is a collection of key/value pairs.
    /// Insert a value into a bucket, if it doesn't exist already
    /// The Key should be the name of the item, not including the bucket name.
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError>;

    /// Fetch an item from the key-value storage
    /// The Key should be the name of the item, not including the bucket name.
    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError>;

    /// Delete an item from the bucket
    /// The Key should be the name of the item, not including the bucket name.
    async fn delete(&self, key: &Key) -> Result<(), StoreError>;

    /// A stream of items inserted into the bucket.
    /// Every time the stream is polled it will either return a newly created entry, or block until
    /// such time.
    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + '_>>, StoreError>;

    /// The entries in this bucket.
    /// The Key includes the full path including the bucket name.
    /// That means you cannot directory get a Key from `entries` and pass it to `get` or `delete`.
    async fn entries(&self) -> Result<HashMap<Key, bytes::Bytes>, StoreError>;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StoreOutcome {
    /// The operation succeeded and created a new entry with this revision.
    /// Note that "create" also means update, because each new revision is a "create".
    Created(u64),
    /// The operation did not do anything, the value was already present, with this revision.
    Exists(u64),
}
impl fmt::Display for StoreOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreOutcome::Created(revision) => write!(f, "Created at {revision}"),
            StoreOutcome::Exists(revision) => write!(f, "Exists at {revision}"),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum StoreError {
    #[error("Could not find bucket '{0}'")]
    MissingBucket(String),

    #[error("Could not find key '{0}'")]
    MissingKey(String),

    #[error("Internal storage error: '{0}'")]
    ProviderError(String),

    #[error("Internal NATS error: {0}")]
    NATSError(String),

    #[error("Internal etcd error: {0}")]
    EtcdError(String),

    #[error("Internal filesystem error: {0}")]
    FilesystemError(String),

    #[error("Key Value Error: {0} for bucket '{1}'")]
    KeyValueError(String, String),

    #[error("Error decoding bytes: {0}")]
    JSONDecodeError(#[from] serde_json::error::Error),

    #[error("Race condition, retry the call")]
    Retry,
}

/// A trait allowing to get/set a revision on an object.
/// NATS uses this to ensure atomic updates.
pub trait Versioned {
    fn revision(&self) -> u64;
    fn set_revision(&mut self, r: u64);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use futures::{StreamExt, pin_mut};

    const BUCKET_NAME: &str = "v1/mdc";

    /// Convert the value returned by `watch()` into a broadcast stream that multiple
    /// clients can listen to.
    #[allow(dead_code)]
    pub struct TappableStream {
        tx: tokio::sync::broadcast::Sender<WatchEvent>,
    }

    #[allow(dead_code)]
    impl TappableStream {
        async fn new<T>(stream: T, max_size: usize) -> Self
        where
            T: futures::Stream<Item = WatchEvent> + Send + 'static,
        {
            let (tx, _) = tokio::sync::broadcast::channel(max_size);
            let tx2 = tx.clone();
            tokio::spawn(async move {
                pin_mut!(stream);
                while let Some(x) = stream.next().await {
                    let _ = tx2.send(x);
                }
            });
            TappableStream { tx }
        }

        fn subscribe(&self) -> tokio::sync::broadcast::Receiver<WatchEvent> {
            self.tx.subscribe()
        }
    }

    fn init() {
        crate::logging::init();
    }

    #[tokio::test]
    async fn test_memory_storage() -> anyhow::Result<()> {
        init();

        let s = Arc::new(MemoryStore::new());
        let s2 = Arc::clone(&s);

        let bucket = s.get_or_create_bucket(BUCKET_NAME, None).await?;
        let res = bucket.insert(&"test1".into(), "value1".into(), 0).await?;
        assert_eq!(res, StoreOutcome::Created(0));

        let mut expected = Vec::with_capacity(3);
        for i in 1..=3 {
            let item = WatchEvent::Put(KeyValue::new(
                Key::new(format!("test{i}")),
                format!("value{i}").into(),
            ));
            expected.push(item);
        }

        let (got_first_tx, got_first_rx) = tokio::sync::oneshot::channel();
        let ingress = tokio::spawn(async move {
            let b2 = s2.get_or_create_bucket(BUCKET_NAME, None).await?;
            let mut stream = b2.watch().await?;

            // Put in before starting the watch-all
            let v = stream.next().await.unwrap();
            assert_eq!(v, expected[0]);

            got_first_tx.send(()).unwrap();

            // Put in after
            let v = stream.next().await.unwrap();
            assert_eq!(v, expected[1]);

            let v = stream.next().await.unwrap();
            assert_eq!(v, expected[2]);

            Ok::<_, StoreError>(())
        });

        // MemoryStore uses a HashMap with no inherent ordering, so we must ensure test1 is
        // fetched before test2 is inserted, otherwise they can come out in any order, and we
        // wouldn't be testing the watch behavior.
        got_first_rx.await?;

        let res = bucket.insert(&"test2".into(), "value2".into(), 0).await?;
        assert_eq!(res, StoreOutcome::Created(0));

        // Repeat a key and revision. Ignored.
        let res = bucket.insert(&"test2".into(), "value2".into(), 0).await?;
        assert_eq!(res, StoreOutcome::Exists(0));

        // Increment revision
        let res = bucket.insert(&"test2".into(), "value2".into(), 1).await?;
        assert_eq!(res, StoreOutcome::Created(1));

        let res = bucket.insert(&"test3".into(), "value3".into(), 0).await?;
        assert_eq!(res, StoreOutcome::Created(0));

        // ingress exits once it has received all values
        let _ = ingress.await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_stream() -> anyhow::Result<()> {
        init();

        let s: &'static _ = Box::leak(Box::new(MemoryStore::new()));
        let bucket: &'static _ =
            Box::leak(Box::new(s.get_or_create_bucket(BUCKET_NAME, None).await?));

        let res = bucket.insert(&"test1".into(), "value1".into(), 0).await?;
        assert_eq!(res, StoreOutcome::Created(0));

        let stream = bucket.watch().await?;
        let tap = TappableStream::new(stream, 10).await;

        let mut rx1 = tap.subscribe();
        let mut rx2 = tap.subscribe();

        let item = WatchEvent::Put(KeyValue::new(Key::new("test1".to_string()), "GK".into()));
        let item_clone = item.clone();
        let handle1 = tokio::spawn(async move {
            let b = rx1.recv().await.unwrap();
            assert_eq!(b, item_clone);
        });
        let handle2 = tokio::spawn(async move {
            let b = rx2.recv().await.unwrap();
            assert_eq!(b, item);
        });

        bucket.insert(&"test1".into(), "GK".into(), 1).await?;

        let _ = futures::join!(handle1, handle2);
        Ok(())
    }
}
