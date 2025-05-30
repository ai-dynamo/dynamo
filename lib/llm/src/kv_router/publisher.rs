// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::kv_router::{
    indexer::{compute_block_hash_for_seq, RouterEvent},
    protocols::*,
    KV_EVENT_SUBJECT, KV_METRICS_ENDPOINT,
};
use async_trait::async_trait;
use dynamo_runtime::traits::{events::EventPublisher, DistributedRuntimeProvider};
use dynamo_runtime::{
    component::Component,
    pipeline::{
        network::Ingress, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream,
        SingleIn,
    },
    protocols::annotated::Annotated,
    Error, Result,
};
use futures::stream;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use rmp_serde as rmps;
use serde::Deserialize;
use serde::Serialize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use zeromq::{Socket, SocketRecv, SubSocket};

// -------------------------------------------------------------------------
// KV Event Publishers -----------------------------------------------------
// -------------------------------------------------------------------------

pub enum KvEventInternalSourceConfig {
    Zmq { endpoint: String, topic: String },
}

enum KvEventInternalSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
        processor_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventInternalSource {
    fn start(
        component: Component,
        worker_id: i64,
        cancellation_token: CancellationToken,
        kv_block_size: usize,
        source_config: KvEventInternalSourceConfig,
    ) -> Result<Self> {
        match source_config {
            KvEventInternalSourceConfig::Zmq { endpoint, topic } => {
                let (raw_tx, raw_rx) = mpsc::unbounded_channel::<(u64, Vec<u8>)>();
                let warning_count = Arc::new(AtomicU32::new(0));

                let zmq_handle = component
                    .drt()
                    .runtime()
                    .secondary()
                    .spawn(start_zmq_listener(
                        endpoint,
                        topic,
                        raw_tx,
                        cancellation_token.clone(),
                    ));

                let processor_handle = component.drt().runtime().secondary().spawn(start_zmq_event_processor(
                    raw_rx,
                    component,
                    worker_id,
                    kv_block_size,
                    warning_count,
                    cancellation_token,
                ));

                Ok(KvEventInternalSource::Zmq {
                    zmq_handle,
                    processor_handle,
                })
            }
        }
    }

    fn shutdown(&self) {
        match self {
            KvEventInternalSource::Zmq { zmq_handle, processor_handle } => {
                zmq_handle.abort();
                processor_handle.abort();
            }
        }
    }
}

pub enum KvEventSourceConfig {
    Internal(KvEventInternalSourceConfig),
    External,
}

enum KvEventSource {
    Internal(KvEventInternalSource),
    External {
        tx: mpsc::UnboundedSender<KvCacheEvent>,
    },
}

impl KvEventSource {
    fn start(
        component: Component,
        worker_id: i64,
        cancellation_token: CancellationToken,
        kv_block_size: usize,
        source_config: KvEventSourceConfig,
    ) -> Result<Self> {
        tracing::info!("Publishing KV Events to subject: {}", KV_EVENT_SUBJECT);
        match source_config {
            KvEventSourceConfig::Internal(internal_source_config) => {
                let source = KvEventInternalSource::start(
                    component,
                    worker_id,
                    cancellation_token,
                    kv_block_size,
                    internal_source_config,
                )?;
                Ok(Self::Internal(source))
            }
            KvEventSourceConfig::External => {
                let (tx, mut rx) = mpsc::unbounded_channel::<KvCacheEvent>();
                _ = component.drt().runtime().secondary().spawn(async move {
                    loop {
                        tokio::select! {
                            _ = cancellation_token.cancelled() => {
                                tracing::info!("KV Event source received cancellation signal");
                                break;
                            }
                            Some(event) = rx.recv() => {
                                let router_event = RouterEvent::new(worker_id, event);
                                component
                                    .publish(KV_EVENT_SUBJECT, &router_event)
                                    .await
                                    .unwrap();
                            }
                        }
                    }
                });

                Ok(Self::External { tx })
            }
        }
    }
}

impl KvEventSource {
    fn publish(&self, event: KvCacheEvent) -> Result<()> {
        match self {
            KvEventSource::Internal(_) => {
                anyhow::bail!("Internal source does not support manual publishing");
            }

            KvEventSource::External { tx } => {
                tx.send(event)?;
                Ok(())
            }
        }
    }

    fn shutdown(&self) {
        match self {
            KvEventSource::Internal(source) => {
                source.shutdown();
            }
            KvEventSource::External { tx: _ } => {}
        }
    }
}

pub struct KvEventPublisher {
    kv_block_size: usize,
    source: KvEventSource,
    cancellation_token: CancellationToken,
}

impl KvEventPublisher {
    pub fn new(
        component: Component,
        worker_id: i64,
        kv_block_size: usize,
        source_config: KvEventSourceConfig,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();

        let source = KvEventSource::start(
            component,
            worker_id,
            cancellation_token.clone(),
            kv_block_size,
            source_config,
        )?;

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<()> {
        tracing::trace!("Publish event: {:?}", event);
        self.source.publish(event)
    }

    pub fn kv_block_size(&self) -> usize {
        self.kv_block_size
    }

    pub fn shutdown(&self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }
        self.source.shutdown();
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn start_zmq_event_processor<P: EventPublisher>(
    mut raw_rx: mpsc::UnboundedReceiver<(u64, Vec<u8>)>,
    component: P,
    worker_id: i64,
    kv_block_size: usize,
    warning_count: Arc<AtomicU32>,
    cancellation_token: dynamo_runtime::CancellationToken,
) {
    loop {
        tokio::select! {
            biased;

            // Check for cancellation
            _ = cancellation_token.cancelled() => {
                tracing::debug!("Event processor received cancellation signal");
                break;
            }

            // Process incoming messages
            msg = raw_rx.recv() => {
                let Some((seq, payload)) = msg else {
                    tracing::debug!("Event processor channel closed");
                    break;
                };

                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!(error=%e, "Failed to decode KVEventBatch msgpack");
                    continue;
                };

                for raw_evt in batch.events.into_iter() {
                    let Some(event) = convert_event(raw_evt, seq, kv_block_size, &warning_count) else {
                        // Case where convert_event returns None
                        continue;
                    };

                    let router_event = RouterEvent::new(worker_id, event);
                    if let Err(e) = component.publish(KV_EVENT_SUBJECT, &router_event).await {
                        tracing::warn!(error=%e, "Failed to publish router event.");
                    }
                }
            }
        }
    }
    tracing::debug!("Event processor exiting");
}

// Error handling configuration for ZMQ operations
const INITIAL_BACKOFF_MS: u64 = 10;
const MAX_BACKOFF_MS: u64 = 5000;
const MAX_CONSECUTIVE_ERRORS: u32 = 10;
const MAX_BACKOFF_EXPONENT: u32 = 8; // Cap at 2^8 = 256x multiplier to prevent overflow

/// Calculate exponential backoff duration based on consecutive error count
fn calculate_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_BACKOFF_EXPONENT)),
        MAX_BACKOFF_MS,
    )
}

async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    raw_tx: mpsc::UnboundedSender<(u64, Vec<u8>)>,
    zmq_token: dynamo_runtime::CancellationToken,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let mut socket = SubSocket::new();

    // Subscribe to the requested topic (empty string == all topics)
    if let Err(e) = socket.subscribe(&zmq_topic).await {
        tracing::error!("Failed to subscribe on ZMQ socket: {}", e);
        return;
    }

    if let Err(e) = socket.connect(&zmq_endpoint).await {
        tracing::error!("Failed to connect ZMQ SUB socket: {}", e);
        return;
    }

    let mut consecutive_errors = 0u32;

    loop {
        tokio::select! {
            biased;

            // Check for cancellation
            _ = zmq_token.cancelled() => {
                tracing::info!("ZMQ listener received cancellation signal");
                break;
            }

            // Receive message
            msg_result = socket.recv() => {
                let Ok(msg) = msg_result else {
                    let e = msg_result.unwrap_err();
                    consecutive_errors += 1;

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        tracing::error!(
                            error=%e,
                            consecutive_errors=%consecutive_errors,
                            "Too many consecutive ZMQ errors, terminating listener"
                        );
                        break;
                    }

                    // Simple exponential backoff with max exponent to prevent overflow
                    let backoff_ms = calculate_backoff_ms(consecutive_errors);

                    tracing::warn!(
                        error=%e,
                        consecutive_errors=%consecutive_errors,
                        backoff_ms=%backoff_ms,
                        "Error reading from ZMQ socket, applying exponential backoff"
                    );

                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    continue;
                };

                // Reset error count on successful message
                consecutive_errors = 0;

                // We expect multipart frames: [topic, seq, payload]
                let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

                if frames.len() != 3 {
                    tracing::warn!(expected=3, actual=%frames.len(), "Received unexpected ZMQ frame count");
                    continue;
                }
                let payload = frames.remove(2);
                let seq_bytes = frames.remove(1);

                if seq_bytes.len() != 8 {
                    tracing::warn!(expected=8, actual=%seq_bytes.len(), "Invalid sequence number byte length");
                    continue;
                }

                let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());
                if raw_tx.send((seq, payload)).is_err() {
                    tracing::warn!("Failed to send message to channel - receiver dropped");
                    break;
                }
            }
        }
    }
    tracing::debug!("ZMQ listener exiting");
}

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router. Returns `None` when the
/// event cannot be represented with the current protocol (e.g., we ignore
/// `AllBlocksCleared` until a concrete format is defined).
fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: usize,
    warning_count: &Arc<AtomicU32>,
) -> Option<KvCacheEvent> {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
        } => {
            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            Some(KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash.map(ExternalSequenceBlockHash::from),
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes,
                        lora_id.unwrap_or(0),
                        warning_count,
                    ),
                }),
            })
        }
        RawKvEvent::BlockRemoved { block_hashes } => {
            let hashes = block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash::from)
                .collect();
            Some(KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
            })
        }
        RawKvEvent::AllBlocksCleared => {
            tracing::debug!("Received AllBlocksCleared event – currently ignored");
            None
        }
    }
}

pub fn create_stored_block_from_parts(
    kv_block_size: usize,
    block_hash: i64,
    token_ids: &[u32],
    _lora_id: u64,
) -> KvCacheStoredBlockData {
    let tokens_hash = compute_block_hash_for_seq(token_ids, kv_block_size)[0];
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
    }
}

pub fn create_stored_blocks(
    kv_block_size: usize,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[i64],
    lora_id: u64,
    warning_count: &Arc<AtomicU32>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (num_tokens_it, block_hash_it) in num_block_tokens.iter().zip(block_hashes.iter()) {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..(token_offset + *num_tokens_it as usize)];
        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_id,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

// -------------------------------------------------------------------------
// Types mirroring the Python msgspec-defined structures -------------------
// -------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
struct KvEventBatch {
    ts: f64,
    events: Vec<RawKvEvent>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")] // msgspec encodes variant tag as a string when `tag=True`
enum RawKvEvent {
    BlockStored {
        block_hashes: Vec<i64>,
        parent_block_hash: Option<i64>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_id: Option<u64>,
    },
    BlockRemoved {
        block_hashes: Vec<i64>,
    },
    AllBlocksCleared,
}

// -------------------------------------------------------------------------
// Metrics Publishers ------------------------------------------------------
// -------------------------------------------------------------------------

pub struct KvMetricsPublisher {
    tx: tokio::sync::watch::Sender<Arc<ForwardPassMetrics>>,
    rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl KvMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Arc::new(ForwardPassMetrics::default()));
        Ok(KvMetricsPublisher { tx, rx })
    }

    pub fn publish(
        &self,
        metrics: Arc<ForwardPassMetrics>,
    ) -> Result<(), tokio::sync::watch::error::SendError<Arc<ForwardPassMetrics>>> {
        tracing::trace!("Publish metrics: {metrics:?}");
        self.tx.send(metrics)
    }

    pub async fn create_endpoint(&self, component: Component) -> Result<()> {
        let mut metrics_rx = self.rx.clone();
        let handler = Arc::new(KvLoadEndpoingHander::new(metrics_rx.clone()));
        let handler = Ingress::for_engine(handler)?;

        component
            .endpoint(KV_METRICS_ENDPOINT)
            .endpoint_builder()
            .stats_handler(move |_| {
                let metrics = metrics_rx.borrow_and_update().clone();
                serde_json::to_value(&*metrics).unwrap()
            })
            .handler(handler)
            .start()
            .await
    }
}

struct KvLoadEndpoingHander {
    metrics_rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl KvLoadEndpoingHander {
    pub fn new(metrics_rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>) -> Self {
        Self { metrics_rx }
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<()>, ManyOut<Annotated<ForwardPassMetrics>>, Error>
    for KvLoadEndpoingHander
{
    async fn generate(
        &self,
        request: SingleIn<()>,
    ) -> Result<ManyOut<Annotated<ForwardPassMetrics>>> {
        let context = request.context();
        let metrics = self.metrics_rx.borrow().clone();
        let metrics = (*metrics).clone();
        let stream = stream::iter(vec![Annotated::from_data(metrics)]);
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

// -------------------------------------------------------------------------
// Testing -----------------------------------------------------------------
// -------------------------------------------------------------------------

#[cfg(test)]
mod test_event_processing {
    use super::*;
    use crate::kv_router::indexer::compute_block_hash_for_seq;

    // ---------------------------------------------------------------------
    // create_stored_block_from_parts --------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored = create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, 0);

        assert_eq!(stored.block_hash.0, blk_hash as u64);
        let expected_hash = compute_block_hash_for_seq(&token_ids, 4)[0];
        assert_eq!(stored.tokens_hash, expected_hash);
    }

    // ---------------------------------------------------------------------
    // create_stored_blocks -------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_blocks_ok() {
        let kv_block_size = 4;
        // two blocks, each of size 4
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let num_block_tokens = vec![4_u64, 4_u64];
        let block_hashes = vec![111_i64, 222_i64];

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            /*lora_id=*/ 0,
            &Arc::new(AtomicU32::new(0)),
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);
    }

    #[test]
    fn test_create_stored_blocks_wrong_size_triggers_warning() {
        let kv_block_size = 4;
        // second block is the wrong size
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7];
        let num_block_tokens = vec![4_u64, 3_u64];
        let block_hashes = vec![111_i64, 222_i64];
        let warning_count = Arc::new(AtomicU32::new(0));

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            /*lora_id=*/ 0,
            &warning_count,
        );

        // should early-exit as second has mismatch
        assert!(blocks.len() == 1);
        assert!(warning_count.load(Ordering::Relaxed) == 1)
    }

    // ---------------------------------------------------------------------
    // convert_event --------------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_convert_event_block_stored() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockStored {
            block_hashes: vec![10, 11],
            parent_block_hash: Some(99),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            block_size: 4,
            lora_id: Some(0),
        };

        let out = convert_event(raw_evt, 42, kv_block_size, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.unwrap().data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![123, 456],
        };
        let out = convert_event(raw_evt, 7, kv_block_size, &Arc::new(AtomicU32::new(0)));

        assert!(matches!(out.unwrap().data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        assert!(convert_event(raw_evt, 1, kv_block_size, &Arc::new(AtomicU32::new(0))).is_none());
    }
}

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use async_trait;
    use bytes::Bytes;
    use std::sync::{Arc, Mutex};
    use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

    // Type alias to resolve clippy::type_complexity warning
    type PublishedEvents = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    //--------------------------------------------------------------------
    // A tiny stand-in for Component that just records every publish call
    //--------------------------------------------------------------------
    #[derive(Default)]
    struct MockComponent {
        published: PublishedEvents,
    }

    impl MockComponent {
        fn new() -> (Self, PublishedEvents) {
            let published = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    published: published.clone(),
                },
                published,
            )
        }
    }

    #[async_trait::async_trait]
    impl EventPublisher for MockComponent {
        async fn publish(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            event: &(impl serde::Serialize + Send + Sync),
        ) -> dynamo_runtime::Result<()> {
            let bytes = rmp_serde::to_vec(event).unwrap();
            self.published
                .lock()
                .unwrap()
                .push((event_name.as_ref().to_string(), bytes));
            Ok(())
        }

        async fn publish_bytes(
            &self,
            event_name: impl AsRef<str> + Send + Sync,
            bytes: Vec<u8>,
        ) -> dynamo_runtime::Result<()> {
            self.published
                .lock()
                .unwrap()
                .push((event_name.as_ref().to_string(), bytes));
            Ok(())
        }

        fn subject(&self) -> String {
            "mock.subject".into()
        }
    }

    //--------------------------------------------------------------------
    // Test start_zmq_event_processor in isolation
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_event_processor_sends_router_event() {
        let kv_block_size = 4;
        let worker_id = 99;

        // 1) build a one-item KvEventBatch and msgpack-encode it
        let batch = KvEventBatch {
            ts: 0.0,
            events: vec![RawKvEvent::BlockRemoved {
                block_hashes: vec![1, 2],
            }],
        };
        let payload = rmps::to_vec(&batch).unwrap();

        let token = dynamo_runtime::CancellationToken::new();

        // 2) channel feeding the processor
        let (tx, rx) = mpsc::unbounded_channel::<(u64, Vec<u8>)>();
        tx.send((123, payload.clone())).unwrap(); // seq = 123
        drop(tx);

        // 3) mock component to capture output
        let (comp, published) = MockComponent::new();

        // 4) run the function under test (let it consume exactly one msg)
        let handle = tokio::spawn(start_zmq_event_processor(
            rx,
            comp,
            worker_id,
            kv_block_size,
            Arc::new(AtomicU32::new(0)),
            token,
        ));

        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // 5) assert we have exactly one RouterEvent pushed with right worker_id
        let published = published.lock().unwrap();
        let (subject, bytes) = &published[0];

        assert_eq!(subject, &KV_EVENT_SUBJECT.to_string());
        assert_eq!(bytes.first(), payload.first())
    }

    //--------------------------------------------------------------------
    // Test start_zmq_listener without a real socket
    //   (feed it frames through a ZMQ PAIR tcp socket)
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_listener_pushes_to_channel() {
        // Prepare channel that listener should fill
        let (tx, mut rx) = mpsc::unbounded_channel::<(u64, Vec<u8>)>();

        // ZMQ TCP endpoint using localhost with fixed port
        let endpoint = "tcp://127.0.0.1:15555";
        let topic = "".to_string(); // subscribe to all

        // Publisher side - set up first
        let mut pub_socket = PubSocket::new();
        pub_socket.bind(endpoint).await.unwrap();

        // Cancellation token so we can stop the listener
        let token = dynamo_runtime::CancellationToken::new();

        // Spawn async listener
        let listener_handle = tokio::spawn({
            let token = token.clone();
            start_zmq_listener(endpoint.to_string(), topic, tx, token)
        });

        // Give time for the connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send synthetic 3-frame message: [topic, seq(8B), payload]
        let seq: u64 = 77;
        let payload = Bytes::from("hello");

        let frames = vec![
            Bytes::from(""),
            Bytes::from(seq.to_be_bytes().to_vec()),
            payload.clone(),
        ];

        // Create a proper multipart message
        let msg = ZmqMessage::try_from(frames).expect("Failed to create ZmqMessage");

        // Send the multipart message
        pub_socket.send(msg).await.unwrap();

        // Wait for message to be received
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that we received the message
        let (got_seq, got_payload) = rx.try_recv().expect("no message received");
        assert_eq!(got_seq, seq);
        assert_eq!(got_payload, payload);

        // Stop the listener
        token.cancel();
        let _ = listener_handle.await;
    }
}

#[cfg(test)]
mod test_exponential_backoff {
    use super::*;

    #[test]
    fn test_backoff_calculation_progression() {
        // Test the exponential progression
        assert_eq!(calculate_backoff_ms(0), 10); // 10 * 2^0 = 10
        assert_eq!(calculate_backoff_ms(1), 20); // 10 * 2^1 = 20
        assert_eq!(calculate_backoff_ms(2), 40); // 10 * 2^2 = 40
        assert_eq!(calculate_backoff_ms(3), 80); // 10 * 2^3 = 80
        assert_eq!(calculate_backoff_ms(4), 160); // 10 * 2^4 = 160
        assert_eq!(calculate_backoff_ms(5), 320); // 10 * 2^5 = 320
        assert_eq!(calculate_backoff_ms(6), 640); // 10 * 2^6 = 640
        assert_eq!(calculate_backoff_ms(7), 1280); // 10 * 2^7 = 1280
        assert_eq!(calculate_backoff_ms(8), 2560); // 10 * 2^8 = 2560
    }

    #[test]
    fn test_backoff_caps_at_max_exponent() {
        // After MAX_BACKOFF_EXPONENT, should stay at 2^8 = 2560ms
        assert_eq!(calculate_backoff_ms(8), 2560);
        assert_eq!(calculate_backoff_ms(9), 2560); // Same as 8
        assert_eq!(calculate_backoff_ms(100), 2560); // Same as 8
    }

    #[test]
    fn test_backoff_never_exceeds_max() {
        // Even if we somehow had a huge exponent, never exceed MAX_BACKOFF_MS
        for i in 0..20 {
            assert!(calculate_backoff_ms(i) <= MAX_BACKOFF_MS);
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_backoff_constants_are_sane() {
        // Verify our constants make sense together
        assert!(INITIAL_BACKOFF_MS > 0);
        assert!(MAX_BACKOFF_MS > INITIAL_BACKOFF_MS);
        assert!(MAX_BACKOFF_EXPONENT <= 10); // Prevent crazy exponents
        assert!(MAX_CONSECUTIVE_ERRORS > 0);

        // Max calculated value should be less than MAX_BACKOFF_MS
        let max_calculated = INITIAL_BACKOFF_MS * 2_u64.pow(MAX_BACKOFF_EXPONENT);
        assert!(max_calculated <= MAX_BACKOFF_MS);
    }
}
