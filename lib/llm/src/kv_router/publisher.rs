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

/// Configure the source of KV events.
/// Currently, only ZMQ is supported.
pub enum KvEventSourceConfig {
    Zmq { endpoint: String, topic: String },
}

/// The source of KV events.
enum KvEventSource {
    Zmq {
        zmq_handle: tokio::task::JoinHandle<()>,
    },
}

impl KvEventSource {
    /// Start the event source from a [`KvEventSourceConfig`].
    fn start(
        component: Component,
        kv_block_size: usize,
        source_config: KvEventSourceConfig,
        cancellation_token: CancellationToken,
        tx: mpsc::UnboundedSender<KvCacheEvent>,
    ) -> Result<Self> {
        match source_config {
            KvEventSourceConfig::Zmq { endpoint, topic } => {
                let zmq_handle = component
                    .drt()
                    .runtime()
                    .secondary()
                    .spawn(start_zmq_listener(
                        endpoint,
                        topic,
                        tx,
                        cancellation_token.clone(),
                        kv_block_size,
                    ));

                Ok(KvEventSource::Zmq { zmq_handle })
            }
        }
    }

    fn shutdown(&self) {
        match self {
            KvEventSource::Zmq { zmq_handle } => {
                zmq_handle.abort();
            }
        }
    }
}

/// A publisher of KV events.
pub struct KvEventPublisher {
    /// The size of the KV block.
    kv_block_size: usize,
    /// The source of KV events.
    /// Can be `None` if all events provided through [`KvEventPublisher::publish`].
    source: Option<KvEventSource>,
    /// The cancellation token.
    cancellation_token: CancellationToken,
    /// The channel to send events to.
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvEventPublisher {
    pub fn new(
        component: Component,
        worker_id: i64,
        kv_block_size: usize,
        source_config: Option<KvEventSourceConfig>,
    ) -> Result<Self> {
        let cancellation_token = CancellationToken::new();

        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();

        // Create our event source (if any)
        let mut source = None;
        if let Some(config) = source_config {
            source = Some(KvEventSource::start(
                component.clone(),
                kv_block_size,
                config,
                cancellation_token.clone(),
                tx.clone(),
            )?);
        }

        component
            .drt()
            .runtime()
            .secondary()
            .spawn(start_event_processor(
                component,
                worker_id,
                cancellation_token.clone(),
                rx,
            ));

        Ok(Self {
            kv_block_size,
            source,
            cancellation_token,
            tx,
        })
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        tracing::trace!("Publish event: {:?}", event);
        self.tx.send(event)
    }

    pub fn kv_block_size(&self) -> usize {
        self.kv_block_size
    }

    pub fn shutdown(&mut self) {
        if !self.cancellation_token.is_cancelled() {
            self.cancellation_token.cancel();
        }

        if let Some(source) = self.source.take() {
            source.shutdown();
        }
    }
}

impl Drop for KvEventPublisher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

async fn start_event_processor<P: EventPublisher + Send + Sync + 'static>(
    publisher: P,
    worker_id: i64,
    cancellation_token: CancellationToken,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => {
                tracing::info!("KV Event source received cancellation signal");
                break;
            }
            event = rx.recv() => {
                let Some(event) = event else {
                    tracing::debug!("Event processor channel closed.");
                    break;
                };

                // Encapsulate in a router event and publish.
                let router_event = RouterEvent::new(worker_id, event);
                if let Err(e) = publisher.publish(KV_EVENT_SUBJECT, &router_event).await {
                    tracing::error!("Failed to publish event: {}", e);
                }
            }
        }
    }
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

pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    cancellation_token: CancellationToken,
    kv_block_size: usize,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );

    let warning_count = Arc::new(AtomicU32::new(0));

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
            _ = cancellation_token.cancelled() => {
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

                // Extract the payload and sequence number.
                let payload = frames.pop().unwrap();
                let seq_bytes = frames.pop().unwrap();

                if seq_bytes.len() != 8 {
                    tracing::warn!(expected=8, actual=%seq_bytes.len(), "Invalid sequence number byte length");
                    continue;
                }

                let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());

                // Decode our batch of events.
                let batch_result = rmps::from_slice::<KvEventBatch>(&payload);
                let Ok(batch) = batch_result else {
                    let e = batch_result.unwrap_err();
                    tracing::warn!(error=%e, "Failed to decode KVEventBatch msgpack");
                    continue;
                };

                // For each of our events, convert them to [`KvCacheEvent`] and send to the event_processor.
                for raw_event in batch.events.into_iter() {
                    let event = convert_event(raw_event, seq, kv_block_size, &warning_count);
                    if tx.send(event).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        return;
                    }
                }
            }
        }
        tracing::debug!("ZMQ listener exiting");
    }
}

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router.
fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: usize,
    warning_count: &Arc<AtomicU32>,
) -> KvCacheEvent {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
        } => {
            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            KvCacheEvent {
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
            }
        }
        RawKvEvent::BlockRemoved { block_hashes } => {
            let hashes = block_hashes
                .into_iter()
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
        },
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

pub struct WorkerMetricsPublisher {
    tx: tokio::sync::watch::Sender<Arc<ForwardPassMetrics>>,
    rx: tokio::sync::watch::Receiver<Arc<ForwardPassMetrics>>,
}

impl WorkerMetricsPublisher {
    pub fn new() -> Result<Self> {
        let (tx, rx) = tokio::sync::watch::channel(Arc::new(ForwardPassMetrics::default()));
        Ok(WorkerMetricsPublisher { tx, rx })
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
        assert!(matches!(out.data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![123, 456],
        };
        let out = convert_event(raw_evt, 7, kv_block_size, &Arc::new(AtomicU32::new(0)));

        assert!(matches!(out.data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        let out = convert_event(raw_evt, 1, kv_block_size, &Arc::new(AtomicU32::new(0)));
        assert!(matches!(out.data, KvCacheEventData::Cleared));
    }
}

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use crate::kv_router::protocols::ExternalSequenceBlockHash;
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
    // Test start_event_processor
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor() {
        let (component, published) = MockComponent::new();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
            }),
        };

        let token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        tx.send(event).unwrap();
        drop(tx);

        let handle = tokio::spawn(start_event_processor(component, 1, token, rx));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        let published = published.lock().unwrap();
        assert_eq!(published.len(), 1);
        let (subject, _) = &published[0];
        assert_eq!(subject, &KV_EVENT_SUBJECT.to_string());
    }

    //--------------------------------------------------------------------
    // Test start_zmq_listener without a real socket
    //   (feed it frames through a ZMQ PAIR tcp socket)
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_listener_pushes_to_channel() {
        // Prepare channel that listener should fill
        let (tx, mut rx) = mpsc::unbounded_channel::<KvCacheEvent>();

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
            start_zmq_listener(endpoint.to_string(), topic, tx, token, 4)
        });

        // Give time for the connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send synthetic 3-frame message: [topic, seq(8B), payload]
        let seq: u64 = 77;

        let events = vec![RawKvEvent::BlockStored {
            block_hashes: vec![42],
            parent_block_hash: None,
            token_ids: vec![0, 1, 2, 3],
            block_size: 4,
            lora_id: None,
        }];

        let batch = KvEventBatch { ts: 0.0, events };

        let payload = Bytes::from(rmps::to_vec(&batch).unwrap());

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
        let event = rx.try_recv().expect("no message received");

        let KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks,
        }) = event.data
        else {
            panic!("expected KvCacheStoreData");
        };

        assert!(parent_hash.is_none());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_hash.0, 42);

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
