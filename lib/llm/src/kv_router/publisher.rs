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
use dynamo_runtime::traits::{events::EventPublisher, DistributedRuntimeProvider, RuntimeProvider};
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

use rmp_serde as rmps;
use serde::Deserialize;
use serde::Serialize;
use std::sync::atomic::{AtomicU32, Ordering};
use zeromq::{Socket, SocketRecv, SubSocket};

static WARNING_COUNT: AtomicU32 = AtomicU32::new(0);

// -------------------------------------------------------------------------
// KV Event Publishers -----------------------------------------------------
// -------------------------------------------------------------------------

pub struct KvEventPublisher {
    kv_block_size: usize,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
}

impl KvEventPublisher {
    pub fn new(component: Component, worker_id: i64, kv_block_size: usize) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel::<KvCacheEvent>();
        let p = KvEventPublisher { tx, kv_block_size };

        start_publish_task(component, worker_id, rx);
        Ok(p)
    }

    pub fn publish(&self, event: KvCacheEvent) -> Result<(), mpsc::error::SendError<KvCacheEvent>> {
        tracing::info!("Publish event: {:?}", event);
        self.tx.send(event)
    }

    pub fn kv_block_size(&self) -> usize {
        self.kv_block_size
    }
}

fn start_publish_task(
    component: Component,
    worker_id: i64,
    mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
) {
    let component_clone = component.clone();
    tracing::info!("Publishing KV Events to subject: {}", KV_EVENT_SUBJECT);

    _ = component.drt().runtime().secondary().spawn(async move {
        while let Some(event) = rx.recv().await {
            let router_event = RouterEvent::new(worker_id, event);
            component_clone
                .publish(KV_EVENT_SUBJECT, &router_event)
                .await
                .unwrap();
        }
    });
}

pub struct KvEventPublisherFromZmq {
    kv_block_size: usize,
    zmq_handle: Option<tokio::task::JoinHandle<()>>,
    zmq_token: Option<dynamo_runtime::CancellationToken>,
}

impl KvEventPublisherFromZmq {
    pub fn new(kv_block_size: usize) -> Self {
        Self {
            kv_block_size,
            zmq_handle: None,
            zmq_token: None,
        }
    }

    pub fn start_background_task(
        &mut self,
        component: Component,
        worker_id: i64,
        zmq_endpoint: String,
        zmq_topic: String,
    ) {
        let kv_block_size = self.kv_block_size;
        let (raw_tx, raw_rx) = mpsc::unbounded_channel::<(u64, Vec<u8>)>();

        let zmq_token = component.rt().child_token();
        self.zmq_token = Some(zmq_token.clone());

        // Spawn async ZMQ listener
        self.zmq_handle = Some(
            component
                .drt()
                .runtime()
                .secondary()
                .spawn(start_zmq_listener(
                    zmq_endpoint,
                    zmq_topic,
                    raw_tx,
                    zmq_token,
                )),
        );

        component
            .drt()
            .runtime()
            .secondary()
            .spawn(start_event_processor(
                raw_rx,
                component,
                worker_id,
                kv_block_size,
            ));
    }

    pub fn shutdown(&mut self) {
        if let Some(token) = self.zmq_token.take() {
            token.cancel();
        }
        if let Some(handle) = self.zmq_handle.take() {
            handle.abort();
        }
    }
}

async fn start_event_processor<P: EventPublisher>(
    mut raw_rx: mpsc::UnboundedReceiver<(u64, Vec<u8>)>,
    component: P,
    worker_id: i64,
    kv_block_size: usize,
) {
    while let Some((seq, payload)) = raw_rx.recv().await {
        match rmps::from_slice::<KvEventBatch>(&payload) {
            Ok(batch) => {
                for raw_evt in batch.events.into_iter() {
                    if let Some(event) = convert_event(raw_evt, seq, kv_block_size) {
                        let router_event = RouterEvent::new(worker_id, event);
                        if let Err(e) = component.publish(KV_EVENT_SUBJECT, &router_event).await {
                            tracing::warn!("Failed to publish router event: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Failed to decode KVEventBatch msgpack: {}", e);
            }
        }
    }
}

async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    raw_tx: mpsc::UnboundedSender<(u64, Vec<u8>)>,
    zmq_token: dynamo_runtime::CancellationToken,
) {
    tracing::info!(
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

    loop {
        tokio::select! {
            // Check for cancellation
            _ = zmq_token.cancelled() => {
                tracing::info!("ZMQ listener received cancellation signal");
                break;
            }

            // Receive message
            msg_result = socket.recv() => {
                match msg_result {
                    Ok(msg) => {
                        // We expect multipart frames: [topic, seq, payload]
                        let mut frames: Vec<Vec<u8>> = msg.into_vec().into_iter().map(|frame| frame.to_vec()).collect();

                        if frames.len() == 3 {
                            let payload = frames.remove(2);
                            let seq_bytes = frames.remove(1);

                            if seq_bytes.len() != 8 {
                                tracing::warn!("Invalid sequence number frame len={}", seq_bytes.len());
                                continue;
                            }

                            let seq = u64::from_be_bytes(seq_bytes.try_into().unwrap());
                            if raw_tx.send((seq, payload)).is_err() {
                                tracing::warn!("Failed to send message to channel - receiver dropped");
                                break;
                            }
                        } else {
                            tracing::warn!("Received unexpected ZMQ frame count: {}", frames.len());
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Error reading from ZMQ socket: {}", e);
                        // Brief sleep to avoid tight error loop
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }
                }
            }
        }
    }
    tracing::info!("ZMQ listener exiting");
}

/// Convert a raw event coming from the ZMQ channel into the internal
/// [`KvCacheEvent`] representation used by the router. Returns `None` when the
/// event cannot be represented with the current protocol (e.g., we ignore
/// `AllBlocksCleared` until a concrete format is defined).
fn convert_event(raw: RawKvEvent, event_id: u64, kv_block_size: usize) -> Option<KvCacheEvent> {
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
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for (num_tokens_it, block_hash_it) in num_block_tokens.iter().zip(block_hashes.iter()) {
        if *num_tokens_it != kv_block_size as u64 {
            if WARNING_COUNT.fetch_add(1, Ordering::Relaxed) < 3 {
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
    #[serde(rename = "BlockStored")]
    BlockStored {
        block_hashes: Vec<i64>,
        parent_block_hash: Option<i64>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_id: Option<u64>,
    },
    #[serde(rename = "BlockRemoved")]
    BlockRemoved { block_hashes: Vec<i64> },
    #[serde(rename = "AllBlocksCleared")]
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

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            /*lora_id=*/ 0,
        );

        // should early-exit as second has mismatch
        assert!(blocks.len() == 1);
        assert!(WARNING_COUNT.load(Ordering::Relaxed) == 1)
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

        let out = convert_event(raw_evt, 42, kv_block_size);
        assert!(matches!(out.unwrap().data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![123, 456],
        };
        let out = convert_event(raw_evt, 7, kv_block_size);

        assert!(matches!(out.unwrap().data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        assert!(convert_event(raw_evt, 1, kv_block_size).is_none());
    }
}

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use async_trait;
    use bytes::Bytes;
    use std::sync::{Arc, Mutex};
    use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

    //--------------------------------------------------------------------
    // A tiny stand-in for Component that just records every publish call
    //--------------------------------------------------------------------
    #[derive(Default)]
    struct MockComponent {
        published: Arc<Mutex<Vec<(String, Vec<u8>)>>>,
    }

    impl MockComponent {
        fn new() -> (Self, Arc<Mutex<Vec<(String, Vec<u8>)>>>) {
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
    // Test start_event_processor in isolation
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor_sends_router_event() {
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

        // 2) channel feeding the processor
        let (tx, rx) = mpsc::unbounded_channel::<(u64, Vec<u8>)>();
        tx.send((123, payload.clone())).unwrap(); // seq = 123
        drop(tx);

        // 3) mock component to capture output
        let (comp, published) = MockComponent::new();

        // 4) run the function under test (let it consume exactly one msg)
        let handle = tokio::spawn(start_event_processor(rx, comp, worker_id, kv_block_size));

        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // 5) assert we have exactly one RouterEvent pushed with right worker_id
        let published: std::sync::MutexGuard<'_, Vec<(String, Vec<u8>)>> =
            published.lock().unwrap();
        let (subject, bytes) = &published[0];

        assert_eq!(subject, &KV_EVENT_SUBJECT.to_string());
        assert_eq!(bytes.get(0), payload.get(0))
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
