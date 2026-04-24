// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-owned conditional-disaggregation session protocol and helpers.
//!
//! Hub-visible request metadata lives in `kvbm-disagg-protocol`. This module
//! owns the engine-to-engine session messages, remote block-set model, metadata
//! cache, and velo-streaming bidi session implementation.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Context as _, Result, anyhow};
use dashmap::{DashMap, DashSet};
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture};
use kvbm_disagg_protocol::{SessionEndpoint, SessionId};
use kvbm_logical::blocks::ImmutableBlock;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as TokioMutex, mpsc, oneshot};

use crate::{BlockId, G2, InstanceId, LogicalLayoutHandle, SequenceHash};

/// Session endpoint kind for a bidirectional session composed from two
/// velo-streaming SPSC anchors.
pub const CONDITIONAL_DISAGG_STREAM_SCHEMA: &str = "kvbm_conditional_disagg_v1";

/// Serializable reference to a block made available through a disagg session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBlockRef {
    pub block_id: BlockId,
    #[serde(with = "serde_hash::single")]
    pub sequence_hash: SequenceHash,
}

/// A set of blocks that all share the same remote logical source layout.
///
/// Workers resolve `(peer instance, worker rank, source_layout)` through
/// imported `SerializedLayout` metadata before executing the transfer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBlockSet {
    pub source_layout: LogicalLayoutHandle,
    pub blocks: Vec<RemoteBlockRef>,
}

/// Select all session blocks, or a specific hash subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashSelection {
    All,
    Hashes(Vec<SequenceHash>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HashSelectionKind {
    All,
    Hashes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct HashSelectionWire {
    #[serde(rename = "type")]
    kind: HashSelectionKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty", with = "serde_hash::vec")]
    hashes: Vec<SequenceHash>,
}

impl Serialize for HashSelection {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let wire = match self {
            Self::All => HashSelectionWire {
                kind: HashSelectionKind::All,
                hashes: Vec::new(),
            },
            Self::Hashes(hashes) => HashSelectionWire {
                kind: HashSelectionKind::Hashes,
                hashes: hashes.clone(),
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for HashSelection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = HashSelectionWire::deserialize(deserializer)?;
        match wire.kind {
            HashSelectionKind::All => Ok(Self::All),
            HashSelectionKind::Hashes => Ok(Self::Hashes(wire.hashes)),
        }
    }
}

/// Ask the holder for ready block sets for all or a subset of hashes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSetRequest {
    pub request_id: String,
    pub hashes: HashSelection,
}

/// Holder response to a block-set request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSetResponse {
    pub request_id: String,
    pub ready: Vec<RemoteBlockSet>,
    #[serde(with = "serde_hash::vec")]
    pub pending_hashes: Vec<SequenceHash>,
}

/// Backward-compatible names for the initial descriptor terminology.
#[deprecated(since = "0.1.0", note = "use BlockSetRequest instead")]
pub type DescriptorRequest = BlockSetRequest;
#[deprecated(since = "0.1.0", note = "use BlockSetRequest instead")]
pub type DescriptorResponse = BlockSetResponse;

/// Request that the peer release session-held pins for all or a subset of
/// blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnpinRequest {
    pub request_id: String,
    pub hashes: HashSelection,
}

/// Mandatory acknowledgement after a session unpin request has been applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnpinAck {
    pub request_id: String,
    pub hashes: HashSelection,
}

/// The puller completed an RDMA pull. The corresponding `PullAck` proves the
/// holder remained live long enough to observe completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullComplete {
    pub pull_id: u64,
    #[serde(with = "serde_hash::vec")]
    pub hashes: Vec<SequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullAck {
    pub pull_id: u64,
}

/// Frames sent from a decode worker to a prefill worker over a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecodeToPrefillFrame {
    BlockSetResponse(BlockSetResponse),
    /// Compatibility variant for older descriptor terminology.
    DescriptorResponse(DescriptorResponse),
    UnpinRequest(UnpinRequest),
    UnpinAck(UnpinAck),
    PullComplete(PullComplete),
    PullAck(PullAck),
    BlockSetsReady {
        block_sets: Vec<RemoteBlockSet>,
    },
    /// Compatibility variant for older descriptor terminology.
    BlocksReady {
        blocks: Vec<RemoteBlockRef>,
    },
    OutputBlocksPulled {
        #[serde(with = "serde_hash::vec")]
        hashes: Vec<SequenceHash>,
    },
    Detach,
    Error {
        message: String,
    },
}

/// Frames sent from a prefill worker to a decode worker over a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PrefillToDecodeFrame {
    Attach {
        #[serde(with = "serde_instance_id_string")]
        prefill_instance_id: InstanceId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefill_endpoint: Option<SessionEndpoint>,
    },
    BlockSetRequest(BlockSetRequest),
    /// Compatibility variant for older descriptor terminology.
    DescriptorRequest(DescriptorRequest),
    UnpinRequest(UnpinRequest),
    UnpinAck(UnpinAck),
    PullComplete(PullComplete),
    PullAck(PullAck),
    InitialBlocksPulled {
        #[serde(with = "serde_hash::vec")]
        hashes: Vec<SequenceHash>,
    },
    OutputBlockSetsReady {
        block_sets: Vec<RemoteBlockSet>,
    },
    /// Compatibility variant for older descriptor terminology.
    OutputBlocksReady {
        blocks: Vec<RemoteBlockRef>,
    },
    Detach,
    Error {
        message: String,
    },
}

mod serde_instance_id_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;
    use velo::InstanceId;

    pub fn serialize<S>(id: &InstanceId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InstanceId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value)
            .map(InstanceId::from)
            .map_err(D::Error::custom)
    }
}

/// Serde helpers for [`SequenceHash`] in velo wire frames.
///
/// rmp-serde (MessagePack) has no u128 type. We serialize each hash as a
/// `[u64; 2]` pair (hi word, lo word), which every serializer handles natively.
///
/// SAFETY for `from_u128`: `SequenceHash = PositionalLineageHash(u128)` is a
/// single-field tuple struct. Rust guarantees this has the same size and
/// alignment as u128, making the transmute sound for any bit pattern.
mod serde_hash {
    use super::SequenceHash;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};

    #[inline]
    fn to_pair(hash: SequenceHash) -> [u64; 2] {
        let v = hash.as_u128();
        [(v >> 64) as u64, v as u64]
    }

    #[inline]
    fn from_pair([hi, lo]: [u64; 2]) -> SequenceHash {
        let v: u128 = ((hi as u128) << 64) | lo as u128;
        // SAFETY: see module-level comment.
        unsafe { std::mem::transmute::<u128, SequenceHash>(v) }
    }

    pub mod single {
        use super::*;

        pub fn serialize<S: Serializer>(hash: &SequenceHash, s: S) -> Result<S::Ok, S::Error> {
            to_pair(*hash).serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SequenceHash, D::Error> {
            let pair = <[u64; 2]>::deserialize(d)?;
            Ok(from_pair(pair))
        }
    }

    pub mod vec {
        use super::*;

        pub fn serialize<S: Serializer>(
            hashes: &Vec<SequenceHash>,
            s: S,
        ) -> Result<S::Ok, S::Error> {
            let pairs: Vec<[u64; 2]> = hashes.iter().map(|h| to_pair(*h)).collect();
            pairs.serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            d: D,
        ) -> Result<Vec<SequenceHash>, D::Error> {
            let pairs = Vec::<[u64; 2]>::deserialize(d)
                .map_err(|e| D::Error::custom(format!("serde_hash::vec: {e}")))?;
            Ok(pairs.into_iter().map(from_pair).collect())
        }
    }
}

/// Stream of session events consumed by connector/coordinator monitor tasks.
pub type SessionEventStream = Pin<Box<dyn Stream<Item = SessionEvent> + Send + 'static>>;

// /// Convert native KVBM sequence hashes to the JSON-safe protocol form.
// pub fn hash_to_wire(hash: SequenceHash) -> SequenceHash {
//     hash.as_u128().to_string()
// }

pub fn hashes_to_wire(hashes: impl IntoIterator<Item = SequenceHash>) -> Vec<SequenceHash> {
    hashes.into_iter().collect()
}

/// Blocks used to seed a decode-side remote-prefill session.
#[derive(Debug, Clone, Default)]
pub struct SessionBlocks {
    /// Blocks already ready in local G2 and available for block-set requests.
    pub ready_g2: Vec<ImmutableBlock<G2>>,
    /// Hashes known to this side but not yet ready as G2 block refs.
    pub pending_hashes: Vec<SequenceHash>,
}

impl SessionBlocks {
    pub fn new(ready_g2: Vec<ImmutableBlock<G2>>, pending_hashes: Vec<SequenceHash>) -> Self {
        Self {
            ready_g2,
            pending_hashes,
        }
    }

    pub fn ready_hashes_wire(&self) -> Vec<SequenceHash> {
        self.ready_g2
            .iter()
            .map(|block| block.sequence_hash())
            .collect()
    }

    pub fn pending_hashes_wire(&self) -> Vec<SequenceHash> {
        hashes_to_wire(self.pending_hashes.iter().copied())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    Attached {
        peer_instance_id: InstanceId,
    },
    BlockSetRequest(BlockSetRequest),
    #[deprecated(since = "0.1.0", note = "use BlockSetRequest instead")]
    DescriptorRequest(BlockSetRequest),
    UnpinRequested(UnpinRequest),
    UnpinAcked(UnpinAck),
    PullComplete(PullComplete),
    PullAcked(PullAck),
    BlockSetsAdded {
        block_sets: Vec<RemoteBlockSet>,
    },
    Detached {
        reason: Option<String>,
    },
    Failed {
        reason: String,
    },
}

/// Abstraction over a bidirectional decode/prefill session.
pub trait PrefillSession: Send + Sync {
    fn session_id(&self) -> SessionId;

    fn endpoint(&self) -> Option<SessionEndpoint>;

    fn add_ready_blocks(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()>;

    fn add_pending_hashes(&self, hashes: Vec<SequenceHash>) -> Result<()>;

    // TODO(cd-review): seems like this can only be called once? perhaps the return type shoudl be an Option/Result?
    fn subscribe(&self) -> SessionEventStream;

    fn respond_to_block_set_request(&self, response: BlockSetResponse) -> Result<()>;

    /// Release session-owned pins matching `selection`. This must not release
    /// coordinator-owned references to the same blocks.
    fn release_session_pins(&self, selection: &HashSelection) -> Result<Vec<SequenceHash>>;

    fn ack_unpin(&self, ack: UnpinAck) -> Result<()>;

    /// Local-initiated unpin protocol. Included now so future decode output
    /// pull paths can require a peer ack before releasing session pins.
    fn request_unpin(&self, request: UnpinRequest) -> BoxFuture<'static, Result<UnpinAck>>;

    fn close(&self, reason: Option<String>);
}

pub trait PrefillSessionFactory: Send + Sync {
    fn create_decode(&self, session_id: SessionId) -> Result<Arc<dyn PrefillSession>>;
}

struct EventStream {
    receiver: mpsc::UnboundedReceiver<SessionEvent>,
}

impl Stream for EventStream {
    type Item = SessionEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

#[derive(Default)]
struct SessionBlockState {
    ready_blocks: Vec<ImmutableBlock<G2>>,
    pending_hashes: Vec<SequenceHash>,
}

struct PendingState {
    block_sets: HashMap<String, oneshot::Sender<Result<BlockSetResponse>>>,
    unpins: HashMap<String, oneshot::Sender<Result<UnpinAck>>>,
    pulls: HashMap<u64, oneshot::Sender<Result<PullAck>>>,
}

impl PendingState {
    fn new() -> Self {
        Self {
            block_sets: HashMap::new(),
            unpins: HashMap::new(),
            pulls: HashMap::new(),
        }
    }
}

struct DisaggSessionInner {
    session_id: SessionId,
    velo: Arc<velo::Velo>,
    endpoint: SessionEndpoint,
    event_tx: mpsc::UnboundedSender<SessionEvent>,
    event_rx: ParkingMutex<Option<mpsc::UnboundedReceiver<SessionEvent>>>,
    block_state: ParkingMutex<SessionBlockState>,
    pending: ParkingMutex<PendingState>,
    decode_tx: TokioMutex<Option<velo::StreamSender<DecodeToPrefillFrame>>>,
    prefill_tx: TokioMutex<Option<velo::StreamSender<PrefillToDecodeFrame>>>,
}

/// Concrete conditional-disaggregation session backed by two velo SPSC streams.
#[derive(Clone)]
pub struct DisaggSession {
    inner: Arc<DisaggSessionInner>,
}

impl DisaggSession {
    /// Create the decode side of a CD bidi session. The returned endpoint is
    /// carried in `RemotePrefillRequest.decode_endpoint`.
    pub fn create_decode(velo: Arc<velo::Velo>, session_id: SessionId) -> Arc<Self> {
        let anchor = velo.create_anchor::<PrefillToDecodeFrame>();
        let endpoint = endpoint_from_handle(anchor.handle());
        let session = Self::new(velo, session_id, endpoint);
        spawn_prefill_to_decode_monitor(Arc::clone(&session.inner), anchor);
        Arc::new(session)
    }

    /// Create and attach the prefill side to a decode endpoint pulled from
    /// the hub queue.
    pub async fn attach_prefill(
        velo: Arc<velo::Velo>,
        session_id: SessionId,
        decode_endpoint: &SessionEndpoint,
    ) -> Result<Arc<Self>> {
        let decode_handle = handle_from_endpoint(decode_endpoint)?;
        let p_to_d = velo
            .attach_anchor::<PrefillToDecodeFrame>(decode_handle)
            .await
            .context("attach prefill sender to decode endpoint")?;

        let anchor = velo.create_anchor::<DecodeToPrefillFrame>();
        let endpoint = endpoint_from_handle(anchor.handle());
        let session = Self::new(Arc::clone(&velo), session_id, endpoint.clone());
        {
            let mut sender = session.inner.prefill_tx.lock().await;
            *sender = Some(p_to_d);
        }
        spawn_decode_to_prefill_monitor(Arc::clone(&session.inner), anchor);

        session
            .send_prefill(PrefillToDecodeFrame::Attach {
                prefill_instance_id: velo.instance_id(),
                prefill_endpoint: Some(endpoint),
            })
            .await
            .context("send prefill attach frame")?;

        Ok(Arc::new(session))
    }

    fn new(velo: Arc<velo::Velo>, session_id: SessionId, endpoint: SessionEndpoint) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        Self {
            inner: Arc::new(DisaggSessionInner {
                session_id,
                velo,
                endpoint,
                event_tx,
                event_rx: ParkingMutex::new(Some(event_rx)),
                block_state: ParkingMutex::new(SessionBlockState::default()),
                pending: ParkingMutex::new(PendingState::new()),
                decode_tx: TokioMutex::new(None),
                prefill_tx: TokioMutex::new(None),
            }),
        }
    }

    pub fn session_id(&self) -> SessionId {
        self.inner.session_id
    }

    pub fn endpoint(&self) -> SessionEndpoint {
        self.inner.endpoint.clone()
    }

    pub fn subscribe(&self) -> SessionEventStream {
        let receiver = self
            .inner
            .event_rx
            .lock()
            .take()
            .expect("DisaggSession::subscribe called twice");
        Box::pin(EventStream { receiver })
    }

    pub async fn request_block_sets(&self, request: BlockSetRequest) -> Result<BlockSetResponse> {
        let request_id = request.request_id.clone();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.inner.pending.lock();
            if pending.block_sets.insert(request_id.clone(), tx).is_some() {
                anyhow::bail!("block-set request already pending: {request_id}");
            }
        }
        if let Err(err) = self
            .send_prefill(PrefillToDecodeFrame::BlockSetRequest(request))
            .await
        {
            self.inner.pending.lock().block_sets.remove(&request_id);
            return Err(err);
        }
        rx.await
            .context("block-set response channel closed before ack")?
    }

    pub async fn request_descriptors(&self, request: BlockSetRequest) -> Result<BlockSetResponse> {
        self.request_block_sets(request).await
    }

    pub async fn respond_to_block_set_request(&self, response: BlockSetResponse) -> Result<()> {
        self.send_decode(DecodeToPrefillFrame::BlockSetResponse(response))
            .await
    }

    pub async fn respond_to_descriptor_request(&self, response: BlockSetResponse) -> Result<()> {
        self.respond_to_block_set_request(response).await
    }

    pub async fn request_unpin_from_prefill(&self, request: UnpinRequest) -> Result<UnpinAck> {
        let request_id = request.request_id.clone();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.inner.pending.lock();
            if pending.unpins.insert(request_id.clone(), tx).is_some() {
                anyhow::bail!("unpin request already pending: {request_id}");
            }
        }
        if let Err(err) = self
            .send_prefill(PrefillToDecodeFrame::UnpinRequest(request))
            .await
        {
            self.inner.pending.lock().unpins.remove(&request_id);
            return Err(err);
        }
        rx.await.context("unpin ack channel closed before ack")?
    }

    pub async fn ack_unpin_from_decode(&self, ack: UnpinAck) -> Result<()> {
        self.send_decode(DecodeToPrefillFrame::UnpinAck(ack)).await
    }

    pub async fn publish_output_block_sets(&self, block_sets: Vec<RemoteBlockSet>) -> Result<()> {
        self.send_prefill(PrefillToDecodeFrame::OutputBlockSetsReady { block_sets })
            .await
    }

    pub async fn pull_complete_from_decode(&self, complete: PullComplete) -> Result<PullAck> {
        let pull_id = complete.pull_id;
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.inner.pending.lock();
            if pending.pulls.insert(pull_id, tx).is_some() {
                anyhow::bail!("pull already pending: {pull_id}");
            }
        }
        if let Err(err) = self
            .send_decode(DecodeToPrefillFrame::PullComplete(complete))
            .await
        {
            self.inner.pending.lock().pulls.remove(&pull_id);
            return Err(err);
        }
        rx.await.context("pull ack channel closed before ack")?
    }

    pub async fn ack_pull_from_prefill(&self, ack: PullAck) -> Result<()> {
        self.send_prefill(PrefillToDecodeFrame::PullAck(ack)).await
    }

    pub async fn finalize(&self) -> Result<()> {
        let prefill_tx = self.inner.prefill_tx.lock().await.take();
        let decode_tx = self.inner.decode_tx.lock().await.take();
        if let Some(tx) = prefill_tx {
            tx.finalize()
                .map_err(|err| anyhow!("finalize prefill sender: {err}"))?;
        }
        if let Some(tx) = decode_tx {
            tx.finalize()
                .map_err(|err| anyhow!("finalize decode sender: {err}"))?;
        }
        Ok(())
    }

    async fn send_decode(&self, frame: DecodeToPrefillFrame) -> Result<()> {
        let sender = self.inner.decode_tx.lock().await;
        let Some(sender) = sender.as_ref() else {
            anyhow::bail!("decode->prefill stream is not attached");
        };
        sender
            .send(frame)
            .await
            .map_err(|err| anyhow!("send decode->prefill frame: {err}"))
    }

    async fn send_prefill(&self, frame: PrefillToDecodeFrame) -> Result<()> {
        let sender = self.inner.prefill_tx.lock().await;
        let Some(sender) = sender.as_ref() else {
            anyhow::bail!("prefill->decode stream is not attached");
        };
        sender
            .send(frame)
            .await
            .map_err(|err| anyhow!("send prefill->decode frame: {err}"))
    }
}

impl PrefillSession for DisaggSession {
    fn session_id(&self) -> SessionId {
        self.session_id()
    }

    fn endpoint(&self) -> Option<SessionEndpoint> {
        Some(self.endpoint())
    }

    fn add_ready_blocks(&self, blocks: Vec<ImmutableBlock<G2>>) -> Result<()> {
        self.inner.block_state.lock().ready_blocks.extend(blocks);
        Ok(())
    }

    fn add_pending_hashes(&self, hashes: Vec<SequenceHash>) -> Result<()> {
        self.inner.block_state.lock().pending_hashes.extend(hashes);
        Ok(())
    }

    fn subscribe(&self) -> SessionEventStream {
        self.subscribe()
    }

    fn respond_to_block_set_request(&self, response: BlockSetResponse) -> Result<()> {
        let session = self.clone();
        tokio::spawn(async move {
            if let Err(err) = session.respond_to_block_set_request(response).await {
                session.emit_failed(format!("block-set response failed: {err}"));
            }
        });
        Ok(())
    }

    fn release_session_pins(&self, selection: &HashSelection) -> Result<Vec<SequenceHash>> {
        let mut state = self.inner.block_state.lock();
        let mut released = Vec::new();
        match selection {
            HashSelection::All => {
                released = state
                    .ready_blocks
                    .iter()
                    .map(|block| block.sequence_hash())
                    .collect();
                state.ready_blocks.clear();
            }
            HashSelection::Hashes(hashes) => {
                let selected: std::collections::HashSet<_> = hashes.iter().cloned().collect();
                state.ready_blocks.retain(|block| {
                    let hash = block.sequence_hash();
                    if selected.contains(&hash) {
                        released.push(hash);
                        false
                    } else {
                        true
                    }
                });
            }
        }
        released.sort();
        Ok(released)
    }

    fn ack_unpin(&self, ack: UnpinAck) -> Result<()> {
        let session = self.clone();
        tokio::spawn(async move {
            if let Err(err) = session.ack_unpin_from_decode(ack).await {
                session.emit_failed(format!("unpin ack failed: {err}"));
            }
        });
        Ok(())
    }

    fn request_unpin(&self, request: UnpinRequest) -> BoxFuture<'static, Result<UnpinAck>> {
        let session = self.clone();
        async move {
            let request_id = request.request_id.clone();
            let (tx, rx) = oneshot::channel();
            {
                let mut pending = session.inner.pending.lock();
                if pending.unpins.insert(request_id.clone(), tx).is_some() {
                    anyhow::bail!("unpin request already pending: {request_id}");
                }
            }
            if let Err(err) = session
                .send_decode(DecodeToPrefillFrame::UnpinRequest(request))
                .await
            {
                session.inner.pending.lock().unpins.remove(&request_id);
                return Err(err);
            }
            rx.await.context("unpin ack channel closed before ack")?
        }
        .boxed()
    }

    fn close(&self, reason: Option<String>) {
        let session = self.clone();
        tokio::spawn(async move {
            if let Some(reason) = reason {
                session
                    .emit_event(SessionEvent::Detached {
                        reason: Some(reason),
                    })
                    .ok();
            }
            if let Err(err) = session.finalize().await {
                session.emit_failed(format!("session close failed: {err}"));
            }
        });
    }
}

/// Factory for decode-side `DisaggSession`s.
pub struct VeloPrefillSessionFactory {
    velo: Arc<velo::Velo>,
}

impl VeloPrefillSessionFactory {
    pub fn new(velo: Arc<velo::Velo>) -> Arc<Self> {
        Arc::new(Self { velo })
    }
}

impl PrefillSessionFactory for VeloPrefillSessionFactory {
    fn create_decode(&self, session_id: SessionId) -> Result<Arc<dyn PrefillSession>> {
        Ok(DisaggSession::create_decode(
            Arc::clone(&self.velo),
            session_id,
        ))
    }
}

impl DisaggSession {
    fn emit_event(&self, event: SessionEvent) -> Result<()> {
        self.inner
            .event_tx
            .send(event)
            .map_err(|err| anyhow!("session event receiver closed: {err}"))
    }

    fn emit_failed(&self, reason: String) {
        let _ = self.emit_event(SessionEvent::Failed { reason });
    }
}

impl DisaggSessionInner {
    fn emit_event(&self, event: SessionEvent) {
        let _ = self.event_tx.send(event);
    }

    fn fail_pending(&self, reason: &str) {
        let mut pending = self.pending.lock();
        for (_, tx) in pending.block_sets.drain() {
            let _ = tx.send(Err(anyhow!(reason.to_string())));
        }
        for (_, tx) in pending.unpins.drain() {
            let _ = tx.send(Err(anyhow!(reason.to_string())));
        }
        for (_, tx) in pending.pulls.drain() {
            let _ = tx.send(Err(anyhow!(reason.to_string())));
        }
    }
}

fn endpoint_from_handle(handle: velo::StreamAnchorHandle) -> SessionEndpoint {
    SessionEndpoint {
        kind: CONDITIONAL_DISAGG_STREAM_SCHEMA.to_string(),
        payload: serde_json::to_value(handle).expect("serialize stream anchor handle"),
    }
}

fn handle_from_endpoint(endpoint: &SessionEndpoint) -> Result<velo::StreamAnchorHandle> {
    if endpoint.kind != CONDITIONAL_DISAGG_STREAM_SCHEMA {
        anyhow::bail!(
            "unsupported disagg session endpoint kind: {}",
            endpoint.kind
        );
    }
    serde_json::from_value(endpoint.payload.clone()).context("decode stream anchor handle")
}

fn spawn_prefill_to_decode_monitor(
    inner: Arc<DisaggSessionInner>,
    mut anchor: velo::StreamAnchor<PrefillToDecodeFrame>,
) {
    tokio::spawn(async move {
        while let Some(frame) = anchor.next().await {
            match frame {
                Ok(velo::StreamFrame::Item(frame)) => {
                    handle_prefill_to_decode_frame(Arc::clone(&inner), frame).await;
                }
                Ok(velo::StreamFrame::Finalized) => {
                    inner.emit_event(SessionEvent::Detached { reason: None });
                    inner.fail_pending("prefill->decode stream finalized");
                    break;
                }
                Ok(velo::StreamFrame::Detached) => {
                    inner.emit_event(SessionEvent::Detached {
                        reason: Some("prefill->decode stream detached".to_string()),
                    });
                    break;
                }
                Ok(other) => {
                    let reason = format!("unexpected prefill->decode stream frame: {other:?}");
                    inner.emit_event(SessionEvent::Failed {
                        reason: reason.clone(),
                    });
                    inner.fail_pending(&reason);
                    break;
                }
                Err(err) => {
                    let reason = format!("prefill->decode stream error: {err}");
                    inner.emit_event(SessionEvent::Failed {
                        reason: reason.clone(),
                    });
                    inner.fail_pending(&reason);
                    break;
                }
            }
        }
    });
}

async fn handle_prefill_to_decode_frame(
    inner: Arc<DisaggSessionInner>,
    frame: PrefillToDecodeFrame,
) {
    match frame {
        PrefillToDecodeFrame::Attach {
            prefill_instance_id,
            prefill_endpoint,
        } => {
            let Some(endpoint) = prefill_endpoint else {
                inner.emit_event(SessionEvent::Failed {
                    reason: "prefill attach omitted endpoint".to_string(),
                });
                return;
            };
            match attach_decode_sender(&inner, &endpoint).await {
                Ok(()) => inner.emit_event(SessionEvent::Attached {
                    peer_instance_id: prefill_instance_id,
                }),
                Err(err) => inner.emit_event(SessionEvent::Failed {
                    reason: format!("decode failed to attach prefill endpoint: {err}"),
                }),
            }
        }
        PrefillToDecodeFrame::BlockSetRequest(request) => {
            inner.emit_event(SessionEvent::BlockSetRequest(request));
        }
        PrefillToDecodeFrame::DescriptorRequest(request) => {
            inner.emit_event(SessionEvent::DescriptorRequest(request));
        }
        PrefillToDecodeFrame::UnpinRequest(request) => {
            inner.emit_event(SessionEvent::UnpinRequested(request));
        }
        PrefillToDecodeFrame::UnpinAck(ack) => {
            complete_unpin(&inner, ack.clone());
            inner.emit_event(SessionEvent::UnpinAcked(ack));
        }
        PrefillToDecodeFrame::PullComplete(complete) => {
            inner.emit_event(SessionEvent::PullComplete(complete));
        }
        PrefillToDecodeFrame::PullAck(ack) => {
            complete_pull(&inner, ack.clone());
            inner.emit_event(SessionEvent::PullAcked(ack));
        }
        PrefillToDecodeFrame::InitialBlocksPulled { hashes } => {
            inner.emit_event(SessionEvent::PullComplete(PullComplete {
                pull_id: 0,
                hashes,
            }));
        }
        PrefillToDecodeFrame::OutputBlockSetsReady { block_sets } => {
            inner.emit_event(SessionEvent::BlockSetsAdded { block_sets });
        }
        PrefillToDecodeFrame::OutputBlocksReady { blocks } => {
            inner.emit_event(SessionEvent::BlockSetsAdded {
                block_sets: vec![RemoteBlockSet {
                    source_layout: LogicalLayoutHandle::G2,
                    blocks,
                }],
            });
        }
        PrefillToDecodeFrame::Detach => {
            inner.emit_event(SessionEvent::Detached {
                reason: Some("prefill requested detach".to_string()),
            });
        }
        PrefillToDecodeFrame::Error { message } => {
            inner.emit_event(SessionEvent::Failed { reason: message });
        }
    }
}

async fn attach_decode_sender(
    inner: &Arc<DisaggSessionInner>,
    endpoint: &SessionEndpoint,
) -> Result<()> {
    let handle = handle_from_endpoint(endpoint)?;
    let sender = inner
        .velo
        .attach_anchor::<DecodeToPrefillFrame>(handle)
        .await
        .context("attach decode sender")?;
    let mut slot = inner.decode_tx.lock().await;
    *slot = Some(sender);
    Ok(())
}

fn spawn_decode_to_prefill_monitor(
    inner: Arc<DisaggSessionInner>,
    mut anchor: velo::StreamAnchor<DecodeToPrefillFrame>,
) {
    tokio::spawn(async move {
        while let Some(frame) = anchor.next().await {
            match frame {
                Ok(velo::StreamFrame::Item(frame)) => handle_decode_to_prefill_frame(&inner, frame),
                Ok(velo::StreamFrame::Finalized) => {
                    inner.emit_event(SessionEvent::Detached { reason: None });
                    inner.fail_pending("decode->prefill stream finalized");
                    break;
                }
                Ok(velo::StreamFrame::Detached) => {
                    inner.emit_event(SessionEvent::Detached {
                        reason: Some("decode->prefill stream detached".to_string()),
                    });
                    break;
                }
                Ok(other) => {
                    let reason = format!("unexpected decode->prefill stream frame: {other:?}");
                    inner.emit_event(SessionEvent::Failed {
                        reason: reason.clone(),
                    });
                    inner.fail_pending(&reason);
                    break;
                }
                Err(err) => {
                    let reason = format!("decode->prefill stream error: {err}");
                    inner.emit_event(SessionEvent::Failed {
                        reason: reason.clone(),
                    });
                    inner.fail_pending(&reason);
                    break;
                }
            }
        }
    });
}

fn handle_decode_to_prefill_frame(inner: &Arc<DisaggSessionInner>, frame: DecodeToPrefillFrame) {
    match frame {
        DecodeToPrefillFrame::BlockSetResponse(response)
        | DecodeToPrefillFrame::DescriptorResponse(response) => {
            if let Some(tx) = inner.pending.lock().block_sets.remove(&response.request_id) {
                let _ = tx.send(Ok(response));
            }
        }
        DecodeToPrefillFrame::UnpinRequest(request) => {
            inner.emit_event(SessionEvent::UnpinRequested(request));
        }
        DecodeToPrefillFrame::UnpinAck(ack) => {
            complete_unpin(inner, ack.clone());
            inner.emit_event(SessionEvent::UnpinAcked(ack));
        }
        DecodeToPrefillFrame::PullComplete(complete) => {
            inner.emit_event(SessionEvent::PullComplete(complete));
        }
        DecodeToPrefillFrame::PullAck(ack) => {
            complete_pull(inner, ack.clone());
            inner.emit_event(SessionEvent::PullAcked(ack));
        }
        DecodeToPrefillFrame::BlockSetsReady { block_sets } => {
            inner.emit_event(SessionEvent::BlockSetsAdded { block_sets });
        }
        DecodeToPrefillFrame::BlocksReady { blocks } => {
            inner.emit_event(SessionEvent::BlockSetsAdded {
                block_sets: vec![RemoteBlockSet {
                    source_layout: LogicalLayoutHandle::G2,
                    blocks,
                }],
            });
        }
        DecodeToPrefillFrame::OutputBlocksPulled { hashes } => {
            inner.emit_event(SessionEvent::PullComplete(PullComplete {
                pull_id: 0,
                hashes,
            }));
        }
        DecodeToPrefillFrame::Detach => {
            inner.emit_event(SessionEvent::Detached {
                reason: Some("decode requested detach".to_string()),
            });
        }
        DecodeToPrefillFrame::Error { message } => {
            inner.emit_event(SessionEvent::Failed { reason: message });
        }
    }
}

fn complete_unpin(inner: &Arc<DisaggSessionInner>, ack: UnpinAck) {
    if let Some(tx) = inner.pending.lock().unpins.remove(&ack.request_id) {
        let _ = tx.send(Ok(ack));
    }
}

fn complete_pull(inner: &Arc<DisaggSessionInner>, ack: PullAck) {
    if let Some(tx) = inner.pending.lock().pulls.remove(&ack.pull_id) {
        let _ = tx.send(Ok(ack));
    }
}

/// Ensures this leader has imported transfer metadata for a remote peer.
pub trait PeerMetadataCache: Send + Sync {
    fn ensure_remote_metadata(&self, peer: InstanceId) -> BoxFuture<'static, Result<()>>;

    fn invalidate(&self, _peer: InstanceId) {}
}

/// Metadata cache used by tests and local-only code paths that do not execute
/// real remote transfers.
#[derive(Debug, Default)]
pub struct NoopPeerMetadataCache;

impl NoopPeerMetadataCache {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl PeerMetadataCache for NoopPeerMetadataCache {
    fn ensure_remote_metadata(&self, _peer: InstanceId) -> BoxFuture<'static, Result<()>> {
        async { Ok(()) }.boxed()
    }
}

/// Coalesces concurrent metadata imports and caches successful imports by peer.
pub struct CoalescingPeerMetadataCache {
    inner: Arc<dyn PeerMetadataCache>,
    ready: Arc<DashSet<InstanceId>>,
    locks: Arc<DashMap<InstanceId, Arc<TokioMutex<()>>>>,
}

impl CoalescingPeerMetadataCache {
    pub fn new(inner: Arc<dyn PeerMetadataCache>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            ready: Arc::new(DashSet::new()),
            locks: Arc::new(DashMap::new()),
        })
    }
}

impl PeerMetadataCache for CoalescingPeerMetadataCache {
    fn ensure_remote_metadata(&self, peer: InstanceId) -> BoxFuture<'static, Result<()>> {
        if self.ready.contains(&peer) {
            return async { Ok(()) }.boxed();
        }

        let inner = Arc::clone(&self.inner);
        let ready = Arc::clone(&self.ready);
        let lock = self
            .locks
            .entry(peer)
            .or_insert_with(|| Arc::new(TokioMutex::new(())))
            .clone();

        async move {
            let _guard = lock.lock().await;
            if ready.contains(&peer) {
                return Ok(());
            }
            inner.ensure_remote_metadata(peer).await?;
            ready.insert(peer);
            Ok(())
        }
        .boxed()
    }

    fn invalidate(&self, peer: InstanceId) {
        self.ready.remove(&peer);
        self.inner.invalidate(peer);
    }
}

/// Adapter from disagg code to the engine leader metadata import path.
pub struct EnginePeerMetadataCache {
    leader: Arc<crate::leader::InstanceLeader>,
}

impl EnginePeerMetadataCache {
    pub fn new(leader: Arc<crate::leader::InstanceLeader>) -> Arc<Self> {
        Arc::new(Self { leader })
    }
}

impl PeerMetadataCache for EnginePeerMetadataCache {
    fn ensure_remote_metadata(&self, peer: InstanceId) -> BoxFuture<'static, Result<()>> {
        let leader = Arc::clone(&self.leader);
        async move { leader.ensure_remote_metadata(peer).await }.boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn hash(position: u64) -> SequenceHash {
        SequenceHash::new(position, None, position)
    }

    #[test]
    fn session_frames_round_trip_json() {
        let block = RemoteBlockRef {
            block_id: 7,
            sequence_hash: hash(0),
        };
        let frame = PrefillToDecodeFrame::OutputBlockSetsReady {
            block_sets: vec![RemoteBlockSet {
                source_layout: LogicalLayoutHandle::G2,
                blocks: vec![block],
            }],
        };

        let encoded = serde_json::to_vec(&frame).unwrap();
        let decoded: PrefillToDecodeFrame = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, frame);
    }

    #[test]
    fn block_set_and_unpin_frames_round_trip_msgpack() {
        let request = PrefillToDecodeFrame::BlockSetRequest(BlockSetRequest {
            request_id: "blocks".to_string(),
            hashes: HashSelection::All,
        });
        let encoded = rmp_serde::to_vec(&request).unwrap();
        let decoded: PrefillToDecodeFrame = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded, request);

        let unpin = PrefillToDecodeFrame::UnpinRequest(UnpinRequest {
            request_id: "unpin".to_string(),
            hashes: HashSelection::Hashes(vec![hash(1), hash(2)]),
        });
        let encoded = rmp_serde::to_vec(&unpin).unwrap();
        let decoded: PrefillToDecodeFrame = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded, unpin);
    }

    #[derive(Default)]
    struct CountingMetadataCache {
        calls: Arc<AtomicUsize>,
    }

    impl PeerMetadataCache for CountingMetadataCache {
        fn ensure_remote_metadata(&self, _peer: InstanceId) -> BoxFuture<'static, Result<()>> {
            let calls = Arc::clone(&self.calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                Ok(())
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn coalescing_cache_imports_once_for_concurrent_same_peer() {
        let inner = Arc::new(CountingMetadataCache::default());
        let calls = Arc::clone(&inner.calls);
        let cache = CoalescingPeerMetadataCache::new(inner);
        let peer = uuid::Uuid::new_v4().into();

        let (a, b, c) = tokio::join!(
            cache.ensure_remote_metadata(peer),
            cache.ensure_remote_metadata(peer),
            cache.ensure_remote_metadata(peer),
        );

        a.unwrap();
        b.unwrap();
        c.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        cache.ensure_remote_metadata(peer).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
