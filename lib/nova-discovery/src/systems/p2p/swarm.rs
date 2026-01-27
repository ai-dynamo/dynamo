// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P2P discovery implementation using libp2p Kademlia DHT.

use anyhow::{Context, Result, anyhow};
use futures::{StreamExt, future::BoxFuture};
use libp2p::{
    Multiaddr, PeerId, StreamProtocol, Transport,
    core::upgrade,
    identity, noise,
    pnet::{PnetConfig, PreSharedKey},
    swarm::{NetworkBehaviour, Swarm, SwarmEvent},
    tcp, yamux,
};
use libp2p_kad::{
    Behaviour as Kademlia, Config as KademliaConfig, Event as KademliaEvent, Mode, QueryResult,
    Quorum, Record, RecordKey, store::MemoryStore,
};
use libp2p_mdns as mdns;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::peer::{
    DiscoveryError, DiscoveryQueryError, InstanceId, PeerDiscovery, PeerInfo, WorkerAddress,
    WorkerId,
};

/// Dynamo Kademlia protocol name.
const DYNAMO_KAD_PROTOCOL: &str = "/dynamo/kad/1.0.0";

/// Generate Pre-Shared Key from a cluster_id string.
pub fn generate_psk_from_cluster_id(cluster_id: &str) -> PreSharedKey {
    use blake2::{Blake2b512, Digest};

    let mut hasher = Blake2b512::new();
    hasher.update(cluster_id.as_bytes());
    let hash = hasher.finalize();

    let mut psk_bytes = [0u8; 32];
    psk_bytes.copy_from_slice(&hash[..32]);

    PreSharedKey::new(psk_bytes)
}

/// Helper error type for DHT get operations.
#[derive(Debug)]
enum GetRecordError {
    NotFound,
    Backend(anyhow::Error),
}

impl From<GetRecordError> for DiscoveryQueryError {
    fn from(err: GetRecordError) -> Self {
        match err {
            GetRecordError::NotFound => DiscoveryQueryError::NotFound,
            GetRecordError::Backend(err) => DiscoveryQueryError::Backend(Arc::new(err)),
        }
    }
}

/// Network behaviour combining Kademlia DHT and mDNS.
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "DynamoBehaviourEvent")]
struct DynamoBehaviour {
    kad: Kademlia<MemoryStore>,
    mdns: libp2p::swarm::behaviour::toggle::Toggle<mdns::tokio::Behaviour>,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum DynamoBehaviourEvent {
    Kad(KademliaEvent),
    Mdns(mdns::Event),
}

impl From<KademliaEvent> for DynamoBehaviourEvent {
    fn from(event: KademliaEvent) -> Self {
        DynamoBehaviourEvent::Kad(event)
    }
}

impl From<mdns::Event> for DynamoBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        DynamoBehaviourEvent::Mdns(event)
    }
}

type ProviderList = Vec<(PeerId, Vec<Multiaddr>)>;

enum SwarmCommand {
    PutRecord {
        key: RecordKey,
        value: Vec<u8>,
        reply: oneshot::Sender<Result<()>>,
    },
    GetRecord {
        key: RecordKey,
        reply: oneshot::Sender<Result<Vec<u8>, GetRecordError>>,
    },
    #[allow(dead_code)]
    StartProviding {
        key: RecordKey,
        reply: oneshot::Sender<Result<()>>,
    },
    #[allow(dead_code)]
    GetProviders {
        key: RecordKey,
        reply: oneshot::Sender<Result<ProviderList>>,
    },
    #[allow(dead_code)]
    Shutdown,
}

type PendingGetQueries =
    Arc<RwLock<HashMap<libp2p_kad::QueryId, oneshot::Sender<Result<Vec<u8>, GetRecordError>>>>>;

type PendingProviderQueries =
    Arc<RwLock<HashMap<libp2p_kad::QueryId, oneshot::Sender<Result<ProviderList>>>>>;

#[derive(Clone)]
pub(super) struct P2pDiscovery {
    local_peer_id: PeerId,
    command_tx: mpsc::Sender<SwarmCommand>,
    #[allow(dead_code)]
    pending_get_queries: PendingGetQueries,
    #[allow(dead_code)]
    pending_provider_queries: PendingProviderQueries,
}

impl std::fmt::Debug for P2pDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("P2pDiscovery")
            .field("local_peer_id", &self.local_peer_id)
            .finish()
    }
}

impl P2pDiscovery {
    fn decode_peer(value: &[u8]) -> Result<PeerInfo, GetRecordError> {
        if value.is_empty() {
            return Err(GetRecordError::NotFound);
        }

        serde_json::from_slice(value).map_err(|err| GetRecordError::Backend(err.into()))
    }

    fn create_behaviour(
        key: &identity::Keypair,
        replication_factor: usize,
        enable_mdns: bool,
        record_ttl_secs: u64,
        publication_interval_secs: Option<u64>,
        provider_publication_interval_secs: Option<u64>,
    ) -> DynamoBehaviour {
        let local_peer_id = key.public().to_peer_id();

        let store = MemoryStore::new(local_peer_id);
        let protocol = StreamProtocol::try_from_owned(DYNAMO_KAD_PROTOCOL.to_string())
            .expect("Valid protocol name");
        let mut kad_config = KademliaConfig::new(protocol);

        kad_config
            .set_replication_factor(
                NonZeroUsize::new(replication_factor).expect("Replication factor must be non-zero"),
            )
            .set_parallelism(NonZeroUsize::new(10).unwrap())
            .set_query_timeout(Duration::from_secs(30))
            .set_publication_interval(publication_interval_secs.map(Duration::from_secs))
            .set_provider_publication_interval(
                provider_publication_interval_secs.map(Duration::from_secs),
            )
            .set_record_ttl(Some(Duration::from_secs(record_ttl_secs)))
            .set_provider_record_ttl(Some(Duration::from_secs(record_ttl_secs)));

        let mut kad = Kademlia::with_config(local_peer_id, store, kad_config);
        kad.set_mode(Some(Mode::Server));

        // Conditionally enable mDNS based on configuration
        let mdns = if enable_mdns {
            let behaviour = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)
                .expect("Failed to create mDNS behaviour");
            libp2p::swarm::behaviour::toggle::Toggle::from(Some(behaviour))
        } else {
            libp2p::swarm::behaviour::toggle::Toggle::from(None)
        };

        DynamoBehaviour { kad, mdns }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn new(
        cluster_id: String,
        listen_port: u16,
        bootstrap_peers: Vec<String>,
        replication_factor: usize,
        enable_mdns: bool,
        record_ttl_secs: u64,
        publication_interval_secs: Option<u64>,
        provider_publication_interval_secs: Option<u64>,
    ) -> Result<Self> {
        let keypair = identity::Keypair::generate_ed25519();
        let local_peer_id = keypair.public().to_peer_id();

        info!(
            "Initializing P2P discovery for peer {} with cluster_id '{}'",
            local_peer_id, cluster_id
        );

        let psk = generate_psk_from_cluster_id(&cluster_id);

        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_other_transport(move |key| {
                let tcp = tcp::tokio::Transport::default();
                let pnet_tcp = tcp.and_then(move |socket, _| {
                    let psk_clone = psk;
                    async move { PnetConfig::new(psk_clone).handshake(socket).await }
                });

                pnet_tcp
                    .upgrade(upgrade::Version::V1)
                    .authenticate(
                        noise::Config::new(key)
                            .expect("Failed to create noise config with valid keypair"),
                    )
                    .multiplex(yamux::Config::default())
                    .boxed()
            })?
            .with_behaviour(|key| {
                Self::create_behaviour(
                    key,
                    replication_factor,
                    enable_mdns,
                    record_ttl_secs,
                    publication_interval_secs,
                    provider_publication_interval_secs,
                )
            })?
            .build();

        if enable_mdns {
            info!("mDNS enabled for local peer discovery");
        }

        let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", listen_port)
            .parse()
            .context("Invalid listen address")?;
        swarm
            .listen_on(listen_addr.clone())
            .with_context(|| format!("Failed to listen on {}", listen_addr))?;
        info!("Listening on {}", listen_addr);

        for peer_str in &bootstrap_peers {
            let addr: Multiaddr = format!(
                "/ip4/{}/tcp/{}",
                peer_str.split(':').next().unwrap_or("127.0.0.1"),
                peer_str.split(':').nth(1).unwrap_or("4001")
            )
            .parse()
            .with_context(|| format!("Invalid bootstrap peer address: {}", peer_str))?;

            if let Err(e) = swarm.dial(addr.clone()) {
                warn!("Failed to dial bootstrap peer {}: {:?}", peer_str, e);
            } else {
                info!("Dialing bootstrap peer at {}", addr);
            }
        }

        if !bootstrap_peers.is_empty() {
            if let Err(e) = swarm.behaviour_mut().kad.bootstrap() {
                warn!("Failed to bootstrap Kademlia DHT: {:?}", e);
            } else {
                info!("Started DHT bootstrap");
            }
        }

        let (command_tx, command_rx) = mpsc::channel(100);
        let pending_get_queries = Arc::new(RwLock::new(HashMap::new()));
        let pending_provider_queries = Arc::new(RwLock::new(HashMap::new()));

        let pending_get_queries_clone = Arc::clone(&pending_get_queries);
        let pending_provider_queries_clone = Arc::clone(&pending_provider_queries);
        tokio::spawn(async move {
            Self::swarm_event_loop(
                swarm,
                command_rx,
                pending_get_queries_clone,
                pending_provider_queries_clone,
            )
            .await;
        });

        Ok(Self {
            local_peer_id,
            command_tx,
            pending_get_queries,
            pending_provider_queries,
        })
    }

    async fn swarm_event_loop(
        mut swarm: Swarm<DynamoBehaviour>,
        mut command_rx: mpsc::Receiver<SwarmCommand>,
        pending_get_queries: PendingGetQueries,
        pending_provider_queries: PendingProviderQueries,
    ) {
        loop {
            tokio::select! {
                Some(cmd) = command_rx.recv() => {
                    match cmd {
                        SwarmCommand::PutRecord { key, value, reply } => {
                            let record = Record {
                                key,
                                value,
                                publisher: None,
                                expires: None,
                            };

                            match swarm.behaviour_mut().kad.put_record(record, Quorum::One) {
                                Ok(_) => {
                                    let _ = reply.send(Ok(()));
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(anyhow!("Failed to put record: {:?}", e)));
                                }
                            }
                        }
                        SwarmCommand::GetRecord { key, reply } => {
                            let query_id = swarm.behaviour_mut().kad.get_record(key);
                            pending_get_queries.write().await.insert(query_id, reply);
                        }
                        SwarmCommand::StartProviding { key, reply } => {
                            match swarm.behaviour_mut().kad.start_providing(key) {
                                Ok(_) => {
                                    let _ = reply.send(Ok(()));
                                }
                                Err(e) => {
                                    let _ = reply.send(Err(anyhow!("Failed to start providing: {:?}", e)));
                                }
                            }
                        }
                        SwarmCommand::GetProviders { key, reply } => {
                            let query_id = swarm.behaviour_mut().kad.get_providers(key);
                            pending_provider_queries.write().await.insert(query_id, reply);
                        }
                        SwarmCommand::Shutdown => {
                            info!("Shutting down P2P swarm");
                            break;
                        }
                    }
                }

                event = swarm.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(DynamoBehaviourEvent::Kad(kad_event)) => {
                            Self::handle_kad_event(
                                kad_event,
                                &pending_get_queries,
                                &pending_provider_queries,
                            ).await;
                        }
                        SwarmEvent::Behaviour(DynamoBehaviourEvent::Mdns(mdns_event)) => {
                            Self::handle_mdns_event(mdns_event, &mut swarm);
                        }
                        SwarmEvent::NewListenAddr { address, .. } => {
                            info!("Listening on {}", address);
                        }
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            debug!("Connection established with peer {}", peer_id);
                        }
                        SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                            debug!("Connection closed with peer {}: {:?}", peer_id, cause);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    async fn handle_kad_event(
        event: KademliaEvent,
        pending_get_queries: &PendingGetQueries,
        pending_provider_queries: &PendingProviderQueries,
    ) {
        match event {
            KademliaEvent::OutboundQueryProgressed {
                id,
                result: QueryResult::GetRecord(Ok(libp2p_kad::GetRecordOk::FoundRecord(record))),
                ..
            } => {
                if let Some(sender) = pending_get_queries.write().await.remove(&id) {
                    let _ = sender.send(Ok(record.record.value.clone()));
                }
            }
            KademliaEvent::OutboundQueryProgressed {
                id,
                result: QueryResult::GetRecord(Err(err)),
                ..
            } => {
                if let Some(sender) = pending_get_queries.write().await.remove(&id) {
                    let mapped = match err {
                        libp2p_kad::GetRecordError::NotFound { .. } => {
                            Err(GetRecordError::NotFound)
                        }
                        other => Err(GetRecordError::Backend(anyhow!(
                            "Get record failed: {:?}",
                            other
                        ))),
                    };
                    let _ = sender.send(mapped);
                }
            }
            KademliaEvent::OutboundQueryProgressed {
                id,
                result:
                    QueryResult::GetProviders(Ok(libp2p_kad::GetProvidersOk::FoundProviders {
                        providers,
                        ..
                    })),
                ..
            } => {
                let provider_addrs: Vec<(PeerId, Vec<Multiaddr>)> = providers
                    .into_iter()
                    .map(|peer_id| (peer_id, Vec::new()))
                    .collect();

                if let Some(sender) = pending_provider_queries.write().await.remove(&id) {
                    let _ = sender.send(Ok(provider_addrs));
                }
            }
            KademliaEvent::OutboundQueryProgressed {
                id,
                result: QueryResult::GetProviders(Err(e)),
                ..
            } => {
                if let Some(sender) = pending_provider_queries.write().await.remove(&id) {
                    let _ = sender.send(Err(anyhow!("Get providers failed: {:?}", e)));
                }
            }
            KademliaEvent::OutboundQueryProgressed {
                result: QueryResult::Bootstrap(Ok(_)),
                ..
            } => {
                info!("Kademlia bootstrap completed successfully");
            }
            KademliaEvent::OutboundQueryProgressed {
                result: QueryResult::Bootstrap(Err(e)),
                ..
            } => {
                warn!("Kademlia bootstrap failed: {:?}", e);
            }
            KademliaEvent::RoutingUpdated { peer, .. } => {
                debug!("Routing table updated with peer {}", peer);
            }
            _ => {}
        }
    }

    fn handle_mdns_event(event: mdns::Event, swarm: &mut Swarm<DynamoBehaviour>) {
        match event {
            mdns::Event::Discovered(peers) => {
                for (peer_id, addr) in peers {
                    debug!("mDNS discovered peer {} at {}", peer_id, addr);
                    swarm.behaviour_mut().kad.add_address(&peer_id, addr);
                }
            }
            mdns::Event::Expired(peers) => {
                for (peer_id, _addr) in peers {
                    debug!("mDNS expired peer {}", peer_id);
                }
            }
        }
    }

    async fn put_record(&self, key: RecordKey, value: Vec<u8>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(SwarmCommand::PutRecord {
                key,
                value,
                reply: tx,
            })
            .await
            .context("Failed to send put record command")?;

        rx.await.context("Put record command cancelled")?
    }

    async fn get_record(&self, key: RecordKey) -> Result<Vec<u8>, GetRecordError> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(SwarmCommand::GetRecord { key, reply: tx })
            .await
            .map_err(|e| {
                GetRecordError::Backend(anyhow!("Failed to send get record command: {e}"))
            })?;

        let response = tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .map_err(|_| GetRecordError::Backend(anyhow!("Get record timed out")))?;

        response.map_err(|_| GetRecordError::Backend(anyhow!("Get record command cancelled")))?
    }

    /// Start providing a content key in the DHT.
    #[allow(dead_code)]
    pub async fn start_providing(&self, key: &str) -> Result<()> {
        let record_key = RecordKey::new(&key.as_bytes());
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(SwarmCommand::StartProviding {
                key: record_key,
                reply: tx,
            })
            .await
            .context("Failed to send start providing command")?;

        rx.await.context("Start providing command cancelled")?
    }

    /// Get all providers for a content key from the DHT.
    #[allow(dead_code)]
    pub async fn get_providers(&self, key: &str) -> Result<ProviderList> {
        let record_key = RecordKey::new(&key.as_bytes());
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(SwarmCommand::GetProviders {
                key: record_key,
                reply: tx,
            })
            .await
            .context("Failed to send get providers command")?;

        let response = tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .context("Get providers timed out")?;

        response.context("Get providers command cancelled")?
    }

    pub(super) fn shutdown(&self) {
        let command_tx = self.command_tx.clone();
        tokio::spawn(async move {
            if let Err(err) = command_tx.send(SwarmCommand::Shutdown).await {
                warn!("Failed to send P2P shutdown command: {:?}", err);
            }
        });
    }
}

impl PeerDiscovery for P2pDiscovery {
    fn discover_by_worker_id(
        &self,
        worker_id: WorkerId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move {
            let key = RecordKey::new(&worker_id.as_u64().to_be_bytes());
            let value = this
                .get_record(key)
                .await
                .map_err(DiscoveryQueryError::from)?;

            let peer_info = Self::decode_peer(&value).map_err(DiscoveryQueryError::from)?;

            Ok(peer_info)
        })
    }

    fn discover_by_instance_id(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<PeerInfo, DiscoveryQueryError>> {
        let this = self.clone();
        Box::pin(async move {
            let key = RecordKey::new(instance_id.as_bytes());
            let value = this
                .get_record(key)
                .await
                .map_err(DiscoveryQueryError::from)?;

            let peer_info = Self::decode_peer(&value).map_err(DiscoveryQueryError::from)?;

            Ok(peer_info)
        })
    }

    fn register_instance(
        &self,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move {
            let worker_id = instance_id.worker_id();
            let desired_peer = PeerInfo::new(instance_id, worker_address.clone());

            // Collision detection on worker_id
            let worker_key = RecordKey::new(&worker_id.as_u64().to_be_bytes());
            match this.get_record(worker_key.clone()).await {
                Ok(existing) => match Self::decode_peer(&existing) {
                    Ok(stored) => {
                        if stored.instance_id != instance_id {
                            return Err(DiscoveryError::WorkerIdCollision(
                                worker_id,
                                stored.instance_id,
                                instance_id,
                            ));
                        }

                        if stored.address_checksum() != desired_peer.address_checksum() {
                            return Err(DiscoveryError::ChecksumMismatch(
                                instance_id,
                                stored.address_checksum(),
                                desired_peer.address_checksum(),
                            ));
                        }

                        return Err(DiscoveryError::Backend(anyhow!(
                            "Instance {instance_id} already registered"
                        )));
                    }
                    Err(GetRecordError::NotFound) => {}
                    Err(GetRecordError::Backend(err)) => return Err(DiscoveryError::Backend(err)),
                },
                Err(GetRecordError::NotFound) => {}
                Err(GetRecordError::Backend(err)) => return Err(DiscoveryError::Backend(err)),
            }

            // Check existing instance record for checksum mismatch
            let instance_key = RecordKey::new(instance_id.as_bytes());
            match this.get_record(instance_key.clone()).await {
                Ok(existing) => match Self::decode_peer(&existing) {
                    Ok(stored) => {
                        if stored.address_checksum() != desired_peer.address_checksum() {
                            return Err(DiscoveryError::ChecksumMismatch(
                                instance_id,
                                stored.address_checksum(),
                                desired_peer.address_checksum(),
                            ));
                        }

                        // Identical record already exists, treat as success (idempotent)
                        return Ok(());
                    }
                    Err(GetRecordError::NotFound) => {}
                    Err(GetRecordError::Backend(err)) => return Err(DiscoveryError::Backend(err)),
                },
                Err(GetRecordError::NotFound) => {}
                Err(GetRecordError::Backend(err)) => return Err(DiscoveryError::Backend(err)),
            }

            let payload = serde_json::to_vec(&desired_peer)
                .context("Failed to serialize PeerInfo")
                .map_err(DiscoveryError::Backend)?;

            this.put_record(worker_key, payload.clone())
                .await
                .map_err(DiscoveryError::Backend)?;
            this.put_record(instance_key, payload)
                .await
                .map_err(DiscoveryError::Backend)?;

            Ok(())
        })
    }

    fn unregister_instance(
        &self,
        instance_id: InstanceId,
    ) -> BoxFuture<'static, Result<(), DiscoveryError>> {
        let this = self.clone();
        Box::pin(async move {
            let worker_key = RecordKey::new(&instance_id.worker_id().as_u64().to_be_bytes());
            let instance_key = RecordKey::new(instance_id.as_bytes());

            this.put_record(worker_key, Vec::new())
                .await
                .map_err(DiscoveryError::Backend)?;
            this.put_record(instance_key, Vec::new())
                .await
                .map_err(DiscoveryError::Backend)?;

            debug!(
                "Published tombstone for instance {} (worker_id {})",
                instance_id,
                instance_id.worker_id()
            );

            Ok(())
        })
    }
}
