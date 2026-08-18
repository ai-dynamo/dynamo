// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use dynamo_kv_router::identity::{
    CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
};
use dynamo_kv_router::protocols::{
    ActiveLoad, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
};
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::{
    DistributedRuntime, Runtime, distributed::DistributedConfig, traits::DistributedRuntimeProvider,
};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;
use tonic::Streaming;
use tonic::transport::{
    Certificate, Channel, ClientTlsConfig, Endpoint, Identity as ClientIdentity,
};

use super::super::discovery::{
    DcMembershipView, DomainWorkerTopology, EndpointMembership, KvCacheDomainKey,
};
use super::super::identity::{
    CanonicalModelId, CanonicalModelRegistration, DcRelayIdentity, KvQueryHashFormat,
    KvQuerySemantics, ModelAlias, WorkerRole,
};
use super::super::pool_registry::{
    PoolActorConfig, PoolAttachRequest, PoolAttachment, PoolRegistry, PoolWanFacts,
};
use super::super::protocol as proto;
use super::super::protocol::wire::images::{self, FilterFormat, ImagesFrame, SnapshotAssembly};
use super::super::topology::TopologyPublisher;
use super::super::transport_config::KvDcRelayTransportConfig;
use super::server::KvDcRelayTransport;
use super::source::WanPublicationSource;
use super::test_support::{TestPki, test_pki, tls_test_config};
use crate::kv_dc_relay::actor::KvDcRelayHandle;
use crate::local_model::runtime_config::ModelRuntimeConfig;

const IO_TIMEOUT: Duration = Duration::from_secs(5);
const WORKER_ID: u64 = 7;

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

struct RelayFixture {
    _temp: TempDir,
    pki: TestPki,
    address: SocketAddr,
    transport: KvDcRelayTransport,
    registry: Arc<PoolRegistry>,
    topology: Arc<TopologyPublisher>,
    attachment: Option<PoolAttachment>,
    pool_id: PoolId,
}

impl RelayFixture {
    async fn start(configure: impl FnOnce(&mut KvDcRelayTransportConfig)) -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = format!("kv-dc-relay-conformance-{fixture_id}");
        let component = drt
            .namespace(&namespace)
            .unwrap()
            .component("relay")
            .unwrap();
        let relay_identity =
            DcRelayIdentity::new(component.drt().connection_id(), fixture_id + 100);
        let registry = Arc::new(PoolRegistry::new(
            relay_identity,
            PoolActorConfig {
                expected_unique_blocks: 32,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
        ));
        let pool_id = pool_id(fixture_id as u8);
        let attachment = attach_pool(&registry, pool_id).await;
        let endpoint = EndpointId::from("backend.generate");
        let topology = Arc::new(TopologyPublisher::new(
            membership_view(endpoint.clone(), pool_id.indexer_domain()),
            &registry.catalog(),
        ));
        topology.claim_availability(endpoint.clone(), 1);
        topology.replace_availability(endpoint, 1, Some(HashSet::from([WORKER_ID])));
        assert!(registry.observe_load(
            pool_id,
            attachment.layout_generation,
            ActiveLoad {
                worker_id: WORKER_ID,
                dp_rank: 0,
                active_decode_blocks: Some(30),
                active_prefill_tokens: Some(512),
                kv_used_blocks: Some(40),
            },
        ));

        let lifecycle = CancellationToken::new();
        let source = WanPublicationSource::new(
            component,
            registry.clone(),
            topology.clone(),
            relay_identity,
            lifecycle,
        );
        let pki = test_pki();
        let temp = TempDir::new().unwrap();
        let mut config = tls_test_config(&temp, &pki);
        configure(&mut config);
        let transport = KvDcRelayTransport::start(source, config).await.unwrap();
        let address = transport
            .health()
            .bound_address
            .expect("transport must expose its bound address");
        Self {
            _temp: temp,
            pki,
            address,
            transport,
            registry,
            topology,
            attachment: Some(attachment),
            pool_id,
        }
    }

    async fn client(&self) -> proto::KvEventRelayClient<Channel> {
        proto::KvEventRelayClient::new(
            connect(
                self.address,
                &self.pki,
                Some((&self.pki.client_cert_pem, &self.pki.client_key_pem)),
            )
            .await
            .unwrap(),
        )
    }

    fn actor(&self) -> &KvDcRelayHandle {
        &self
            .attachment
            .as_ref()
            .expect("pool attachment must be active")
            .handle
    }

    async fn replace_pool(&mut self) {
        let old = self
            .attachment
            .take()
            .expect("pool attachment must be active");
        self.registry.detach(old).await.unwrap();
        self.attachment = Some(attach_pool(&self.registry, self.pool_id).await);
        self.topology.replace_catalog(&self.registry.catalog());
    }

    async fn shutdown(mut self) {
        self.transport.shutdown().await;
        if let Some(attachment) = self.attachment.take() {
            self.registry.detach(attachment).await.unwrap();
        }
        self.registry.shutdown().await;
    }
}

fn pool_id(seed: u8) -> PoolId {
    PoolId::new(
        IndexerDomainId::new(
            CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
            RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
        ),
        DcId::new(3),
    )
}

async fn attach_pool(registry: &PoolRegistry, pool_id: PoolId) -> PoolAttachment {
    let runtime_config = ModelRuntimeConfig {
        total_kv_blocks: Some(100),
        max_num_batched_tokens: Some(2_048),
        ..ModelRuntimeConfig::default()
    };
    registry
        .attach(PoolAttachRequest {
            pool_id,
            endpoint: EndpointId::from("backend.generate"),
            registrations: vec![CanonicalModelRegistration::new(
                CanonicalModelId::new("llama").unwrap(),
                vec![ModelAlias::new("chat").unwrap()],
            )],
            query_semantics: KvQuerySemantics::new(64, KvQueryHashFormat::DynamoStandardV1)
                .unwrap(),
            roles: vec![WorkerRole::Legacy],
            wan_facts: Some(PoolWanFacts {
                runtime_configs: HashMap::from([(WORKER_ID, runtime_config)]),
            }),
        })
        .await
        .unwrap()
}

fn membership_view(endpoint: EndpointId, indexer_domain: IndexerDomainId) -> DcMembershipView {
    let registration = CanonicalModelRegistration::new(
        CanonicalModelId::new("llama").unwrap(),
        vec![ModelAlias::new("chat").unwrap()],
    );
    let query_semantics = KvQuerySemantics::new(64, KvQueryHashFormat::DynamoStandardV1).unwrap();
    let domain = KvCacheDomainKey {
        id: indexer_domain,
        diagnostic_model_artifact: "llama".to_string(),
        query_semantics,
    };
    let membership = EndpointMembership {
        endpoint: endpoint.clone(),
        generation: 1,
        domain: Some(domain),
        namespace: endpoint.namespace.clone(),
        registrations: vec![registration],
        models: vec!["llama".to_string()],
        aliases: vec!["chat".to_string()],
        roles: vec![WorkerRole::Legacy],
        runtime_configs: HashMap::new(),
        worker_topology: HashMap::from([(
            WORKER_ID,
            DomainWorkerTopology {
                worker_type: None,
                needs: Vec::new(),
            },
        )]),
        adapters: HashMap::new(),
        conflicts: Vec::new(),
    };
    DcMembershipView {
        endpoints: Arc::new(HashMap::from([(endpoint, membership)])),
    }
}

fn stored(event_id: u64, hash: u64) -> RouterEvent {
    const EXTERNAL_MASK: u64 = 0xC0F0_4A11_5EED_2026;
    RouterEvent::new(
        WORKER_ID,
        KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(hash ^ EXTERNAL_MASK),
                    tokens_hash: LocalBlockHash(hash),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        },
    )
}

async fn connect(
    address: SocketAddr,
    pki: &TestPki,
    identity: Option<(&str, &str)>,
) -> anyhow::Result<Channel> {
    let mut tls = ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(&pki.ca_pem))
        .domain_name("localhost");
    if let Some((cert, key)) = identity {
        tls = tls.identity(ClientIdentity::from_pem(cert, key));
    }
    Ok(
        Endpoint::from_shared(format!("https://127.0.0.1:{}", address.port()))?
            .connect_timeout(IO_TIMEOUT)
            .timeout(IO_TIMEOUT)
            .tls_config(tls)?
            .connect()
            .await?,
    )
}

fn relay_info_request() -> proto::RelayInfoRequest {
    proto::RelayInfoRequest {
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

async fn relay_info_with_identity(
    address: SocketAddr,
    pki: &TestPki,
    identity: Option<(&str, &str)>,
) -> anyhow::Result<proto::RelayInfo> {
    let channel = connect(address, pki, identity).await?;
    let mut client = proto::KvEventRelayClient::new(channel);
    Ok(client
        .get_relay_info(relay_info_request())
        .await?
        .into_inner())
}

async fn next_filter_update(stream: &mut Streaming<proto::FilterUpdate>) -> proto::FilterUpdate {
    let update = tokio::time::timeout(IO_TIMEOUT, stream.message())
        .await
        .expect("pool stream timed out")
        .expect("pool stream failed")
        .expect("pool stream ended");
    proto::validate_protocol_envelope(update.protocol_version, update.contract_marker)
        .expect("valid pool update envelope");
    update
}

fn filter_format(producer: &proto::ProducerIdentity) -> FilterFormat {
    let format = producer
        .ckf_format
        .as_ref()
        .expect("producer must declare CKF format");
    FilterFormat::new(
        format.seed,
        usize::try_from(format.bucket_count).expect("bucket count must fit usize"),
    )
    .expect("catalog must contain a valid CBI1 format")
}

async fn receive_snapshot(
    stream: &mut Streaming<proto::FilterUpdate>,
    producer: &proto::ProducerIdentity,
) -> (u64, Vec<u64>) {
    let format = filter_format(producer);
    let mut assembly = SnapshotAssembly::new(format);
    loop {
        let update = next_filter_update(stream).await;
        assert_eq!(update.producer.as_ref(), Some(producer));
        assert_eq!(
            proto::FilterUpdateKind::try_from(update.kind).unwrap(),
            proto::FilterUpdateKind::SnapshotChunk
        );
        assert_eq!(update.base_sequence, update.sequence);
        let frame = images::decode(format, &update.payload).expect("valid CBI1 snapshot chunk");
        if let Some((epoch, images)) = assembly.absorb(&frame).expect("ordered snapshot chunks") {
            assert_eq!(epoch, update.sequence);
            let mut mirror = vec![0; format.bucket_count];
            for image in images {
                mirror[image.bucket as usize] = image.value;
            }
            return (epoch, mirror);
        }
    }
}

async fn receive_delta(
    stream: &mut Streaming<proto::FilterUpdate>,
    producer: &proto::ProducerIdentity,
    mirror: &mut [u64],
) -> proto::FilterUpdate {
    let update = next_filter_update(stream).await;
    assert_eq!(update.producer.as_ref(), Some(producer));
    assert_eq!(
        proto::FilterUpdateKind::try_from(update.kind).unwrap(),
        proto::FilterUpdateKind::Delta
    );
    let frame = images::decode(filter_format(producer), &update.payload).expect("valid CBI1 delta");
    let ImagesFrame::Delta {
        header,
        base_epoch,
        images,
    } = frame
    else {
        panic!("delta envelope must carry a CBI1 delta");
    };
    assert_eq!(header.epoch, update.sequence);
    assert_eq!(base_epoch, update.base_sequence);
    for image in images {
        mirror[image.bucket as usize] = image.value;
    }
    update
}

async fn initial_catalog(
    client: &mut proto::KvEventRelayClient<Channel>,
) -> (
    Streaming<proto::KvPoolCatalogUpdate>,
    proto::ProducerIdentity,
) {
    let mut stream = client
        .watch_kv_pool_catalog(proto::WatchKvPoolCatalogRequest {
            subscriber_id: "catalog-test".to_string(),
            contract_marker: proto::RELAY_CONTRACT_MARKER,
        })
        .await
        .unwrap()
        .into_inner();
    let catalog = tokio::time::timeout(IO_TIMEOUT, stream.message())
        .await
        .expect("catalog stream timed out")
        .unwrap()
        .expect("catalog stream ended");
    proto::validate_protocol_envelope(catalog.protocol_version, catalog.contract_marker)
        .expect("valid catalog envelope");
    let pools = &catalog.snapshot.expect("catalog snapshot").pools;
    assert_eq!(pools.len(), 1);
    let descriptor = &pools[0];
    proto::validate_pool_descriptor(descriptor).expect("valid pool descriptor");
    assert_eq!(descriptor.registrations[0].canonical_model_id, "llama");
    assert_eq!(descriptor.registrations[0].aliases, ["chat"]);
    assert_eq!(descriptor.pool_roles, [proto::WorkerRole::Legacy as i32]);
    assert_eq!(
        descriptor
            .query_semantics
            .as_ref()
            .expect("query semantics")
            .kv_block_size,
        64
    );
    (
        stream,
        descriptor.producer.clone().expect("catalog producer"),
    )
}

async fn subscribe_pool(
    client: &mut proto::KvEventRelayClient<Channel>,
    producer: proto::ProducerIdentity,
) -> Result<Streaming<proto::FilterUpdate>, tonic::Status> {
    client
        .subscribe_kv_pool(proto::SubscribeKvPoolRequest {
            subscriber_id: "pool-test".to_string(),
            expected_producer: Some(producer),
            contract_marker: proto::RELAY_CONTRACT_MARKER,
        })
        .await
        .map(tonic::Response::into_inner)
}

#[tokio::test]
async fn actual_relay_transport_requires_a_trusted_client_certificate() {
    let fixture = RelayFixture::start(|_| {}).await;

    let missing = tokio::time::timeout(
        IO_TIMEOUT,
        relay_info_with_identity(fixture.address, &fixture.pki, None),
    )
    .await
    .expect("missing-certificate request timed out");
    assert!(missing.is_err());

    let wrong = tokio::time::timeout(
        IO_TIMEOUT,
        relay_info_with_identity(
            fixture.address,
            &fixture.pki,
            Some((
                &fixture.pki.wrong_client_cert_pem,
                &fixture.pki.wrong_client_key_pem,
            )),
        ),
    )
    .await
    .expect("wrong-certificate request timed out");
    assert!(wrong.is_err());

    let mut client = fixture.client().await;
    let info = client
        .get_relay_info(relay_info_request())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.protocol_version, proto::RELAY_PROTOCOL_VERSION);
    assert_eq!(info.contract_marker, proto::RELAY_CONTRACT_MARKER);

    drop(client);
    fixture.shutdown().await;
}

#[tokio::test]
async fn actual_relay_transport_serves_all_rpc_surfaces_and_contiguous_cbi1() {
    let fixture = RelayFixture::start(|_| {}).await;
    let mut client = fixture.client().await;

    client.get_relay_info(relay_info_request()).await.unwrap();
    let (_catalog_stream, producer) = initial_catalog(&mut client).await;
    let mut pool_stream = subscribe_pool(&mut client, producer.clone()).await.unwrap();
    let (snapshot_sequence, mut mirror) = receive_snapshot(&mut pool_stream, &producer).await;
    assert!(mirror.iter().all(|&bucket| bucket == 0));

    let mut readiness = client
        .subscribe_serving_readiness(proto::SubscribeServingReadinessRequest {
            subscriber_id: "readiness-test".to_string(),
            contract_marker: proto::RELAY_CONTRACT_MARKER,
        })
        .await
        .unwrap()
        .into_inner();
    let readiness = tokio::time::timeout(IO_TIMEOUT, readiness.message())
        .await
        .expect("readiness stream timed out")
        .unwrap()
        .expect("readiness stream ended");
    proto::validate_protocol_envelope(readiness.protocol_version, readiness.contract_marker)
        .expect("valid readiness envelope");
    assert_eq!(readiness.entries.len(), 1);
    proto::validate_topology_entry(&readiness.entries[0]).expect("valid topology entry");
    assert_eq!(readiness.entries[0].canonical_model_id, "llama");
    assert_eq!(
        readiness.entries[0].state,
        proto::ServingReadinessState::Ready as i32
    );
    assert!(readiness.entries[0].legacy_fallback_active);
    assert_eq!(readiness.entries[0].members.len(), 1);
    assert_eq!(
        readiness.entries[0].members[0].roles,
        [proto::WorkerRole::Legacy as i32]
    );
    assert!(readiness.entries[0].members[0].pool_id.is_some());

    let mut load = client
        .subscribe_kv_pool_load(proto::SubscribeKvPoolLoadRequest {
            subscriber_id: "load-test".to_string(),
            contract_marker: proto::RELAY_CONTRACT_MARKER,
        })
        .await
        .unwrap()
        .into_inner();
    let load = tokio::time::timeout(IO_TIMEOUT, load.message())
        .await
        .expect("load stream timed out")
        .unwrap()
        .expect("load stream ended");
    proto::validate_protocol_envelope(load.protocol_version, load.contract_marker)
        .expect("valid load envelope");
    assert_eq!(load.pools.len(), 1);
    assert_eq!(load.pools[0].producer.as_ref(), Some(&producer));
    assert_eq!(load.pools[0].kv_used_blocks, 40);
    assert_eq!(load.pools[0].total_kv_blocks, 100);
    assert_eq!(load.pools[0].active_decode_blocks, 30);
    assert_eq!(load.pools[0].active_prefill_tokens, 512);

    fixture.actor().admit_event(1, stored(1, 99)).await.unwrap();
    fixture.actor().flush().await.unwrap();
    let delta = receive_delta(&mut pool_stream, &producer, &mut mirror).await;
    assert_eq!(delta.base_sequence, snapshot_sequence);
    assert_eq!(delta.sequence, snapshot_sequence + 1);
    assert!(mirror.iter().any(|&bucket| bucket != 0));

    drop((client, pool_stream));
    fixture.shutdown().await;
}

#[tokio::test]
async fn replacement_generation_rejects_stale_subscribe_and_requires_a_new_snapshot() {
    let mut fixture = RelayFixture::start(|_| {}).await;
    let mut client = fixture.client().await;
    let (mut catalog_stream, first_producer) = initial_catalog(&mut client).await;
    let mut first_stream = subscribe_pool(&mut client, first_producer.clone())
        .await
        .unwrap();
    let (_, mut first_mirror) = receive_snapshot(&mut first_stream, &first_producer).await;
    fixture
        .actor()
        .admit_event(1, stored(1, 123))
        .await
        .unwrap();
    fixture.actor().flush().await.unwrap();
    let old_delta = receive_delta(&mut first_stream, &first_producer, &mut first_mirror).await;

    fixture.replace_pool().await;
    let second_producer = loop {
        let catalog = tokio::time::timeout(IO_TIMEOUT, catalog_stream.message())
            .await
            .expect("replacement catalog timed out")
            .unwrap()
            .expect("catalog stream ended during replacement");
        let pools = catalog.snapshot.expect("catalog snapshot").pools;
        if let Some(producer) = pools.into_iter().find_map(|descriptor| descriptor.producer) {
            break producer;
        }
    };
    assert_ne!(first_producer, second_producer);
    assert_ne!(old_delta.producer.as_ref(), Some(&second_producer));

    let stale = subscribe_pool(&mut client, first_producer.clone())
        .await
        .unwrap_err();
    assert_eq!(stale.code(), tonic::Code::FailedPrecondition);

    let mut second_stream = subscribe_pool(&mut client, second_producer.clone())
        .await
        .unwrap();
    let (second_sequence, second_mirror) =
        receive_snapshot(&mut second_stream, &second_producer).await;
    assert_eq!(second_sequence, 0);
    assert!(second_mirror.iter().all(|&bucket| bucket == 0));

    tokio::time::timeout(IO_TIMEOUT, async {
        while let Ok(Some(update)) = first_stream.message().await {
            assert_eq!(update.producer.as_ref(), Some(&first_producer));
            assert_ne!(update.producer.as_ref(), Some(&second_producer));
        }
    })
    .await
    .expect("retired stream did not terminate");

    drop((client, first_stream, second_stream));
    fixture.shutdown().await;
}

#[tokio::test]
async fn pool_stream_limit_returns_resource_exhausted_and_allows_resubscribe() {
    let fixture = RelayFixture::start(|config| config.max_pool_streams_total = 1).await;
    let mut client = fixture.client().await;
    let (_catalog_stream, producer) = initial_catalog(&mut client).await;
    let mut first_stream = subscribe_pool(&mut client, producer.clone()).await.unwrap();
    receive_snapshot(&mut first_stream, &producer).await;

    let rejected = subscribe_pool(&mut client, producer.clone())
        .await
        .unwrap_err();
    assert_eq!(rejected.code(), tonic::Code::ResourceExhausted);

    drop(first_stream);
    let mut replacement_stream = tokio::time::timeout(IO_TIMEOUT, async {
        loop {
            match subscribe_pool(&mut client, producer.clone()).await {
                Ok(stream) => break stream,
                Err(status) if status.code() == tonic::Code::ResourceExhausted => {
                    tokio::task::yield_now().await;
                }
                Err(status) => panic!("resubscribe failed: {status}"),
            }
        }
    })
    .await
    .expect("pool stream permit was not released");
    receive_snapshot(&mut replacement_stream, &producer).await;

    drop((client, replacement_stream));
    fixture.shutdown().await;
}
