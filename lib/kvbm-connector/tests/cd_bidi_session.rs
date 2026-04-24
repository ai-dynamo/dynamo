// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use kvbm_common::LogicalLayoutHandle;
use kvbm_connector::connector::leader::disagg::{DisaggSession, SessionEvent};
use kvbm_disagg_protocol::{DISAGG_PROTOCOL_VERSION, RemotePrefillRequest};
use kvbm_engine::disagg::{
    BlockSetRequest, BlockSetResponse, HashSelection, PullAck, PullComplete, RemoteBlockRef,
    RemoteBlockSet, UnpinAck, UnpinRequest,
};
use kvbm_hub::{ConditionalDisaggClient, ConditionalDisaggManager};
use velo::backend::tcp::TcpTransportBuilder;
use velo::discovery::PeerDiscovery;

fn new_velo_transport() -> Arc<velo::backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

async fn start_server_with_cd() -> kvbm_hub::HubServer {
    let transport = new_velo_transport();
    let cd_manager: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());

    kvbm_hub::create_server_builder()
        .bind_addr(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(cd_manager as Arc<dyn kvbm_hub::FeatureManager>)
        .serve()
        .await
        .expect("start test server with CD")
}

fn build_client(server: &kvbm_hub::HubServer) -> Arc<kvbm_hub::HubClient> {
    kvbm_hub::create_client_builder()
        .host(server.discovery_addr().ip().to_string())
        .discovery_port(server.discovery_addr().port())
        .control_port(server.control_addr().port())
        .build()
        .expect("build hub client")
}

async fn next_event(
    stream: &mut kvbm_connector::connector::leader::disagg::SessionEventStream,
) -> SessionEvent {
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("timed out waiting for session event")
        .expect("session event stream closed")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cd_prefill_and_decode_coordinate_with_bidi_control_stream() {
    let server = start_server_with_cd().await;

    let p_velo = new_velo().await;
    let d_velo = new_velo().await;

    let p_hub = build_client(&server);
    let d_hub = build_client(&server);
    p_hub.register_handlers(&p_velo).unwrap();
    d_hub.register_handlers(&d_velo).unwrap();

    let p_cd = ConditionalDisaggClient::new(
        Arc::clone(&p_hub),
        Arc::clone(&p_velo),
        kvbm_hub::ConditionalDisaggRole::Prefill,
    );
    let d_cd = ConditionalDisaggClient::new(
        Arc::clone(&d_hub),
        Arc::clone(&d_velo),
        kvbm_hub::ConditionalDisaggRole::Decode,
    );

    let p_hub_id = p_cd
        .register(p_velo.peer_info())
        .await
        .unwrap()
        .expect("hub velo id");
    let d_hub_id = d_cd
        .register(d_velo.peer_info())
        .await
        .unwrap()
        .expect("hub velo id");
    assert_eq!(p_hub_id, d_hub_id);

    tokio::time::sleep(Duration::from_millis(100)).await;

    let p_id = p_velo.instance_id();
    let d_id = d_velo.instance_id();

    let hub_peer = p_hub.discover_by_instance_id(p_hub_id).await.unwrap();
    p_velo.register_peer(hub_peer.clone()).unwrap();
    d_velo.register_peer(hub_peer).unwrap();

    let d_peer = p_cd
        .await_peer_of_role(
            kvbm_hub::ConditionalDisaggRole::Decode,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(d_peer.instance_id(), d_id);
    p_velo.register_peer(d_peer).unwrap();

    let p_peer = d_cd
        .await_peer_of_role(
            kvbm_hub::ConditionalDisaggRole::Prefill,
            Duration::from_millis(50),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    assert_eq!(p_peer.instance_id(), p_id);
    d_velo.register_peer(p_peer).unwrap();

    let session_id = uuid::Uuid::new_v4();
    let decode_session = DisaggSession::create_decode(Arc::clone(&d_velo), session_id);
    let mut decode_events = decode_session.subscribe();

    let req = RemotePrefillRequest {
        protocol_version: DISAGG_PROTOCOL_VERSION,
        request_id: "req-bidi-1".to_string(),
        session_id,
        initiator_instance_id: d_id,
        decode_endpoint: Some(decode_session.endpoint()),
        sequence_hashes: vec!["100".to_string(), "101".to_string()],
        token_ids: vec![1, 2, 3, 4],
        num_computed_tokens: 16,
    };
    d_cd.push_prefill_request(&req).await.unwrap();

    let pulled = p_cd
        .pull_prefill_request(Duration::from_secs(2))
        .await
        .unwrap()
        .expect("prefill should dequeue request");
    assert_eq!(pulled, req);

    let prefill_session = DisaggSession::attach_prefill(
        Arc::clone(&p_velo),
        pulled.session_id,
        pulled
            .decode_endpoint
            .as_ref()
            .expect("decode endpoint carried through queue"),
    )
    .await
    .expect("prefill attaches to decode session");
    let mut prefill_events = prefill_session.subscribe();

    match next_event(&mut decode_events).await {
        SessionEvent::Attached { peer_instance_id } => assert_eq!(peer_instance_id, p_id),
        other => panic!("decode expected attach event, got {other:?}"),
    }

    let block_set_task = {
        let session = Arc::clone(&prefill_session);
        tokio::spawn(async move {
            session
                .request_block_sets(BlockSetRequest {
                    request_id: "blocks-initial".to_string(),
                    hashes: HashSelection::All,
                })
                .await
        })
    };

    match next_event(&mut decode_events).await {
        SessionEvent::BlockSetRequest(request) => assert_eq!(
            request,
            BlockSetRequest {
                request_id: "blocks-initial".to_string(),
                hashes: HashSelection::All,
            }
        ),
        other => panic!("decode expected block-set request, got {other:?}"),
    }

    let selected_hashes = vec!["100".to_string(), "101".to_string()];
    let initial_block_set = RemoteBlockSet {
        source_layout: LogicalLayoutHandle::G2,
        blocks: vec![
            RemoteBlockRef {
                block_id: 7,
                sequence_hash: selected_hashes[0].clone(),
            },
            RemoteBlockRef {
                block_id: 8,
                sequence_hash: selected_hashes[1].clone(),
            },
        ],
    };
    decode_session
        .respond_to_block_set_request(BlockSetResponse {
            request_id: "blocks-initial".to_string(),
            ready: vec![initial_block_set.clone()],
            pending_hashes: Vec::new(),
        })
        .await
        .unwrap();

    let block_set_response = block_set_task.await.unwrap().unwrap();
    assert_eq!(
        block_set_response,
        BlockSetResponse {
            request_id: "blocks-initial".to_string(),
            ready: vec![initial_block_set.clone()],
            pending_hashes: Vec::new(),
        }
    );

    let unpin_task = {
        let session = Arc::clone(&prefill_session);
        tokio::spawn(async move {
            session
                .request_unpin_from_prefill(UnpinRequest {
                    request_id: "unpin-initial".to_string(),
                    hashes: HashSelection::All,
                })
                .await
        })
    };

    match next_event(&mut decode_events).await {
        SessionEvent::UnpinRequested(request) => assert_eq!(
            request,
            UnpinRequest {
                request_id: "unpin-initial".to_string(),
                hashes: HashSelection::All,
            }
        ),
        other => panic!("decode expected unpin request, got {other:?}"),
    }

    decode_session
        .ack_unpin_from_decode(UnpinAck {
            request_id: "unpin-initial".to_string(),
            hashes: HashSelection::All,
        })
        .await
        .unwrap();

    assert_eq!(
        unpin_task.await.unwrap().unwrap(),
        UnpinAck {
            request_id: "unpin-initial".to_string(),
            hashes: HashSelection::All,
        }
    );
    match next_event(&mut prefill_events).await {
        SessionEvent::UnpinAcked(ack) => assert_eq!(
            ack,
            UnpinAck {
                request_id: "unpin-initial".to_string(),
                hashes: HashSelection::All,
            }
        ),
        other => panic!("prefill expected unpin ack event, got {other:?}"),
    }

    let output_block_sets = vec![RemoteBlockSet {
        source_layout: LogicalLayoutHandle::G2,
        blocks: vec![RemoteBlockRef {
            block_id: 99,
            sequence_hash: "200".to_string(),
        }],
    }];
    prefill_session
        .publish_output_block_sets(output_block_sets.clone())
        .await
        .unwrap();

    match next_event(&mut decode_events).await {
        SessionEvent::BlockSetsAdded { block_sets } => assert_eq!(block_sets, output_block_sets),
        other => panic!("decode expected output block sets, got {other:?}"),
    }

    let pull_task = {
        let session = Arc::clone(&decode_session);
        tokio::spawn(async move {
            session
                .pull_complete_from_decode(PullComplete {
                    pull_id: 42,
                    hashes: vec!["200".to_string()],
                })
                .await
        })
    };

    match next_event(&mut prefill_events).await {
        SessionEvent::PullComplete(complete) => assert_eq!(
            complete,
            PullComplete {
                pull_id: 42,
                hashes: vec!["200".to_string()],
            }
        ),
        other => panic!("prefill expected pull complete, got {other:?}"),
    }

    prefill_session
        .ack_pull_from_prefill(PullAck { pull_id: 42 })
        .await
        .unwrap();

    assert_eq!(pull_task.await.unwrap().unwrap(), PullAck { pull_id: 42 });

    prefill_session.finalize().await.unwrap();
    decode_session.finalize().await.unwrap();

    server.shutdown().await.unwrap();
}
