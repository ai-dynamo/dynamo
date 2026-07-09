// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::lease::{renewal_schedule, renewal_should_stop};
use super::super::*;
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, StorageTier,
};

#[test]
fn admission_dimensions_match_the_module_wire_contract() {
    assert!(
        validate_reservation_dimensions(
            MAX_ADMISSION_DOMAIN_LENGTH,
            MAX_ADMISSION_PREFIX_HASHES,
            MAX_ADMISSION_CANDIDATES,
            MAX_ADMISSION_LEASE_MS,
        )
        .is_ok()
    );
    assert!(
        validate_reservation_dimensions(
            MAX_ADMISSION_DOMAIN_LENGTH,
            MAX_ADMISSION_PREFIX_HASHES + 1,
            MAX_ADMISSION_CANDIDATES,
            MAX_ADMISSION_LEASE_MS,
        )
        .is_err()
    );
    assert!(
        validate_reservation_dimensions(
            MAX_ADMISSION_DOMAIN_LENGTH,
            MAX_ADMISSION_PREFIX_HASHES,
            MAX_ADMISSION_CANDIDATES + 1,
            MAX_ADMISSION_LEASE_MS,
        )
        .is_err()
    );
}

#[test]
fn parses_valkey_endpoint_variants() {
    assert_eq!(parse_endpoint("127.0.0.1:6379").unwrap(), "127.0.0.1:6379");
    assert_eq!(
        parse_endpoint("valkey://localhost:6380/").unwrap(),
        "localhost:6380"
    );
    assert!(parse_endpoint("valkey://user@localhost:6379").is_err());
    assert!(parse_endpoint("localhost").is_err());
}

#[test]
fn indexer_rejects_duplicate_or_unbounded_endpoint_lists() {
    let build = |urls: &str| {
        ValkeyIndexer::new(
            urls,
            1,
            None,
            "namespace",
            "component",
            Some("scope"),
            None,
            16,
            CancellationToken::new(),
        )
    };
    assert!(build("router:6379,valkey://router:6379").is_err());
    let too_many = (0..65)
        .map(|port| format!("router:{}", 10_000 + port))
        .collect::<Vec<_>>()
        .join(",");
    assert!(build(&too_many).is_err());
}

#[test]
fn index_key_isolates_component_and_encodes_tuple_segments() {
    let indexer = ValkeyIndexer::new(
        "127.0.0.1:6379",
        1,
        None,
        "team:blue",
        "back/end%1",
        Some("shared:index/α"),
        Some("model/available-only-to-frontend"),
        16,
        CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(
        std::str::from_utf8(&indexer.inner.index_key).unwrap(),
        "dynamo:kv-router:team%3Ablue:component-back%2Fend%251:scope-shared%3Aindex%2F%CE%B1:block-size-16"
    );

    let worker_indexer = ValkeyIndexer::new(
        "127.0.0.1:6379",
        1,
        None,
        "team:blue",
        "back/end%1",
        Some("shared:index/α"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(indexer.inner.index_key, worker_indexer.inner.index_key);

    let other_component = ValkeyIndexer::new(
        "127.0.0.1:6379",
        1,
        None,
        "team:blue",
        "other-backend",
        Some("shared:index/α"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    assert_ne!(
        worker_indexer.inner.index_key,
        other_component.inner.index_key
    );
}

#[test]
fn index_key_rejects_empty_tuple_segments() {
    for (namespace, component, scope) in [
        (" ", "backend", Some("shared")),
        ("dynamo", " ", Some("shared")),
        ("dynamo", "backend", Some(" ")),
    ] {
        assert!(
            ValkeyIndexer::new(
                "127.0.0.1:6379",
                1,
                None,
                namespace,
                component,
                scope,
                None,
                16,
                CancellationToken::new(),
            )
            .is_err()
        );
    }
}

#[test]
fn required_replica_acks_preserve_legacy_inference_and_allow_stable_primary() {
    assert_eq!(resolve_required_replica_acks(None, 1).unwrap(), 0);
    assert_eq!(resolve_required_replica_acks(None, 2).unwrap(), 1);
    assert_eq!(resolve_required_replica_acks(None, 8).unwrap(), 1);
    assert_eq!(resolve_required_replica_acks(Some(1), 1).unwrap(), 1);
    assert_eq!(resolve_required_replica_acks(Some(0), 2).unwrap(), 0);
    assert!(resolve_required_replica_acks(None, 0).is_err());
    assert!(resolve_required_replica_acks(Some(MAX_REQUIRED_REPLICA_ACKS + 1), 1).is_err());
}

#[test]
fn event_and_match_wire_formats_round_trip() {
    let event = RouterEvent::with_storage_tier(
        7,
        KvCacheEvent {
            event_id: 9,
            dp_rank: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(101),
                    tokens_hash: LocalBlockHash(11),
                    mm_extra_info: None,
                }],
            }),
        },
        StorageTier::Device,
    );
    let encoded = encode_event(&event);
    assert_eq!(encoded[0], WIRE_VERSION);
    assert_eq!(encoded[1], EVENT_STORE);

    let response = [
        &[WIRE_VERSION][..],
        &(1_u32.to_be_bytes()),
        &(7_u64.to_be_bytes()),
        &(2_u32.to_be_bytes()),
        &(1_u32.to_be_bytes()),
        &(101_u64.to_be_bytes()),
    ]
    .concat();
    let details = decode_match(&response).unwrap();
    let worker = WorkerWithDpRank::new(7, 2);
    assert_eq!(details.overlap_scores.scores[&worker], 1);
    assert_eq!(
        details.last_matched_hashes[&worker],
        ExternalSequenceBlockHash(101)
    );
}

#[test]
fn accepts_idempotent_module_noop() {
    assert!(ensure_ok(Ok(RespValue::Simple("NOOP".to_string()))).is_ok());
    assert!(retryable_write_error(&RespError::Timeout));
    assert!(!retryable_write_error(&RespError::Server(
        "DYNKV_INVALID_EVENT".to_string()
    )));
    for message in [
        "READONLY You can't write against a read only replica.",
        "MASTERDOWN Link with MASTER is down",
        "DYNKV_NOT_PRIMARY",
        "ERR WAIT cannot be used with replica instances",
    ] {
        let error = RespError::Server(message.to_string());
        assert!(topology_error(&error));
        assert!(retryable_write_error(&error));
        assert!(retryable_primary_read_error(&error));
    }
}

#[test]
fn worker_rank_set_is_sorted_unique_and_bounded() {
    validate_worker_ranks(&[3, 4, u32::MAX]).unwrap();
    assert!(validate_worker_ranks(&[]).is_err());
    assert!(validate_worker_ranks(&[4, 4]).is_err());
    assert!(validate_worker_ranks(&[5, 4]).is_err());
}

#[test]
fn worker_lease_wire_encodes_owner_duration_and_rank_set() {
    let owner = 0x0102_0304_0506_0708;
    let expected_generation = 0x1112_1314_1516_1718;
    let encoded =
        encode_worker_lease_registration(owner, 30_000, expected_generation, &[3, 4]).unwrap();
    let expected = [
        &[WORKER_LEASED_REGISTRATION_WIRE_VERSION][..],
        &owner.to_be_bytes(),
        &30_000_u64.to_be_bytes(),
        &expected_generation.to_be_bytes(),
        &2_u32.to_be_bytes(),
        &3_u32.to_be_bytes(),
        &4_u32.to_be_bytes(),
    ]
    .concat();
    assert_eq!(encoded, expected);

    let renew = encode_worker_lease_control(77, owner, Some(30_000)).unwrap();
    assert_eq!(renew[0], WORKER_LEASE_CONTROL_VERSION);
    assert_eq!(&renew[1..9], &77_u64.to_be_bytes());
    assert_eq!(&renew[9..17], &owner.to_be_bytes());
    assert_eq!(&renew[17..25], &30_000_u64.to_be_bytes());
    let unregister = encode_worker_lease_control(77, owner, None).unwrap();
    assert_eq!(&unregister, &renew[..17]);

    assert!(encode_worker_lease_registration(0, 30_000, 0, &[3]).is_err());
    assert!(encode_worker_lease_registration(owner, 0, 0, &[3]).is_err());
    assert!(encode_worker_lease_registration(owner, 30_000, 0, &[3, 3]).is_err());
}

#[test]
fn registration_and_gc_replies_are_strictly_decoded() {
    let generation = 0x0102_0304_0506_0708_u64;
    assert_eq!(
        decode_u64_bulk(
            RespValue::Bulk(generation.to_be_bytes().to_vec()),
            "DYNKV.REGISTRATION_GENERATION"
        )
        .unwrap(),
        generation
    );
    assert!(decode_u64_bulk(RespValue::Bulk(vec![0; 7]), "test").is_err());

    let reply = RespValue::Array((0_i64..8_i64).map(RespValue::Integer).collect::<Vec<_>>());
    assert_eq!(decode_gc_reply(reply).unwrap(), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(decode_gc_reply(RespValue::Array(vec![])).is_err());
    assert!(
        decode_gc_reply(RespValue::Array(
            (0..8).map(|_| RespValue::Integer(-1)).collect()
        ))
        .is_err()
    );
}

fn reservation_request() -> ReservationRequest {
    ReservationRequest {
        domain: b"decode".to_vec(),
        nonce: ReservationNonce {
            client_nonce: 0x0102_0304_0506_0708,
            request_nonce: 0x1112_1314_1516_1718,
        },
        lease_ms: 30_000,
        block_hashes: vec![LocalBlockHash(7), LocalBlockHash(11)],
        candidates: vec![
            ValkeyReservationCandidate {
                worker: WorkerWithDpRank::new(10, 2),
                capacity: 64,
            },
            ValkeyReservationCandidate {
                worker: WorkerWithDpRank::new(12, 3),
                capacity: 128,
            },
        ],
    }
}

#[test]
fn reservation_wire_encodes_identity_prefix_and_registered_capacity() {
    let request = reservation_request();
    let encoded = encode_select_reserve(&request).unwrap();
    let expected = [
        &[ADMISSION_WIRE_VERSION][..],
        &(6_u32.to_be_bytes()),
        b"decode",
        &(request.nonce.client_nonce.to_be_bytes()),
        &(request.nonce.request_nonce.to_be_bytes()),
        &(30_000_u64.to_be_bytes()),
        &(2_u32.to_be_bytes()),
        &(7_u64.to_be_bytes()),
        &(11_u64.to_be_bytes()),
        &(2_u32.to_be_bytes()),
        &(10_u64.to_be_bytes()),
        &(2_u32.to_be_bytes()),
        &(64_u32.to_be_bytes()),
        &(12_u64.to_be_bytes()),
        &(3_u32.to_be_bytes()),
        &(128_u32.to_be_bytes()),
    ]
    .concat();
    assert_eq!(encoded, expected);

    let release = encode_release(&request, 99);
    assert_eq!(
        &release[..1 + 4 + 6 + 8 + 8],
        &expected[..1 + 4 + 6 + 8 + 8]
    );
    assert_eq!(&release[release.len() - 8..], &99_u64.to_be_bytes());

    let renew = encode_renew(&request, 99);
    assert_eq!(&renew[..release.len()], &release);
    assert_eq!(&renew[renew.len() - 8..], &30_000_u64.to_be_bytes());
}

#[test]
fn reservation_reply_validates_nonce_and_decodes_grant() {
    let request = reservation_request();
    let payload = [
        &[ADMISSION_WIRE_VERSION, ADMISSION_RESERVED][..],
        &(request.nonce.client_nonce.to_be_bytes()),
        &(request.nonce.request_nonce.to_be_bytes()),
        &(12_u64.to_be_bytes()),
        &(3_u32.to_be_bytes()),
        &(123_456_u64.to_be_bytes()),
        &(9_u32.to_be_bytes()),
        &(5_u32.to_be_bytes()),
    ]
    .concat();
    let grant = decode_reservation_reply(&payload, request.nonce)
        .unwrap()
        .expect("reserved reply");
    assert_eq!(grant.worker, WorkerWithDpRank::new(12, 3));
    assert_eq!(grant.expires_at_ms, 123_456);
    assert_eq!(grant.matched_blocks, 9);
    assert_eq!(grant.active_reservations_at_grant, 5);
    assert!(
        decode_reservation_reply(
            &[ADMISSION_WIRE_VERSION, ADMISSION_NO_CAPACITY],
            request.nonce
        )
        .unwrap()
        .is_none()
    );
    assert!(decode_reservation_reply(&payload, ReservationNonce::random()).is_err());
    assert!(decode_admission_status(&[ADMISSION_WIRE_VERSION, ADMISSION_RESERVED]).unwrap());
    assert!(!decode_admission_status(&[ADMISSION_WIRE_VERSION, ADMISSION_NO_CAPACITY]).unwrap());
}

#[test]
fn retries_transient_valkey_server_errors() {
    for message in [
        "NOREPLICAS Not enough good replicas to write.",
        "LOADING Redis is loading the dataset in memory",
        "TRYAGAIN Multiple keys request during rehashing of slot",
        "MASTERDOWN Link with MASTER is down",
        "CLUSTERDOWN The cluster is down",
    ] {
        assert!(retryable_write_error(&RespError::Server(
            message.to_string()
        )));
    }
    assert!(!retryable_write_error(&RespError::Server(
        "DYNKV_MISSING_PARENT".to_string()
    )));
}

#[test]
fn recognizes_an_expired_reservation_as_an_idempotent_lease_outcome() {
    let expired: anyhow::Error =
        RespError::Server("DYNKV_RESERVATION_EXPIRED stale lease".to_string()).into();
    assert!(reservation_expired_error(&expired));

    let other: anyhow::Error = RespError::Server("DYNKV_UNKNOWN_LEASE".to_string()).into();
    assert!(!reservation_expired_error(&other));
}

#[test]
fn recognizes_only_the_exact_worker_cleanup_pending_error() {
    let pending: anyhow::Error =
        RespError::Server("DYNKV_WORKER_CLEANUP_PENDING worker 91".to_string()).into();
    assert!(worker_cleanup_pending_error(&pending));

    let other: anyhow::Error =
        RespError::Server("DYNKV_STALE_REGISTRATION_GENERATION".to_string()).into();
    assert!(!worker_cleanup_pending_error(&other));
}

#[test]
fn frontend_mutation_lanes_do_not_scale_with_match_reader_pool() {
    for pool_size in [1, 8, 64] {
        let primary = ValkeyPrimary::new("127.0.0.1:6379".to_string(), pool_size);
        assert!(primary.direct_event_writers.is_empty());
        assert_eq!(primary.admission_select_writers.len(), 4);
        assert_eq!(primary.admission_lifecycle_writers.len(), 4);
    }
}

#[test]
fn renewal_schedule_is_deterministic_and_attempts_early() {
    let nonce = ReservationNonce {
        client_nonce: 0x0123_4567_89ab_cdef,
        request_nonce: 0xfedc_ba98_7654_3210,
    };
    let first = renewal_schedule(30_000, nonce);
    let repeated = renewal_schedule(30_000, nonce);

    assert_eq!(first, repeated);
    assert!(first.0 >= Duration::from_millis(30_000 / 16));
    assert!(first.0 <= Duration::from_millis(30_000 / 4));
    assert_eq!(first.1, Duration::from_millis(30_000 / 3));

    // The minimum supported lease keeps the same deadline margin: the
    // first renewal starts well before one quarter of the lease expires.
    let (minimum_first, minimum_interval) = renewal_schedule(MIN_ADMISSION_LEASE_MS, nonce);
    assert!(minimum_first >= Duration::from_millis(MIN_ADMISSION_LEASE_MS / 16));
    assert!(minimum_first <= Duration::from_millis(MIN_ADMISSION_LEASE_MS / 4));
    assert_eq!(
        minimum_interval,
        Duration::from_millis(MIN_ADMISSION_LEASE_MS / 3)
    );
}

#[test]
fn detached_release_cancels_renewal_before_its_release_task_can_run() {
    let request = reservation_request();
    let grant = ReservationGrant {
        worker: request.candidates[0].worker,
        expires_at_ms: 30_000,
        matched_blocks: 0,
        active_reservations_at_grant: 0,
    };
    let renewal_cancel = CancellationToken::new();
    let indexer = ValkeyIndexer::new(
        "127.0.0.1:6379",
        1,
        None,
        "test",
        "frontend",
        None,
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    let inner = Arc::new(ReservationLeaseInner {
        indexer,
        request,
        grant,
        state: Mutex::new(ReservationLeaseState {
            expires_at_ms: grant.expires_at_ms,
            released: false,
        }),
        lifecycle: Mutex::new(()),
        renewal_cancel: renewal_cancel.clone(),
        renewal_started: AtomicBool::new(false),
        release_started: AtomicBool::new(false),
    });

    // A plain unit test has no Tokio runtime, so this also verifies that
    // cancellation happens before the no-runtime release fallback.
    ValkeyReservationLease {
        inner: Arc::clone(&inner),
    }
    .release_detached();

    assert!(renewal_cancel.is_cancelled());
    assert!(inner.release_started.load(Ordering::Acquire));
    let state = inner
        .state
        .try_lock()
        .expect("release task was not spawned");
    assert!(renewal_should_stop(&inner, &state));
}

#[tokio::test]
async fn detached_admission_cleanup_tasks_are_bounded_and_expire() {
    let indexer = ValkeyIndexer::new(
        "127.0.0.1:1",
        1,
        None,
        "test",
        "frontend",
        None,
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(
        indexer.inner.admission_cleanup_permits.available_permits(),
        MAX_PENDING_ADMISSION_CLEANUPS
    );

    let permits = (0..MAX_PENDING_ADMISSION_CLEANUPS)
        .map(|_| {
            Arc::clone(&indexer.inner.admission_cleanup_permits)
                .try_acquire_owned()
                .expect("configured cleanup permit")
        })
        .collect::<Vec<_>>();
    assert!(
        Arc::clone(&indexer.inner.admission_cleanup_permits)
            .try_acquire_owned()
            .is_err(),
        "cleanup task count must have a hard cap"
    );
    drop(permits);

    let mut request = reservation_request();
    request.lease_ms = 25;
    drop(PendingValkeyReservation {
        indexer: indexer.clone(),
        request: request.clone(),
        armed: true,
    });
    assert_eq!(
        indexer.inner.admission_cleanup_permits.available_permits(),
        MAX_PENDING_ADMISSION_CLEANUPS - 1
    );
    wait_for_cleanup_permits(&indexer).await;

    let grant = ReservationGrant {
        worker: request.candidates[0].worker,
        expires_at_ms: 30_000,
        matched_blocks: 0,
        active_reservations_at_grant: 0,
    };
    ValkeyReservationLease {
        inner: Arc::new(ReservationLeaseInner {
            indexer: indexer.clone(),
            request,
            grant,
            state: Mutex::new(ReservationLeaseState {
                expires_at_ms: grant.expires_at_ms,
                released: false,
            }),
            lifecycle: Mutex::new(()),
            renewal_cancel: CancellationToken::new(),
            renewal_started: AtomicBool::new(false),
            release_started: AtomicBool::new(false),
        }),
    }
    .release_detached();
    assert_eq!(
        indexer.inner.admission_cleanup_permits.available_permits(),
        MAX_PENDING_ADMISSION_CLEANUPS - 1
    );
    wait_for_cleanup_permits(&indexer).await;
}

#[tokio::test]
async fn release_does_not_hold_state_lock_while_waiting_for_lifecycle_io() {
    let indexer = ValkeyIndexer::new(
        "127.0.0.1:1",
        1,
        None,
        "test",
        "frontend",
        None,
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    let mut request = reservation_request();
    request.lease_ms = 25;
    let grant = ReservationGrant {
        worker: request.candidates[0].worker,
        expires_at_ms: 30_000,
        matched_blocks: 0,
        active_reservations_at_grant: 0,
    };
    let inner = Arc::new(ReservationLeaseInner {
        indexer,
        request,
        grant,
        state: Mutex::new(ReservationLeaseState {
            expires_at_ms: grant.expires_at_ms,
            released: false,
        }),
        lifecycle: Mutex::new(()),
        renewal_cancel: CancellationToken::new(),
        renewal_started: AtomicBool::new(false),
        release_started: AtomicBool::new(false),
    });
    let lifecycle = inner.lifecycle.lock().await;
    let release = tokio::spawn(super::super::lease::release_reservation_lease(Arc::clone(
        &inner,
    )));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !inner.release_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("release task should start");

    assert!(
        inner.state.try_lock().is_ok(),
        "remote lifecycle serialization must not hold the state lock"
    );
    drop(lifecycle);
    assert!(release.await.unwrap().is_err());
}

async fn wait_for_cleanup_permits(indexer: &ValkeyIndexer) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if indexer.inner.admission_cleanup_permits.available_permits()
                == MAX_PENDING_ADMISSION_CLEANUPS
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("detached cleanup must stop after its lease-sized deadline");
}
