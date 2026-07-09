// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{super::*, common::*};
use tokio::{
    net::TcpListener,
    sync::{Barrier, oneshot},
};

async fn assert_cancelled_ordinary_lane_reconnects(lane: OrdinaryTestLane) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let (partial_sent, partial_received) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut first).await,
            vec![b"PING".to_vec()]
        );
        first.write_all(b"+STALE").await.unwrap();
        first.flush().await.unwrap();
        partial_sent.send(()).unwrap();

        assert_test_connection_closed(&mut first, "cancelled command").await;

        let (mut second, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut second).await,
            vec![b"PING".to_vec()]
        );
        second.write_all(b"+PONG\r\n").await.unwrap();
    });

    let primary = Arc::new(ValkeyPrimary::new(endpoint, 1));
    let command_primary = Arc::clone(&primary);
    let command = tokio::spawn(async move { ordinary_test_command(&command_primary, lane).await });
    partial_received.await.unwrap();
    command.abort();
    assert!(command.await.unwrap_err().is_cancelled());

    let response = timeout(
        Duration::from_secs(2),
        ordinary_test_command(&primary, lane),
    )
    .await
    .expect("replacement command timed out")
    .unwrap();
    assert!(matches!(response, RespValue::Simple(value) if value == "PONG"));
    server.await.unwrap();
}

async fn assert_cancelled_wait_lane_reconnects(lane: WaitTestLane) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let (partial_sent, partial_received) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        let mutation = read_test_resp_request(&mut first).await;
        assert_eq!(mutation[0], b"DYNKV.APPLY");
        first.write_all(b"+OK\r\n").await.unwrap();
        let wait = read_test_resp_request(&mut first).await;
        assert_eq!(wait[0], b"WAIT");
        first.write_all(b":").await.unwrap();
        first.flush().await.unwrap();
        partial_sent.send(()).unwrap();

        assert_test_connection_closed(&mut first, "cancelled WAIT").await;

        let (mut second, _) = listener.accept().await.unwrap();
        let mutation = read_test_resp_request(&mut second).await;
        assert_eq!(mutation[0], b"DYNKV.APPLY");
        second.write_all(b"+OK\r\n").await.unwrap();
        let wait = read_test_resp_request(&mut second).await;
        assert_eq!(wait[0], b"WAIT");
        second.write_all(b":1\r\n").await.unwrap();
    });

    let primary = Arc::new(ValkeyPrimary::new(endpoint, 1));
    let command_primary = Arc::clone(&primary);
    let command = tokio::spawn(async move { wait_test_command(&command_primary, lane).await });
    partial_received.await.unwrap();
    command.abort();
    assert!(command.await.unwrap_err().is_cancelled());

    let response = timeout(Duration::from_secs(2), wait_test_command(&primary, lane))
        .await
        .expect("replacement mutation/WAIT timed out")
        .unwrap();
    assert!(matches!(response, RespValue::Simple(value) if value == "OK"));
    server.await.unwrap();
}

#[tokio::test]
async fn cancellation_discards_partial_response_on_every_ordinary_pool() {
    for lane in [
        OrdinaryTestLane::Writer,
        OrdinaryTestLane::Reader,
        OrdinaryTestLane::AdmissionSelect,
        OrdinaryTestLane::AdmissionLifecycle,
    ] {
        assert_cancelled_ordinary_lane_reconnects(lane).await;
    }
}

#[tokio::test]
async fn cancellation_discards_partial_wait_on_every_mutation_pool() {
    for lane in [
        WaitTestLane::Writer,
        WaitTestLane::AdmissionSelect,
        WaitTestLane::AdmissionLifecycle,
    ] {
        assert_cancelled_wait_lane_reconnects(lane).await;
    }
}

#[tokio::test]
async fn direct_event_lanes_run_apply_and_wait_in_parallel() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (first, _) = listener.accept().await.unwrap();
        let (second, _) = listener.accept().await.unwrap();
        let both_mutations_received = Arc::new(Barrier::new(2));

        let serve = |mut stream: TcpStream, barrier: Arc<Barrier>| async move {
            let mutation = read_test_resp_request(&mut stream).await;
            assert_eq!(mutation[0], b"DYNKV.APPLY_OWNED");
            barrier.wait().await;
            stream.write_all(b"+OK\r\n").await.unwrap();

            let wait = read_test_resp_request(&mut stream).await;
            assert_eq!(wait[0], b"WAIT");
            assert_eq!(wait[1], b"1");
            stream.write_all(b":1\r\n").await.unwrap();
        };
        tokio::join!(
            serve(first, Arc::clone(&both_mutations_received)),
            serve(second, both_mutations_received)
        );
    });

    let indexer = ValkeyIndexer::new_worker(
        &endpoint,
        2,
        Some(1),
        "test",
        "worker",
        Some("parallel-direct-events"),
        16,
        CancellationToken::new(),
    )
    .unwrap();
    let first_indexer = indexer.clone();
    let first = tokio::spawn(async move {
        first_indexer
            .apply_event_owned(&direct_event_test_event(1, 0), 11)
            .await
    });
    let second = tokio::spawn(async move {
        indexer
            .apply_event_owned(&direct_event_test_event(2, 1), 11)
            .await
    });

    let (first, second) = timeout(Duration::from_secs(2), async {
        tokio::join!(first, second)
    })
    .await
    .expect("two direct event lanes should make progress together");
    first.unwrap().unwrap();
    second.unwrap().unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn direct_event_batch_pipelines_every_apply_before_one_wait() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        for _ in 0..3 {
            let apply = read_test_resp_request(&mut stream).await;
            assert_eq!(apply[0], b"DYNKV.APPLY_OWNED");
        }
        let wait = read_test_resp_request(&mut stream).await;
        assert_eq!(wait[0], b"WAIT");
        assert_eq!(wait[1], b"1");
        // No reply is sent until the complete APPLY×N,WAIT sequence has
        // arrived, proving the client did not insert read RTTs between it.
        stream
            .write_all(b"+OK\r\n+OK\r\n+OK\r\n:1\r\n")
            .await
            .unwrap();
    });

    let indexer = ValkeyIndexer::new_worker(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("pipeline-order"),
        16,
        CancellationToken::new(),
    )
    .unwrap();
    let events = (1..=3)
        .map(|event_id| direct_event_test_event(event_id, event_id as u32 - 1))
        .collect::<Vec<_>>();
    indexer.apply_events_owned(&events, 11).await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn direct_event_batch_retry_replays_all_applies_then_one_barrier_and_wait() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let first_apply = read_test_resp_request(&mut stream).await;
        let second_apply = read_test_resp_request(&mut stream).await;
        assert_eq!(first_apply[0], b"DYNKV.APPLY_OWNED");
        assert_eq!(second_apply[0], b"DYNKV.APPLY_OWNED");
        assert_eq!(read_test_resp_request(&mut stream).await[0], b"WAIT");
        stream.write_all(b"+OK\r\n+OK\r\n:0\r\n").await.unwrap();

        assert_eq!(read_test_resp_request(&mut stream).await, first_apply);
        assert_eq!(read_test_resp_request(&mut stream).await, second_apply);
        assert_eq!(
            read_test_resp_request(&mut stream).await[0],
            b"DYNKV.BARRIER"
        );
        assert_eq!(read_test_resp_request(&mut stream).await[0], b"WAIT");
        stream
            .write_all(b"+NOOP\r\n+NOOP\r\n+OK\r\n:1\r\n")
            .await
            .unwrap();
    });

    let indexer = ValkeyIndexer::new_worker(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("pipeline-retry"),
        16,
        CancellationToken::new(),
    )
    .unwrap();
    indexer
        .apply_events_owned(
            &[direct_event_test_event(1, 0), direct_event_test_event(2, 1)],
            11,
        )
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn cancellation_discards_partial_direct_event_batch_pipeline() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let (partial_sent, partial_received) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut first, _) = listener.accept().await.unwrap();
        for expected in [b"DYNKV.APPLY_OWNED".as_slice(); 2] {
            assert_eq!(read_test_resp_request(&mut first).await[0], expected);
        }
        assert_eq!(read_test_resp_request(&mut first).await[0], b"WAIT");
        first.write_all(b"+OK\r\n+PARTIAL").await.unwrap();
        first.flush().await.unwrap();
        partial_sent.send(()).unwrap();
        assert_test_connection_closed(&mut first, "cancelled APPLY pipeline").await;

        let (mut second, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut second).await[0],
            b"DYNKV.APPLY_OWNED"
        );
        assert_eq!(
            read_test_resp_request(&mut second).await[0],
            b"DYNKV.APPLY_OWNED"
        );
        assert_eq!(read_test_resp_request(&mut second).await[0], b"WAIT");
        second.write_all(b"+OK\r\n+OK\r\n:1\r\n").await.unwrap();
    });

    let indexer = Arc::new(
        ValkeyIndexer::new_worker(
            &endpoint,
            1,
            Some(1),
            "test",
            "worker",
            Some("pipeline-cancel"),
            16,
            CancellationToken::new(),
        )
        .unwrap(),
    );
    let events = vec![direct_event_test_event(1, 0), direct_event_test_event(2, 1)];
    let cancelled_indexer = Arc::clone(&indexer);
    let cancelled_events = events.clone();
    let command = tokio::spawn(async move {
        cancelled_indexer
            .apply_events_owned(&cancelled_events, 11)
            .await
    });
    partial_received.await.unwrap();
    command.abort();
    assert!(command.await.unwrap_err().is_cancelled());

    timeout(
        Duration::from_secs(2),
        indexer.apply_events_owned(&events, 11),
    )
    .await
    .expect("replacement APPLY pipeline timed out")
    .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn resp_parser_accepts_nested_module_arrays() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        assert_eq!(
            read_test_resp_request(&mut stream).await,
            vec![b"DYNKV.GC".to_vec()]
        );
        stream
            .write_all(b"*3\r\n:7\r\n$3\r\nabc\r\n*2\r\n+OK\r\n$-1\r\n")
            .await
            .unwrap();
    });

    let primary = ValkeyPrimary::new(endpoint, 1);
    let response = primary.command_read(&[b"DYNKV.GC"]).await.unwrap();
    let RespValue::Array(values) = response else {
        panic!("expected an array response");
    };
    assert!(matches!(values.first(), Some(RespValue::Integer(7))));
    assert!(matches!(values.get(1), Some(RespValue::Bulk(value)) if value == b"abc"));
    assert!(matches!(
        values.get(2),
        Some(RespValue::Array(nested))
            if matches!(nested.first(), Some(RespValue::Simple(value)) if value == "OK")
                && matches!(nested.get(1), Some(RespValue::Null))
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn gc_step_uses_current_watermark_and_waits_for_its_replica() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let gc = read_test_resp_request(&mut stream).await;
        assert_eq!(gc[0], b"DYNKV.GC");
        assert_eq!(gc[2], b"CURRENT");
        assert_eq!(gc[3], 257_u32.to_be_bytes());
        stream
            .write_all(b"*8\r\n:257\r\n:5\r\n:1\r\n:1\r\n:1\r\n:1\r\n:0\r\n:2\r\n")
            .await
            .unwrap();

        let wait = read_test_resp_request(&mut stream).await;
        assert_eq!(wait[0], b"WAIT");
        assert_eq!(wait[1], b"1");
        assert_eq!(wait[2], REPLICATION_WAIT_TIMEOUT_MS.to_string().as_bytes());
        stream.write_all(b":1\r\n").await.unwrap();
    });

    let indexer = ValkeyIndexer::new(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("gc"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(
        indexer.gc_step(257).await.unwrap(),
        [257, 5, 1, 1, 1, 1, 0, 2]
    );
    server.await.unwrap();
}

#[tokio::test]
async fn gc_step_does_not_retry_an_ambiguous_wait() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let gc = read_test_resp_request(&mut stream).await;
        assert_eq!(gc[0], b"DYNKV.GC");
        stream
            .write_all(b"*8\r\n:1\r\n:1\r\n:1\r\n:0\r\n:0\r\n:0\r\n:0\r\n:1\r\n")
            .await
            .unwrap();
        let wait = read_test_resp_request(&mut stream).await;
        assert_eq!(wait[0], b"WAIT");
        stream.write_all(b":0\r\n").await.unwrap();

        let mut next = [0_u8; 1];
        assert!(
            timeout(Duration::from_millis(100), stream.read(&mut next))
                .await
                .is_err(),
            "GC retried on the same connection"
        );
    });

    let indexer = ValkeyIndexer::new(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("gc-no-retry"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    let error = indexer.gc_step(1).await.unwrap_err();
    assert!(error.to_string().contains("replication quorum not met"));
    server.await.unwrap();
}

#[tokio::test]
async fn worker_registration_refreshes_a_stale_v3_generation_and_waits() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let owner = 0x0102_0304_0506_0708_u64;
    let worker_id = 77_u64;
    let server = tokio::spawn(async move {
        // Registration-generation reads and lifecycle writes deliberately
        // use independent cancellation-safe pool lanes.
        let (mut reader, _) = listener.accept().await.unwrap();
        let first_query = read_test_resp_request(&mut reader).await;
        assert_eq!(first_query[0], b"DYNKV.REGISTRATION_GENERATION");
        assert_eq!(first_query[2], worker_id.to_be_bytes());
        reader
            .write_all(b"$8\r\n\0\0\0\0\0\0\0\x05\r\n")
            .await
            .unwrap();

        let (mut writer, _) = listener.accept().await.unwrap();
        let first_register = read_test_resp_request(&mut writer).await;
        assert_eq!(first_register[0], b"DYNKV.REGISTER_WORKER_RANKS");
        assert_eq!(first_register[2], worker_id.to_be_bytes());
        assert_eq!(
            first_register[3][0],
            WORKER_LEASED_REGISTRATION_WIRE_VERSION
        );
        assert_eq!(&first_register[3][1..9], &owner.to_be_bytes());
        assert_eq!(&first_register[3][17..25], &5_u64.to_be_bytes());
        writer
            .write_all(b"-DYNKV_STALE_REGISTRATION_GENERATION\r\n")
            .await
            .unwrap();
        assert_test_connection_closed(&mut writer, "stale registration CAS").await;

        let second_query = read_test_resp_request(&mut reader).await;
        assert_eq!(second_query[0], b"DYNKV.REGISTRATION_GENERATION");
        reader
            .write_all(b"$8\r\n\0\0\0\0\0\0\0\x09\r\n")
            .await
            .unwrap();

        let (mut writer, _) = listener.accept().await.unwrap();
        let second_register = read_test_resp_request(&mut writer).await;
        assert_eq!(&second_register[3][17..25], &9_u64.to_be_bytes());
        writer.write_all(b"+OK\r\n").await.unwrap();
        let wait = read_test_resp_request(&mut writer).await;
        assert_eq!(wait[0], b"WAIT");
        writer.write_all(b":1\r\n").await.unwrap();
    });

    let indexer = ValkeyIndexer::new(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("registration-v3"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    indexer
        .register_worker_lease(worker_id, owner, 30_000, 257, &[0, 1])
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn worker_registration_advances_pending_cleanup_before_retrying() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let owner = 0x1112_1314_1516_1718_u64;
    let worker_id = 91_u64;
    let server = tokio::spawn(async move {
        let (mut reader, _) = listener.accept().await.unwrap();
        let first_query = read_test_resp_request(&mut reader).await;
        assert_eq!(first_query[0], b"DYNKV.REGISTRATION_GENERATION");
        reader
            .write_all(b"$8\r\n\0\0\0\0\0\0\0\x05\r\n")
            .await
            .unwrap();

        let (mut writer, _) = listener.accept().await.unwrap();
        let first_register = read_test_resp_request(&mut writer).await;
        assert_eq!(first_register[0], b"DYNKV.REGISTER_WORKER_RANKS");
        writer
            .write_all(b"-DYNKV_WORKER_CLEANUP_PENDING\r\n")
            .await
            .unwrap();
        assert_test_connection_closed(&mut writer, "pending registration cleanup").await;

        let (mut cleanup_writer, _) = listener.accept().await.unwrap();
        let gc = read_test_resp_request(&mut cleanup_writer).await;
        assert_eq!(gc[0], b"DYNKV.GC");
        assert_eq!(gc[2], b"CURRENT");
        assert_eq!(gc[3], 257_u32.to_be_bytes());
        cleanup_writer
            .write_all(b"*8\r\n:257\r\n:1\r\n:0\r\n:0\r\n:0\r\n:0\r\n:1\r\n:2\r\n")
            .await
            .unwrap();
        let gc_wait = read_test_resp_request(&mut cleanup_writer).await;
        assert_eq!(gc_wait[0], b"WAIT");
        cleanup_writer.write_all(b":1\r\n").await.unwrap();

        let second_query = read_test_resp_request(&mut reader).await;
        assert_eq!(second_query[0], b"DYNKV.REGISTRATION_GENERATION");
        reader
            .write_all(b"$8\r\n\0\0\0\0\0\0\0\x09\r\n")
            .await
            .unwrap();

        let second_register = read_test_resp_request(&mut cleanup_writer).await;
        assert_eq!(second_register[0], b"DYNKV.REGISTER_WORKER_RANKS");
        assert_eq!(&second_register[3][17..25], &9_u64.to_be_bytes());
        cleanup_writer.write_all(b"+OK\r\n").await.unwrap();
        let register_wait = read_test_resp_request(&mut cleanup_writer).await;
        assert_eq!(register_wait[0], b"WAIT");
        cleanup_writer.write_all(b":1\r\n").await.unwrap();
    });

    let indexer = ValkeyIndexer::new(
        &endpoint,
        1,
        Some(1),
        "test",
        "worker",
        Some("registration-cleanup"),
        None,
        16,
        CancellationToken::new(),
    )
    .unwrap();
    indexer
        .register_worker_lease(worker_id, owner, 30_000, 257, &[0, 1])
        .await
        .unwrap();
    server.await.unwrap();
}
