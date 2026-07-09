// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, StorageTier,
};
use tokio::net::TcpListener;

#[derive(Clone, Copy, Debug)]
pub(super) enum OrdinaryTestLane {
    Writer,
    Reader,
    AdmissionSelect,
    AdmissionLifecycle,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum WaitTestLane {
    Writer,
    AdmissionSelect,
    AdmissionLifecycle,
}

pub(super) fn direct_event_test_event(event_id: u64, dp_rank: u32) -> RouterEvent {
    RouterEvent::with_storage_tier(
        7,
        KvCacheEvent {
            event_id,
            dp_rank,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(event_id),
                    tokens_hash: LocalBlockHash(event_id),
                    mm_extra_info: None,
                }],
            }),
        },
        StorageTier::Device,
    )
}

#[test]
pub(super) fn rank_snapshot_wire_is_store_only_and_rank_scoped() {
    let first = direct_event_test_event(1, 3);
    let second = direct_event_test_event(2, 3);
    let encoded_first = encode_event(&first);
    let encoded_second = encode_event(&second);
    let snapshot = encode_rank_snapshot(7, 3, &[first, second]).unwrap();

    assert_eq!(snapshot[0], WIRE_VERSION);
    assert_eq!(u32::from_be_bytes(snapshot[1..5].try_into().unwrap()), 2);
    let mut offset = 5;
    for encoded in [&encoded_first, &encoded_second] {
        let length = u32::from_be_bytes(snapshot[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        assert_eq!(length, encoded.len());
        assert_eq!(&snapshot[offset..offset + length], encoded.as_slice());
        offset += length;
    }
    assert_eq!(offset, snapshot.len());

    assert!(encode_rank_snapshot(7, 4, &[direct_event_test_event(3, 3)]).is_err());
    let mut host = direct_event_test_event(4, 3);
    host.storage_tier = StorageTier::HostPinned;
    assert!(encode_rank_snapshot(7, 3, &[host]).is_err());
}

#[test]
pub(super) fn stale_rank_generation_is_not_a_retryable_transport_error() {
    let error = RespError::Server("DYNKV_STALE_GENERATION".to_string());
    assert!(stale_rank_generation_error(&error));
    assert!(!retryable_write_error(&error));
}

pub(super) async fn read_test_resp_line(stream: &mut TcpStream) -> Vec<u8> {
    let mut line = Vec::new();
    loop {
        let mut byte = [0_u8; 1];
        stream.read_exact(&mut byte).await.unwrap();
        line.push(byte[0]);
        if line.ends_with(b"\r\n") {
            line.truncate(line.len() - 2);
            return line;
        }
    }
}

pub(super) async fn read_test_resp_request(stream: &mut TcpStream) -> Vec<Vec<u8>> {
    let mut marker = [0_u8; 1];
    stream.read_exact(&mut marker).await.unwrap();
    assert_eq!(marker[0], b'*');
    let count = String::from_utf8(read_test_resp_line(stream).await)
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let mut arguments = Vec::with_capacity(count);
    for _ in 0..count {
        stream.read_exact(&mut marker).await.unwrap();
        assert_eq!(marker[0], b'$');
        let length = String::from_utf8(read_test_resp_line(stream).await)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let mut argument = vec![0_u8; length];
        stream.read_exact(&mut argument).await.unwrap();
        let mut terminator = [0_u8; 2];
        stream.read_exact(&mut terminator).await.unwrap();
        assert_eq!(&terminator, b"\r\n");
        arguments.push(argument);
    }
    arguments
}

pub(super) fn test_sentinel_reply(primary: &str) -> Vec<u8> {
    let (host, port) = primary
        .rsplit_once(':')
        .expect("test primary endpoint contains a port");
    format!(
        "*2\r\n${}\r\n{}\r\n${}\r\n{}\r\n",
        host.len(),
        host,
        port.len(),
        port
    )
    .into_bytes()
}

pub(super) async fn accept_test_primary_role(listener: &TcpListener) {
    let (mut stream, _) = listener.accept().await.unwrap();
    assert_eq!(
        read_test_resp_request(&mut stream).await,
        vec![b"ROLE".to_vec()]
    );
    stream.write_all(b"*1\r\n$6\r\nmaster\r\n").await.unwrap();
}

pub(super) async fn spawn_test_sentinel(
    primary_responses: Vec<String>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let server = tokio::spawn(async move {
        for primary in primary_responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            assert_eq!(
                read_test_resp_request(&mut stream).await,
                vec![
                    b"SENTINEL".to_vec(),
                    b"GET-MASTER-ADDR-BY-NAME".to_vec(),
                    b"dynamo-primary".to_vec(),
                ]
            );
            stream
                .write_all(&test_sentinel_reply(&primary))
                .await
                .unwrap();
        }
    });
    (endpoint, server)
}

pub(super) async fn assert_test_connection_closed(stream: &mut TcpStream, context: &str) {
    let mut byte = [0_u8; 1];
    match timeout(Duration::from_secs(2), stream.read(&mut byte))
        .await
        .unwrap_or_else(|_| panic!("{context} did not close its socket"))
    {
        Ok(0) => {}
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
            ) => {}
        Ok(received) => panic!("{context} reused its socket and sent {received} bytes"),
        Err(error) => panic!("{context} socket close failed: {error}"),
    }
}

pub(super) async fn ordinary_test_command(
    primary: &ValkeyPrimary,
    lane: OrdinaryTestLane,
) -> std::result::Result<RespValue, RespError> {
    match lane {
        OrdinaryTestLane::Writer => primary.command_write(&[b"PING"]).await,
        OrdinaryTestLane::Reader => primary.command_read(&[b"PING"]).await,
        OrdinaryTestLane::AdmissionSelect => primary.command_admission_select(&[b"PING"]).await,
        OrdinaryTestLane::AdmissionLifecycle => {
            primary.command_admission_lifecycle(&[b"PING"]).await
        }
    }
}

pub(super) async fn wait_test_command(
    primary: &ValkeyPrimary,
    lane: WaitTestLane,
) -> std::result::Result<RespValue, RespError> {
    let arguments = [b"DYNKV.APPLY".as_slice(), b"key", b"payload"];
    match lane {
        WaitTestLane::Writer => primary.command_write_and_wait(&arguments, 1, false).await,
        WaitTestLane::AdmissionSelect => {
            primary
                .command_admission_select_and_wait(&arguments, 1, false)
                .await
        }
        WaitTestLane::AdmissionLifecycle => {
            primary
                .command_admission_lifecycle_and_wait(&arguments, 1, false)
                .await
        }
    }
}
