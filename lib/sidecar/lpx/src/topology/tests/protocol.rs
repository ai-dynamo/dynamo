// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire contract: generated types, identity and scope, snapshot validity
//! and the request-plane envelope.

use prost::Message;

use super::*;
use crate::topology::proto::GetSnapshotRequest;
use crate::topology::{check_expected, check_identity, check_snapshot, require_expected};

/// `proto/testdata/get_snapshot_response.txtpb` as `protoc --encode` writes it.
const GOLDEN: &[u8] = include_bytes!("../../../proto/testdata/get_snapshot_response.binpb");

#[test]
fn generated_types_match_the_independent_golden_encoding() {
    assert_eq!(response().encode_to_vec(), GOLDEN);
    assert_eq!(
        GetSnapshotResponse::decode(GOLDEN).expect("the golden encoding decodes"),
        response()
    );
}

fn with_identity(deployment: &str, allocation: &str, build: &str) -> Identity {
    Identity {
        deployment: deployment.to_string(),
        allocation: allocation.to_string(),
        build: build.to_string(),
    }
}

/// The allocation scope is
/// `printf 'lpx.topology.v1\0%s\0%s' DEPLOYMENT ALLOCATION | sha256sum | cut -c1-16` and the
/// build scope is
/// `printf 'lpx.topology.v1\0%s\0%s\0%s' DEPLOYMENT ALLOCATION BUILD | sha256sum | cut -c1-16`.
#[test]
fn identity_fields_are_non_empty_bounded_and_free_of_control_characters() {
    let bounded = "x".repeat(512);
    // 512 bytes in 256 characters.
    let multibyte = "é".repeat(256);
    for valid in [
        identity(),
        with_identity(&bounded, &bounded, &bounded),
        with_identity(&multibyte, "llama-0-afd-0", "b"),
    ] {
        check_identity(&valid).unwrap_or_else(|error| panic!("{valid:?}: {error}"));
    }

    let too_long = "x".repeat(513);
    let too_many_bytes = "é".repeat(257);
    for (invalid, field) in [
        (with_identity("", "a", "b"), "deployment"),
        (with_identity("d", &too_long, "b"), "allocation"),
        (with_identity("d", "a", &too_many_bytes), "build"),
        (with_identity("d\0", "a", "b"), "deployment"),
        (with_identity("d", "a\n", "b"), "allocation"),
        (with_identity("d", "a", "b\u{7f}"), "build"),
        (with_identity("\td", "a", "b"), "deployment"),
    ] {
        let error = check_identity(&invalid).expect_err(&format!("{invalid:?} is invalid"));
        assert!(error.contains(field), "{invalid:?}: {error}");
    }
}

#[test]
fn a_conductor_side_request_needs_a_complete_expected_identity() {
    let request = |expected| GetSnapshotRequest { expected };
    let full = request(Some(identity()));
    assert_eq!(require_expected(&full), Ok(&identity()));

    let error = require_expected(&request(None)).expect_err("no expected identity");
    assert!(error.contains("expected"), "{error}");
    for (partial, field) in [
        (with_identity("", "a", "b"), "deployment"),
        (with_identity("d", "", "b"), "allocation"),
        (with_identity("d", "a", ""), "build"),
        (with_identity("d", "a", "b\n"), "build"),
    ] {
        let error =
            require_expected(&request(Some(partial.clone()))).expect_err("a partial identity");
        assert!(error.contains(field), "{partial:?}: {error}");
    }
}

#[test]
fn only_the_expected_fields_that_are_set_must_match() {
    let actual = identity();
    check_expected(&actual, None).expect("nothing expected");
    check_expected(&actual, Some(&Identity::default())).expect("nothing set");
    check_expected(&actual, Some(&actual)).expect("everything matches");
    let allocation_only = with_identity("", "llama-0-afd-0", "");
    check_expected(&actual, Some(&allocation_only)).expect("the set field matches");

    for (expected, field) in [
        (with_identity("other", "", ""), "deployment"),
        (with_identity("", "llama-0-afd-1", ""), "allocation"),
        (
            Identity {
                build: "main/llama3_2-1b-lpx-lp20/0000".to_string(),
                ..identity()
            },
            "build",
        ),
    ] {
        let error = check_expected(&actual, Some(&expected)).expect_err("a mismatch");
        assert!(error.contains(field), "{expected:?}: {error}");
    }
}

fn mutated(mutate: impl FnOnce(&mut Snapshot)) -> Snapshot {
    let mut snapshot = snapshot();
    mutate(&mut snapshot);
    snapshot
}

/// The execution of the partition at `index` of [`snapshot`].
fn execution_at(snapshot: &mut Snapshot, index: usize) -> &mut Execution {
    match snapshot.partitions[index].placement.as_mut() {
        Some(Placement::Execution(execution)) => execution,
        placement => panic!("partition {index} does not execute: {placement:?}"),
    }
}

fn control_at(snapshot: &mut Snapshot, index: usize) -> &mut Endpoint {
    execution_at(snapshot, index)
        .control
        .as_mut()
        .expect("a control endpoint")
}

#[test]
fn complete_snapshots_are_accepted_in_any_order() {
    let label = "a".repeat(63);
    // Three 63-byte labels, a 61-byte one and their dots: 253 bytes.
    let longest_name = format!("{label}.{label}.{label}.{}", "b".repeat(61));
    assert_eq!(longest_name.len(), 253);
    let mixed_case = "Agt-0.Llama-0.svc";
    for (name, snapshot) in [
        ("the golden snapshot", snapshot()),
        (
            "reversed partitions",
            mutated(|snapshot| snapshot.partitions.reverse()),
        ),
        (
            "an IPv6 literal",
            mutated(|snapshot| control_at(snapshot, 3).host = "fd00::12".to_string()),
        ),
        (
            "an LPUD control endpoint",
            mutated(|snapshot| {
                let control = control_at(snapshot, 3);
                control.port = 12346;
                control.protocol = Protocol::LpudRemoted as i32;
            }),
        ),
        (
            "the longest DNS name",
            mutated(|snapshot| control_at(snapshot, 1).host = longest_name.clone()),
        ),
        (
            "a mixed-case DNS name",
            mutated(|snapshot| control_at(snapshot, 1).host = mixed_case.to_string()),
        ),
        (
            "an all-digit label before the last",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt.0.svc".to_string()),
        ),
        (
            "a last label of digits and a letter",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-0.0a".to_string()),
        ),
        (
            "a last label that is 0x without hex digits",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-0.0x".to_string()),
        ),
        (
            "a last label that is 0x and a non-hex letter",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-0.0xg".to_string()),
        ),
        (
            "the port range's ends",
            mutated(|snapshot| {
                control_at(snapshot, 1).port = 1;
                control_at(snapshot, 3).port = 65535;
            }),
        ),
    ] {
        let identity = check_snapshot(&snapshot).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(identity, &super::identity(), "{name}");
    }
}

#[test]
fn incomplete_or_unusable_snapshots_are_rejected() {
    let long_label = "a".repeat(64);
    let label = "a".repeat(63);
    let too_long_name = format!("{label}.{label}.{label}.{}", "b".repeat(62));
    let cases: Vec<(&str, Snapshot, &[&str])> = vec![
        (
            "no identity",
            mutated(|snapshot| snapshot.identity = None),
            &["identity"],
        ),
        (
            "an invalid identity",
            mutated(|snapshot| snapshot.identity.as_mut().unwrap().build.clear()),
            &["build"],
        ),
        (
            "no partitions",
            mutated(|snapshot| snapshot.partitions.clear()),
            &["partitions"],
        ),
        (
            "a duplicate partition",
            mutated(|snapshot| snapshot.partitions.push(member(4, 1))),
            &["partition 4"],
        ),
        (
            "no placement",
            mutated(|snapshot| snapshot.partitions[2].placement = None),
            &["partition 5", "placement"],
        ),
        (
            "no nodes",
            mutated(|snapshot| execution_at(snapshot, 1).node_count = 0),
            &["partition 4", "node_count"],
        ),
        (
            "no control endpoint",
            mutated(|snapshot| execution_at(snapshot, 1).control = None),
            &["partition 4", "control"],
        ),
        (
            "an empty host",
            mutated(|snapshot| control_at(snapshot, 1).host.clear()),
            &["partition 4", "host"],
        ),
        (
            "a bracketed IPv6 literal",
            mutated(|snapshot| control_at(snapshot, 1).host = "[fd00::12]".to_string()),
            &["partition 4", "host"],
        ),
        (
            "an IPv6 zone",
            mutated(|snapshot| control_at(snapshot, 1).host = "fe80::1%eth0".to_string()),
            &["partition 4", "host"],
        ),
        (
            "an empty label",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-0..svc".to_string()),
            &["partition 4", "host"],
        ),
        (
            "a trailing dot",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-0.svc.".to_string()),
            &["partition 4", "host"],
        ),
        (
            "a leading hyphen",
            mutated(|snapshot| control_at(snapshot, 1).host = "-agt.svc".to_string()),
            &["partition 4", "host"],
        ),
        (
            "a trailing hyphen",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt-.svc".to_string()),
            &["partition 4", "host"],
        ),
        (
            "an underscore",
            mutated(|snapshot| control_at(snapshot, 1).host = "agt_0.svc".to_string()),
            &["partition 4", "host"],
        ),
        (
            "a 64-byte label",
            mutated(|snapshot| control_at(snapshot, 1).host = format!("{long_label}.svc")),
            &["partition 4", "host"],
        ),
        (
            "a 254-byte name",
            mutated(|snapshot| control_at(snapshot, 1).host = too_long_name.clone()),
            &["partition 4", "host"],
        ),
        (
            "port 0",
            mutated(|snapshot| control_at(snapshot, 1).port = 0),
            &["partition 4", "port"],
        ),
        (
            "port 65536",
            mutated(|snapshot| control_at(snapshot, 1).port = 65536),
            &["partition 4", "port"],
        ),
        (
            "an unspecified control protocol",
            mutated(|snapshot| control_at(snapshot, 1).protocol = Protocol::Unspecified as i32),
            &["partition 4", "protocol"],
        ),
        (
            "a state protocol on the control endpoint",
            mutated(|snapshot| control_at(snapshot, 1).protocol = STATE as i32),
            &["partition 4", "protocol"],
        ),
        (
            "an unknown control protocol",
            mutated(|snapshot| control_at(snapshot, 1).protocol = 9),
            &["partition 4", "protocol"],
        ),
        (
            "a control protocol on the state-ingress endpoint",
            mutated(|snapshot| {
                execution_at(snapshot, 0)
                    .state_ingress
                    .as_mut()
                    .unwrap()
                    .protocol = CONTROL as i32
            }),
            &["partition 1", "state_ingress", "protocol"],
        ),
        (
            "an unusable state-ingress port",
            mutated(|snapshot| {
                execution_at(snapshot, 0)
                    .state_ingress
                    .as_mut()
                    .unwrap()
                    .port = 0
            }),
            &["partition 1", "state_ingress", "port"],
        ),
        (
            "a chain member that owns itself",
            mutated(|snapshot| snapshot.partitions[2] = member(5, 5)),
            &["partition 5", "owner"],
        ),
        (
            "a chain member whose owner is missing",
            mutated(|snapshot| snapshot.partitions[2] = member(5, 7)),
            &["partition 5", "owner"],
        ),
        (
            "a chain member owned by another member",
            mutated(|snapshot| snapshot.partitions.push(member(6, 5))),
            &["partition 6", "owner"],
        ),
    ];
    for (name, snapshot, expect) in cases {
        let error = check_snapshot(&snapshot).expect_err(name);
        for needle in expect {
            assert!(error.contains(needle), "{name}: {error:?} lacks {needle:?}");
        }
    }

    // A name whose last label is all digits, or `0x` and hex digits, is no
    // DNS name, and none of these is an IP literal in standard form, though
    // libc resolves most of them as IPv4 shorthand: `1.2.3` as 1.2.0.3,
    // `010.0.0.1` as 8.0.0.1, `0x7f000001` as 127.0.0.1.
    for host in [
        "0",
        "123",
        "1.1",
        "1.2.3",
        "01.02.03.04",
        "1.2.3.256",
        "1.2.3.4.5",
        "010.0.0.1",
        "agt-0.123",
        "0x7f000001",
        "0X7F000001",
        "0x7f",
        "1.0x2",
        "1.2.3.0x4",
    ] {
        let snapshot = mutated(|snapshot| control_at(snapshot, 1).host = host.to_string());
        let error = check_snapshot(&snapshot).expect_err(host);
        for needle in ["partition 4", "host"] {
            assert!(error.contains(needle), "{host}: {error:?} lacks {needle:?}");
        }
    }
}
