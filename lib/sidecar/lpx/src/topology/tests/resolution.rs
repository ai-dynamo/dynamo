// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How the consumer finds the authority for the identity a conductor
//! expects: what the registered endpoint names say, and which of the
//! registered authorities' answers wins.

use tonic::{Code, Status};

use super::*;
use crate::topology::consumer::{Registered, choose, registered};

/// The endpoint of [`identity`], and of the same allocation with build
/// `main/llama3_2-1b-lpx-lp20/0000` and of allocation `llama-0-afd-1`, from
/// `printf 'lpx.topology.v1\0%s\0%s' ...` and
/// `printf 'lpx.topology.v1\0%s\0%s\0%s' ...` through `sha256sum | cut -c1-16`.
const EXPECTED: &str = "snapshot-776f7e41bb573162-7e63d7f9aae8ebe8";
const OTHER_BUILD: &str = "snapshot-776f7e41bb573162-eb97fd18615215ea";
const OTHER_ALLOCATION: &str = "snapshot-768000ca136539ec-d98d44ff7e110505";

#[test]
fn registrations_are_sorted_by_allocation_then_build() {
    assert_eq!(registered(&identity(), []), Registered::Nothing);
    // Another allocation's authority, and names without the allocation's
    // whole prefix, are not this allocation's.
    assert_eq!(
        registered(
            &identity(),
            [
                (OTHER_ALLOCATION, 1),
                ("generate", 2),
                ("snapshot-776f7e41bb573162", 3)
            ]
        ),
        Registered::Nothing
    );
    assert_eq!(
        registered(&identity(), [(OTHER_BUILD, 4), (OTHER_ALLOCATION, 1)]),
        Registered::OtherBuilds(vec!["eb97fd18615215ea".to_string()])
    );
    // The expected build's authorities, each once, whatever else the
    // allocation has registered.
    assert_eq!(
        registered(
            &identity(),
            [
                (OTHER_BUILD, 4),
                (EXPECTED, 9),
                (EXPECTED, 7),
                (EXPECTED, 9)
            ]
        ),
        Registered::Authorities(vec![7, 9])
    );
}

/// A leader that is shutting down, a registration its crashed publisher left
/// behind, or a refusal from another authority never hides the one
/// authority that answers.
#[test]
fn exactly_one_ok_answer_wins() {
    for answers in [
        vec![(1, Ok(response()))],
        vec![
            (1, Err(Status::unavailable("shutting down"))),
            (2, Ok(response())),
        ],
        vec![
            (2, Ok(response())),
            (
                1,
                Err(Status::unavailable(
                    "the topology authority is unreachable: connection refused",
                )),
            ),
        ],
        vec![
            (1, Err(Status::failed_precondition("drifted"))),
            (2, Ok(response())),
        ],
    ] {
        let chosen = choose(&identity(), answers).unwrap_or_else(|status| panic!("{status:?}"));
        assert_eq!(chosen, response());
    }
}

#[test]
fn more_than_one_ok_answer_is_ambiguous() {
    let status = choose(
        &identity(),
        vec![
            (1, Ok(response())),
            (2, Err(Status::unavailable("shutting down"))),
            (3, Ok(response())),
        ],
    )
    .expect_err("two OK answers");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    for needle in ["2 topology authorities", "ambiguous"] {
        assert!(status.message().contains(needle), "{status:?}");
    }
}

/// Without an OK answer, a refusal is final and anything else is retried
/// with every authority's reason.
#[test]
fn without_an_ok_answer_a_refusal_fails_and_anything_else_is_retried() {
    let status = choose(
        &identity(),
        vec![
            (1, Err(Status::unavailable("shutting down"))),
            (2, Err(Status::failed_precondition("the build differs"))),
        ],
    )
    .expect_err("a refusal");
    assert_eq!(status.code(), Code::FailedPrecondition, "{status:?}");
    assert!(status.message().contains("the build differs"), "{status:?}");

    let status = choose(
        &identity(),
        vec![
            (1, Err(Status::unavailable("shutting down"))),
            (2, Err(Status::internal("the Agent failed"))),
        ],
    )
    .expect_err("no OK answer");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    for needle in ["shutting down", "the Agent failed"] {
        assert!(status.message().contains(needle), "{status:?}");
    }
}
