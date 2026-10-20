// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::test_fixtures::*;

/// The handoff codec is symmetric: the decode worker requires exactly what the
/// prefill worker wrote. A field lost in transit must fail by name rather than
/// decode into a plausible session. `attributes` matters most -- the server
/// reads the session's location and rank out of it, and accepts an endpoint in
/// place of an `opaque_state`, so a handoff that lost it would pass the
/// server's own guard and resume a session with no opaque state at rank 0.
#[test]
fn a_handoff_missing_a_field_is_rejected() {
    let mut session = fake_session();
    session.dp_rank = 3;
    let encoded = crate::disagg::session_to_json(session).expect("encode");

    let decoded = crate::disagg::session_from_json(&encoded).expect("round trip");
    assert_eq!(decoded.dp_rank, 3, "the rank must survive the round trip");

    for field in [
        "attributes",
        "dp_rank",
        "endpoints",
        "transfer_backend",
        "session_id",
    ] {
        let mut mangled = encoded.clone();
        mangled
            .as_object_mut()
            .expect("handoff object")
            .remove(field)
            .expect("field is present");
        let error = crate::disagg::session_from_json(&mangled)
            .expect_err("a handoff missing a field is not usable");
        assert!(
            error.to_string().contains(field),
            "the error must name {field}: {error}"
        );
    }
}

/// A newer prefill worker may add fields this decode worker has never heard of.
/// Rejecting them would fail every new-prefill/old-decode request during a
/// rolling upgrade, as a non-migratable 400, with no version to negotiate on.
#[test]
fn a_handoff_carrying_an_unknown_field_still_decodes() {
    let encoded = crate::disagg::session_to_json(fake_session()).expect("encode");
    let mut newer = encoded.clone();
    newer
        .as_object_mut()
        .expect("handoff object")
        .insert("schedule_style".to_string(), serde_json::json!(2));

    crate::disagg::session_from_json(&newer)
        .expect("a handoff from a newer peer must still decode");
}

/// The prefill leg refuses to emit a handoff the decode leg could not resolve,
/// so the failure names the prefill worker rather than surfacing one hop later.
#[test]
fn a_prefill_ready_without_attributes_is_rejected() {
    let mut session = fake_session();
    session.attributes_struct = None;

    let error = crate::disagg::session_to_json(session)
        .expect_err("a session with no attributes cannot locate the context worker");
    assert!(
        error.to_string().contains("attributes_struct"),
        "the error must name the missing field: {error}"
    );
}

/// The same rolling-upgrade rule one level down: a newer peer may add fields
/// inside `endpoints[]`. `deny_unknown_fields` there reversed the module's
/// stated policy and would fail that request as a non-migratable 400.
#[test]
fn an_unknown_field_inside_an_endpoint_still_decodes() {
    let encoded = crate::disagg::session_to_json(fake_session()).expect("encode");
    let mut newer = encoded.clone();
    newer
        .as_object_mut()
        .expect("handoff object")
        .get_mut("endpoints")
        .expect("endpoints")
        .as_array_mut()
        .expect("endpoints array")
        .first_mut()
        .expect("at least one endpoint")
        .as_object_mut()
        .expect("endpoint object")
        .insert(
            "future_field".to_string(),
            serde_json::json!("from a newer peer"),
        );

    crate::disagg::session_from_json(&newer)
        .expect("an endpoint from a newer peer must still decode");
}
