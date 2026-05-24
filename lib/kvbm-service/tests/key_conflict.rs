// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Key-conflict and reset-after-detach behavior.

use std::sync::Arc;

use kvbm_service::error::ServiceError;
use kvbm_service::instance::{KvbmInstance, LayoutShape, RegistrationInstance};
use kvbm_service::metrics::ServiceMetrics;
use kvbm_service::mode::ServiceMode;
use kvbm_service::registry::{NoopLifecycle, Registry, StreamLifecycle};

fn noop() -> Arc<dyn StreamLifecycle> {
    Arc::new(NoopLifecycle)
}

fn kvbm(model: &str, block: u32, layout_byte: u8) -> RegistrationInstance {
    RegistrationInstance::Kvbm(KvbmInstance {
        model_name: model.into(),
        layout: LayoutShape::UniversalTp1Canonical {
            bytes: vec![layout_byte],
        },
        tp_size: 2,
        block_size: block,
        mode: ServiceMode::Kvbm,
    })
}

#[test]
fn different_model_name_conflicts() {
    let r = Registry::new(8, ServiceMetrics::new());
    r.try_register(kvbm("llm-a", 64, 0), "c1".into()).unwrap();
    let err = r
        .try_register(kvbm("llm-b", 64, 0), "c2".into())
        .unwrap_err();
    assert!(matches!(err, ServiceError::KeyConflict(_)));
}

#[test]
fn different_block_size_conflicts() {
    let r = Registry::new(8, ServiceMetrics::new());
    r.try_register(kvbm("llm", 64, 0), "c1".into()).unwrap();
    let err = r.try_register(kvbm("llm", 32, 0), "c2".into()).unwrap_err();
    assert!(matches!(err, ServiceError::KeyConflict(_)));
}

#[test]
fn different_layout_bytes_conflict() {
    let r = Registry::new(8, ServiceMetrics::new());
    r.try_register(kvbm("llm", 64, 0xaa), "c1".into()).unwrap();
    let err = r
        .try_register(kvbm("llm", 64, 0xbb), "c2".into())
        .unwrap_err();
    assert!(matches!(err, ServiceError::KeyConflict(_)));
}

#[test]
fn reset_after_last_detach_accepts_new_key() {
    let r = Registry::new(8, ServiceMetrics::new());
    let entry = r.try_register(kvbm("llm-a", 64, 0), "c1".into()).unwrap();
    r.commit_register(entry.id, noop()).unwrap();
    assert!(r.unregister(entry.id));
    // After the last detach, a completely different key should now be accepted.
    r.try_register(kvbm("llm-b", 32, 0xff), "c2".into())
        .unwrap();
    let snap = r.snapshot();
    assert_eq!(snap.state, "SingleKey");
    assert!(snap.instance.is_some());
}
