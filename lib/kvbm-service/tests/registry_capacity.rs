// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slot-accounting behavior of the registry state machine.

use kvbm_service::error::ServiceError;
use kvbm_service::instance::{KvbmInstance, LayoutShape, RegistrationInstance};
use kvbm_service::metrics::ServiceMetrics;
use kvbm_service::mode::ServiceMode;
use kvbm_service::registry::Registry;

fn kvbm(tp: u32) -> RegistrationInstance {
    RegistrationInstance::Kvbm(KvbmInstance {
        model_name: "llm".into(),
        layout: LayoutShape::UniversalTp1Canonical {
            bytes: vec![0xab, 0xcd],
        },
        tp_size: tp,
        block_size: 64,
        mode: ServiceMode::Kvbm,
    })
}

#[test]
fn capacity_8_tp4_fits_two_and_rejects_third() {
    let r = Registry::new(8, ServiceMetrics::new());
    r.try_register(kvbm(4), "c1".into()).unwrap();
    r.try_register(kvbm(4), "c2".into()).unwrap();
    let err = r.try_register(kvbm(4), "c3".into()).unwrap_err();
    assert!(matches!(err, ServiceError::NoCapacity(_)));
}

#[test]
fn capacity_4_tp1_fits_four_then_full() {
    let r = Registry::new(4, ServiceMetrics::new());
    for i in 0..4 {
        r.try_register(kvbm(1), format!("c-{i}")).unwrap();
    }
    assert_eq!(r.snapshot().used_slots, 4);
    let err = r.try_register(kvbm(1), "c-5".into()).unwrap_err();
    assert!(matches!(err, ServiceError::NoCapacity(_)));
}

#[test]
fn tp_not_power_of_two_rejected_pre_registration() {
    let inst = kvbm(3);
    let err = inst.validate(8).unwrap_err();
    assert!(matches!(err, ServiceError::InvalidArgument(_)));
}

#[test]
fn tp_greater_than_capacity_rejected_pre_registration() {
    let inst = kvbm(16);
    let err = inst.validate(8).unwrap_err();
    assert!(matches!(err, ServiceError::InvalidArgument(_)));
}
