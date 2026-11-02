// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::ErrorClass;

use super::{
    super::{error, metrics},
    DispatchError,
};

pub(super) fn classify(cause: &(dyn std::error::Error + 'static)) -> DispatchError {
    if metrics::request_was_cancelled(cause) {
        return DispatchError::cancelled();
    }
    // A router class queue drops a branch when its due time passes while it waits or a class
    // limit refuses it. A drop is the overload status without `Retry-After`, so it bypasses the
    // capacity response, which promises a retry interval.
    if metrics::request_deadline_exceeded(cause) {
        return queue_drop(
            metrics::REQUEST_DEADLINE_EXCEEDED_MESSAGE,
            "router queue deadline exceeded".to_string(),
        );
    }
    if let Some(rejection) = metrics::queue_rejection(cause) {
        return queue_drop("router queue limit reached", rejection.to_string());
    }
    if metrics::request_was_rejected(cause) {
        return DispatchError {
            status: error::overload_status_code(),
            public_message: "SGLang scoring worker capacity is exhausted",
            detail: "backend capacity exhausted".to_string(),
            retry_after: true,
            is_capacity_exhausted: true,
        };
    }
    if let Some(error) = error::find_canonical_error_in_chain(cause) {
        if error.reason().as_str() == "runtime.unclassified" {
            return DispatchError::unavailable("native scoring transport failed");
        }
        if let error::ClientErrorAction::Respond {
            status,
            public_message,
        } = error::http_action_for_error(error)
        {
            return DispatchError {
                status,
                public_message,
                detail: error.reason().as_str().to_string(),
                retry_after: error.class() == ErrorClass::CapacityExhausted,
                is_capacity_exhausted: error.class() == ErrorClass::CapacityExhausted,
            };
        }
    }
    DispatchError::unavailable("native scoring engine failed")
}

fn queue_drop(public_message: &'static str, detail: String) -> DispatchError {
    DispatchError {
        status: error::overload_status_code(),
        public_message,
        detail,
        retry_after: false,
        is_capacity_exhausted: false,
    }
}
