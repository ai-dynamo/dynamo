// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::ErrorClass;

use super::{
    super::{error, metrics},
    DispatchError,
};

pub(super) fn classify(
    cause: &(dyn std::error::Error + 'static),
    is_pinned: bool,
) -> DispatchError {
    if metrics::request_was_cancelled(cause) {
        return DispatchError::cancelled();
    }
    if is_pinned
        && std::iter::successors(Some(cause), |error| error.source())
            .any(|error| error.is::<crate::kv_router::UnknownWorkerTarget>())
    {
        return DispatchError::unavailable(
            "server-selected scoring worker rejected pinned placement",
        );
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
