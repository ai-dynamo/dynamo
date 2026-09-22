// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, ErrorType};

use super::status_to_dynamo;

#[test]
fn maps_transport_statuses_to_backend_errors() {
    for (code, expected) in [
        (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
        (tonic::Code::NotFound, BackendError::InvalidArgument),
        (tonic::Code::OutOfRange, BackendError::InvalidArgument),
        (
            tonic::Code::FailedPrecondition,
            BackendError::InvalidArgument,
        ),
        (tonic::Code::AlreadyExists, BackendError::InvalidArgument),
        (tonic::Code::Unknown, BackendError::Unknown),
        (tonic::Code::Unimplemented, BackendError::Unknown),
        (tonic::Code::ResourceExhausted, BackendError::Unknown),
        (tonic::Code::PermissionDenied, BackendError::Unknown),
        (tonic::Code::Unauthenticated, BackendError::Unknown),
        (tonic::Code::Aborted, BackendError::Unknown),
        (tonic::Code::DataLoss, BackendError::Unknown),
        (tonic::Code::Ok, BackendError::Unknown),
        (tonic::Code::Unavailable, BackendError::CannotConnect),
        (tonic::Code::Cancelled, BackendError::Cancelled),
        (
            tonic::Code::DeadlineExceeded,
            BackendError::ConnectionTimeout,
        ),
        (tonic::Code::Internal, BackendError::Unknown),
    ] {
        let error = status_to_dynamo("Test", tonic::Status::new(code, "failure"));
        assert_eq!(error.error_type(), ErrorType::Backend(expected));
        assert!(error.to_string().contains("Test: failure"));
        assert!(error.to_string().contains(&format!("{code:?}")));
        #[cfg(feature = "tonic-v14")]
        {
            let v14 = super::status_to_dynamo_v14(
                "Test",
                tonic_v14::Status::new(tonic_v14::Code::from_i32(code as i32), "failure"),
            );
            assert_eq!(v14.error_type(), error.error_type());
            assert_eq!(v14.to_string(), error.to_string());
        }
    }
}
