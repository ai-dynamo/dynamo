// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod shared_prefix_error_tests {
    use super::*;

    #[test]
    fn missing_b_pin_is_typed_public_invalid_argument() {
        let error = classifier_error(ThunderAgentError::NonFinalRequiresHardPin);
        let typed = error
            .as_ref()
            .downcast_ref::<DynamoError>()
            .expect("missing pin should use the frontend's typed rejection path");
        assert_eq!(typed.error_type(), ErrorType::InvalidArgument);
        assert_eq!(
            typed.public_message(),
            Some(
                "shared-prefix budget requires a hard Worker/DP pin for non-final session requests"
            )
        );

        let internal = classifier_error(ThunderAgentError::RequestLimitExceeded { limit: 1 });
        assert!(internal.as_ref().downcast_ref::<DynamoError>().is_none());
    }
}
