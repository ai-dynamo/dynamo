// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions};

pub(crate) fn minimal_request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("served-model".to_string())
        .token_ids(vec![11, 22, 33])
        .sampling_options(SamplingOptions::default())
        .stop_conditions(StopConditions::default())
        .output_options(OutputOptions::default())
        .build()
        .expect("minimal request")
}
