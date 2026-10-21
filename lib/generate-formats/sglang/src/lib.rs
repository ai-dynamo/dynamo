// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang's native `/generate`, Dynamo's built-in generate format.

pub mod generate;
pub use inspect::{Inspected, inspect};
mod inspect;
pub mod merge;

pub use merge::{ClientCodec, Encoded, Native, SglangFormat, rules};

pub const NAME: &str = "sglang";
/// The runtime capability workers serving SGLang's `/generate` advertise.
pub const CAPABILITY: &str = dynamo_llm::local_model::runtime_config::SGLANG_GENERATE_CAPABILITY;
/// `extra_args` key carrying SGLang's request to the worker.
pub const PAYLOAD_KEY: &str = "sglang_tito";
/// `engine_data` key carrying each SGLang response back.
pub const RESPONSE_KEY: &str = "sglang_response";
/// The frontend restores public IDs, so workers may isolate engine dispatch IDs.
/// Older frontends omit this annotation and keep their existing worker IDs.
pub const PRIVATE_RID_ANNOTATION: &str = "sglang_private_rid";

pub mod response_merge;

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

#[cfg(test)]
mod tests;

mod errors;
