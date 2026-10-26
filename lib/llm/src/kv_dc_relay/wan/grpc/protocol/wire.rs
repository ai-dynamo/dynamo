// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport encodings and validation shared by Relay producers and consumers.

mod identity;
pub mod images;
mod load;

pub use identity::{
    ProducerKey, WireIdentityError, validate_ckf_format, validate_contract_marker,
    validate_endpoint_id, validate_model_registration, validate_pool_descriptor, validate_pool_id,
    validate_producer_identity, validate_protocol_envelope, validate_query_semantics,
    validate_topology_entry, validate_worker_roles,
};
pub use load::{
    validate_model_serving_load, validate_pool_serving_load, validate_serving_load_update,
};
