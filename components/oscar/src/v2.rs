// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Oscar v2 API using dynamo-runtime descriptors
//!
//! This module provides Oscar's next-generation API built on the dynamo-runtime v2
//! descriptor system. It offers improved type safety, validation, and ergonomics
//! compared to the string-based v1 API.

pub mod descriptors;
pub mod keys;

pub use descriptors::{
    CallerContext, CallerContextBuilder, CallerDescriptor, ObjectDescriptor, ObjectDescriptorBuilder, 
    ObjectName, OscarDescriptorError,
};
pub use keys::OscarKeyType as OscarKeyTypeV2;