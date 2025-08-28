// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entity descriptor system for type-safe component identification
//!
//! This module provides descriptors that enforce entity relationships and naming
//! conventions at compile time, replacing string-based identities with structured
//! descriptor types.

pub mod descriptor;
pub mod validation;

pub use descriptor::{
    ComponentDescriptor,
    DescriptorBuilder,
    DescriptorError,
    EntityDescriptor,
    EndpointDescriptor,
    InstanceDescriptor, 
    InstanceType,
    NamespaceDescriptor,
    PathDescriptor,
    ToPath,
};