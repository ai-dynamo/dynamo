// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo Runtime v2 API
//!
//! This module provides the next generation of Dynamo runtime APIs, built around
//! entity descriptors for improved type safety, validation, and developer experience.
//! 
//! The v2 API introduces:
//! - Strong compile-time guarantees about entity relationships
//! - Comprehensive validation with detailed error messages  
//! - Type-safe descriptor system for components, endpoints, and instances
//! - Improved ergonomics through fluent builder patterns
//! - Forward compatibility for descriptor-based systems

pub mod entity;

pub use entity::{
    ComponentDescriptor, DescriptorBuilder, DescriptorError, EndpointDescriptor, 
    InstanceDescriptor, NamespaceDescriptor, PathDescriptor,
};