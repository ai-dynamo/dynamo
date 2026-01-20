// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local layout wrapper with handle and metadata.

use std::ops::Deref;

use super::handle::LayoutHandle;
use crate::block_manager::v2::physical::layout::PhysicalLayout;

/// A local physical layout with an assigned handle.
///
/// This wraps a `PhysicalLayout` that exists on the local worker,
/// associating it with a unique handle that combines the worker_id
/// and a locally-assigned layout_id.
///
/// This type is cheap to clone as `PhysicalLayout` contains `Arc` internally.
#[derive(Debug, Clone)]
pub struct LocalLayout {
    handle: LayoutHandle,
    layout: PhysicalLayout,
}

#[allow(dead_code)]
impl LocalLayout {
    /// Create a new local layout.
    ///
    /// # Arguments
    /// * `handle` - Unique handle for this layout
    /// * `layout` - The physical layout
    pub fn new(handle: LayoutHandle, layout: PhysicalLayout) -> Self {
        Self { handle, layout }
    }

    /// Get the handle for this layout.
    pub fn handle(&self) -> LayoutHandle {
        self.handle
    }

    /// Get a reference to the physical layout.
    pub fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    /// Get the worker_id from the handle.
    pub fn worker_id(&self) -> u64 {
        self.handle.worker_id()
    }

    /// Get the layout_id from the handle.
    pub fn layout_id(&self) -> u16 {
        self.handle.layout_id()
    }

    /// Consume this local layout and return the physical layout.
    pub fn into_layout(self) -> PhysicalLayout {
        self.layout
    }
}

impl Deref for LocalLayout {
    type Target = PhysicalLayout;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

#[cfg(all(test, feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::v2::physical::layout::{LayoutConfig, PhysicalLayout};
    use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;

    fn create_test_agent(name: &str) -> NixlAgent {
        NixlAgent::require_backends(name, &[]).expect("failed to create wrapped agent")
    }

    fn make_test_layout() -> PhysicalLayout {
        let agent = create_test_agent("test-local");
        let config = LayoutConfig::builder()
            .num_blocks(2)
            .num_layers(2)
            .outer_dim(2)
            .page_size(4)
            .inner_dim(8)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        PhysicalLayout::builder(agent)
            .with_config(config)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap()
    }

    #[test]
    fn test_local_layout_creation() {
        let handle = LayoutHandle::new(42, 100);
        let layout = make_test_layout();
        let local = LocalLayout::new(handle, layout);

        assert_eq!(local.handle(), handle);
        assert_eq!(local.worker_id(), 42);
        assert_eq!(local.layout_id(), 100);
    }

    #[test]
    fn test_local_layout_into_layout() {
        let handle = LayoutHandle::new(1, 2);
        let layout = make_test_layout();
        let local = LocalLayout::new(handle, layout);

        let _recovered = local.into_layout();
        // Successfully consumed and returned the layout
    }
}
