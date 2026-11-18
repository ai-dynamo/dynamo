// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the storage module (re-exported from dynamo-memory).

use super::*;

#[test]
fn test_type_erasure() {
    let storage = SystemStorage::new(1024).unwrap();
    let erased: OwnedMemoryRegion = erase_storage(storage);

    assert_eq!(erased.size(), 1024);
    assert_eq!(erased.storage_kind(), StorageKind::System);
}

#[cfg(feature = "testing-cuda")]
mod cuda_tests {
    use super::*;
}

#[cfg(all(feature = "testing-nixl", feature = "testing-cuda"))]
mod nixl_tests {
    use super::*;
    use nixl_sys::Agent as NixlAgent;

    #[test]
    fn test_nixl_registration() {
        let pinned = PinnedStorage::new(2048).unwrap();
        let agent = NixlAgent::new("test_agent").unwrap();
        let (_, ucx_params) = agent.get_plugin_params("UCX").unwrap_or_else(|e| {
            let available: Vec<String> = agent
                .get_available_plugins()
                .map(|p| p.iter().filter_map(Result::ok).map(String::from).collect())
                .unwrap_or_default();
            panic!("UCX plugin unavailable ({e}); NIXL reports {available:?}");
        });
        agent.create_backend("UCX", &ucx_params).unwrap();
        let registered = register_with_nixl(pinned, &agent, None).unwrap();
        assert_eq!(registered.agent_name(), "test_agent");
    }
}
