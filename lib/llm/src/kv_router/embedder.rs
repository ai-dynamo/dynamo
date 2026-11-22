// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use dynamo_kv_router::carrier_lookup::CarrierLookupSource;

static CARRIER_LOOKUP_SOURCE: OnceLock<Arc<dyn CarrierLookupSource>> = OnceLock::new();

/// Install once per process, before the router is built. Errors if already installed.
pub fn install_carrier_lookup_source(source: Arc<dyn CarrierLookupSource>) -> anyhow::Result<()> {
    CARRIER_LOOKUP_SOURCE
        .set(source)
        .map_err(|_| anyhow::anyhow!("carrier lookup source is already installed"))
}

pub fn carrier_lookup_source() -> Option<Arc<dyn CarrierLookupSource>> {
    CARRIER_LOOKUP_SOURCE.get().cloned()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dynamo_kv_router::carrier_lookup::{CarrierLookup, CarrierLookupSource};

    use super::{carrier_lookup_source, install_carrier_lookup_source};

    struct EmptyLookupSource;

    impl CarrierLookupSource for EmptyLookupSource {
        fn lookup(&self, _hub_url: &str) -> Option<Arc<dyn CarrierLookup>> {
            None
        }
    }

    #[test]
    fn carrier_lookup_source_can_only_be_installed_once() {
        let source: Arc<dyn CarrierLookupSource> = Arc::new(EmptyLookupSource);

        assert!(install_carrier_lookup_source(Arc::clone(&source)).is_ok());
        assert!(carrier_lookup_source().is_some());
        assert!(install_carrier_lookup_source(source).is_err());
    }
}
