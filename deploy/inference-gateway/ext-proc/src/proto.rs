// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf types from the Envoy ext_proc service protos.
//!
//! The module hierarchy matches the proto package structure so that prost's
//! `super::` cross-package references resolve correctly.

#[allow(clippy::all)]
#[allow(non_camel_case_types)]
pub mod envoy {
    pub mod config {
        pub mod core {
            pub mod v3 {
                tonic::include_proto!("envoy.config.core.v3");
            }
        }
    }

    pub mod r#type {
        pub mod v3 {
            tonic::include_proto!("envoy.type.v3");
        }
    }

    pub mod extensions {
        pub mod filters {
            pub mod http {
                pub mod ext_proc {
                    pub mod v3 {
                        tonic::include_proto!(
                            "envoy.extensions.filters.http.ext_proc.v3"
                        );
                    }
                }
            }
        }
    }

    pub mod service {
        pub mod ext_proc {
            pub mod v3 {
                tonic::include_proto!("envoy.service.ext_proc.v3");
            }
        }
    }
}
