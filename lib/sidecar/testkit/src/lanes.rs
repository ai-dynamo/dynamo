// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

macro_rules! sidecar_test {
    (lane: pre_merge; $($test:tt)*) => {
        sidecar_test!(@case __sidecar_lane_pre_merge; $($test)*);
    };
    (lane: post_merge; $($test:tt)*) => {
        sidecar_test!(@case __sidecar_lane_post_merge; $($test)*);
    };
    (lane: nightly; $($test:tt)*) => {
        sidecar_test!(@case __sidecar_lane_nightly; $($test)*);
    };
    (@case $lane:ident; $(#[$attribute:meta])* fn $name:ident() $(-> $result:ty)? $body:block) => {
        mod $name {
            use super::*;

            $(#[$attribute])*
            fn $lane() $(-> $result)? $body
        }
    };
    (@case $lane:ident; $(#[$attribute:meta])* async fn $name:ident() $(-> $result:ty)? $body:block) => {
        mod $name {
            use super::*;

            $(#[$attribute])*
            async fn $lane() $(-> $result)? $body
        }
    };
}
