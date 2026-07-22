// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const OPENENGINE_COMMIT: &str = "b0cf2a4826d246192dc65b055dab6d2b38d2d67e";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");
}
