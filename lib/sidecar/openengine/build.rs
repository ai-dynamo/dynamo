// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const OPENENGINE_COMMIT: &str = "a66ff6f73a65e262a7c3edd5ea6fd0d8701d402f";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");
}
