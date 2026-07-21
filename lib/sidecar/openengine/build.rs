// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const OPENENGINE_COMMIT: &str = "f1a7189311770f8aa1f0dd787561df809847595d";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");
}
