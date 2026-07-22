// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const OPENENGINE_COMMIT: &str = "df3a9be24a2a36a4ff7a6d4fef9f1d7480ae210d";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-env=OPENENGINE_PROTO_COMMIT={OPENENGINE_COMMIT}");
}
