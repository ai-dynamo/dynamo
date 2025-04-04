// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

fn main() {
    extern crate bindgen;

    use cmake::Config;
    use std::env;
    use std::path::PathBuf;
    let installed_headers = "/usr/local/include/nvidia/nvllm/nvllm_trt.h";
    let local_headers = "../../bindings/cpp/nvllm-trt/include/nvidia/nvllm/nvllm_trt.h";
    let headers_path;

    if PathBuf::from(installed_headers).exists() {
        headers_path = installed_headers;
        println!("cargo:warning=nvllm found. Building with installed version...");
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=/opt/tensorrt_llm/lib");
        println!("cargo:rustc-link-lib=dylib=nvllm_trt");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm_nvrtc_wrapper");
        println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
        println!("cargo:rustc-link-lib=dylib=decoder_attention");

        println!("cargo:rerun-if-changed=/usr/local/lib");
    } else if PathBuf::from(local_headers).exists() {
        headers_path = local_headers;
        println!("cargo:warning=nvllm not found. Building stub version...");

        let dst = Config::new("../../bindings/cpp/nvllm-trt")
            .define("USE_STUBS", "ON")
            .no_build_target(true)
            .build();

        println!("cargo:warning=building stubs in {}", dst.display());
        let dst = dst.canonicalize().unwrap();

        println!("cargo:rustc-link-search=native={}/build", dst.display());
        println!("cargo:rustc-link-lib=dylib=nvllm_trt");
        println!("cargo:rustc-link-lib=dylib=tensorrt_llm");

        println!("cargo:rerun-if-changed=../bindings/cpp/nvllm-trt");
    } else {
        panic!("nvllm_trt.h not found");
    }

    // generate bindings for the trtllm c api
    let bindings = bindgen::Builder::default()
        .header(headers_path)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to a file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Could not write bindings!");

    // // Build protobuf
    // tonic_build::configure()
    //     .build_server(false)
    //     .compile_protos(&["../../proto/trtllm.proto"], &["../../proto"])
    //     .expect("Failed to compile protos");
}
