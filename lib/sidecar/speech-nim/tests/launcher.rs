// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(unix)]

use std::{fs, os::unix::fs::PermissionsExt, path::Path, time::Duration};

use tokio::process::Command;

async fn launch(variables: &[(&str, &str)]) -> (tempfile::TempDir, std::process::Output) {
    let directory = tempfile::tempdir().unwrap();
    let capture = directory.path().join("calls");
    fs::create_dir(&capture).unwrap();
    for name in ["speech-sidecar", "python"] {
        let executable = directory.path().join(name);
        fs::write(
            &executable,
            r#"#!/bin/bash
name=$1
if [[ $1 == -m ]]; then name=$2; fi
printf '%s\n' "$@" > "$SPEECH_LAUNCH_CAPTURE_DIR/$name.tmp"
mv "$SPEECH_LAUNCH_CAPTURE_DIR/$name.tmp" "$SPEECH_LAUNCH_CAPTURE_DIR/$name"
for ((attempt=0; attempt<200; attempt++)); do
    if [[ -f $SPEECH_LAUNCH_CAPTURE_DIR/asr && -f $SPEECH_LAUNCH_CAPTURE_DIR/tts &&
          -f $SPEECH_LAUNCH_CAPTURE_DIR/dynamo.vllm && -f $SPEECH_LAUNCH_CAPTURE_DIR/dynamo.frontend ]]; then
        exit 0
    fi
    sleep 0.01
done
exit 1
"#,
        )
        .unwrap();
        fs::set_permissions(executable, fs::Permissions::from_mode(0o755)).unwrap();
    }
    let output = tokio::time::timeout(
        Duration::from_secs(5),
        Command::new("bash")
            .process_group(0)
            .arg(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../examples/nemotron_speech_cascaded_pipeline/launch_workers.sh"),
            )
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", directory.path().display()),
            )
            .env(
                "SPEECH_NIM_SIDECAR_BIN",
                directory.path().join("speech-sidecar"),
            )
            .env("SPEECH_LAUNCH_CAPTURE_DIR", &capture)
            .envs(variables.iter().copied())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("launcher timed out")
    .unwrap();
    (directory, output)
}

#[tokio::test]
async fn launcher_rejects_legacy_servers_before_starting_children() {
    for (legacy, replacement) in [
        ("ASR_NIM_SERVER", "ASR_NIM_PORT"),
        ("TTS_NIM_SERVER", "TTS_NIM_PORT"),
    ] {
        for also_set_new_port in [false, true] {
            let mut variables = vec![(legacy, "localhost:55152")];
            if also_set_new_port {
                variables.push((replacement, "55152"));
            }
            let (directory, output) = launch(&variables).await;
            assert!(!output.status.success());
            let error = String::from_utf8(output.stderr).unwrap();
            assert!(
                error.contains(legacy) && error.contains(replacement),
                "{error}"
            );
            assert_eq!(
                fs::read_dir(directory.path().join("calls"))
                    .unwrap()
                    .count(),
                0
            );
        }
    }
}

#[tokio::test]
async fn launcher_uses_default_and_explicit_local_ports() {
    for (variables, asr, tts) in [
        (vec![], "50152", "50151"),
        (
            vec![("ASR_NIM_SERVER", ""), ("TTS_NIM_SERVER", "")],
            "50152",
            "50151",
        ),
        (
            vec![("ASR_NIM_PORT", "55152"), ("TTS_NIM_PORT", "55151")],
            "55152",
            "55151",
        ),
    ] {
        let (directory, output) = launch(&variables).await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        for (mode, port) in [("asr", asr), ("tts", tts)] {
            let arguments = fs::read_to_string(directory.path().join("calls").join(mode)).unwrap();
            assert!(
                arguments.contains(&format!("--grpc-endpoint\n127.0.0.1:{port}\n")),
                "{arguments}"
            );
        }
        assert!(directory.path().join("calls/dynamo.vllm").exists());
        assert!(directory.path().join("calls/dynamo.frontend").exists());
    }
}
