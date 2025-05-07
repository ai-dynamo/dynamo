// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use std::sync::LazyLock;

use anyhow::Context;
use regex::Regex;
use tokio::io::AsyncBufReadExt;

use dynamo_llm::engines::MultiNodeConfig;

pub mod sglang;
pub mod vllm;

/// Internal endpoint to connect the subprocess over etcd/nats
pub const ENDPOINT: &str = "dyn://dynamo.internal.worker";

pub async fn start(
    // The Python code to run
    py_script: &'static str,
    // Path to folder or file with model weights
    model_path: &Path,
    // How many GPUs to use
    tensor_parallel_size: u32,
    // sglang which GPU to start from, on a multi-GPU system
    // vllm uses CUDA_VISIBLE_DEVICES
    base_gpu_id: Option<u32>,
    // sglang multi-node config. vllm uses `ray` externally
    multi_node_config: Option<MultiNodeConfig>,
    // Path to a JSON file containing extra arguments to the backend engine
    extra_engine_args: Option<&Path>,
) -> anyhow::Result<(tempfile::TempPath, tokio::process::Child)> {
    let mut tmp = tempfile::NamedTempFile::new()?;
    // Writes on Linux don't block
    tmp.write_all(py_script.as_bytes())?;
    let script_path = tmp.into_temp_path();

    let mut args = vec![
        script_path.to_string_lossy().to_string(),
        "--endpoint".to_string(),
        ENDPOINT.to_string(),
        "--model".to_string(),
        model_path.to_string_lossy().to_string(),
        "--tensor-parallel-size".to_string(),
        tensor_parallel_size.to_string(),
    ];
    // sglang only
    if let Some(base_gpu_id) = base_gpu_id {
        args.push("--base-gpu-id".to_string());
        args.push(base_gpu_id.to_string());
    }
    // sglang only
    if let Some(multi_node_config) = multi_node_config {
        args.push("--nnodes".to_string());
        args.push(multi_node_config.num_nodes.to_string());
        args.push("--node-rank".to_string());
        args.push(multi_node_config.node_rank.to_string());
        args.push("--dist-init-addr".to_string());
        args.push(multi_node_config.leader_addr);
    }
    if let Some(extra_engine_args) = extra_engine_args {
        args.push("--extra-engine-args".to_string());
        args.push(extra_engine_args.to_string_lossy().to_string());
    }
    let mut cmd = tokio::process::Command::new("python3");
    cmd.kill_on_drop(false)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("Failed running: '{}'", pretty_cmd(&cmd)))?;
    // Safety: We set stdout/stderr a few lines above
    let stdout = tokio::io::BufReader::new(child.stdout.take().unwrap());
    let stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());

    tokio::spawn(async move {
        let mut lines = stdout.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("{}", strip_log_prefix(&line));
        }
    });
    tokio::spawn(async move {
        let mut lines = stderr.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("{}", strip_log_prefix(&line));
        }
    });

    // We must keep temp path alive, it deletes on drop
    Ok((script_path, child))
}

pub fn pretty_cmd(c: &tokio::process::Command) -> String {
    format!(
        "{} {}",
        c.as_std().get_program().to_string_lossy(),
        c.as_std()
            .get_args()
            .map(|x| x.to_string_lossy())
            .collect::<Vec<std::borrow::Cow<'_, str>>>()
            .join(" ")
    )
}

// Thanks Gemini
static LOG_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:(?:[A-Z]+ \d{2}-\d{2} \d{2}:\d{2}:\d{2})|(?:\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])) (.*)$"
    ).unwrap()
    // ^                                  Start of the line.
    // (?:                                Non-capturing group for the two prefix alternatives.
    //      (?:                           Non-capturing group for the first prefix type.
    //          [A-Z]+                    One or more uppercase letters (log level).
    //            (single space)          A literal space.
    //          \d{2}-\d{2}               Date like MM-DD.
    //            (single space)          A literal space.
    //          \d{2}:\d{2}:\d{2}         Time like HH:MM:SS.
    //      )                             End of first prefix type group.
    //      |                             OR
    //      (?:                           Non-capturing group for the second prefix type.
    //          \[                        A literal opening square bracket.
    //          \d{4}-\d{2}-\d{2}         Date like YYYY-MM-DD.
    //            (single space)          A literal space.
    //          \d{2}:\d{2}:\d{2}         Time like HH:MM:SS.
    //          \]                        A literal closing square bracket.
    //      )                             End of second prefix type group.
    // )                                  End of the alternatives group.
    //   (single space)                   A literal space. This is the space BEFORE the message.
    // (.*)                               Capture group 1: The rest of the line (the message).
    // $                                  End of the line.
});

/// Strips the log level, date, and time from the start of a log line.
///
/// # Examples
/// ```
/// let line = "INFO 05-06 09:38:50 [async_llm.py:252] Added request 1";
/// assert_eq!(strip_log_prefix(line), "[async_llm.py:252] Added request 1");
///
/// let line_no_prefix = "This is a normal line.";
/// assert_eq!(strip_log_prefix(line_no_prefix), "This is a normal line.");
/// ```
fn strip_log_prefix(line: &str) -> Cow<'_, str> {
    if let Some(captures) = LOG_PREFIX_RE.captures(line) {
        // `captures.get(0)` would be the entire matched prefix + message.
        // `captures.get(1)` is the first capture group, which is `(.*)`, the message itself.
        if let Some(message_match) = captures.get(1) {
            return Cow::Borrowed(message_match.as_str());
        }
    }
    // If the regex doesn't match, or somehow the capture group is not found (shouldn't happen with (.*))
    // return the original line.
    Cow::Borrowed(line)
}

#[cfg(test)]
mod tests {
    use super::strip_log_prefix;

    #[test]
    fn test_strip_log_prefix() {
        let line = "INFO 05-06 09:38:50 [async_llm.py:252] Added request 1";
        let expected = "[async_llm.py:252] Added request 1";
        assert_eq!(strip_log_prefix(line), expected);

        let line = "Just a regular line.";
        assert_eq!(strip_log_prefix(line), line);

        let line = "INFO this is not a full prefix";
        assert_eq!(strip_log_prefix(line), line);

        let line = "[2025-05-06 11:58:51] Capture cuda graph bs [1, 2, 4, 8]";
        assert_eq!(strip_log_prefix(line), "Capture cuda graph bs [1, 2, 4, 8]");
    }
}
