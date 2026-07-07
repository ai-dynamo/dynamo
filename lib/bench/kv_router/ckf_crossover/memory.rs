// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(target_os = "linux"))]
use std::process::Command;

#[derive(Clone, Copy, Debug, Default)]
pub struct MemorySnapshot {
    pub rss_bytes: u64,
    pub pss_bytes: Option<u64>,
    pub uss_bytes: Option<u64>,
}

pub fn memory_snapshot() -> anyhow::Result<MemorySnapshot> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/self/smaps_rollup")?;
        let value = |name: &str| -> Option<u64> {
            contents.lines().find_map(|line| {
                let (key, rest) = line.split_once(':')?;
                (key == name).then(|| {
                    rest.split_whitespace()
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|kb| kb * 1024)
                })?
            })
        };
        let private_clean = value("Private_Clean").unwrap_or(0);
        let private_dirty = value("Private_Dirty").unwrap_or(0);
        return Ok(MemorySnapshot {
            rss_bytes: value("Rss").unwrap_or(0),
            pss_bytes: value("Pss"),
            uss_bytes: Some(private_clean + private_dirty),
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()?;
        let rss_kb = String::from_utf8(output.stdout)?.trim().parse::<u64>()?;
        Ok(MemorySnapshot {
            rss_bytes: rss_kb * 1024,
            pss_bytes: None,
            uss_bytes: None,
        })
    }
}
