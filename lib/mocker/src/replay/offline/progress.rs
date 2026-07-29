// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use indicatif::{ProgressBar, ProgressStyle};

pub(super) struct ReplayProgress {
    bar: ProgressBar,
}

impl ReplayProgress {
    pub(super) fn new(total_requests: usize, label: &'static str) -> Self {
        let bar = ProgressBar::new(total_requests as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}",
            )
            .expect("progress bar template must be valid")
            .progress_chars("#>-"),
        );
        bar.set_message(label);
        Self { bar }
    }

    pub(super) fn inc_completed(&self) {
        self.bar.inc(1);
    }

    pub(super) fn finish(&self) {
        self.bar.finish_and_clear();
    }
}

impl Drop for ReplayProgress {
    fn drop(&mut self) {
        if !self.bar.is_finished() {
            self.bar.finish_and_clear();
        }
    }
}
