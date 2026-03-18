// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::time::Duration;

use llm_rs::preprocessor::media::{MediaDecoder as RsMediaDecoder, MediaFetcher as RsMediaFetcher};

#[pyclass]
#[derive(Clone)]
pub struct MediaDecoder {
    pub(crate) inner: RsMediaDecoder,
}

#[pymethods]
impl MediaDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMediaDecoder::default(),
        }
    }

    fn enable_image(&mut self, decoder_options: &Bound<'_, PyDict>) -> PyResult<()> {
        let decoder_options = pythonize::depythonize(decoder_options).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to parse image decoder config: {}", err))
        })?;
        self.inner.image = Some(decoder_options);
        Ok(())
    }

    #[cfg(feature = "media-ffmpeg")]
    fn enable_video(&mut self, decoder_options: &Bound<'_, PyDict>) -> PyResult<()> {
        let decoder_options = pythonize::depythonize(decoder_options).map_err(|err| {
            PyErr::new::<PyException, _>(format!("Failed to parse video decoder config: {}", err))
        })?;
        self.inner.video = Some(decoder_options);
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct MediaFetcher {
    pub(crate) inner: RsMediaFetcher,
}

#[pymethods]
impl MediaFetcher {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsMediaFetcher::default(),
        }
    }
    fn user_agent(&mut self, user_agent: String) {
        self.inner.user_agent = user_agent;
    }

    fn allow_direct_ip(&mut self, allow: bool) {
        self.inner.allow_direct_ip = allow;
    }

    fn allow_direct_port(&mut self, allow: bool) {
        self.inner.allow_direct_port = allow;
    }

    fn allowed_media_domains(&mut self, domains: Vec<String>) {
        self.inner.allowed_media_domains = Some(domains.into_iter().collect());
    }

    fn timeout_ms(&mut self, timeout_ms: u64) {
        self.inner.timeout = Some(Duration::from_millis(timeout_ms));
    }
}
