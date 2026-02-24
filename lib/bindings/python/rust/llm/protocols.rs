// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for LLM protocol types (images, videos).
//!
//! These wrappers expose the Rust protocol types to Python with a Pydantic-compatible API

use pyo3::prelude::*;
use pyo3::types::PyDict;

use dynamo_llm::protocols::openai::{
    images::{self as rs_images, NvExt as RsImageNvExt},
    videos::{self as rs_videos, NvExt as RsVideoNvExt},
};

use super::super::to_pyerr;

// ============================================================================
// Image Protocol Types
// ============================================================================

/// NVIDIA extensions for image generation requests.
#[pyclass(name = "ImageNvExt")]
#[derive(Clone)]
pub struct ImageNvExt {
    pub(crate) inner: RsImageNvExt,
}

#[pymethods]
impl ImageNvExt {
    #[new]
    #[pyo3(signature = (*, annotations=None, negative_prompt=None, num_inference_steps=None, guidance_scale=None, seed=None))]
    fn new(
        annotations: Option<Vec<String>>,
        negative_prompt: Option<String>,
        num_inference_steps: Option<u8>,
        guidance_scale: Option<f32>,
        seed: Option<u32>,
    ) -> Self {
        Self {
            inner: RsImageNvExt {
                annotations,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                seed,
            },
        }
    }

    #[getter]
    fn annotations(&self) -> Option<Vec<String>> {
        self.inner.annotations.clone()
    }

    #[getter]
    fn negative_prompt(&self) -> Option<String> {
        self.inner.negative_prompt.clone()
    }

    #[getter]
    fn num_inference_steps(&self) -> Option<u8> {
        self.inner.num_inference_steps
    }

    #[getter]
    fn guidance_scale(&self) -> Option<f32> {
        self.inner.guidance_scale
    }

    #[getter]
    fn seed(&self) -> Option<u32> {
        self.inner.seed
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: RsImageNvExt = serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: RsImageNvExt = pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageNvExt(annotations={:?}, negative_prompt={:?}, num_inference_steps={:?}, guidance_scale={:?}, seed={:?})",
            self.inner.annotations, self.inner.negative_prompt, self.inner.num_inference_steps,
            self.inner.guidance_scale, self.inner.seed,
        )
    }
}

/// Request for image generation (/v1/images/generations endpoint).
#[pyclass(name = "NvCreateImageRequest")]
#[derive(Clone)]
pub struct NvCreateImageRequest {
    pub(crate) inner: rs_images::NvCreateImageRequest,
}

#[pymethods]
impl NvCreateImageRequest {
    #[new]
    #[pyo3(signature = (prompt, *, model=None, n=None, quality=None, response_format=None, size=None, style=None, user=None, moderation=None, nvext=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt: String,
        model: Option<String>,
        n: Option<u8>,
        quality: Option<String>,
        response_format: Option<String>,
        size: Option<String>,
        style: Option<String>,
        user: Option<String>,
        moderation: Option<String>,
        nvext: Option<ImageNvExt>,
    ) -> PyResult<Self> {
        // Build the inner CreateImageRequest by serializing to JSON and back,
        // which lets serde handle enum conversions (ImageModel, ImageQuality, etc.)
        let mut map = serde_json::Map::new();
        map.insert("prompt".into(), serde_json::Value::String(prompt));
        if let Some(v) = model {
            map.insert("model".into(), serde_json::Value::String(v));
        }
        if let Some(v) = n {
            map.insert("n".into(), serde_json::Value::Number(v.into()));
        }
        if let Some(v) = quality {
            map.insert("quality".into(), serde_json::Value::String(v));
        }
        if let Some(v) = response_format {
            map.insert("response_format".into(), serde_json::Value::String(v));
        }
        if let Some(v) = size {
            map.insert("size".into(), serde_json::Value::String(v));
        }
        if let Some(v) = style {
            map.insert("style".into(), serde_json::Value::String(v));
        }
        if let Some(v) = user {
            map.insert("user".into(), serde_json::Value::String(v));
        }
        if let Some(v) = moderation {
            map.insert("moderation".into(), serde_json::Value::String(v));
        }
        if let Some(ext) = &nvext {
            let nvext_val = serde_json::to_value(&ext.inner).map_err(to_pyerr)?;
            map.insert("nvext".into(), nvext_val);
        }
        let inner: rs_images::NvCreateImageRequest =
            serde_json::from_value(serde_json::Value::Object(map)).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.inner.prompt
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.inner.model.as_ref().map(|m| {
            serde_json::to_value(m)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn n(&self) -> Option<u8> {
        self.inner.inner.n
    }

    #[getter]
    fn quality(&self) -> Option<String> {
        self.inner.inner.quality.as_ref().map(|q| {
            serde_json::to_value(q)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn response_format(&self) -> Option<String> {
        self.inner.inner.response_format.as_ref().map(|r| {
            serde_json::to_value(r)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn size(&self) -> Option<String> {
        self.inner.inner.size.as_ref().map(|s| {
            serde_json::to_value(s)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn style(&self) -> Option<String> {
        self.inner.inner.style.as_ref().map(|s| {
            serde_json::to_value(s)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn user(&self) -> Option<String> {
        self.inner.inner.user.clone()
    }

    #[getter]
    fn moderation(&self) -> Option<String> {
        self.inner.inner.moderation.as_ref().map(|m| {
            serde_json::to_value(m)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default()
        })
    }

    #[getter]
    fn nvext(&self) -> Option<ImageNvExt> {
        self.inner.nvext.clone().map(|n| ImageNvExt { inner: n })
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: rs_images::NvCreateImageRequest =
            serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: rs_images::NvCreateImageRequest =
            pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "NvCreateImageRequest(prompt={:?}, model={:?})",
            self.inner.inner.prompt,
            self.model(),
        )
    }
}

/// Individual image data in a response.
#[pyclass(name = "ImageData")]
#[derive(Clone)]
pub struct ImageData {
    pub(crate) url: Option<String>,
    pub(crate) b64_json: Option<String>,
    pub(crate) revised_prompt: Option<String>,
}

#[pymethods]
impl ImageData {
    #[new]
    #[pyo3(signature = (*, url=None, b64_json=None, revised_prompt=None))]
    fn new(
        url: Option<String>,
        b64_json: Option<String>,
        revised_prompt: Option<String>,
    ) -> Self {
        Self {
            url,
            b64_json,
            revised_prompt,
        }
    }

    #[getter]
    fn url(&self) -> Option<String> {
        self.url.clone()
    }

    #[getter]
    fn b64_json(&self) -> Option<String> {
        self.b64_json.clone()
    }

    #[getter]
    fn revised_prompt(&self) -> Option<String> {
        self.revised_prompt.clone()
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let val: serde_json::Value = serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self::from_json_value(&val))
    }

    fn model_dump_json(&self) -> PyResult<String> {
        let val = self.to_json_value();
        serde_json::to_string(&val).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let val: serde_json::Value = pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self::from_json_value(&val))
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let val = self.to_json_value();
        let obj = pythonize::pythonize(py, &val).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageData(url={:?}, b64_json={:?}, revised_prompt={:?})",
            self.url, self.b64_json, self.revised_prompt,
        )
    }
}

impl ImageData {
    /// Convert to a JSON value that matches the Python Pydantic schema.
    fn to_json_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if let Some(ref u) = self.url {
            map.insert("url".into(), serde_json::Value::String(u.clone()));
        } else {
            map.insert("url".into(), serde_json::Value::Null);
        }
        if let Some(ref b) = self.b64_json {
            map.insert("b64_json".into(), serde_json::Value::String(b.clone()));
        } else {
            map.insert("b64_json".into(), serde_json::Value::Null);
        }
        if let Some(ref r) = self.revised_prompt {
            map.insert(
                "revised_prompt".into(),
                serde_json::Value::String(r.clone()),
            );
        } else {
            map.insert("revised_prompt".into(), serde_json::Value::Null);
        }
        serde_json::Value::Object(map)
    }

    /// Parse from a JSON value.
    fn from_json_value(val: &serde_json::Value) -> Self {
        Self {
            url: val
                .get("url")
                .and_then(|v| v.as_str())
                .map(String::from),
            b64_json: val
                .get("b64_json")
                .and_then(|v| v.as_str())
                .map(String::from),
            revised_prompt: val
                .get("revised_prompt")
                .and_then(|v| v.as_str())
                .map(String::from),
        }
    }
}

/// Response structure for image generation.
#[pyclass(name = "NvImagesResponse")]
#[derive(Clone)]
pub struct NvImagesResponse {
    pub(crate) created: i64,
    pub(crate) data: Vec<ImageData>,
}

#[pymethods]
impl NvImagesResponse {
    #[new]
    #[pyo3(signature = (created, *, data=None))]
    fn new(created: i64, data: Option<Vec<ImageData>>) -> Self {
        Self {
            created,
            data: data.unwrap_or_default(),
        }
    }

    #[getter]
    fn created(&self) -> i64 {
        self.created
    }

    #[getter]
    fn data(&self) -> Vec<ImageData> {
        self.data.clone()
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let val: serde_json::Value = serde_json::from_str(json_str).map_err(to_pyerr)?;
        Self::from_json_value(&val)
    }

    fn model_dump_json(&self) -> PyResult<String> {
        let val = self.to_json_value();
        serde_json::to_string(&val).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let val: serde_json::Value = pythonize::depythonize(obj).map_err(to_pyerr)?;
        Self::from_json_value(&val)
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let val = self.to_json_value();
        let obj = pythonize::pythonize(py, &val).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "NvImagesResponse(created={}, data=[{} items])",
            self.created,
            self.data.len(),
        )
    }
}

impl NvImagesResponse {
    fn from_json_value(val: &serde_json::Value) -> PyResult<Self> {
        let created = val
            .get("created")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| to_pyerr("missing or invalid 'created' field"))?;
        let data = val
            .get("data")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(ImageData::from_json_value).collect())
            .unwrap_or_default();
        Ok(Self { created, data })
    }

    fn to_json_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "created".into(),
            serde_json::Value::Number(self.created.into()),
        );
        let data_arr: Vec<serde_json::Value> =
            self.data.iter().map(|d| d.to_json_value()).collect();
        map.insert("data".into(), serde_json::Value::Array(data_arr));
        serde_json::Value::Object(map)
    }
}

// ============================================================================
// Video Protocol Types
// ============================================================================

/// NVIDIA extensions for video generation requests.
#[pyclass(name = "VideoNvExt")]
#[derive(Clone)]
pub struct VideoNvExt {
    pub(crate) inner: RsVideoNvExt,
}

#[pymethods]
impl VideoNvExt {
    #[new]
    #[pyo3(signature = (*, annotations=None, fps=None, num_frames=None, negative_prompt=None, num_inference_steps=None, guidance_scale=None, seed=None))]
    fn new(
        annotations: Option<Vec<String>>,
        fps: Option<i32>,
        num_frames: Option<i32>,
        negative_prompt: Option<String>,
        num_inference_steps: Option<i32>,
        guidance_scale: Option<f32>,
        seed: Option<i64>,
    ) -> Self {
        Self {
            inner: RsVideoNvExt {
                annotations,
                fps,
                num_frames,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                seed,
            },
        }
    }

    #[getter]
    fn annotations(&self) -> Option<Vec<String>> {
        self.inner.annotations.clone()
    }

    #[getter]
    fn fps(&self) -> Option<i32> {
        self.inner.fps
    }

    #[getter]
    fn num_frames(&self) -> Option<i32> {
        self.inner.num_frames
    }

    #[getter]
    fn negative_prompt(&self) -> Option<String> {
        self.inner.negative_prompt.clone()
    }

    #[getter]
    fn num_inference_steps(&self) -> Option<i32> {
        self.inner.num_inference_steps
    }

    #[getter]
    fn guidance_scale(&self) -> Option<f32> {
        self.inner.guidance_scale
    }

    #[getter]
    fn seed(&self) -> Option<i64> {
        self.inner.seed
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: RsVideoNvExt = serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: RsVideoNvExt = pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoNvExt(annotations={:?}, fps={:?}, num_frames={:?})",
            self.inner.annotations, self.inner.fps, self.inner.num_frames,
        )
    }
}

/// Request for video generation (/v1/videos endpoint).
#[pyclass(name = "NvCreateVideoRequest")]
#[derive(Clone)]
pub struct NvCreateVideoRequest {
    pub(crate) inner: rs_videos::NvCreateVideoRequest,
}

#[pymethods]
impl NvCreateVideoRequest {
    #[new]
    #[pyo3(signature = (prompt, model, *, input_reference=None, seconds=None, size=None, user=None, response_format=None, nvext=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt: String,
        model: String,
        input_reference: Option<String>,
        seconds: Option<i32>,
        size: Option<String>,
        user: Option<String>,
        response_format: Option<String>,
        nvext: Option<VideoNvExt>,
    ) -> Self {
        Self {
            inner: rs_videos::NvCreateVideoRequest {
                prompt,
                model,
                input_reference,
                seconds,
                size,
                user,
                response_format,
                nvext: nvext.map(|n| n.inner),
            },
        }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn input_reference(&self) -> Option<String> {
        self.inner.input_reference.clone()
    }

    #[getter]
    fn seconds(&self) -> Option<i32> {
        self.inner.seconds
    }

    #[getter]
    fn size(&self) -> Option<String> {
        self.inner.size.clone()
    }

    #[getter]
    fn user(&self) -> Option<String> {
        self.inner.user.clone()
    }

    #[getter]
    fn response_format(&self) -> Option<String> {
        self.inner.response_format.clone()
    }

    #[getter]
    fn nvext(&self) -> Option<VideoNvExt> {
        self.inner.nvext.clone().map(|n| VideoNvExt { inner: n })
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: rs_videos::NvCreateVideoRequest =
            serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: rs_videos::NvCreateVideoRequest =
            pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "NvCreateVideoRequest(prompt={:?}, model={:?})",
            self.inner.prompt, self.inner.model,
        )
    }
}

/// Video data in response.
#[pyclass(name = "VideoData")]
#[derive(Clone)]
pub struct VideoData {
    pub(crate) inner: rs_videos::VideoData,
}

#[pymethods]
impl VideoData {
    #[new]
    #[pyo3(signature = (*, url=None, b64_json=None))]
    fn new(url: Option<String>, b64_json: Option<String>) -> Self {
        Self {
            inner: rs_videos::VideoData { url, b64_json },
        }
    }

    #[getter]
    fn url(&self) -> Option<String> {
        self.inner.url.clone()
    }

    #[getter]
    fn b64_json(&self) -> Option<String> {
        self.inner.b64_json.clone()
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: rs_videos::VideoData = serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: rs_videos::VideoData = pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoData(url={:?}, b64_json={:?})",
            self.inner.url, self.inner.b64_json,
        )
    }
}

/// Response structure for video generation.
#[pyclass(name = "NvVideosResponse")]
#[derive(Clone)]
pub struct NvVideosResponse {
    pub(crate) inner: rs_videos::NvVideosResponse,
}

#[pymethods]
impl NvVideosResponse {
    #[new]
    #[pyo3(signature = (id, model, created, *, object="video".to_string(), status="completed".to_string(), progress=100, data=None, error=None, inference_time_s=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        model: String,
        created: i64,
        object: String,
        status: String,
        progress: i32,
        data: Option<Vec<VideoData>>,
        error: Option<String>,
        inference_time_s: Option<f64>,
    ) -> Self {
        Self {
            inner: rs_videos::NvVideosResponse {
                id,
                object,
                model,
                status,
                progress,
                created,
                data: data.into_iter().flatten().map(|d| d.inner).collect(),
                error,
                inference_time_s,
            },
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn object(&self) -> &str {
        &self.inner.object
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn status(&self) -> &str {
        &self.inner.status
    }

    #[getter]
    fn progress(&self) -> i32 {
        self.inner.progress
    }

    #[getter]
    fn created(&self) -> i64 {
        self.inner.created
    }

    #[getter]
    fn data(&self) -> Vec<VideoData> {
        self.inner
            .data
            .iter()
            .map(|d| VideoData { inner: d.clone() })
            .collect()
    }

    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    #[getter]
    fn inference_time_s(&self) -> Option<f64> {
        self.inner.inference_time_s
    }

    #[staticmethod]
    fn model_validate_json(json_str: &str) -> PyResult<Self> {
        let inner: rs_videos::NvVideosResponse =
            serde_json::from_str(json_str).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(to_pyerr)
    }

    #[staticmethod]
    fn model_validate(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner: rs_videos::NvVideosResponse =
            pythonize::depythonize(obj).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    fn model_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let obj = pythonize::pythonize(py, &self.inner).map_err(to_pyerr)?;
        obj.downcast_into::<PyDict>()
            .map_err(|e| to_pyerr(format!("expected dict, got {}", e)))
    }

    fn __repr__(&self) -> String {
        format!(
            "NvVideosResponse(id={:?}, model={:?}, status={:?}, data=[{} items])",
            self.inner.id,
            self.inner.model,
            self.inner.status,
            self.inner.data.len(),
        )
    }
}
