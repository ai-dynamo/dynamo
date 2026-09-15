// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use llm_rs::model_card::ModelDeploymentCard as RsModelDeploymentCard;
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Clone)]
pub(crate) struct ModelDeploymentCard {
    pub(crate) inner: RsModelDeploymentCard,
}

fn aggregated_proxy_card(
    source: &RsModelDeploymentCard,
    model_name: &str,
    model_type: llm_rs::model_type::ModelType,
) -> anyhow::Result<RsModelDeploymentCard> {
    // Round-trip instead of cloning so a checksum previously cached on the
    // source card cannot survive the proxy-owned field changes below.
    let mut card = RsModelDeploymentCard::load_from_json_str(&source.to_json()?)?;
    card.set_name(model_name);
    card.model_type = model_type;
    card.model_input = llm_rs::model_type::ModelInput::Tokens;
    card.worker_type = Some(llm_rs::worker_type::WorkerType::Aggregated);
    card.needs.clear();
    Ok(card)
}

#[pymethods]
impl ModelDeploymentCard {
    // Previously called "from_local_path"
    #[staticmethod]
    fn load(path: String, model_name: String) -> PyResult<ModelDeploymentCard> {
        let mut card = RsModelDeploymentCard::load_from_disk(&path, None).map_err(to_pyerr)?;
        card.set_name(&model_name);
        Ok(ModelDeploymentCard { inner: card })
    }

    #[staticmethod]
    fn from_json_str(json: String) -> PyResult<ModelDeploymentCard> {
        let card = RsModelDeploymentCard::load_from_json_str(&json).map_err(to_pyerr)?;
        Ok(ModelDeploymentCard { inner: card })
    }

    fn to_json_str(&self) -> PyResult<String> {
        let json = self.inner.to_json().map_err(to_pyerr)?;
        Ok(json)
    }

    fn source_path(&self) -> &str {
        self.inner.source_path()
    }

    /// Resolved metadata directory (post-`download_config`).
    fn local_dir(&self) -> PyResult<String> {
        self.inner
            .local_dir()
            .into_os_string()
            .into_string()
            .map_err(|os| {
                PyValueError::new_err(format!("MDC local_dir contains non-UTF-8 bytes: {os:?}"))
            })
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_type(&self) -> ModelType {
        ModelType {
            inner: self.inner.model_type,
        }
    }

    fn runtime_config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let rc = pythonize::pythonize(py, &self.inner.runtime_config).map_err(to_pyerr)?;
        Ok(rc.unbind())
    }

    /// Preserve model metadata while changing the fields owned by an aggregated proxy.
    fn for_aggregated_proxy(
        &self,
        model_name: String,
        model_type: ModelType,
    ) -> PyResult<ModelDeploymentCard> {
        Ok(ModelDeploymentCard {
            inner: aggregated_proxy_card(&self.inner, &model_name, model_type.inner)
                .map_err(to_pyerr)?,
        })
    }

    /// Register this exact model card on another endpoint.
    fn register<'p>(&self, py: Python<'p>, endpoint: Endpoint) -> PyResult<Bound<'p, PyAny>> {
        let card = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            register_model_card(&endpoint.inner, &card)
                .await
                .map_err(to_pyerr)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregated_proxy_preserves_backend_metadata() {
        let mut source = RsModelDeploymentCard::with_name_only("internal-model");
        source.kv_cache_block_size = 64;
        source.migration_limit = 7;
        source.aliases = vec!["backend-alias".to_string()];
        source.user_data = Some(serde_json::json!({"owner": "backend"}));
        source.runtime_config.context_length = Some(131_072);
        source.runtime_config.runtime_data.insert(
            "token_budget".to_string(),
            serde_json::json!({
                "combined_limit": 131_072,
                "reject_prompt_overflow": true,
                "reject_total_overflow": true,
            }),
        );
        source.worker_type = Some(llm_rs::worker_type::WorkerType::Decode);
        source.needs = vec![vec![llm_rs::worker_type::WorkerType::Prefill]];
        let source_checksum = source.mdcsum().to_string();

        let proxy = aggregated_proxy_card(
            &source,
            "public-model",
            llm_rs::model_type::ModelType::Chat | llm_rs::model_type::ModelType::Completions,
        )
        .unwrap();

        assert_eq!(proxy.name(), "public-model");
        assert_eq!(proxy.kv_cache_block_size, source.kv_cache_block_size);
        assert_eq!(proxy.runtime_config, source.runtime_config);
        assert_ne!(proxy.mdcsum(), source_checksum);
        assert_eq!(
            proxy.worker_type,
            Some(llm_rs::worker_type::WorkerType::Aggregated)
        );
        assert!(proxy.needs.is_empty());
        assert_eq!(source.name(), "internal-model");
        assert_eq!(
            source.worker_type,
            Some(llm_rs::worker_type::WorkerType::Decode)
        );

        let mut source_json: serde_json::Value =
            serde_json::from_str(&source.to_json().unwrap()).unwrap();
        let mut proxy_json: serde_json::Value =
            serde_json::from_str(&proxy.to_json().unwrap()).unwrap();
        for field in [
            "display_name",
            "slug",
            "model_type",
            "model_input",
            "worker_type",
            "needs",
        ] {
            source_json.as_object_mut().unwrap().remove(field);
            proxy_json.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(proxy_json, source_json);
    }
}
