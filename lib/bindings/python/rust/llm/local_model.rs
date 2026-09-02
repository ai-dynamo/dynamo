// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};

use super::*;
use dynamo_kv_router::protocols::{
    KvTransferEnforcement as RsKvTransferEnforcement, RoutingConstraints as RsRoutingConstraints,
};
use dynamo_runtime::protocols::EndpointId;
use llm_rs::local_model::runtime_config::DisaggregatedEndpoint as RsDisaggregatedEndpoint;
use llm_rs::local_model::runtime_config::ModelRuntimeConfig as RsModelRuntimeConfig;
use llm_rs::local_model::runtime_config::StructuralTagMode as RsStructuralTagMode;
use llm_rs::local_model::runtime_config::StructuralTagSchemaMode as RsStructuralTagSchemaMode;
use llm_rs::local_model::runtime_config::StructuralTagScope as RsStructuralTagScope;
use llm_rs::local_model::runtime_config::TokenizerBackend as RsTokenizerBackend;
use llm_rs::protocols::external_speculation::{
    DraftTransportDescriptorV1 as RsDraftTransportDescriptorV1,
    new_external_speculation_incarnation,
};
use llm_rs::protocols::tensor::TensorModelConfig;
use pyo3::exceptions::PyValueError;

fn validate_model_runtime_config(config: &RsModelRuntimeConfig) -> PyResult<()> {
    config.validate_config().map_err(PyValueError::new_err)
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct RoutingConstraints {
    #[pyo3(get, set)]
    pub required_taints: HashSet<String>,
    #[pyo3(get, set)]
    pub preferred_taints: HashMap<String, f32>,
}

#[pymethods]
impl RoutingConstraints {
    #[new]
    #[pyo3(signature = (required_taints=None, preferred_taints=None))]
    fn new(
        required_taints: Option<HashSet<String>>,
        preferred_taints: Option<HashMap<String, f32>>,
    ) -> Self {
        Self {
            required_taints: required_taints.unwrap_or_default(),
            preferred_taints: preferred_taints.unwrap_or_default(),
        }
    }
}

impl From<RoutingConstraints> for RsRoutingConstraints {
    fn from(value: RoutingConstraints) -> Self {
        Self {
            required_taints: value.required_taints,
            preferred_taints: value.preferred_taints,
        }
    }
}

impl From<RsRoutingConstraints> for RoutingConstraints {
    fn from(value: RsRoutingConstraints) -> Self {
        Self {
            required_taints: value.required_taints,
            preferred_taints: value.preferred_taints,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct ModelRuntimeConfig {
    pub(crate) inner: RsModelRuntimeConfig,
}

impl ModelRuntimeConfig {
    pub(crate) fn validate_config(&self) -> PyResult<()> {
        validate_model_runtime_config(&self.inner)
    }

    fn require_selectable_dp_rank(&self, dp_rank: u32) -> PyResult<()> {
        let end = self
            .inner
            .data_parallel_start_rank
            .checked_add(self.inner.data_parallel_size)
            .ok_or_else(|| PyValueError::new_err("data-parallel rank range overflows u32"))?;
        if dp_rank < self.inner.data_parallel_start_rank || dp_rank >= end {
            return Err(PyValueError::new_err(format!(
                "DP rank {dp_rank} is outside [{}, {end})",
                self.inner.data_parallel_start_rank
            )));
        }
        Ok(())
    }
}

fn insert_immutable_descriptor<T: PartialEq>(
    descriptors: &mut BTreeMap<u32, T>,
    dp_rank: u32,
    descriptor: T,
    name: &str,
) -> PyResult<()> {
    validate_immutable_descriptor(descriptors, dp_rank, &descriptor, name)?;
    if descriptors.contains_key(&dp_rank) {
        return Ok(());
    }
    descriptors.insert(dp_rank, descriptor);
    Ok(())
}

fn validate_immutable_descriptor<T: PartialEq>(
    descriptors: &BTreeMap<u32, T>,
    dp_rank: u32,
    descriptor: &T,
    name: &str,
) -> PyResult<()> {
    if descriptors
        .get(&dp_rank)
        .is_some_and(|existing| existing != descriptor)
    {
        return Err(PyValueError::new_err(format!(
            "{name} for DP rank {dp_rank} is immutable for this runtime configuration"
        )));
    }
    Ok(())
}

pub(crate) fn parse_tensor_model_config(
    tensor_model_config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<TensorModelConfig>> {
    tensor_model_config
        .map(|config| {
            pythonize::depythonize(config).map_err(|err| {
                PyErr::new::<PyException, _>(format!(
                    "Failed to convert tensor_model_config: {err}"
                ))
            })
        })
        .transpose()
}

#[pymethods]
impl ModelRuntimeConfig {
    #[new]
    fn new() -> PyResult<Self> {
        let config = Self {
            inner: RsModelRuntimeConfig::new(),
        };
        config.validate_config()?;
        Ok(config)
    }

    #[setter]
    fn set_total_kv_blocks(&mut self, total_kv_blocks: u64) {
        self.inner.total_kv_blocks = Some(total_kv_blocks);
    }

    #[setter]
    fn set_context_length(&mut self, context_length: Option<u32>) {
        self.inner.context_length = context_length;
    }

    #[setter]
    fn set_max_num_seqs(&mut self, max_num_seqs: u64) {
        self.inner.max_num_seqs = Some(max_num_seqs);
    }

    #[setter]
    fn set_max_num_batched_tokens(&mut self, max_num_batched_tokens: u64) {
        self.inner.max_num_batched_tokens = Some(max_num_batched_tokens);
    }

    #[setter]
    fn set_tool_call_parser(&mut self, tool_call_parser: Option<String>) {
        self.inner.tool_call_parser = tool_call_parser;
    }

    #[setter]
    fn set_reasoning_parser(&mut self, reasoning_parser: Option<String>) {
        self.inner.reasoning_parser = reasoning_parser;
    }

    #[setter]
    fn set_tokenizer_backend(&mut self, tokenizer_backend: Option<String>) -> PyResult<()> {
        self.inner.tokenizer_backend = tokenizer_backend
            .map(|backend| {
                backend
                    .parse::<RsTokenizerBackend>()
                    .map_err(PyValueError::new_err)
            })
            .transpose()?;
        Ok(())
    }

    #[setter]
    fn set_data_parallel_start_rank(&mut self, data_parallel_start_rank: u32) {
        self.inner.data_parallel_start_rank = data_parallel_start_rank;
    }

    /// Advertise one immutable external draft transport for a global DP rank.
    #[pyo3(signature = (dp_rank, protocol, address, orphan_cleanup_timeout_ms, draft_incarnation_id=None))]
    fn add_external_draft_transport(
        &mut self,
        dp_rank: u32,
        protocol: String,
        address: String,
        orphan_cleanup_timeout_ms: u32,
        draft_incarnation_id: Option<u64>,
    ) -> PyResult<u64> {
        self.require_selectable_dp_rank(dp_rank)?;
        let draft_incarnation_id = match draft_incarnation_id {
            Some(value) => value,
            None => new_external_speculation_incarnation()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        };
        let descriptor = RsDraftTransportDescriptorV1 {
            protocol,
            address,
            draft_incarnation_id,
            orphan_cleanup_timeout_ms,
        };
        descriptor
            .validate()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        insert_immutable_descriptor(
            &mut self.inner.external_draft_transports,
            dp_rank,
            descriptor,
            "external draft transport",
        )?;
        Ok(draft_incarnation_id)
    }

    #[setter]
    fn set_data_parallel_size(&mut self, data_parallel_size: u32) {
        self.inner.data_parallel_size = data_parallel_size;
    }

    #[setter]
    fn set_enable_local_indexer(&mut self, enable_local_indexer: bool) {
        self.inner.enable_local_indexer = enable_local_indexer;
    }

    #[setter]
    fn set_kv_event_publishing_enabled(&mut self, enabled: Option<bool>) {
        self.inner.kv_event_publishing_enabled = enabled;
    }

    #[setter]
    fn set_kv_event_source_mode(&mut self, mode: Option<String>) {
        self.inner.kv_event_source_mode = mode;
    }

    #[setter]
    fn set_kv_state_endpoint(&mut self, kv_state_endpoint: Option<String>) {
        self.inner.kv_state_endpoint = kv_state_endpoint.as_deref().map(EndpointId::from);
    }

    #[setter]
    fn set_exclude_tools_when_tool_choice_none(
        &mut self,
        exclude_tools_when_tool_choice_none: bool,
    ) {
        self.inner.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none;
    }

    #[setter]
    fn set_enable_eagle(&mut self, enable_eagle: bool) {
        self.inner.enable_eagle = enable_eagle;
    }

    #[setter]
    fn set_taints(&mut self, taints: HashSet<String>) {
        self.inner.taints = taints;
    }

    #[setter]
    fn set_stable_routing_id(&mut self, stable_routing_id: Option<String>) {
        self.inner.stable_routing_id = stable_routing_id;
    }

    #[getter]
    fn get_stable_routing_id(&self) -> Option<String> {
        self.inner.stable_routing_id.clone()
    }

    fn set_engine_specific(&mut self, key: &str, value: String) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(&value).map_err(to_pyerr)?;
        self.inner
            .set_engine_specific(key, value)
            .map_err(to_pyerr)?;
        Ok(())
    }

    #[getter]
    fn total_kv_blocks(&self) -> Option<u64> {
        self.inner.total_kv_blocks
    }

    #[getter]
    fn context_length(&self) -> Option<u32> {
        self.inner.context_length
    }

    #[getter]
    fn max_num_seqs(&self) -> Option<u64> {
        self.inner.max_num_seqs
    }

    #[getter]
    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.inner.max_num_batched_tokens
    }

    #[getter]
    fn tool_call_parser(&self) -> Option<String> {
        self.inner.tool_call_parser.clone()
    }

    #[getter]
    fn reasoning_parser(&self) -> Option<String> {
        self.inner.reasoning_parser.clone()
    }

    #[getter]
    fn tokenizer_backend(&self) -> Option<String> {
        self.inner
            .tokenizer_backend
            .map(|backend| backend.as_str().to_string())
    }

    #[getter]
    fn data_parallel_start_rank(&self) -> u32 {
        self.inner.data_parallel_start_rank
    }

    #[getter]
    fn data_parallel_size(&self) -> u32 {
        self.inner.data_parallel_size
    }

    #[getter]
    fn enable_local_indexer(&self) -> bool {
        self.inner.enable_local_indexer
    }

    #[getter]
    fn kv_event_publishing_enabled(&self) -> Option<bool> {
        self.inner.kv_event_publishing_enabled
    }

    #[getter]
    fn kv_event_source_mode(&self) -> Option<String> {
        self.inner.kv_event_source_mode.clone()
    }

    #[getter]
    fn kv_state_endpoint(&self) -> Option<String> {
        self.inner
            .kv_state_endpoint
            .as_ref()
            .map(ToString::to_string)
    }

    #[getter]
    fn exclude_tools_when_tool_choice_none(&self) -> bool {
        self.inner.exclude_tools_when_tool_choice_none
    }

    #[getter]
    fn runtime_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in self.inner.runtime_data.clone() {
            dict.set_item(key, value.to_string())?;
        }
        Ok(dict.into())
    }

    fn get_engine_specific(&self, key: &str) -> PyResult<Option<String>> {
        self.inner.get_engine_specific(key).map_err(to_pyerr)
    }

    fn set_structural_tag_mode(&mut self, mode: &str) -> PyResult<()> {
        self.inner.structural_tag_mode = match mode {
            "off" => RsStructuralTagMode::Off,
            "on" => RsStructuralTagMode::On,
            _ => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "Invalid structural_tag_mode: {mode}. Expected 'off' or 'on'."
                )));
            }
        };
        Ok(())
    }

    /// Set the structural tag scope ("auto" or "always").
    fn set_structural_tag_scope(&mut self, scope: &str) -> PyResult<()> {
        self.inner.structural_tag_scope = match scope {
            "auto" => RsStructuralTagScope::Auto,
            "always" => RsStructuralTagScope::Always,
            _ => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "Invalid structural_tag_scope: {scope}. Expected 'auto' or 'always'."
                )));
            }
        };
        Ok(())
    }

    /// Set the structural tag schema mode ("auto" or "strict").
    fn set_structural_tag_schema(&mut self, schema: &str) -> PyResult<()> {
        self.inner.structural_tag_schema = match schema {
            "auto" => RsStructuralTagSchemaMode::Auto,
            "strict" => RsStructuralTagSchemaMode::Strict,
            _ => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "Invalid structural_tag_schema: {schema}. Expected 'auto' or 'strict'."
                )));
            }
        };
        Ok(())
    }

    #[pyo3(signature = (bootstrap_host=None, bootstrap_port=None))]
    fn set_disaggregated_endpoint(
        &mut self,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) {
        self.inner.disaggregated_endpoint = Some(RsDisaggregatedEndpoint {
            bootstrap_host,
            bootstrap_port,
        });
    }

    #[getter]
    fn bootstrap_host(&self) -> Option<String> {
        self.inner
            .disaggregated_endpoint
            .as_ref()
            .and_then(|e| e.bootstrap_host.clone())
    }

    #[getter]
    fn bootstrap_port(&self) -> Option<u16> {
        self.inner
            .disaggregated_endpoint
            .as_ref()
            .and_then(|e| e.bootstrap_port)
    }

    #[getter]
    fn enable_eagle(&self) -> bool {
        self.inner.enable_eagle
    }

    #[getter]
    fn taints(&self) -> HashSet<String> {
        self.inner.taints.clone()
    }

    #[getter]
    fn topology_domains(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.topology_domains {
            dict.set_item(key, value)?;
        }
        Ok(dict.into())
    }

    #[setter]
    fn set_topology_domains(&mut self, topology_domains: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut new_topology_domains = HashMap::new();
        for (key, value) in topology_domains.iter() {
            let key_str: String = key.extract()?;
            let value_str: String = value.extract()?;
            new_topology_domains.insert(key_str, value_str);
        }
        self.inner.topology_domains = new_topology_domains;
        Ok(())
    }

    #[getter]
    fn kv_transfer_domain(&self) -> Option<String> {
        self.inner.kv_transfer_domain.clone()
    }

    #[setter]
    fn set_kv_transfer_domain(&mut self, kv_transfer_domain: Option<String>) {
        self.inner.kv_transfer_domain = kv_transfer_domain;
    }

    #[getter]
    fn kv_transfer_enforcement(&self) -> Option<String> {
        self.inner
            .kv_transfer_enforcement
            .map(|enforcement| match enforcement {
                RsKvTransferEnforcement::Required => "required".to_string(),
                RsKvTransferEnforcement::Preferred => "preferred".to_string(),
            })
    }

    #[setter]
    fn set_kv_transfer_enforcement(
        &mut self,
        kv_transfer_enforcement: Option<String>,
    ) -> PyResult<()> {
        let Some(kv_transfer_enforcement) = kv_transfer_enforcement else {
            self.inner.kv_transfer_enforcement = None;
            return Ok(());
        };

        self.inner.kv_transfer_enforcement = match kv_transfer_enforcement.as_str() {
            "" => None,
            "required" => Some(RsKvTransferEnforcement::Required),
            "preferred" => Some(RsKvTransferEnforcement::Preferred),
            value => {
                return Err(PyValueError::new_err(format!(
                    "kv_transfer_enforcement must be 'required' or 'preferred', got {value:?}"
                )));
            }
        };
        Ok(())
    }

    #[getter]
    fn kv_transfer_preferred_weight(&self) -> Option<f32> {
        self.inner.kv_transfer_preferred_weight
    }

    #[setter]
    fn set_kv_transfer_preferred_weight(
        &mut self,
        kv_transfer_preferred_weight: Option<f32>,
    ) -> PyResult<()> {
        self.inner.kv_transfer_preferred_weight = kv_transfer_preferred_weight;
        Ok(())
    }
}
