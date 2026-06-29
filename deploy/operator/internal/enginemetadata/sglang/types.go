/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import "encoding/json"

// These structs decode SGLang HTTP metadata responses, so pointer fields are
// intentional where the upstream response can distinguish missing/null from a
// real false or zero value. Source contracts:
// /model_info: https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/entrypoints/http_server.py#L648-L665
// /server_info: https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/entrypoints/http_server.py#L688-L708
// ServerArgs fields: https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/server_args.py
// kv_events descriptor: https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/server_args.py#L7092-L7175

type MetadataSnapshot struct {
	Models     ModelsResponse
	ModelInfo  ModelInfo
	ServerInfo ServerInfo
}

type ModelsResponse struct {
	Data []Model `json:"data"`
}

type Model struct {
	ID string `json:"id"`
}

type ModelInfo struct {
	ModelPath    string `json:"model_path"`
	IsGeneration *bool  `json:"is_generation"`
}

type ServerInfo struct {
	ServedModelName      string              `json:"served_model_name"`
	PageSize             *uint32             `json:"page_size"`
	DPSize               *uint32             `json:"dp_size"`
	MaxTotalNumTokens    *uint64             `json:"max_total_num_tokens"`
	MaxPrefillTokens     *uint64             `json:"max_prefill_tokens"`
	DisaggregationMode   *string             `json:"disaggregation_mode"`
	SpeculativeAlgorithm *string             `json:"speculative_algorithm"`
	KVEvents             *KVEventsDescriptor `json:"kv_events"`
	KVEventsConfig       json.RawMessage     `json:"kv_events_config"`
}

type KVEventsDescriptor struct {
	Publisher        string  `json:"publisher"`
	EndpointHost     string  `json:"endpoint_host"`
	EndpointPortBase *uint32 `json:"endpoint_port_base"`
	BlockSize        *uint32 `json:"block_size"`
	DPSize           *uint32 `json:"dp_size"`
}

type KVEventsConfig struct {
	Publisher string `json:"publisher"`
	Endpoint  string `json:"endpoint"`
}
