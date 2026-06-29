/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	testKVEventsEndpoint = "tcp://worker-0.svc:5557"
	testModelName        = "qwen"
	testQwen3ModelName   = "Qwen/Qwen3-0.6B"
)

func testSnapshot(t *testing.T, overrides map[string]any) MetadataSnapshot {
	t.Helper()
	serverInfo := map[string]any{
		"page_size":             16,
		"dp_size":               2,
		"max_total_num_tokens":  4096,
		"max_prefill_tokens":    1024,
		"served_model_name":     "server-model",
		"disaggregation_mode":   "null",
		"speculative_algorithm": "EAGLE",
		"kv_events": map[string]any{
			"publisher":          "zmq",
			"endpoint_host":      "*",
			"endpoint_port_base": 5557,
			"topic":              "",
			"block_size":         16,
			"dp_size":            2,
		},
	}
	for key, value := range overrides {
		if value == nil {
			delete(serverInfo, key)
			continue
		}
		serverInfo[key] = value
	}
	return MetadataSnapshot{
		Models:     ModelsResponse{Data: []Model{{ID: testModelName}}},
		ModelInfo:  ModelInfo{ModelPath: "/models/qwen", IsGeneration: boolRef(true)},
		ServerInfo: jsonRoundTripServerInfo(t, serverInfo),
	}
}

func jsonRoundTripServerInfo(t *testing.T, values map[string]any) ServerInfo {
	t.Helper()
	payload, err := json.Marshal(values)
	require.NoError(t, err)
	var result ServerInfo
	require.NoError(t, json.Unmarshal(payload, &result))
	return result
}

func boolRef(value bool) *bool {
	return &value
}

func testRegistration() WorkerRegistration {
	return WorkerRegistration{
		WorkerID:       7,
		WorkerEndpoint: "http://worker-0.svc:30000",
	}
}

func requireBuildWorkerRequestErrorContains(t *testing.T, reg WorkerRegistration, metadata MetadataSnapshot, want string) {
	t.Helper()
	_, err := BuildWorkerRequest(reg, metadata)
	require.ErrorContains(t, err, want)
}

func fixtureBytes(t *testing.T, name string) []byte {
	t.Helper()
	payload, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return payload
}

func fixtureJSON[T any](t *testing.T, name string) T {
	t.Helper()
	payload := fixtureBytes(t, name)
	var result T
	if err := json.Unmarshal(payload, &result); err != nil {
		t.Fatalf("decode fixture %s: %v", name, err)
	}
	return result
}

func realFixtureSnapshot(t *testing.T) MetadataSnapshot {
	t.Helper()
	return fixtureSnapshot(t, "server_info_qwen3_huggingface_legacy_kv_events_config.json")
}

func realStructuredKVEventsFixtureSnapshot(t *testing.T) MetadataSnapshot {
	t.Helper()
	return fixtureSnapshot(t, "server_info_qwen3_huggingface_structured_kv_events.json")
}

func fixtureSnapshot(t *testing.T, serverInfoFixture string) MetadataSnapshot {
	t.Helper()
	return MetadataSnapshot{
		Models:     fixtureJSON[ModelsResponse](t, "models.json"),
		ModelInfo:  fixtureJSON[ModelInfo](t, "model_info_generation.json"),
		ServerInfo: fixtureJSON[ServerInfo](t, serverInfoFixture),
	}
}

func TestBuildWorkerRequestReadsRealLegacyKVEventsConfigMetadata(t *testing.T) {
	reg := testRegistration()
	reg.RequireKVEvents = true
	reg.StableRoutingID = "pod-uid"
	reg.TopologyDomains = map[string]string{"zone": "zone-a"}

	worker, err := BuildWorkerRequest(reg, realFixtureSnapshot(t))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}

	if worker.WorkerID != reg.WorkerID || worker.ModelName != testQwen3ModelName || worker.TenantID != defaultTenantID {
		t.Fatalf("unexpected worker identity: %#v", worker)
	}
	if worker.Endpoint != "http://worker-0.svc:30000" {
		t.Fatalf("endpoint = %q", worker.Endpoint)
	}
	if worker.BlockSize != 16 || worker.DataParallelStartRank != 0 || worker.DataParallelSize != 1 {
		t.Fatalf("unexpected topology: block=%d dpStart=%d dpSize=%d",
			worker.BlockSize, worker.DataParallelStartRank, worker.DataParallelSize)
	}
	if worker.MaxNumBatchedTokens != 16384 || worker.TotalKVBlocks != 10570 {
		t.Fatalf("unexpected capacity: maxBatch=%d totalBlocks=%d",
			worker.MaxNumBatchedTokens, worker.TotalKVBlocks)
	}
	if worker.KVEventsEndpoints[0] != testKVEventsEndpoint {
		t.Fatalf("kv events endpoints = %#v", worker.KVEventsEndpoints)
	}
	if worker.StableRoutingID != "pod-uid" || worker.TopologyDomains["zone"] != "zone-a" {
		t.Fatalf("routing metadata = stable:%q topology:%#v", worker.StableRoutingID, worker.TopologyDomains)
	}
}

func TestBuildWorkerRequestReadsRealStructuredKVEventsMetadata(t *testing.T) {
	reg := testRegistration()
	reg.RequireKVEvents = true

	metadata := realStructuredKVEventsFixtureSnapshot(t)
	if len(metadata.ServerInfo.KVEventsConfig) == 0 {
		t.Fatalf("fixture missing kv_events_config metadata")
	}
	metadata.ServerInfo.KVEvents = nil
	worker, err := BuildWorkerRequest(reg, metadata)
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}

	if worker.ModelName != testQwen3ModelName || worker.BlockSize != 16 || worker.DataParallelSize != 1 {
		t.Fatalf("unexpected worker metadata: %#v", worker)
	}
	if worker.KVEventsEndpoints[0] != testKVEventsEndpoint {
		t.Fatalf("kv events endpoints = %#v", worker.KVEventsEndpoints)
	}
}

func TestBuildWorkerRequestValidatesKVEventsConfigDPSize(t *testing.T) {
	reg := testRegistration()
	reg.RequireKVEvents = true
	metadata := testSnapshot(t, map[string]any{
		"kv_events":        nil,
		"dp_size":          2,
		"kv_events_config": `{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}`,
	})
	requireBuildWorkerRequestErrorContains(t, reg, metadata, "data_parallel_size=1")
}

func TestBuildWorkerRequestReadsStructuredKVEventsMetadata(t *testing.T) {
	reg := testRegistration()
	reg.RequireKVEvents = true

	worker, err := BuildWorkerRequest(reg, testSnapshot(t, nil))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.IsEagle == nil || !*worker.IsEagle {
		t.Fatalf("is_eagle = %v, want true for EAGLE3", worker.IsEagle)
	}
	if worker.KVEventsEndpoints[0] != testKVEventsEndpoint || worker.KVEventsEndpoints[1] != "tcp://worker-0.svc:5558" {
		t.Fatalf("kv events endpoints = %#v", worker.KVEventsEndpoints)
	}
}

func TestBuildWorkerRequestFailsClosedForFixtureMetadata(t *testing.T) {
	for _, tt := range []struct {
		name        string
		mutate      func(*MetadataSnapshot)
		wantMessage string
	}{
		{
			name: "missing block size",
			mutate: func(metadata *MetadataSnapshot) {
				metadata.ServerInfo.PageSize = nil
			},
			wantMessage: "page_size",
		},
		{
			name: "disaggregated worker",
			mutate: func(metadata *MetadataSnapshot) {
				mode := "prefill"
				metadata.ServerInfo.DisaggregationMode = &mode
			},
			wantMessage: "aggregated",
		},
		{
			name: "empty model list",
			mutate: func(metadata *MetadataSnapshot) {
				metadata.Models = ModelsResponse{}
			},
			wantMessage: "/v1/models",
		},
		{
			name: "empty model id",
			mutate: func(metadata *MetadataSnapshot) {
				metadata.Models = ModelsResponse{Data: []Model{{}}}
			},
			wantMessage: "/v1/models",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			metadata := realFixtureSnapshot(t)
			tt.mutate(&metadata)
			requireBuildWorkerRequestErrorContains(t, testRegistration(), metadata, tt.wantMessage)
		})
	}
}

func TestBuildWorkerRequestUsesGlobalDPCapacityAndKVEvents(t *testing.T) {
	worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, nil))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}

	if worker.WorkerID != 7 || worker.ModelName != testModelName || worker.TenantID != defaultTenantID {
		t.Fatalf("unexpected worker identity: %#v", worker)
	}
	if got := worker.BlockSize; got != 16 {
		t.Fatalf("block size = %d, want 16", got)
	}
	if got := worker.DataParallelStartRank; got != 0 {
		t.Fatalf("dp start = %d, want 0", got)
	}
	if got := worker.DataParallelSize; got != 2 {
		t.Fatalf("dp size = %d, want 2", got)
	}
	if got := worker.MaxNumBatchedTokens; got != 1024 {
		t.Fatalf("max batched tokens = %d, want 1024", got)
	}
	if got := worker.TotalKVBlocks; got != 256 {
		t.Fatalf("total kv blocks = %d, want 256", got)
	}
	if worker.IsEagle == nil || !*worker.IsEagle {
		t.Fatalf("is_eagle = %v, want true", worker.IsEagle)
	}
	if got := worker.KVEventsEndpoints[0]; got != testKVEventsEndpoint {
		t.Fatalf("kv endpoint rank 0 = %q", got)
	}
	if got := worker.KVEventsEndpoints[1]; got != "tcp://worker-0.svc:5558" {
		t.Fatalf("kv endpoint rank 1 = %q", got)
	}
}

func TestBuildWorkerRequestSelectsModelFromMultipleModels(t *testing.T) {
	t.Run("served model name", func(t *testing.T) {
		metadata := testSnapshot(t, nil)
		metadata.Models = ModelsResponse{Data: []Model{{ID: "wrong-first-model"}, {ID: "server-model"}}}

		worker, err := BuildWorkerRequest(testRegistration(), metadata)
		if err != nil {
			t.Fatalf("BuildWorkerRequest() error = %v", err)
		}
		if worker.ModelName != "server-model" {
			t.Fatalf("model name = %q, want server-model", worker.ModelName)
		}
	})

	t.Run("model path", func(t *testing.T) {
		metadata := testSnapshot(t, map[string]any{"served_model_name": nil})
		metadata.Models = ModelsResponse{Data: []Model{{ID: "wrong-first-model"}, {ID: testModelName}}}
		metadata.ModelInfo.ModelPath = "/models/qwen"

		worker, err := BuildWorkerRequest(testRegistration(), metadata)
		if err != nil {
			t.Fatalf("BuildWorkerRequest() error = %v", err)
		}
		if worker.ModelName != testModelName {
			t.Fatalf("model name = %q, want qwen", worker.ModelName)
		}
	})

	t.Run("no matching hint", func(t *testing.T) {
		metadata := testSnapshot(t, map[string]any{"served_model_name": nil})
		metadata.Models = ModelsResponse{Data: []Model{{ID: "wrong-first-model"}, {ID: testModelName}}}
		metadata.ModelInfo.ModelPath = "/models/other"

		requireBuildWorkerRequestErrorContains(t, testRegistration(), metadata, "multiple model ids")
	})
}

func TestBuildWorkerRequestDoesNotUseTotalCapacityAsBatchCapacity(t *testing.T) {
	worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, map[string]any{
		"max_prefill_tokens": nil,
	}))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.MaxNumBatchedTokens != 0 {
		t.Fatalf("max batched tokens = %d, want omitted", worker.MaxNumBatchedTokens)
	}
	if worker.TotalKVBlocks != 256 {
		t.Fatalf("total kv blocks = %d, want 256", worker.TotalKVBlocks)
	}
}

func TestBuildWorkerRequestUsesFloorTotalKVBlocks(t *testing.T) {
	worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, map[string]any{
		"page_size":            16,
		"max_total_num_tokens": 4097,
	}))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.TotalKVBlocks != 256 {
		t.Fatalf("total kv blocks = %d, want floor(4097/16)=256", worker.TotalKVBlocks)
	}
}

func TestBuildWorkerRequestClassifiesSpeculativeAlgorithm(t *testing.T) {
	for _, tt := range []struct {
		name        string
		algorithm   string
		wantIsEagle bool
	}{
		{name: "eagle3", algorithm: "EAGLE3", wantIsEagle: true},
		{name: "frozen kv mtp", algorithm: "FROZEN_KV_MTP", wantIsEagle: true},
		{name: "nextn alias", algorithm: "NEXTN", wantIsEagle: true},
		{name: "ngram", algorithm: "NGRAM", wantIsEagle: false},
		{name: "unknown", algorithm: "FUTURE_SPECULATIVE", wantIsEagle: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, map[string]any{
				"speculative_algorithm": tt.algorithm,
			}))
			if err != nil {
				t.Fatalf("BuildWorkerRequest() error = %v", err)
			}
			if worker.IsEagle == nil || *worker.IsEagle != tt.wantIsEagle {
				t.Fatalf("is_eagle = %v, want %v", worker.IsEagle, tt.wantIsEagle)
			}
		})
	}
}

func TestBuildWorkerRequestSupportsRegistrationMetadata(t *testing.T) {
	reg := testRegistration()
	reg.TenantID = "tenant-a"
	reg.StableRoutingID = "pod-123"
	reg.TopologyDomains = map[string]string{"zone": "us-west"}

	worker, err := BuildWorkerRequest(reg, testSnapshot(t, nil))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.ModelName != testModelName || worker.TenantID != "tenant-a" || worker.StableRoutingID != "pod-123" {
		t.Fatalf("registration metadata not applied: %#v", worker)
	}
	if worker.TopologyDomains["zone"] != "us-west" {
		t.Fatalf("topology domains = %#v", worker.TopologyDomains)
	}
}

func TestBuildWorkerRequestIgnoresInvalidOptionalTotalCapacityHint(t *testing.T) {
	worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, map[string]any{
		"max_total_num_tokens": 0,
	}))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.TotalKVBlocks != 0 {
		t.Fatalf("total kv blocks = %v, want omitted", worker.TotalKVBlocks)
	}
}

func TestBuildWorkerRequestFailsClosedWithZeroMaxPrefillTokens(t *testing.T) {
	requireBuildWorkerRequestErrorContains(t, testRegistration(), testSnapshot(t, map[string]any{
		"max_prefill_tokens": 0,
	}), "max_prefill_tokens")
}

func TestBuildWorkerRequestFailsClosedWithoutValidBlockSize(t *testing.T) {
	for _, pageSize := range []any{0, nil} {
		requireBuildWorkerRequestErrorContains(t, testRegistration(), testSnapshot(t, map[string]any{"page_size": pageSize}), "page_size")
	}
}

func TestBuildWorkerRequestRejectsDisaggregatedAndEmbeddingWorkers(t *testing.T) {
	requireBuildWorkerRequestErrorContains(t, testRegistration(), testSnapshot(t, map[string]any{"disaggregation_mode": "prefill"}), "aggregated")

	metadata := testSnapshot(t, nil)
	metadata.ModelInfo = ModelInfo{ModelPath: "/models/embed", IsGeneration: boolRef(false)}
	requireBuildWorkerRequestErrorContains(t, testRegistration(), metadata, "generation")
}

func TestBuildWorkerRequestHandlesOptionalAndRequiredKVEvents(t *testing.T) {
	worker, err := BuildWorkerRequest(testRegistration(), testSnapshot(t, map[string]any{
		"kv_events":        nil,
		"kv_events_config": nil,
	}))
	if err != nil {
		t.Fatalf("optional missing kv_events error = %v", err)
	}
	if len(worker.KVEventsEndpoints) != 0 {
		t.Fatalf("kv events endpoints = %#v, want omitted", worker.KVEventsEndpoints)
	}

	requireBuildWorkerRequestErrorContains(t, testRegistration(), testSnapshot(t, map[string]any{
		"kv_events": map[string]any{"publisher": "null"},
	}), "publisher")

	requireBuildWorkerRequestErrorContains(t, testRegistration(), testSnapshot(t, map[string]any{
		"kv_events":        nil,
		"kv_events_config": `{"publisher":"null","topic":"kv-events","endpoint":"tcp://*:5557"}`,
	}), "publisher")

	reg := testRegistration()
	reg.RequireKVEvents = true
	requireBuildWorkerRequestErrorContains(t, reg, testSnapshot(t, map[string]any{
		"kv_events": map[string]any{
			"publisher":          "zmq",
			"endpoint_host":      "*",
			"endpoint_port_base": 5557,
			"block_size":         32,
			"dp_size":            2,
		},
	}), "block_size")
}

func TestBuildWorkerRequestTreatsStructuredKVEventsAsAuthoritative(t *testing.T) {
	reg := testRegistration()
	reg.RequireKVEvents = true
	worker, err := BuildWorkerRequest(reg, testSnapshot(t, map[string]any{
		"dp_size": 1,
		"kv_events": map[string]any{
			"publisher":          "zmq",
			"endpoint_host":      "*",
			"endpoint_port_base": 5557,
			"block_size":         32,
			"dp_size":            2,
		},
		"kv_events_config": `{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}`,
	}))
	if err != nil {
		t.Fatalf("BuildWorkerRequest() error = %v", err)
	}
	if worker.KVEventsEndpoints[0] != testKVEventsEndpoint {
		t.Fatalf("kv events endpoints = %#v", worker.KVEventsEndpoints)
	}
}
