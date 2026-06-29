/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/selectionservice"
)

const (
	defaultTenantID = "default"
)

type WorkerRegistration struct {
	WorkerID        uint64
	WorkerEndpoint  string
	TenantID        string
	StableRoutingID string
	RequireKVEvents bool
	TopologyDomains map[string]string
}

func BuildWorkerRequest(reg WorkerRegistration, metadata MetadataSnapshot) (selectionservice.WorkerRequest, error) {
	normalizedWorkerEndpoint, err := normalizeSGLangEndpointURL(reg.WorkerEndpoint)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}
	workerEndpoint := normalizedWorkerEndpoint.String()

	modelName, err := modelNameFromMetadata(metadata)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}

	if err := validateGenerationModel(metadata.ModelInfo); err != nil {
		return selectionservice.WorkerRequest{}, err
	}
	if err := validateAggregatedMode(metadata.ServerInfo); err != nil {
		return selectionservice.WorkerRequest{}, err
	}

	blockSize, err := requiredPositiveUint32("page_size", metadata.ServerInfo.PageSize)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}
	dpSize, err := positiveUint32OrDefault("dp_size", metadata.ServerInfo.DPSize, 1)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}
	maxTotalNumTokens := positiveUint64Ptr(metadata.ServerInfo.MaxTotalNumTokens)
	maxNumBatchedTokens, err := optionalPositiveUint64Field("max_prefill_tokens", metadata.ServerInfo.MaxPrefillTokens)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}

	tenantID := reg.TenantID
	if tenantID == "" {
		tenantID = defaultTenantID
	}

	worker := selectionservice.WorkerRequest{
		WorkerID:         reg.WorkerID,
		ModelName:        modelName,
		TenantID:         tenantID,
		Endpoint:         workerEndpoint,
		BlockSize:        blockSize,
		DataParallelSize: dpSize,
		StableRoutingID:  reg.StableRoutingID,
		TopologyDomains:  copyStringMap(reg.TopologyDomains),
	}
	if maxNumBatchedTokens != nil {
		worker.MaxNumBatchedTokens = *maxNumBatchedTokens
	}
	if maxTotalNumTokens != nil {
		worker.TotalKVBlocks = *maxTotalNumTokens / uint64(blockSize)
	}

	if metadata.ServerInfo.SpeculativeAlgorithm != nil {
		isEagle := isEagleAlgorithm(*metadata.ServerInfo.SpeculativeAlgorithm)
		worker.IsEagle = &isEagle
	}

	kvEvents, err := kvEventsEndpoints(metadata.ServerInfo, workerEndpoint, blockSize, dpSize, reg.RequireKVEvents)
	if err != nil {
		return selectionservice.WorkerRequest{}, err
	}
	if len(kvEvents) > 0 {
		worker.KVEventsEndpoints = kvEvents
	}

	return worker, nil
}

func modelNameFromMetadata(metadata MetadataSnapshot) (string, error) {
	modelIDs := make([]string, 0, len(metadata.Models.Data))
	for _, model := range metadata.Models.Data {
		if id := strings.TrimSpace(model.ID); id != "" && !slices.Contains(modelIDs, id) {
			modelIDs = append(modelIDs, id)
		}
	}
	switch len(modelIDs) {
	case 0:
		return "", fmt.Errorf("SGLang /v1/models did not report a model id")
	case 1:
		return modelIDs[0], nil
	}
	for _, hint := range []string{metadata.ServerInfo.ServedModelName, metadata.ModelInfo.ModelPath} {
		if modelID, ok := modelIDMatchingHint(modelIDs, hint); ok {
			return modelID, nil
		}
	}
	return "", fmt.Errorf("SGLang /v1/models reported multiple model ids but none matched served_model_name or model_path")
}

func modelIDMatchingHint(modelIDs []string, hint string) (string, bool) {
	hint = strings.TrimSpace(hint)
	if hint == "" {
		return "", false
	}
	for _, modelID := range modelIDs {
		if hint == modelID || strings.HasSuffix(hint, "/"+modelID) {
			return modelID, true
		}
	}
	return "", false
}

func validateGenerationModel(modelInfo ModelInfo) error {
	if modelInfo.IsGeneration != nil && !*modelInfo.IsGeneration {
		return fmt.Errorf("external SGLang selection adapter requires generation model")
	}
	return nil
}

func validateAggregatedMode(serverInfo ServerInfo) error {
	if serverInfo.DisaggregationMode != nil && !isAggregatedDisaggregationMode(*serverInfo.DisaggregationMode) {
		return fmt.Errorf("external SGLang selection adapter currently supports aggregated workers only")
	}
	return nil
}

func isAggregatedDisaggregationMode(mode string) bool {
	mode = strings.ToLower(mode)
	return mode == "" || mode == "null"
}

func isEagleAlgorithm(algorithm string) bool {
	switch strings.ToUpper(strings.TrimSpace(algorithm)) {
	case "EAGLE", "EAGLE3", "FROZEN_KV_MTP", "NEXTN":
		return true
	default:
		return false
	}
}

func positiveUint32OrDefault(key string, value *uint32, defaultValue uint32) (uint32, error) {
	if value == nil {
		return defaultValue, nil
	}
	return positiveUint32Field(key, *value)
}

func optionalPositiveUint64Field(key string, value *uint64) (*uint64, error) {
	if value == nil {
		return nil, nil
	}
	if *value == 0 {
		return nil, fmt.Errorf("SGLang metadata field %q must be positive", key)
	}
	return value, nil
}

func requiredPositiveUint32(key string, value *uint32) (uint32, error) {
	if value == nil {
		return 0, fmt.Errorf("SGLang metadata field %q must be positive", key)
	}
	return positiveUint32Field(key, *value)
}

func positiveUint32Field(key string, value uint32) (uint32, error) {
	if value == 0 {
		return 0, fmt.Errorf("SGLang metadata field %q must be positive", key)
	}
	return value, nil
}

func copyStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	result := make(map[string]string, len(values))
	for key, value := range values {
		if key != "" && value != "" {
			result[key] = value
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func positiveUint64Ptr(value *uint64) *uint64 {
	if value == nil || *value == 0 {
		return nil
	}
	return value
}
