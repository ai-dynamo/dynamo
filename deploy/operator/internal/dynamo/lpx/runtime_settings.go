/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"maps"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

// maxSpecDecodeNumDrafts bounds the supported speculative-decoding draft fanout.
const maxSpecDecodeNumDrafts = 8

// Preserve the setup-ops discriminator consumed by the LPU runtime image.
const agentSetupOpsFormat = "agent_v2"

// resolveBuildSettings requires a nonnil build; it normalizes and consumes modelSettings, which callers must not reuse.
func resolveBuildSettings(build *Build, modelSettings map[string]any) error {
	// Preserve normalized defaults before reusing the compiler-owned field for model overrides.
	defaults := build.runtimeSettings
	build.runtimeSettings = modelSettings
	splitCount, overridesSplitCount := modelSettings["num_batch_split_divisions"]

	// Overlay model settings before consuming the selected prop-sync and embedding partitions.
	merged, err := mergeRuntimeSettingOverride(defaults, build.runtimeSettings, "settings")
	if err != nil {
		return err
	}
	settings := merged.(map[string]any)

	// Restore the original split count for exact validation after merging settings.
	if overridesSplitCount {
		settings["num_batch_split_divisions"] = splitCount
	}

	// Preserve the selected prop-sync and embedding transformations on the merged settings.
	cpuEmbeddings, err := boolSetting(settings, "cpu_embeddings")
	if err != nil {
		return fmt.Errorf("settings.%w", err)
	}
	if err := build.consumeRuntimeSelectedPropSyncChain(); err != nil {
		return err
	}
	if cpuEmbeddings {
		build.omitStandaloneEmbeddingPartition()
	}
	build.runtimeSettings = settings

	return normalizeLegacyNovaSettings(build.runtimeSettings)
}

func mergeRuntimeSettingOverride(base, override any, path string) (any, error) {
	if override == nil {
		return nil, fmt.Errorf("%s must not be null", path)
	}

	switch typed := override.(type) {
	case map[string]any:
		baseMap, _ := base.(map[string]any)
		merged := make(map[string]any, len(baseMap)+len(typed))
		for key, value := range baseMap {
			if _, overridden := typed[key]; !overridden {
				merged[key] = value
			}
		}
		for _, key := range slices.Sorted(maps.Keys(typed)) {
			overrideValue := typed[key]
			value, err := mergeRuntimeSettingOverride(baseMap[key], overrideValue, path+"."+key)
			if err != nil {
				return nil, err
			}
			if _, numeric := overrideValue.(json.Number); numeric {
				typed[key] = value
			}
			merged[key] = value
		}
		return merged, nil
	case []any:
		if typed == nil {
			return []any{}, nil
		}
		for i, element := range typed {
			value, err := mergeRuntimeSettingOverride(nil, element, fmt.Sprintf("%s[%d]", path, i))
			if err != nil {
				return nil, err
			}
			typed[i] = value
		}
		return typed, nil
	case json.Number:
		number := typed.String()
		if !strings.ContainsAny(number, ".eE") {
			if integer, err := strconv.ParseInt(number, 10, 64); err == nil {
				return integer, nil
			}
		}
		if float, err := strconv.ParseFloat(number, 64); err == nil {
			return float, nil
		}
		return typed, nil
	default:
		return override, nil
	}
}

// validateRuntimeTokenizerSettings validates and normalizes tokenizer settings shared by V2 and V3.
func validateRuntimeTokenizerSettings(iop map[string]any) error {
	tokenizerPath, ok := iop["tokenizer_path"]
	if !ok {
		return fmt.Errorf("iop.tokenizer_path is required; set spec.components[].lpx.settings.tokenizer_path when the capnp manifest does not provide model.tokenizer.path")
	}
	path, ok := tokenizerPath.(string)
	path = strings.TrimSpace(path)
	if !ok || path == "" {
		return fmt.Errorf("iop.tokenizer_path must be a non-empty string")
	}
	iop["tokenizer_path"] = path

	stopTokens, ok := iop["stop_tokens"]
	if !ok {
		return fmt.Errorf("iop.stop_tokens is required; set spec.components[].lpx.settings.stop_tokens when the capnp manifest does not provide model.tokenizer.stopTokens")
	}
	var normalized []uint32
	switch tokens := stopTokens.(type) {
	case []uint32:
		normalized = append([]uint32(nil), tokens...)
	case []any:
		normalized = make([]uint32, len(tokens))
		for i, value := range tokens {
			integer, ok := value.(int64)
			if !ok || integer < 0 || uint64(integer) > uint64(^uint32(0)) {
				return fmt.Errorf("iop.stop_tokens[%d] must be an integer between 0 and %d", i, uint64(^uint32(0)))
			}
			normalized[i] = uint32(integer)
		}
	default:
		return fmt.Errorf("iop.stop_tokens must be an array")
	}
	iop["stop_tokens"] = normalized
	return nil
}

// normalizeLegacyNovaSettings validates legacy Agent V2 controls and removes
// their accepted defaults. It consumes caller-owned settings without touching build geometry.
func normalizeLegacyNovaSettings(iop map[string]any) error {
	// Reject unsupported folding before omitting its accepted false value.
	batchFolding, err := boolSetting(iop, "batch_folding")
	if err != nil {
		return fmt.Errorf("iop.batch_folding must be a boolean")
	}
	if batchFolding {
		return fmt.Errorf("iop.batch_folding=true is not supported with setup_ops_format %q", agentSetupOpsFormat)
	}

	// Accept only zero or one, including JSON numbers from unresolved V2 LPX settings.
	if value, ok := iop["num_batch_split_divisions"]; ok {
		var numBatchSplitDivisions *big.Rat
		switch typed := value.(type) {
		case int64:
			numBatchSplitDivisions = new(big.Rat).SetInt64(typed)
		case float64:
			numBatchSplitDivisions = new(big.Rat).SetFloat64(typed)
		case json.Number:
			numBatchSplitDivisions, _ = new(big.Rat).SetString(typed.String())
		}
		if numBatchSplitDivisions == nil || !numBatchSplitDivisions.IsInt() {
			return fmt.Errorf("iop.num_batch_split_divisions must be an integer")
		}
		if numBatchSplitDivisions.Sign() < 0 || numBatchSplitDivisions.Cmp(big.NewRat(1, 1)) > 0 {
			return fmt.Errorf(
				"iop.num_batch_split_divisions=%v is not supported with setup_ops_format %q",
				value,
				agentSetupOpsFormat,
			)
		}
	}

	// Nova no longer accepts either key, even when the value matches its default.
	delete(iop, "batch_folding")
	delete(iop, "num_batch_split_divisions")
	return nil
}

func boolSetting(values map[string]any, key string) (bool, error) {
	value, ok := values[key]
	if !ok {
		return false, nil
	}
	boolValue, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("%s must be a boolean", key)
	}
	return boolValue, nil
}
