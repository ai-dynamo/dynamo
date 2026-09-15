/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

const (
	sglangEngineGroupBackend          = "sglang"
	sglangEngineGeometryDigestVersion = "sglang-engine-group-geometry/v1"
	sglangDynamoModule                = "dynamo.sglang"
	sglangUpstreamModule              = "sglang.launch_server"
	sglangServeSubcommand             = "serve"

	sglangTensorParallelSizeOption      = "--tp-size"
	sglangPipelineParallelSizeOption    = "--pp-size"
	sglangDataParallelSizeOption        = "--dp-size"
	sglangAttentionContextSizeOption    = "--attn-cp-size"
	sglangMoEDataParallelSizeOption     = "--moe-dp-size"
	sglangExpertParallelSizeOption      = "--ep-size"
	sglangNodesOption                   = "--nnodes"
	sglangElasticBackendOption          = "--elastic-ep-backend"
	sglangMoEA2ABackendOption           = "--moe-a2a-backend"
	sglangElasticInitialSizeOption      = "--elastic-ep-initial-size"
	sglangMaximumEPSizeOption           = "--max-ep-size"
	sglangLoadBalanceMethodOption       = "--load-balance-method"
	sglangTokenizerWorkerCountOption    = "--tokenizer-worker-num"
	sglangJoinModeOption                = "--elastic-ep-join-mode"
	sglangJoinRankOffsetOption          = "--elastic-ep-join-rank-offset"
	sglangDecodeCUDAGraphBackendOption  = "--cuda-graph-backend-decode"
	sglangPrefillCUDAGraphBackendOption = "--cuda-graph-backend-prefill"
	sglangEnableDPAttentionOption       = "--enable-dp-attention"
	sglangEnableDPLMHeadOption          = "--enable-dp-lm-head"
	sglangDisableCUDAGraphOption        = "--disable-cuda-graph"
	sglangUseRayOption                  = "--use-ray"
	sglangEnableElasticBackupOption     = "--enable-elastic-expert-backup"
	sglangDeprecatedElasticRejoinOption = "--elastic-ep-rejoin"
	sglangConfigOption                  = "--config"
	sglangCUDAGraphConfigOption         = "--cuda-graph-config"
)

// ErrUnsupportedSGLangProfileSource classifies SGLang declarations whose geometry cannot be proven statically.
var ErrUnsupportedSGLangProfileSource = errors.New("unsupported SGLang Engine Group profile source")

// UnsupportedSGLangProfileSourceReason is a stable machine-readable explanation for an opaque SGLang source.
type UnsupportedSGLangProfileSourceReason string

const (
	// UnsupportedSGLangProfileSourceReasonCommandForm means the image entrypoint or executable is not explicit.
	UnsupportedSGLangProfileSourceReasonCommandForm UnsupportedSGLangProfileSourceReason = "command-form"
	// UnsupportedSGLangProfileSourceReasonShell means a shell or environment wrapper owns the effective argv.
	UnsupportedSGLangProfileSourceReasonShell UnsupportedSGLangProfileSourceReason = "shell-or-wrapper"
	// UnsupportedSGLangProfileSourceReasonEnvironment means runtime expansion can change the effective declaration.
	UnsupportedSGLangProfileSourceReasonEnvironment UnsupportedSGLangProfileSourceReason = "environment-expansion"
	// UnsupportedSGLangProfileSourceReasonExternalConfig means an external config can override inspected options.
	UnsupportedSGLangProfileSourceReasonExternalConfig UnsupportedSGLangProfileSourceReason = "external-config"
	// UnsupportedSGLangProfileSourceReasonAbbreviatedOption means an option cannot be bound to one exact field.
	UnsupportedSGLangProfileSourceReasonAbbreviatedOption UnsupportedSGLangProfileSourceReason = "abbreviated-option"
	// UnsupportedSGLangProfileSourceReasonMissingGeometry means required geometry relies on engine defaults.
	UnsupportedSGLangProfileSourceReasonMissingGeometry UnsupportedSGLangProfileSourceReason = "missing-geometry"
	// UnsupportedSGLangProfileSourceReasonScaleUpContract means the declaration is not the merged growth-only profile.
	UnsupportedSGLangProfileSourceReasonScaleUpContract UnsupportedSGLangProfileSourceReason = "scale-up-contract"
)

// UnsupportedSGLangProfileSourceError describes why an SGLang declaration cannot produce typed geometry.
type UnsupportedSGLangProfileSourceError struct {
	Reason UnsupportedSGLangProfileSourceReason
	Detail string
}

// Error returns the stable unsupported-source class and reason with human-readable detail.
func (e *UnsupportedSGLangProfileSourceError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%s: %s", ErrUnsupportedSGLangProfileSource, e.Reason)
	}

	return fmt.Sprintf("%s: %s: %s", ErrUnsupportedSGLangProfileSource, e.Reason, e.Detail)
}

// Unwrap classifies an opaque provider source as both source-specific and generically unsupported.
func (e *UnsupportedSGLangProfileSourceError) Unwrap() []error {
	return []error{ErrUnsupportedSGLangProfileSource, enginegroup.ErrUnsupportedProfile}
}

// SGLangProfileGeometrySource contains the declarative inputs needed to resolve the merged SGLang scale-up profile.
// Command and Args must be provider-resolved Kubernetes exec-form argv; only the managed initial target may be
// omitted. WorkloadRevisionDigest must exclude creation-time and live replica targets. Resolution does not mutate
// the slices.
type SGLangProfileGeometrySource struct {
	Command                    []string
	Args                       []string
	InitialReplicas            int32
	MainContainerGPUs          int64
	DedicatedMainGPUAllocation bool
	WorkloadRevisionDigest     string
}

type parsedSGLangProfileGeometry struct {
	tensorParallelSize   int64
	pipelineParallelSize int64
	dataParallelSize     int64
	attentionContextSize int64
	moeDataParallelSize  int64
	expertParallelSize   int64
	nodes                int64
	elasticInitialSize   int64
	elasticMaximumSize   int64
	elasticBackend       string
	moeA2ABackend        string
}

type sglangEngineGeometryProjection struct {
	Version              string `json:"version"`
	ReplicaWidth         int64  `json:"replicaWidth"`
	PipelineParallelSize int64  `json:"pipelineParallelSize"`
	AttentionContextSize int64  `json:"attentionContextParallelSize"`
	MoEDataParallelSize  int64  `json:"moeDataParallelSize"`
	StorageEPSize        int64  `json:"storageEPSize"`
	ElasticMaximumSize   int64  `json:"elasticMaximumSize"`
	ElasticBackend       string `json:"elasticBackend"`
	MoEA2ABackend        string `json:"moeA2ABackend"`
}

type sglangProfileOption struct {
	canonical string
	aliases   []string
	flag      bool
}

var sglangProfileOptions = []sglangProfileOption{
	{canonical: sglangTensorParallelSizeOption, aliases: []string{sglangTensorParallelSizeOption, "--tensor-parallel-size", "--tp"}},
	{canonical: sglangPipelineParallelSizeOption, aliases: []string{sglangPipelineParallelSizeOption, "--pipeline-parallel-size", "--pp"}},
	{canonical: sglangDataParallelSizeOption, aliases: []string{sglangDataParallelSizeOption, "--data-parallel-size", "--dp"}},
	{canonical: sglangAttentionContextSizeOption, aliases: []string{sglangAttentionContextSizeOption, "--attention-context-parallel-size"}},
	{canonical: sglangMoEDataParallelSizeOption, aliases: []string{sglangMoEDataParallelSizeOption, "--moe-data-parallel-size"}},
	{canonical: sglangExpertParallelSizeOption, aliases: []string{sglangExpertParallelSizeOption, "--expert-parallel-size", "--ep"}},
	{canonical: sglangNodesOption, aliases: []string{sglangNodesOption}},
	{canonical: sglangElasticBackendOption, aliases: []string{sglangElasticBackendOption}},
	{canonical: sglangMoEA2ABackendOption, aliases: []string{sglangMoEA2ABackendOption}},
	{canonical: sglangElasticInitialSizeOption, aliases: []string{sglangElasticInitialSizeOption}},
	{canonical: sglangMaximumEPSizeOption, aliases: []string{sglangMaximumEPSizeOption}},
	{canonical: sglangLoadBalanceMethodOption, aliases: []string{sglangLoadBalanceMethodOption}},
	{canonical: sglangTokenizerWorkerCountOption, aliases: []string{sglangTokenizerWorkerCountOption}},
	{canonical: sglangJoinModeOption, aliases: []string{sglangJoinModeOption}},
	{canonical: sglangJoinRankOffsetOption, aliases: []string{sglangJoinRankOffsetOption}},
	{canonical: sglangDecodeCUDAGraphBackendOption, aliases: []string{sglangDecodeCUDAGraphBackendOption}},
	{canonical: sglangPrefillCUDAGraphBackendOption, aliases: []string{sglangPrefillCUDAGraphBackendOption}},
	{canonical: sglangEnableDPAttentionOption, aliases: []string{sglangEnableDPAttentionOption}, flag: true},
	{canonical: sglangEnableDPLMHeadOption, aliases: []string{sglangEnableDPLMHeadOption}, flag: true},
	{canonical: sglangDisableCUDAGraphOption, aliases: []string{sglangDisableCUDAGraphOption}, flag: true},
	{canonical: sglangUseRayOption, aliases: []string{sglangUseRayOption}, flag: true},
	{canonical: sglangEnableElasticBackupOption, aliases: []string{sglangEnableElasticBackupOption}, flag: true},
	{canonical: sglangDeprecatedElasticRejoinOption, aliases: []string{sglangDeprecatedElasticRejoinOption}, flag: true},
}

// ResolveSGLangProfileGeometry parses the merged SGLang scale-up declaration and resolves provider-neutral geometry.
// A successful result establishes physical geometry for growth only; it does not prove bootstrap, rank placement,
// retry-safe membership control, scale-down, or recovery conformance.
func ResolveSGLangProfileGeometry(source SGLangProfileGeometrySource) (enginegroup.ResolvedProfileGeometry, error) {
	// Require a valid creation-time target before comparing it with SGLang's launch-time assertions.
	if source.InitialReplicas <= 0 {
		return enginegroup.ResolvedProfileGeometry{}, fmt.Errorf("initial replicas must be positive, got %d", source.InitialReplicas)
	}

	// Resolve only the exact, merged growth profile whose logical-to-physical mapping is statically known.
	geometry, err := parseSGLangProfileGeometry(source.Command, source.Args)
	if err != nil {
		return enginegroup.ResolvedProfileGeometry{}, err
	}
	if err := validateSGLangScaleUpGeometry(geometry, source.InitialReplicas, source.MainContainerGPUs); err != nil {
		return enginegroup.ResolvedProfileGeometry{}, err
	}

	// Bind immutable engine semantics while excluding creation-time and live replica targets.
	engineGeometryDigest, err := digestSGLangEngineGeometry(geometry)
	if err != nil {
		return enginegroup.ResolvedProfileGeometry{}, err
	}

	// The merged width-one profile maps one logical DP replica to one physical EP rank and one GPU.
	return enginegroup.ResolveProfileGeometry(enginegroup.ProfileGeometryInput{
		Backend:        sglangEngineGroupBackend,
		GPUsPerReplica: 1,
		CapacityRoles: []enginegroup.CapacityRoleGeometry{
			{
				Name:             enginegroup.MainCapacityRoleName,
				Class:            enginegroup.CapacityRoleClassRankOwningCapacity,
				EngineGPUsPerPod: source.MainContainerGPUs,
				DedicatedGPUs:    source.DedicatedMainGPUAllocation,
			},
		},
		EngineGeometryDigest:   engineGeometryDigest,
		WorkloadRevisionDigest: source.WorkloadRevisionDigest,
	})
}

func parseSGLangProfileGeometry(command, args []string) (parsedSGLangProfileGeometry, error) {
	// Resolve the exact engine invocation before inspecting any options.
	argv, err := resolveSGLangProfileArguments(command, args)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}

	// Collect exact aliases without relying on argparse abbreviation or external configuration.
	values, flags, err := collectSGLangProfileOptions(argv)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}

	// Require every creation-time value that identifies the merged Elastic EP growth path.
	required := []string{
		sglangTensorParallelSizeOption,
		sglangDataParallelSizeOption,
		sglangElasticBackendOption,
		sglangMoEA2ABackendOption,
		sglangMaximumEPSizeOption,
		sglangLoadBalanceMethodOption,
	}
	for _, option := range required {
		if _, present := values[option]; !present {
			return parsedSGLangProfileGeometry{}, &UnsupportedSGLangProfileSourceError{
				Reason: UnsupportedSGLangProfileSourceReasonMissingGeometry,
				Detail: fmt.Sprintf("%s must be explicit for the merged Elastic EP scale-up profile", option),
			}
		}
	}

	// Parse positive parallel widths and launch dimensions, using only documented engine defaults.
	tensorParallelSize, err := parsePositiveSGLangInteger(sglangTensorParallelSizeOption, values[sglangTensorParallelSizeOption])
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	dataParallelSize, err := parsePositiveSGLangInteger(sglangDataParallelSizeOption, values[sglangDataParallelSizeOption])
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	pipelineParallelSize, err := parseOptionalPositiveSGLangInteger(values, sglangPipelineParallelSizeOption, 1)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	attentionContextSize, err := parseOptionalPositiveSGLangInteger(values, sglangAttentionContextSizeOption, 1)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	moeDataParallelSize, err := parseOptionalPositiveSGLangInteger(values, sglangMoEDataParallelSizeOption, 1)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	expertParallelSize, err := parseOptionalPositiveSGLangInteger(values, sglangExpertParallelSizeOption, tensorParallelSize)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	nodes, err := parseOptionalPositiveSGLangInteger(values, sglangNodesOption, 1)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	elasticInitialSize, err := parseOptionalPositiveSGLangInteger(values, sglangElasticInitialSizeOption, tensorParallelSize)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	elasticMaximumSize, err := parsePositiveSGLangInteger(sglangMaximumEPSizeOption, values[sglangMaximumEPSizeOption])
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}
	tokenizerWorkers, err := parseOptionalPositiveSGLangInteger(values, sglangTokenizerWorkerCountOption, 1)
	if err != nil {
		return parsedSGLangProfileGeometry{}, err
	}

	// Validate non-geometric switches that make this declaration the merged growth-only implementation.
	if err := validateSGLangScaleUpContract(values, flags, tokenizerWorkers); err != nil {
		return parsedSGLangProfileGeometry{}, err
	}

	return parsedSGLangProfileGeometry{
		tensorParallelSize:   tensorParallelSize,
		pipelineParallelSize: pipelineParallelSize,
		dataParallelSize:     dataParallelSize,
		attentionContextSize: attentionContextSize,
		moeDataParallelSize:  moeDataParallelSize,
		expertParallelSize:   expertParallelSize,
		nodes:                nodes,
		elasticInitialSize:   elasticInitialSize,
		elasticMaximumSize:   elasticMaximumSize,
		elasticBackend:       values[sglangElasticBackendOption],
		moeA2ABackend:        values[sglangMoEA2ABackendOption],
	}, validateSGLangScaleUpSizes(tensorParallelSize, dataParallelSize, elasticInitialSize, elasticMaximumSize)
}

func validateSGLangScaleUpContract(values map[string]string, flags map[string]bool, tokenizerWorkers int64) error {
	// Require the topology and communication modes implemented by the merged scale-up path.
	requirements := []struct {
		valid  bool
		detail string
	}{
		{valid: flags[sglangEnableDPAttentionOption], detail: sglangEnableDPAttentionOption + " is required"},
		{valid: flags[sglangEnableDPLMHeadOption], detail: sglangEnableDPLMHeadOption + " is required"},
		{valid: values[sglangElasticBackendOption] == "mooncake", detail: sglangElasticBackendOption + " must be mooncake"},
		{valid: values[sglangMoEA2ABackendOption] == "nixl", detail: sglangMoEA2ABackendOption + " must be nixl"},
		{valid: values[sglangLoadBalanceMethodOption] == "round_robin", detail: sglangLoadBalanceMethodOption + " must be round_robin"},
		{valid: tokenizerWorkers == 1, detail: sglangTokenizerWorkerCountOption + " must be 1"},
		{valid: !flags[sglangUseRayOption], detail: sglangUseRayOption + " is not supported"},
		{valid: !flags[sglangEnableElasticBackupOption], detail: sglangEnableElasticBackupOption + " is not supported"},
		{valid: !flags[sglangDeprecatedElasticRejoinOption], detail: sglangDeprecatedElasticRejoinOption + " describes recovery, not a primary growth profile"},
	}
	for _, requirement := range requirements {
		if !requirement.valid {
			return &UnsupportedSGLangProfileSourceError{
				Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
				Detail: requirement.detail,
			}
		}
	}

	// The primary profile must not contain operation-specific joining arguments.
	if _, present := values[sglangJoinModeOption]; present {
		return &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
			Detail: sglangJoinModeOption + " is operation-specific and cannot define the primary profile",
		}
	}
	if _, present := values[sglangJoinRankOffsetOption]; present {
		return &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
			Detail: sglangJoinRankOffsetOption + " is operation-specific and cannot define the primary profile",
		}
	}

	// Accept either the legacy all-phase switch or explicit per-phase disabled backends.
	cudaGraphsDisabled := flags[sglangDisableCUDAGraphOption] ||
		(values[sglangDecodeCUDAGraphBackendOption] == "disabled" &&
			values[sglangPrefillCUDAGraphBackendOption] == "disabled")
	if !cudaGraphsDisabled {
		return &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
			Detail: "the merged scale-up profile requires decode and prefill CUDA graphs to be disabled",
		}
	}

	return nil
}

func validateSGLangScaleUpSizes(tp, dp, initial, maximum int64) error {
	// Enforce the width-one mapping implemented by the first merged SGLang scale-up slice.
	requirements := []struct {
		valid  bool
		detail string
	}{
		{valid: tp == dp, detail: fmt.Sprintf("TP %d must equal DP %d", tp, dp)},
		{valid: initial == tp, detail: fmt.Sprintf("initial EP size %d must equal launch TP size %d", initial, tp)},
		{valid: maximum > initial, detail: fmt.Sprintf("maximum EP size %d must exceed initial EP size %d", maximum, initial)},
	}
	for _, requirement := range requirements {
		if !requirement.valid {
			return &UnsupportedSGLangProfileSourceError{
				Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
				Detail: requirement.detail,
			}
		}
	}

	return nil
}

func validateSGLangScaleUpGeometry(geometry parsedSGLangProfileGeometry, initialReplicas int32, mainContainerGPUs int64) error {
	// Bind the Kubernetes logical target to every SGLang launch-time cardinality assertion.
	if geometry.tensorParallelSize != int64(initialReplicas) || geometry.dataParallelSize != int64(initialReplicas) {
		return fmt.Errorf(
			"initial replicas %d conflict with SGLang TP/DP launch size %d",
			initialReplicas,
			geometry.tensorParallelSize,
		)
	}

	// Keep the first profile at replica width one until SGLang's wider DP-attention contract lands.
	if geometry.pipelineParallelSize != 1 || geometry.attentionContextSize != 1 ||
		geometry.moeDataParallelSize != 1 || geometry.expertParallelSize != geometry.tensorParallelSize {
		return &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonScaleUpContract,
			Detail: "PP, attention CP, and MoE DP must be 1 and effective EP must equal TP",
		}
	}

	// Prove that the declared launch cohort can fit in its Kubernetes GPU allocation.
	if geometry.tensorParallelSize%geometry.nodes != 0 {
		return fmt.Errorf("SGLang TP size %d is not divisible by node count %d", geometry.tensorParallelSize, geometry.nodes)
	}
	launchGPUsPerPod := geometry.tensorParallelSize / geometry.nodes
	if launchGPUsPerPod > mainContainerGPUs {
		return fmt.Errorf(
			"SGLang launch needs %d GPUs per pod but the main container allocates %d",
			launchGPUsPerPod,
			mainContainerGPUs,
		)
	}

	return nil
}

func resolveSGLangProfileArguments(command, args []string) ([]string, error) {
	// Preserve caller-owned slices while examining one combined exec-form invocation.
	argv := make([]string, 0, len(command)+len(args))
	argv = append(argv, command...)
	argv = append(argv, args...)
	if len(argv) == 0 {
		return nil, &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonCommandForm,
			Detail: "an explicit SGLang executable is required",
		}
	}

	// Reject launchers that can rewrite or interpolate the effective command.
	executable := filepath.Base(argv[0])
	if isProfileShellOrWrapper(executable) {
		return nil, &UnsupportedSGLangProfileSourceError{
			Reason: UnsupportedSGLangProfileSourceReasonShell,
			Detail: fmt.Sprintf("%s owns the effective SGLang argv", executable),
		}
	}

	// Accept Dynamo's module, SGLang's upstream module, or the documented `sglang serve` command.
	if isSupportedProfilePythonExecutable(executable) && len(argv) >= 3 && argv[1] == "-m" &&
		(argv[2] == sglangDynamoModule || argv[2] == sglangUpstreamModule) {
		return argv[3:], nil
	}
	if executable == "sglang" && len(argv) >= 2 && argv[1] == sglangServeSubcommand {
		return argv[2:], nil
	}

	return nil, &UnsupportedSGLangProfileSourceError{
		Reason: UnsupportedSGLangProfileSourceReasonCommandForm,
		Detail: "expected python -m dynamo.sglang, python -m sglang.launch_server, or sglang serve",
	}
}

func collectSGLangProfileOptions(argv []string) (map[string]string, map[string]bool, error) {
	values := make(map[string]string)
	flags := make(map[string]bool)

	// Stop at the conventional marker because later tokens are positional arguments.
	for index := 0; index < len(argv); index++ {
		token := argv[index]
		if token == "--" {
			break
		}
		if hasKubernetesArgumentExpansion(token) {
			name, _, hasValue := strings.Cut(token, "=")
			if !hasValue || hasKubernetesArgumentExpansion(name) {
				return nil, nil, &UnsupportedSGLangProfileSourceError{
					Reason: UnsupportedSGLangProfileSourceReasonEnvironment,
					Detail: fmt.Sprintf("argument %q contains option-bearing Kubernetes environment expansion", token),
				}
			}
		}
		if isSGLangExternalConfigOption(token) {
			return nil, nil, &UnsupportedSGLangProfileSourceError{
				Reason: UnsupportedSGLangProfileSourceReasonExternalConfig,
				Detail: fmt.Sprintf("%s can override inspected geometry", strings.SplitN(token, "=", 2)[0]),
			}
		}

		// Match only exact documented names and reject abbreviations of sensitive options.
		name, inlineValue, hasInlineValue := strings.Cut(token, "=")
		option, matched := matchSGLangProfileOption(name)
		if !matched {
			if isAbbreviatedSGLangProfileOption(name) {
				return nil, nil, &UnsupportedSGLangProfileSourceError{
					Reason: UnsupportedSGLangProfileSourceReasonAbbreviatedOption,
					Detail: fmt.Sprintf("%s is not an exact supported option name", name),
				}
			}

			continue
		}

		// Reject duplicate aliases because their precedence is not an immutable profile fact.
		if _, duplicateValue := values[option.canonical]; duplicateValue || flags[option.canonical] {
			return nil, nil, fmt.Errorf("SGLang option %s is specified more than once", option.canonical)
		}
		if option.flag {
			if hasInlineValue {
				return nil, nil, fmt.Errorf("SGLang flag %s does not accept a value", name)
			}
			flags[option.canonical] = true
			continue
		}

		// Read value options from either `--name=value` or the following argv token.
		value := inlineValue
		if !hasInlineValue {
			if index+1 >= len(argv) || strings.HasPrefix(argv[index+1], "--") {
				return nil, nil, fmt.Errorf("SGLang option %s requires a value", name)
			}
			index++
			value = argv[index]
		}
		if value == "" {
			return nil, nil, fmt.Errorf("SGLang option %s requires a non-empty value", name)
		}
		values[option.canonical] = value
	}

	return values, flags, nil
}

func matchSGLangProfileOption(name string) (sglangProfileOption, bool) {
	for _, option := range sglangProfileOptions {
		for _, alias := range option.aliases {
			if name == alias {
				return option, true
			}
		}
	}

	return sglangProfileOption{}, false
}

func isAbbreviatedSGLangProfileOption(name string) bool {
	if !strings.HasPrefix(name, "--") {
		return false
	}

	for _, option := range sglangProfileOptions {
		for _, alias := range option.aliases {
			if strings.HasPrefix(alias, name) {
				return true
			}
		}
	}

	for _, option := range []string{sglangConfigOption, sglangCUDAGraphConfigOption} {
		if name != option && strings.HasPrefix(option, name) {
			return true
		}
	}

	return false
}

func isSGLangExternalConfigOption(token string) bool {
	name := strings.SplitN(token, "=", 2)[0]
	return name == sglangConfigOption || name == sglangCUDAGraphConfigOption
}

func parsePositiveSGLangInteger(option, literal string) (int64, error) {
	value, err := strconv.ParseInt(literal, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse SGLang option %s value %q: %w", option, literal, err)
	}
	if value <= 0 {
		return 0, fmt.Errorf("SGLang option %s must be positive, got %d", option, value)
	}

	return value, nil
}

func parseOptionalPositiveSGLangInteger(values map[string]string, option string, defaultValue int64) (int64, error) {
	literal, present := values[option]
	if !present {
		return defaultValue, nil
	}

	return parsePositiveSGLangInteger(option, literal)
}

func digestSGLangEngineGeometry(geometry parsedSGLangProfileGeometry) (string, error) {
	// Exclude mutable replica targets while retaining the initial EP width because it fixes
	// each rank's expert-storage partition for the lifetime of the SGLang engine world.
	projection := sglangEngineGeometryProjection{
		Version:              sglangEngineGeometryDigestVersion,
		ReplicaWidth:         1,
		PipelineParallelSize: geometry.pipelineParallelSize,
		AttentionContextSize: geometry.attentionContextSize,
		MoEDataParallelSize:  geometry.moeDataParallelSize,
		StorageEPSize:        geometry.elasticInitialSize,
		ElasticMaximumSize:   geometry.elasticMaximumSize,
		ElasticBackend:       geometry.elasticBackend,
		MoEA2ABackend:        geometry.moeA2ABackend,
	}
	encoded, err := json.Marshal(projection)
	if err != nil {
		return "", fmt.Errorf("marshal SGLang engine geometry projection: %w", err)
	}

	// Prefix the digest so its encoding is self-describing.
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
