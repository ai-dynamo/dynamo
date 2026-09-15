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
	"errors"
	"regexp"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var sglangProfileFingerprintPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func TestResolveSGLangProfileGeometry(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
	}{
		{
			name:    "Dynamo module with canonical split options",
			command: []string{"python3", "-m", "dynamo.sglang"},
			args: []string{
				"--tp-size", "4",
				"--dp-size", "4",
				"--nnodes", "4",
				"--enable-dp-attention",
				"--enable-dp-lm-head",
				"--elastic-ep-backend", "mooncake",
				"--moe-a2a-backend", "nixl",
				"--elastic-ep-initial-size", "4",
				"--max-ep-size", "8",
				"--load-balance-method", "round_robin",
				"--disable-cuda-graph",
			},
		},
		{
			name:    "Dynamo module split across command and arguments",
			command: []string{"python3"},
			args: append([]string{"-m", "dynamo.sglang"},
				newTestSGLangProfileGeometrySource().Args...),
		},
		{
			name:    "upstream module with documented aliases",
			command: []string{"python", "-m", "sglang.launch_server"},
			args: []string{
				"--tp=4",
				"--dp=4",
				"--pp=1",
				"--attention-context-parallel-size=1",
				"--moe-data-parallel-size=1",
				"--expert-parallel-size=4",
				"--nnodes=4",
				"--enable-dp-attention",
				"--enable-dp-lm-head",
				"--elastic-ep-backend=mooncake",
				"--moe-a2a-backend=nixl",
				"--elastic-ep-initial-size=4",
				"--max-ep-size=8",
				"--load-balance-method=round_robin",
				"--cuda-graph-backend-decode=disabled",
				"--cuda-graph-backend-prefill=disabled",
			},
		},
		{
			name:    "upstream serve command",
			command: []string{"/usr/local/bin/sglang", "serve"},
			args:    newTestSGLangProfileGeometrySource().Args,
		},
		{
			name:    "initial EP defaults to managed creation target",
			command: []string{"python3", "-m", "dynamo.sglang"},
			args:    removeTestSGLangOption(newTestSGLangProfileGeometrySource().Args, sglangElasticInitialSizeOption),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build an inspectable SGLang Elastic EP scale-up source")
			source := newTestSGLangProfileGeometrySource()
			source.Command = append([]string(nil), test.command...)
			source.Args = append([]string(nil), test.args...)

			t.Log("resolve the width-one logical replica into provider-neutral geometry")
			got, err := ResolveSGLangProfileGeometry(source)
			require.NoError(t, err)

			t.Log("verify one independently allocatable GPU pod represents one logical replica")
			fingerprint := got.Fingerprint
			got.Fingerprint = ""
			assert.Equal(t, enginegroup.ResolvedProfileGeometry{
				Backend:        sglangEngineGroupBackend,
				GPUsPerReplica: 1,
				PodsPerReplica: 1,
				CapacityRole: enginegroup.CapacityRoleGeometry{
					Name:             enginegroup.MainCapacityRoleName,
					Class:            enginegroup.CapacityRoleClassRankOwningCapacity,
					EngineGPUsPerPod: 1,
					DedicatedGPUs:    true,
				},
			}, got)
			assert.Regexp(t, sglangProfileFingerprintPattern, fingerprint)
		})
	}
}

func TestResolveSGLangProfileGeometryCanonicalIdentity(t *testing.T) {
	t.Log("resolve equivalent split and alias forms")
	canonical, err := ResolveSGLangProfileGeometry(newTestSGLangProfileGeometrySource())
	require.NoError(t, err)
	aliasSource := newTestSGLangProfileGeometrySource()
	aliasSource.Args = []string{
		"--tensor-parallel-size=4",
		"--data-parallel-size=4",
		"--pipeline-parallel-size=1",
		"--attention-context-parallel-size=1",
		"--moe-data-parallel-size=1",
		"--ep=4",
		"--nnodes=4",
		"--enable-dp-attention",
		"--enable-dp-lm-head",
		"--elastic-ep-backend=mooncake",
		"--moe-a2a-backend=nixl",
		"--elastic-ep-initial-size=4",
		"--max-ep-size=8",
		"--load-balance-method=round_robin",
		"--disable-cuda-graph",
	}
	aliases, err := ResolveSGLangProfileGeometry(aliasSource)
	require.NoError(t, err)

	t.Log("verify syntax aliases do not change immutable profile identity")
	assert.Equal(t, canonical, aliases)
}

func TestResolveSGLangProfileGeometryFingerprintIncludesStorageEPSize(t *testing.T) {
	t.Log("resolve two immutable expert-storage layouts under the same maximum")
	initialFour := newTestSGLangProfileGeometrySource()
	initialFour.Args = setTestSGLangOption(initialFour.Args, sglangMaximumEPSizeOption, "16")
	geometryFour, err := ResolveSGLangProfileGeometry(initialFour)
	require.NoError(t, err)
	initialEight := newTestSGLangProfileGeometrySource()
	initialEight.InitialReplicas = 8
	initialEight.Args = setTestSGLangOption(initialEight.Args, sglangTensorParallelSizeOption, "8")
	initialEight.Args = setTestSGLangOption(initialEight.Args, sglangDataParallelSizeOption, "8")
	initialEight.Args = setTestSGLangOption(initialEight.Args, sglangElasticInitialSizeOption, "8")
	initialEight.Args = setTestSGLangOption(initialEight.Args, sglangNodesOption, "8")
	initialEight.Args = setTestSGLangOption(initialEight.Args, sglangMaximumEPSizeOption, "16")
	geometryEight, err := ResolveSGLangProfileGeometry(initialEight)
	require.NoError(t, err)

	t.Log("verify the initial EP width changes the fixed expert-storage identity")
	assert.NotEqual(t, geometryFour.Fingerprint, geometryEight.Fingerprint)

	t.Log("resolve a different immutable reserved maximum")
	maximumTwelve := newTestSGLangProfileGeometrySource()
	maximumTwelve.Args = setTestSGLangOption(maximumTwelve.Args, sglangMaximumEPSizeOption, "12")
	geometryTwelve, err := ResolveSGLangProfileGeometry(maximumTwelve)
	require.NoError(t, err)

	t.Log("verify the buffer-sizing ceiling participates in profile identity")
	assert.NotEqual(t, geometryFour.Fingerprint, geometryTwelve.Fingerprint)
}

func TestResolveSGLangProfileGeometryConfigurationErrors(t *testing.T) {
	tests := []struct {
		name            string
		initialReplicas int32
		gpusPerPod      int64
		option          string
		value           string
		appendArgs      []string
		wantError       string
	}{
		{
			name:            "invalid creation target",
			initialReplicas: 0,
			gpusPerPod:      1,
			wantError:       "initial replicas must be positive",
		},
		{
			name:            "creation target conflicts with launch size",
			initialReplicas: 8,
			gpusPerPod:      1,
			wantError:       "initial replicas 8 conflict with SGLang TP/DP launch size 4",
		},
		{
			name:            "launch cohort exceeds pod allocation",
			initialReplicas: 4,
			gpusPerPod:      1,
			option:          sglangNodesOption,
			value:           "1",
			wantError:       "launch needs 4 GPUs per pod but the main container allocates 1",
		},
		{
			name:            "TP is not divisible by nodes",
			initialReplicas: 4,
			gpusPerPod:      2,
			option:          sglangNodesOption,
			value:           "3",
			wantError:       "TP size 4 is not divisible by node count 3",
		},
		{
			name:            "malformed integer",
			initialReplicas: 4,
			gpusPerPod:      1,
			option:          sglangNodesOption,
			value:           "many",
			wantError:       "parse SGLang option --nnodes value \"many\"",
		},
		{
			name:            "duplicate alias",
			initialReplicas: 4,
			gpusPerPod:      1,
			appendArgs:      []string{"--tensor-parallel-size", "4"},
			wantError:       "SGLang option --tp-size is specified more than once",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a malformed SGLang scale-up declaration")
			source := newTestSGLangProfileGeometrySource()
			source.InitialReplicas = test.initialReplicas
			source.MainContainerGPUs = test.gpusPerPod
			if test.option != "" {
				source.Args = setTestSGLangOption(source.Args, test.option, test.value)
			}
			source.Args = append(source.Args, test.appendArgs...)

			t.Log("reject invalid declarative input as a configuration error")
			_, err := ResolveSGLangProfileGeometry(source)
			require.ErrorContains(t, err, test.wantError)
			assert.False(t, errors.Is(err, ErrUnsupportedSGLangProfileSource))
		})
	}
}

func TestResolveSGLangProfileGeometryUnsupportedSources(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		remove  string
		append  []string
		reason  UnsupportedSGLangProfileSourceReason
	}{
		{
			name:    "image-owned entrypoint",
			command: []string{},
			reason:  UnsupportedSGLangProfileSourceReasonCommandForm,
		},
		{
			name:    "shell command",
			command: []string{"/bin/bash", "-c"},
			reason:  UnsupportedSGLangProfileSourceReasonShell,
		},
		{
			name:    "environment wrapper",
			command: []string{"/usr/bin/env", "python3"},
			reason:  UnsupportedSGLangProfileSourceReasonShell,
		},
		{
			name:    "unrelated executable",
			command: []string{"worker"},
			reason:  UnsupportedSGLangProfileSourceReasonCommandForm,
		},
		{
			name:    "external server config",
			command: []string{"python3", "-m", "dynamo.sglang"},
			append:  []string{"--config", "/config/server.yaml"},
			reason:  UnsupportedSGLangProfileSourceReasonExternalConfig,
		},
		{
			name:    "opaque CUDA graph config",
			command: []string{"python3", "-m", "dynamo.sglang"},
			append:  []string{"--cuda-graph-config", "@/config/graphs.json"},
			reason:  UnsupportedSGLangProfileSourceReasonExternalConfig,
		},
		{
			name:    "option-bearing environment expansion",
			command: []string{"python3", "-m", "dynamo.sglang"},
			append:  []string{"$(EXTRA_ARGS)"},
			reason:  UnsupportedSGLangProfileSourceReasonEnvironment,
		},
		{
			name:    "abbreviated geometry option",
			command: []string{"python3", "-m", "dynamo.sglang"},
			append:  []string{"--tensor-parallel-s", "4"},
			reason:  UnsupportedSGLangProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "abbreviated CUDA graph config",
			command: []string{"python3", "-m", "dynamo.sglang"},
			append:  []string{"--cuda-graph-conf", "{}"},
			reason:  UnsupportedSGLangProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "missing global DP",
			command: []string{"python3", "-m", "dynamo.sglang"},
			remove:  sglangDataParallelSizeOption,
			reason:  UnsupportedSGLangProfileSourceReasonMissingGeometry,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a source whose effective SGLang declaration cannot be proven")
			source := newTestSGLangProfileGeometrySource()
			source.Command = append([]string(nil), test.command...)
			if test.remove != "" {
				source.Args = removeTestSGLangOption(source.Args, test.remove)
			}
			source.Args = append(source.Args, test.append...)

			t.Log("classify the unsupported source without treating it as transient")
			_, err := ResolveSGLangProfileGeometry(source)
			require.Error(t, err)
			assert.ErrorIs(t, err, ErrUnsupportedSGLangProfileSource)
			assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)

			var sourceError *UnsupportedSGLangProfileSourceError
			require.ErrorAs(t, err, &sourceError)
			assert.Equal(t, test.reason, sourceError.Reason)
		})
	}
}

func TestResolveSGLangProfileGeometryScaleUpContract(t *testing.T) {
	tests := []struct {
		name   string
		remove string
		option string
		value  string
		append []string
	}{
		{name: "DP attention disabled", remove: sglangEnableDPAttentionOption},
		{name: "DP LM head disabled", remove: sglangEnableDPLMHeadOption},
		{name: "wrong Elastic EP backend", option: sglangElasticBackendOption, value: "nixl"},
		{name: "wrong MoE A2A backend", option: sglangMoEA2ABackendOption, value: "deepep"},
		{name: "unsupported load balancing", option: sglangLoadBalanceMethodOption, value: "auto"},
		{name: "multiple tokenizer workers", append: []string{sglangTokenizerWorkerCountOption, "2"}},
		{name: "Ray launch", append: []string{sglangUseRayOption}},
		{name: "elastic expert backup", append: []string{sglangEnableElasticBackupOption}},
		{name: "recovery mode", append: []string{sglangDeprecatedElasticRejoinOption}},
		{name: "join mode in primary profile", append: []string{sglangJoinModeOption, "scale"}},
		{name: "join offset in primary profile", append: []string{sglangJoinRankOffsetOption, "4"}},
		{name: "CUDA graphs not explicitly disabled", remove: sglangDisableCUDAGraphOption},
		{name: "TP and DP differ", option: sglangDataParallelSizeOption, value: "2"},
		{name: "initial EP differs from TP", option: sglangElasticInitialSizeOption, value: "2"},
		{name: "no growth above initial EP", option: sglangMaximumEPSizeOption, value: "4"},
		{name: "pipeline parallelism", append: []string{sglangPipelineParallelSizeOption, "2"}},
		{name: "attention context parallelism", append: []string{sglangAttentionContextSizeOption, "2"}},
		{name: "MoE data parallelism", append: []string{sglangMoEDataParallelSizeOption, "2"}},
		{name: "effective EP differs from TP", append: []string{sglangExpertParallelSizeOption, "2"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("alter one invariant of the merged SGLang scale-up profile")
			source := newTestSGLangProfileGeometrySource()
			if test.remove != "" {
				source.Args = removeTestSGLangOption(source.Args, test.remove)
			}
			if test.option != "" {
				source.Args = setTestSGLangOption(source.Args, test.option, test.value)
			}
			source.Args = append(source.Args, test.append...)

			t.Log("reject the declaration with the stable growth-contract reason")
			_, err := ResolveSGLangProfileGeometry(source)
			require.Error(t, err)

			var sourceError *UnsupportedSGLangProfileSourceError
			require.ErrorAs(t, err, &sourceError)
			assert.Equal(t, UnsupportedSGLangProfileSourceReasonScaleUpContract, sourceError.Reason)
		})
	}
}

func TestResolveSGLangProfileGeometryCommonBoundary(t *testing.T) {
	tests := []struct {
		name          string
		gpusPerPod    int64
		dedicatedGPUs bool
		reason        enginegroup.UnsupportedProfileReason
	}{
		{
			name:          "legacy warm-standby pod packs replicas and reserved GPUs",
			gpusPerPod:    8,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonPackedReplicas,
		},
		{
			name:          "multi-GPU pod packs replicas",
			gpusPerPod:    4,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonPackedReplicas,
		},
		{
			name:          "shared GPU allocation",
			gpusPerPod:    1,
			dedicatedGPUs: false,
			reason:        enginegroup.UnsupportedProfileReasonGPUAllocation,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a valid SGLang profile with an unsupported Kubernetes allocation")
			source := newTestSGLangProfileGeometrySource()
			source.MainContainerGPUs = test.gpusPerPod
			source.DedicatedMainGPUAllocation = test.dedicatedGPUs

			t.Log("preserve the common physical-geometry classification")
			_, err := ResolveSGLangProfileGeometry(source)
			require.Error(t, err)
			assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)
			assert.False(t, errors.Is(err, ErrUnsupportedSGLangProfileSource))

			var profileError *enginegroup.UnsupportedProfileError
			require.ErrorAs(t, err, &profileError)
			assert.Equal(t, test.reason, profileError.Reason)
		})
	}
}

func TestResolveSGLangProfileGeometryDoesNotMutateSource(t *testing.T) {
	t.Log("capture an independently owned copy of the provider source")
	source := newTestSGLangProfileGeometrySource()
	before := cloneTestSGLangProfileGeometrySource(source)

	t.Log("resolve the source through both provider and common layers")
	_, err := ResolveSGLangProfileGeometry(source)
	require.NoError(t, err)

	t.Log("verify neither command nor arguments were changed")
	assert.Equal(t, before, source)
}

func newTestSGLangProfileGeometrySource() SGLangProfileGeometrySource {
	return SGLangProfileGeometrySource{
		Command: []string{"python3", "-m", "dynamo.sglang"},
		Args: []string{
			sglangTensorParallelSizeOption, "4",
			sglangDataParallelSizeOption, "4",
			sglangNodesOption, "4",
			sglangEnableDPAttentionOption,
			sglangEnableDPLMHeadOption,
			sglangElasticBackendOption, "mooncake",
			sglangMoEA2ABackendOption, "nixl",
			sglangElasticInitialSizeOption, "4",
			sglangMaximumEPSizeOption, "8",
			sglangLoadBalanceMethodOption, "round_robin",
			sglangDisableCUDAGraphOption,
		},
		InitialReplicas:            4,
		MainContainerGPUs:          1,
		DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest:     "sha256:test-workload-revision",
	}
}

func cloneTestSGLangProfileGeometrySource(source SGLangProfileGeometrySource) SGLangProfileGeometrySource {
	cloned := source
	cloned.Command = append([]string(nil), source.Command...)
	cloned.Args = append([]string(nil), source.Args...)
	return cloned
}

func setTestSGLangOption(args []string, option, value string) []string {
	updated := append([]string(nil), args...)
	for index := 0; index < len(updated); index++ {
		if updated[index] == option && index+1 < len(updated) {
			updated[index+1] = value
			return updated
		}
		if name, _, present := strings.Cut(updated[index], "="); present && name == option {
			updated[index] = option + "=" + value
			return updated
		}
	}

	return append(updated, option, value)
}

func removeTestSGLangOption(args []string, option string) []string {
	updated := make([]string, 0, len(args))
	for index := 0; index < len(args); index++ {
		name, _, inline := strings.Cut(args[index], "=")
		if name != option {
			updated = append(updated, args[index])
			continue
		}
		if !inline && index+1 < len(args) && !isTestSGLangFlag(option) {
			index++
		}
	}

	return updated
}

func isTestSGLangFlag(option string) bool {
	optionSpec, present := matchSGLangProfileOption(option)
	return present && optionSpec.flag
}
