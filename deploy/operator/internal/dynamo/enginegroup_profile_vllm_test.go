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
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var vllmProfileFingerprintPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func TestResolveVLLMProfileGeometry(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
	}{
		{
			name:    "canonical split arguments with DP omitted",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--model", "test-model",
				"--tensor-parallel-size", "4",
				"--pipeline-parallel-size", "1",
				"--data-parallel-size-local", "1",
			},
		},
		{
			name:    "canonical equals arguments with matching DP and local DP",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--tensor-parallel-size=4",
				"--pipeline-parallel-size=1",
				"--data-parallel-size=8",
				"--data-parallel-size-local=1",
			},
		},
		{
			name:    "explicit unit prefill context parallelism",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--tensor-parallel-size", "4",
				"--pipeline-parallel-size", "1",
				"--prefill-context-parallel-size", "1",
				"--data-parallel-size-local", "1",
			},
		},
		{
			name:    "documented aliases in split form",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"-tp", "4",
				"-pp", "1",
				"-dp", "8",
				"-dpl", "1",
			},
		},
		{
			name:    "documented aliases in equals form",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"-tp=4",
				"-pp=1",
				"-dp=8",
				"-dpl=1",
			},
		},
		{
			name: "literal geometry in command",
			command: []string{
				"python3", "-m", "dynamo.vllm",
				"-tp=4", "-pp", "1",
			},
			args: []string{"-dp", "8", "-dpl", "1"},
		},
		{
			name:    "direct vllm serve invocation",
			command: []string{"/usr/local/bin/vllm", "serve"},
			args: []string{
				"test-model",
				"--tensor-parallel-size", "4",
				"--pipeline-parallel-size", "1",
				"--data-parallel-size-local", "1",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a strict vLLM profile source")
			source := newTestVLLMProfileGeometrySource(test.command, test.args)

			t.Log("resolve provider-owned argv into provider-neutral geometry")
			got, err := ResolveVLLMProfileGeometry(source)
			require.NoError(t, err)

			t.Log("verify the physical facts proven by this source")
			fingerprint := got.Fingerprint
			got.Fingerprint = ""
			assert.Equal(t, enginegroup.ResolvedProfileGeometry{
				Backend:        vllmEngineGroupBackend,
				GPUsPerReplica: 4,
				PodsPerReplica: 1,
				CapacityRole: enginegroup.CapacityRoleGeometry{
					Name:             enginegroup.MainCapacityRoleName,
					Class:            enginegroup.CapacityRoleClassRankOwningCapacity,
					EngineGPUsPerPod: 4,
					DedicatedGPUs:    true,
				},
			}, got)
			assert.Regexp(t, vllmProfileFingerprintPattern, fingerprint)
		})
	}
}

func TestResolveVLLMProfileGeometryCanonicalIdentity(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
	}{
		{
			name:    "canonical split",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--tensor-parallel-size", "4",
				"--pipeline-parallel-size", "1",
				"--data-parallel-size-local", "1",
			},
		},
		{
			name:    "canonical equals with creation-time DP assertion",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--tensor-parallel-size=4",
				"--pipeline-parallel-size=1",
				"--data-parallel-size=8",
				"--data-parallel-size-local=1",
			},
		},
		{
			name:    "aliases with explicit supported local DP",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"-tp", "4",
				"-pp", "1",
				"-dp", "8",
				"-dpl", "1",
				"-pcp", "1",
			},
		},
		{
			name:    "underscore long options",
			command: []string{"python3"},
			args: []string{
				"-m", "dynamo.vllm",
				"--tensor_parallel_size", "4",
				"--pipeline_parallel_size", "1",
				"--prefill_context_parallel_size", "1",
				"--data_parallel_size", "8",
				"--data_parallel_size_local", "1",
			},
		},
	}

	var fingerprint string
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("resolve an equivalent spelling of the same engine geometry")
			resolved, err := ResolveVLLMProfileGeometry(newTestVLLMProfileGeometrySource(test.command, test.args))
			require.NoError(t, err)

			t.Log("verify aliases and creation-time DP syntax do not change profile identity")
			if fingerprint == "" {
				fingerprint = resolved.Fingerprint
			}
			assert.Equal(t, fingerprint, resolved.Fingerprint)
		})
	}
}

func TestResolveVLLMProfileGeometryIncludesPrefillContextParallelism(t *testing.T) {
	t.Log("build a profile whose prefill context axis expands one DP replica")
	source := newTestVLLMProfileGeometrySource(
		[]string{"python3"},
		[]string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "--prefill_context_parallel_size=2", "-dpl", "1"},
	)
	source.MainContainerGPUs = 8

	t.Log("resolve every world-expanding engine axis into physical GPU geometry")
	resolved, err := ResolveVLLMProfileGeometry(source)
	require.NoError(t, err)
	assert.Equal(t, int64(8), resolved.GPUsPerReplica)
	assert.Equal(t, int64(8), resolved.CapacityRole.EngineGPUsPerPod)
}

func TestResolveVLLMProfileGeometryFingerprintExcludesReplicaTarget(t *testing.T) {
	t.Log("resolve geometry at the initial creation target")
	initial := newTestVLLMProfileGeometrySource(
		[]string{"python3", "-m", "dynamo.vllm"},
		[]string{"-tp", "4", "-pp", "1", "-dpl", "1"},
	)
	initialGeometry, err := ResolveVLLMProfileGeometry(initial)
	require.NoError(t, err)

	t.Log("resolve identical geometry at a different creation target")
	resized := initial
	resized.InitialReplicas = 16
	resizedGeometry, err := ResolveVLLMProfileGeometry(resized)
	require.NoError(t, err)

	t.Log("verify a mutable target does not change immutable geometry identity")
	assert.Equal(t, initialGeometry.Fingerprint, resizedGeometry.Fingerprint)
}

func TestResolveVLLMProfileGeometryDPAssertions(t *testing.T) {
	tests := []struct {
		name            string
		initialReplicas int32
		args            []string
		wantError       string
	}{
		{
			name:            "invalid creation target",
			initialReplicas: 0,
			args: []string{
				"-tp", "4",
				"-pp", "1",
			},
			wantError: "initial replicas must be positive",
		},
		{
			name:            "global DP conflicts with creation target",
			initialReplicas: 8,
			args: []string{
				"-tp", "4",
				"-pp", "1",
				"-dp", "4",
			},
			wantError: "initial replicas 8 conflict with vLLM data parallel size 4",
		},
		{
			name:            "local DP exceeds global DP",
			initialReplicas: 1,
			args: []string{
				"-tp", "4",
				"-pp", "1",
				"-dp", "1",
				"-dpl", "2",
			},
			wantError: "vLLM data parallel size local 2 exceeds global data parallel size 1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a vLLM source with an invalid DP declaration")
			source := newTestVLLMProfileGeometrySource([]string{"python3", "-m", "dynamo.vllm"}, test.args)
			source.InitialReplicas = test.initialReplicas

			t.Log("reject the declaration as a normal configuration error")
			_, err := ResolveVLLMProfileGeometry(source)
			require.ErrorContains(t, err, test.wantError)
			assert.False(t, errors.Is(err, ErrUnsupportedVLLMProfileSource))
		})
	}
}

func TestResolveVLLMProfileGeometryUnsupportedSources(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
		reason  UnsupportedVLLMProfileSourceReason
	}{
		{
			name:    "image-owned entrypoint",
			command: []string{},
			args:    []string{"-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonCommandForm,
		},
		{
			name:    "shell executable",
			command: []string{"/bin/bash", "-c"},
			args:    []string{"python3 -m dynamo.vllm -tp 4 -pp 1"},
			reason:  UnsupportedVLLMProfileSourceReasonShell,
		},
		{
			name:    "space-joined shell executable",
			command: []string{"/bin/sh -c"},
			args:    []string{"python3"},
			reason:  UnsupportedVLLMProfileSourceReasonShell,
		},
		{
			name:    "environment wrapper",
			command: []string{"/usr/bin/env", "python3"},
			args:    []string{"-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonShell,
		},
		{
			name:    "python code string",
			command: []string{"python3", "-c"},
			args:    []string{"print('not vllm')", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonCommandForm,
		},
		{
			name:    "unrelated executable",
			command: []string{"worker", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonCommandForm,
		},
		{
			name:    "runtime environment expansion",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "$(TP_SIZE)", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonEnvironment,
		},
		{
			name:    "runtime expansion in an unrecognized option position",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "$(EXTRA_OPTIONS)"},
			reason:  UnsupportedVLLMProfileSourceReasonEnvironment,
		},
		{
			name:    "external config split form",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "--config", "/config/vllm.yaml", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonExternalConfig,
		},
		{
			name:    "external config equals form",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "--config=/config/vllm.yaml", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonExternalConfig,
		},
		{
			name:    "abbreviated geometry option",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "-dpl", "1", "--prefill-context-parallel-s=2"},
			reason:  UnsupportedVLLMProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "abbreviated placement option",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "-dpl", "1", "--nnod=2"},
			reason:  UnsupportedVLLMProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "abbreviated external config option",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "-dpl", "1", "--conf=/config/vllm.yaml"},
			reason:  UnsupportedVLLMProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "geometry after end-of-options marker",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "--", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "geometry omitted in favor of engine defaults",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "--model", "test-model"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "tensor parallel size omitted",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "pipeline parallel size omitted",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "local data parallel layout omitted",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "multi-node launch",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "--nnodes", "2"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "underscore node-rank option",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "--node_rank=0"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "Ray distributed executor",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "--distributed-executor-backend", "ray"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "explicit data-parallel backend",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "-dpb", "mp"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "explicit device placement",
			command: []string{"python3"},
			args:    []string{"-m", "dynamo.vllm", "-tp", "4", "-pp", "1", "--device-ids", "0,1,2,3"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a profile whose effective geometry source is not statically inspectable")
			source := newTestVLLMProfileGeometrySource(test.command, test.args)

			t.Log("classify the expected unsupported source without treating it as transient")
			_, err := ResolveVLLMProfileGeometry(source)
			require.Error(t, err)
			assert.ErrorIs(t, err, ErrUnsupportedVLLMProfileSource)
			assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)

			var sourceError *UnsupportedVLLMProfileSourceError
			require.ErrorAs(t, err, &sourceError)
			assert.Equal(t, test.reason, sourceError.Reason)
		})
	}
}

func TestResolveVLLMProfileGeometryMalformedArguments(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		wantError string
	}{
		{
			name:      "missing split value",
			args:      []string{"-pp", "1", "-tp"},
			wantError: "must be an integer literal",
		},
		{
			name:      "malformed value",
			args:      []string{"-tp", "four", "-pp", "1"},
			wantError: "must be an integer literal",
		},
		{
			name:      "zero value",
			args:      []string{"-tp=0", "-pp", "1"},
			wantError: "must be positive",
		},
		{
			name:      "negative value",
			args:      []string{"-tp", "-4", "-pp", "1"},
			wantError: "must be positive",
		},
		{
			name: "duplicate canonical argument",
			args: []string{
				"--tensor-parallel-size", "4",
				"--tensor-parallel-size=4",
				"-pp", "1",
			},
			wantError: "--tensor-parallel-size is specified more than once",
		},
		{
			name: "conflicting alias duplicate",
			args: []string{
				"--tensor-parallel-size", "4",
				"-tp", "8",
				"-pp", "1",
			},
			wantError: "--tensor-parallel-size is specified more than once",
		},
		{
			name: "duplicate local DP",
			args: []string{
				"-tp", "4",
				"-pp", "1",
				"--data-parallel-size-local", "1",
				"--data-parallel-size-local=1",
			},
			wantError: "--data-parallel-size-local is specified more than once",
		},
		{
			name: "duplicate underscore and dash forms",
			args: []string{
				"--tensor_parallel_size", "4",
				"--tensor-parallel-size", "4",
				"-pp", "1",
			},
			wantError: "--tensor-parallel-size is specified more than once",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a literal but malformed vLLM declaration")
			source := newTestVLLMProfileGeometrySource([]string{"python3", "-m", "dynamo.vllm"}, test.args)

			t.Log("reject malformed or conflicting geometry as a normal configuration error")
			_, err := ResolveVLLMProfileGeometry(source)
			require.ErrorContains(t, err, test.wantError)
			assert.False(t, errors.Is(err, ErrUnsupportedVLLMProfileSource))
			assert.False(t, errors.Is(err, enginegroup.ErrUnsupportedProfile))
		})
	}
}

func TestResolveVLLMProfileGeometryIgnoresUnrelatedLiteralArguments(t *testing.T) {
	t.Log("build a source with fixed-value expansion and literal syntax outside geometry")
	source := newTestVLLMProfileGeometrySource(
		[]string{"python3"},
		[]string{
			"-m", "dynamo.vllm",
			"-tp", "4",
			"-pp", "1",
			"-dpl", "1",
			"--served-model-name=$(MODEL_NAME)",
			"--chat-template=$$(CHAT_TEMPLATE)",
			"--chat-template", "{{ if $message }}hello world{{ end }}",
		},
	)

	t.Log("resolve geometry without interpreting unrelated vLLM arguments")
	_, err := ResolveVLLMProfileGeometry(source)
	require.NoError(t, err)
}

func TestResolveVLLMProfileGeometryCommonGeometryBoundary(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		gpusPerPod    int64
		dedicatedGPUs bool
		reason        enginegroup.UnsupportedProfileReason
	}{
		{
			name:          "packed replicas",
			gpusPerPod:    8,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonPackedReplicas,
		},
		{
			name:          "multi-pod replica",
			gpusPerPod:    2,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonMultiPodReplica,
		},
		{
			name:          "shared GPU allocation",
			gpusPerPod:    4,
			dedicatedGPUs: false,
			reason:        enginegroup.UnsupportedProfileReasonGPUAllocation,
		},
		{
			name:          "control-only local DP process",
			args:          []string{"-tp", "4", "-pp", "1", "-dpl", "0"},
			gpusPerPod:    4,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonCapacityLayout,
		},
		{
			name:          "multiple local DP replicas",
			args:          []string{"-tp", "4", "-pp", "1", "--data_parallel_size_local", "2"},
			gpusPerPod:    8,
			dedicatedGPUs: true,
			reason:        enginegroup.UnsupportedProfileReasonPackedReplicas,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a source whose argv resolves but physical geometry is unsupported")
			args := test.args
			if args == nil {
				args = []string{"-tp", "4", "-pp", "1", "-dpl", "1"}
			}
			source := newTestVLLMProfileGeometrySource(
				[]string{"python3", "-m", "dynamo.vllm"},
				args,
			)
			source.MainContainerGPUs = test.gpusPerPod
			source.DedicatedMainGPUAllocation = test.dedicatedGPUs

			t.Log("preserve the provider-neutral unsupported-profile classification")
			_, err := ResolveVLLMProfileGeometry(source)
			require.Error(t, err)
			assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)
			assert.False(t, errors.Is(err, ErrUnsupportedVLLMProfileSource))

			var profileError *enginegroup.UnsupportedProfileError
			require.ErrorAs(t, err, &profileError)
			assert.Equal(t, test.reason, profileError.Reason)
		})
	}
}

func TestResolveVLLMProfileGeometryFingerprintChangesWithEngineGeometry(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{
			name: "TP four PP one",
			args: []string{"-tp", "4", "-pp", "1", "-dpl", "1"},
		},
		{
			name: "TP two PP two",
			args: []string{"-tp", "2", "-pp", "2", "-dpl", "1"},
		},
	}

	var fingerprint string
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("resolve equal-GPU layouts with different native engine geometry")
			resolved, err := ResolveVLLMProfileGeometry(newTestVLLMProfileGeometrySource(
				[]string{"python3", "-m", "dynamo.vllm"},
				test.args,
			))
			require.NoError(t, err)

			t.Log("verify the semantic engine geometry participates in profile identity")
			if fingerprint == "" {
				fingerprint = resolved.Fingerprint
				return
			}
			assert.NotEqual(t, fingerprint, resolved.Fingerprint)
		})
	}
}

func TestResolveVLLMProfileGeometryDoesNotMutateSource(t *testing.T) {
	t.Log("capture an independently owned copy of the provider source")
	source := newTestVLLMProfileGeometrySource(
		[]string{"python3", "-m", "dynamo.vllm"},
		[]string{"-tp", "4", "-pp", "1", "-dp", "8", "-dpl", "1"},
	)
	before := cloneTestVLLMProfileGeometrySource(source)

	t.Log("resolve the source through both provider and common layers")
	_, err := ResolveVLLMProfileGeometry(source)
	require.NoError(t, err)

	t.Log("verify neither command nor arguments were changed")
	assert.Equal(t, before, source)
}

func newTestVLLMProfileGeometrySource(command, args []string) VLLMProfileGeometrySource {
	return VLLMProfileGeometrySource{
		Command:                    append([]string(nil), command...),
		Args:                       append([]string(nil), args...),
		InitialReplicas:            8,
		MainContainerGPUs:          4,
		DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest:     "sha256:test-workload-revision",
	}
}

func cloneTestVLLMProfileGeometrySource(source VLLMProfileGeometrySource) VLLMProfileGeometrySource {
	cloned := source
	cloned.Command = append([]string(nil), source.Command...)
	cloned.Args = append([]string(nil), source.Args...)
	return cloned
}
