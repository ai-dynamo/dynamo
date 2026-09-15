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
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveVLLMProfileGeometryCurrentOwnershipUnsupported(t *testing.T) {
	tests := []struct {
		name       string
		command    []string
		args       []string
		removeFlag string
		wantDetail string
	}{
		{
			name:    "current internal Ray ownership",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args: []string{
				"-tp", "4",
				"-pp", "1",
				"-dp", "8",
				"-dpl", "1",
				"--data-parallel-backend", "ray",
			},
			wantDetail: "one internal client with Ray-managed DP capacity",
		},
		{
			name:       "Elastic EP disabled",
			command:    []string{"python3", "-m", "dynamo.vllm"},
			args:       []string{"-tp", "4", "-pp", "1", "-dp", "8", "-dpl", "1"},
			removeFlag: enableElasticEPFlag,
			wantDetail: "--enable-elastic-ep is required",
		},
		{
			name:    "external DP load balancing",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args: []string{
				"--tensor-parallel-size=4",
				"--pipeline-parallel-size=1",
				"--data-parallel-size=8",
				"--data-parallel-size-local=1",
				vllmDataParallelExternalLBFlag,
			},
			wantDetail: "incompatible with external or hybrid DP load balancing",
		},
		{
			name:    "hybrid DP load balancing",
			command: []string{"/usr/local/bin/vllm", "serve"},
			args: []string{
				"test-model",
				"--tensor_parallel_size", "4",
				"--pipeline_parallel_size", "1",
				"--data_parallel_size", "8",
				"--data_parallel_size_local", "1",
				vllmDataParallelHybridLBFlag,
			},
			wantDetail: "incompatible with external or hybrid DP load balancing",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := newTestVLLMProfileGeometrySource(test.command, test.args)
			if test.removeFlag != "" {
				source.Args = removeTestVLLMFlag(source.Args, test.removeFlag)
			}

			_, err := ResolveVLLMProfileGeometry(source)
			require.Error(t, err)
			assert.ErrorIs(t, err, ErrUnsupportedVLLMProfileSource)
			assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)
			assert.ErrorContains(t, err, test.wantDetail)

			var sourceError *UnsupportedVLLMProfileSourceError
			require.ErrorAs(t, err, &sourceError)
			assert.Equal(t, UnsupportedVLLMProfileSourceReasonOwnershipMode, sourceError.Reason)
		})
	}
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
			args:            []string{"-tp", "4", "-pp", "1"},
			wantError:       "initial replicas must be positive",
		},
		{
			name:            "global DP conflicts with creation target",
			initialReplicas: 8,
			args:            []string{"-tp", "4", "-pp", "1", "-dp", "4"},
			wantError:       "initial replicas 8 conflict with vLLM data parallel size 4",
		},
		{
			name:            "local DP exceeds global DP",
			initialReplicas: 1,
			args:            []string{"-tp", "4", "-pp", "1", "-dp", "1", "-dpl", "2"},
			wantError:       "vLLM data parallel size local 2 exceeds global data parallel size 1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := newTestVLLMProfileGeometrySource(
				[]string{"python3", "-m", "dynamo.vllm"},
				test.args,
			)
			source.InitialReplicas = test.initialReplicas

			_, err := ResolveVLLMProfileGeometry(source)
			require.ErrorContains(t, err, test.wantError)
			assert.False(t, errors.Is(err, ErrUnsupportedVLLMProfileSource))
		})
	}
}

func TestResolveVLLMProfileGeometryNativeEnvironment(t *testing.T) {
	tests := []struct {
		name                       string
		initialReplicas            int32
		args                       []string
		environment                map[string]string
		hasUnresolvedDPEnvironment bool
		wantError                  string
		wantReason                 UnsupportedVLLMProfileSourceReason
	}{
		{
			name:            "native DP size supplies omitted global assertion",
			initialReplicas: 8,
			args:            []string{"-tp", "4", "-pp", "1", "-dpl", "1"},
			environment:     map[string]string{vllmDPSizeEnvironment: "8"},
			wantReason:      UnsupportedVLLMProfileSourceReasonOwnershipMode,
		},
		{
			name:            "explicit DP one still uses native environment",
			initialReplicas: 1,
			args:            []string{"-tp", "4", "-pp", "1", "-dp", "1", "-dpl", "1"},
			environment:     map[string]string{vllmDPSizeEnvironment: "8"},
			wantError:       "initial replicas 1 conflict with vLLM data parallel size 8",
		},
		{
			name:            "explicit DP above one suppresses native environment",
			initialReplicas: 8,
			args:            []string{"-tp", "4", "-pp", "1", "-dp", "8", "-dpl", "1"},
			environment:     map[string]string{vllmDPSizeEnvironment: "4"},
			wantReason:      UnsupportedVLLMProfileSourceReasonOwnershipMode,
		},
		{
			name:            "local DP zero suppresses native environment",
			initialReplicas: 1,
			args:            []string{"-tp", "4", "-pp", "1", "-dp", "1", "-dpl", "0"},
			environment:     map[string]string{vllmDPSizeEnvironment: "8"},
			wantReason:      UnsupportedVLLMProfileSourceReasonOwnershipMode,
		},
		{
			name:            "malformed native DP size",
			initialReplicas: 8,
			args:            []string{"-tp", "4", "-pp", "1", "-dpl", "1"},
			environment:     map[string]string{vllmDPSizeEnvironment: "many"},
			wantError:       "VLLM_DP_SIZE must be an integer literal",
		},
		{
			name:                       "opaque environment source",
			initialReplicas:            8,
			args:                       []string{"-tp", "4", "-pp", "1", "-dpl", "1"},
			hasUnresolvedDPEnvironment: true,
			wantReason:                 UnsupportedVLLMProfileSourceReasonEnvironment,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := newTestVLLMProfileGeometrySource(
				[]string{"python3", "-m", "dynamo.vllm"},
				test.args,
			)
			source.InitialReplicas = test.initialReplicas
			source.Environment = test.environment
			source.HasUnresolvedDPEnvironment = test.hasUnresolvedDPEnvironment

			_, err := ResolveVLLMProfileGeometry(source)
			require.Error(t, err)
			if test.wantError != "" {
				assert.ErrorContains(t, err, test.wantError)
				return
			}

			var sourceError *UnsupportedVLLMProfileSourceError
			require.ErrorAs(t, err, &sourceError)
			assert.Equal(t, test.wantReason, sourceError.Reason)
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
			name:    "environment wrapper",
			command: []string{"/usr/bin/env", "python3"},
			args:    []string{"-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonShell,
		},
		{
			name:    "runtime environment expansion",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "$(TP_SIZE)", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonEnvironment,
		},
		{
			name:    "external config",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"--config=/config/vllm.yaml", "-tp", "4", "-pp", "1"},
			reason:  UnsupportedVLLMProfileSourceReasonExternalConfig,
		},
		{
			name:    "abbreviated geometry option",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "4", "-pp", "1", "--prefill-context-parallel-s=2"},
			reason:  UnsupportedVLLMProfileSourceReasonAbbreviatedOption,
		},
		{
			name:    "geometry omitted",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"--model", "test-model"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "pipeline parallel size omitted",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "4"},
			reason:  UnsupportedVLLMProfileSourceReasonMissingGeometry,
		},
		{
			name:    "multi-node launch",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "4", "-pp", "1", "--nnodes", "2"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "Ray distributed executor",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "4", "-pp", "1", "--distributed-executor-backend", "ray"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
		{
			name:    "explicit device placement",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"-tp", "4", "-pp", "1", "--device-ids", "0,1,2,3"},
			reason:  UnsupportedVLLMProfileSourceReasonDistributedPlacement,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := newTestVLLMProfileGeometrySource(test.command, test.args)

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
			name:      "duplicate canonical argument",
			args:      []string{"--tensor-parallel-size", "4", "--tensor-parallel-size=4", "-pp", "1"},
			wantError: "--tensor-parallel-size is specified more than once",
		},
		{
			name:      "conflicting alias duplicate",
			args:      []string{"--tensor-parallel-size", "4", "-tp", "8", "-pp", "1"},
			wantError: "--tensor-parallel-size is specified more than once",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			source := newTestVLLMProfileGeometrySource(
				[]string{"python3", "-m", "dynamo.vllm"},
				test.args,
			)

			_, err := ResolveVLLMProfileGeometry(source)
			require.ErrorContains(t, err, test.wantError)
			assert.False(t, errors.Is(err, ErrUnsupportedVLLMProfileSource))
			assert.False(t, errors.Is(err, enginegroup.ErrUnsupportedProfile))
		})
	}
}

func TestResolveVLLMProfileGeometryIgnoresUnrelatedLiteralArguments(t *testing.T) {
	source := newTestVLLMProfileGeometrySource(
		[]string{"python3"},
		[]string{
			"-m", "dynamo.vllm",
			"-tp", "4",
			"-pp", "1",
			"-dp", "8",
			"--served-model-name=$(MODEL_NAME)",
			"--chat-template=$$(CHAT_TEMPLATE)",
			"--chat-template", "{{ if $message }}hello world{{ end }}",
		},
	)

	_, err := ResolveVLLMProfileGeometry(source)
	require.Error(t, err)
	var sourceError *UnsupportedVLLMProfileSourceError
	require.ErrorAs(t, err, &sourceError)
	assert.Equal(t, UnsupportedVLLMProfileSourceReasonOwnershipMode, sourceError.Reason)
}

func TestResolveVLLMProfileGeometryDoesNotMutateSource(t *testing.T) {
	source := newTestVLLMProfileGeometrySource(
		[]string{"python3", "-m", "dynamo.vllm"},
		[]string{"-tp", "4", "-pp", "1", "-dp", "8", "-dpl", "1"},
	)
	source.Environment = map[string]string{vllmDPSizeEnvironment: "4"}
	before := cloneTestVLLMProfileGeometrySource(source)

	_, err := ResolveVLLMProfileGeometry(source)
	require.Error(t, err)
	assert.Equal(t, before, source)
}

func newTestVLLMProfileGeometrySource(command, args []string) VLLMProfileGeometrySource {
	return VLLMProfileGeometrySource{
		Command:                    append([]string(nil), command...),
		Args:                       append(append([]string(nil), args...), enableElasticEPFlag),
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
	if source.Environment != nil {
		cloned.Environment = make(map[string]string, len(source.Environment))
		for name, value := range source.Environment {
			cloned.Environment[name] = value
		}
	}
	return cloned
}

func removeTestVLLMFlag(args []string, flag string) []string {
	updated := make([]string, 0, len(args))
	for _, argument := range args {
		if argument != flag {
			updated = append(updated, argument)
		}
	}

	return updated
}
