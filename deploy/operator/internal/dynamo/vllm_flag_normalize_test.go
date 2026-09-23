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
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

func vllmContainer(args ...string) *corev1.Container {
	return &corev1.Container{
		Name:    "main",
		Command: []string{"python3"},
		Args:    append([]string{"-m", "dynamo.vllm", "--model", "m"}, args...),
	}
}

func TestNormalizeVLLMFlags_EverySpellingReadsTheSame(t *testing.T) {
	for _, tc := range []struct {
		flag  string
		short string
	}{
		{tensorParallelSizeFlag, "-tp"},
		{pipelineParallelSizeFlag, "-pp"},
		{dataParallelSizeFlag, "-dp"},
	} {
		spellings := map[string][]string{
			"long equals":     {tc.flag + "=4"},
			"short separated": {tc.short, "4"},
			"short equals":    {tc.short + "=4"},
		}
		for name, args := range spellings {
			t.Run(fmt.Sprintf("%s/%s", tc.flag, name), func(t *testing.T) {
				got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tc.flag)
				if err != nil {
					t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
				}
				if got != 4 {
					t.Errorf("%s spelled %q read as %d, want 4 -- vLLM accepts all of these forms, "+
						"so every reader in this package must too", tc.flag, args, got)
				}
			})
		}
	}
}

// TestNormalizeVLLMFlags_QualifyingLaunchAlsoSizes pins the asymmetry that motivated this
// fix, at the level available on this branch.
//
// IsElasticEPRayLaunch decides WHETHER a launch is elastic EP; it uses hasArg, which already
// understood "-dpb" and the equals form. The width of that launch is read separately, through
// getFlagValue, which did not. So a command line could qualify as elastic EP and read as one
// rank -- and a one-rank reading renders no extra pods, no local-rank pin and no width gate
// while vLLM still places N ranks, which is the abort those three exist to prevent.
func TestNormalizeVLLMFlags_QualifyingLaunchAlsoSizes(t *testing.T) {
	for name, args := range map[string][]string{
		"equals flags": {"--enable-elastic-ep", dataParallelBackendFlag + "=ray", dataParallelSizeFlag + "=4"},
		"short flags":  {"--enable-elastic-ep", "-dpb", "ray", "-dp", "4"},
	} {
		t.Run(name, func(t *testing.T) {
			container := vllmContainer(args...)
			if !IsElasticEPRayLaunch(container) {
				t.Fatalf("spelling %q should qualify as an elastic-EP Ray launch", args)
			}
			got, err := getFlagValue(getExpandedCommandLine(container), dataParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("qualified as elastic EP but its declared width read as %d, want 4. A shape "+
					"that qualifies must also size correctly, or the leader renders with no width "+
					"gate and vLLM aborts placing 4 ranks on one pod", got)
			}
		})
	}
}

// TestNormalizeVLLMFlags_WorldSizeReadsShortAndEqualsForms covers the multinode consumer.
// vllmLaunchArgs.WorldSize multiplies tensor by pipeline size, and it decides
// data-parallel-size-local and whether multinode coordination is injected at all -- so a
// size silently read as 1 is not a cosmetic miss.
func TestNormalizeVLLMFlags_WorldSizeReadsShortAndEqualsForms(t *testing.T) {
	for name, args := range map[string][]string{
		"equals": {tensorParallelSizeFlag + "=4", pipelineParallelSizeFlag + "=2"},
		"short":  {"-tp", "4", "-pp", "2"},
	} {
		t.Run(name, func(t *testing.T) {
			got := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(args...))).WorldSize()
			if got != 8 {
				t.Errorf("WorldSize() = %d, want 8 (tp 4 x pp 2) for spelling %q", got, args)
			}
		})
	}
}

func TestNormalizeVLLMFlags_LeavesEverythingElseAlone(t *testing.T) {
	in := []string{"python3", "-m", "dynamo.vllm", "--model", "deepseek-ai/DeepSeek-V2-Lite",
		"--some-future-flag=value", "--trust-remote-code", "-dp", "4"}
	got := normalizeVLLMFlags(in)

	want := []string{"python3", "-m", "dynamo.vllm", "--model", "deepseek-ai/DeepSeek-V2-Lite",
		"--some-future-flag=value", "--trust-remote-code", dataParallelSizeFlag, "4"}
	if len(got) != len(want) {
		t.Fatalf("normalizeVLLMFlags(%q) = %q, want %q", in, got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("token %d = %q, want %q (full: %q)", i, got[i], want[i], got)
		}
	}
}

// TestNormalizeVLLMFlags_DoesNotSpoofFlagsFromUnrelatedValues guards the injection this
// restriction prevents: an equals-form value that happens to spell a recognized flag must not
// become a standalone token, or an exact-match reader like hasFlag/IsElasticEPRayLaunch would
// wrongly treat it as the user requesting that flag.
func TestNormalizeVLLMFlags_DoesNotSpoofFlagsFromUnrelatedValues(t *testing.T) {
	container := vllmContainer("--served-model-name=--enable-elastic-ep",
		dataParallelBackendFlag, "ray")
	if IsElasticEPRayLaunch(container) {
		t.Fatal("--served-model-name's value must not be read as --enable-elastic-ep")
	}
}

// TestNormalizeVLLMFlags_UnderscoreSpellingReadsTheSame covers vLLM's FlexibleArgumentParser
// treating "_" and "-" as interchangeable in long option names: "--tensor_parallel_size" must
// read the same as "--tensor-parallel-size", separated or equals-form, or getFlagValue
// silently falls back to 1 -- the same topology mismatch this PR fixes for short/equals forms.
func TestNormalizeVLLMFlags_UnderscoreSpellingReadsTheSame(t *testing.T) {
	for name, args := range map[string][]string{
		"separated": {"--tensor_parallel_size", "4"},
		"equals":    {"--tensor_parallel_size=4"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tensorParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("getFlagValue(%q) = %d, want 4 -- vLLM treats \"_\" and \"-\" as "+
					"interchangeable in long option names", args, got)
			}
		})
	}
}

// TestNormalizeVLLMFlags_UnderscoreSpellingQualifiesElasticEP covers the same underscore
// interchangeability for the boolean --enable_elastic_ep and the --data_parallel_backend
// qualifier IsElasticEPRayLaunch reads via hasFlag/hasArg.
func TestNormalizeVLLMFlags_UnderscoreSpellingQualifiesElasticEP(t *testing.T) {
	container := vllmContainer("--enable_elastic_ep", "--data_parallel_backend=ray")
	if !IsElasticEPRayLaunch(container) {
		t.Fatal("underscore-spelled --enable_elastic_ep/--data_parallel_backend should qualify as an elastic-EP Ray launch")
	}
}

// TestParseVLLMLaunchArgs_EnumFlagsUseLastOccurrence guards against reading an
// enum-style flag's first occurrence instead of vLLM's actual last-value-wins
// resolution. hasArg's first-match semantics would report "ray" is set for
// "--data-parallel-backend ray --data-parallel-backend mp", but vLLM resolves
// the backend to "mp" -- so IsElasticEPRayLaunch must not qualify this as an
// elastic-EP Ray launch, and the symmetric case must not read as mp either.
func TestParseVLLMLaunchArgs_EnumFlagsUseLastOccurrence(t *testing.T) {
	t.Run("data-parallel-backend resolves to the final occurrence, not the first", func(t *testing.T) {
		container := vllmContainer("--enable-elastic-ep",
			dataParallelBackendFlag, "ray", dataParallelBackendFlag, "mp")
		if IsElasticEPRayLaunch(container) {
			t.Fatal("effective --data-parallel-backend is mp (the last occurrence); must not qualify as an elastic-EP Ray launch")
		}
	})
	t.Run("distributed-executor-backend resolves to the final occurrence, not the first", func(t *testing.T) {
		args := parseVLLMLaunchArgs(getExpandedCommandLine(
			vllmContainer(distributedExecutorFlag, "mp", distributedExecutorFlag, "ray")))
		if args.DistributedExecutorBackendIsMp {
			t.Fatal("effective --distributed-executor-backend is ray (the last occurrence); DistributedExecutorBackendIsMp must be false")
		}
	})
}

// TestValidateParallelismSizes_RejectsNonPositive guards against a malformed
// "--tensor-parallel-size 0" or a negative parallelism value: nothing in
// getFlagValue rejects zero or negative integers, and letting one through
// would either divide by zero in injectDataParallelLaunchFlags or silently
// make the multinode gating functions decide no coordination is needed at
// all, letting the malformed flag reach vLLM unmodified with no
// operator-side error.
func TestValidateParallelismSizes_RejectsNonPositive(t *testing.T) {
	for name, rawArgs := range map[string][]string{
		"zero tensor-parallel-size":     {tensorParallelSizeFlag, "0"},
		"negative tensor-parallel-size": {tensorParallelSizeFlag, "-1"},
		"negative pipeline-parallel-size combined": {
			tensorParallelSizeFlag, "2", pipelineParallelSizeFlag, "-1",
		},
		"zero data-parallel-size":     {dataParallelSizeFlag, "0"},
		"negative data-parallel-size": {dataParallelSizeFlag, "-1"},
	} {
		t.Run(name, func(t *testing.T) {
			args := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(rawArgs...)))
			if err := args.ValidateParallelismSizes(); err == nil {
				t.Errorf("ValidateParallelismSizes() = nil, want an error for %q", rawArgs)
			}
		})
	}
}

// TestValidateParallelismSizes_AcceptsAbsentOrPositive confirms an absent
// flag (getFlagValue's default of 1) and explicit positive values are not
// errors -- only an explicit non-positive value is rejected.
func TestValidateParallelismSizes_AcceptsAbsentOrPositive(t *testing.T) {
	for name, rawArgs := range map[string][]string{
		"nothing set":              {},
		"positive tensor-parallel": {tensorParallelSizeFlag, "4"},
		"positive data-parallel":   {dataParallelSizeFlag, "8"},
		"all three set and positive": {
			tensorParallelSizeFlag, "2", pipelineParallelSizeFlag, "2", dataParallelSizeFlag, "4",
		},
	} {
		t.Run(name, func(t *testing.T) {
			args := parseVLLMLaunchArgs(getExpandedCommandLine(vllmContainer(rawArgs...)))
			if err := args.ValidateParallelismSizes(); err != nil {
				t.Errorf("ValidateParallelismSizes() = %v, want nil for %q", err, rawArgs)
			}
		})
	}
}

func TestGetFlagValue_RepeatedFlagUsesLastOccurrence(t *testing.T) {
	for name, args := range map[string][]string{
		"long then long":  {tensorParallelSizeFlag, "1", tensorParallelSizeFlag, "4"},
		"long then short": {tensorParallelSizeFlag, "1", "-tp", "4"},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tensorParallelSizeFlag)
			if err != nil {
				t.Fatalf("getFlagValue(%q) unexpected error: %v", args, err)
			}
			if got != 4 {
				t.Errorf("getFlagValue(%q) = %d, want 4 (last occurrence) -- vLLM's argparse "+
					"applies the last value when a flag is repeated", args, got)
			}
		})
	}
}

// TestGetFlagValue_InvalidValueErrors guards against silently keeping an
// earlier valid value when a later (or earlier) occurrence fails to parse.
// vLLM's own argparse rejects the whole command line the moment it hits an
// unparseable value, so the operator must not compute a topology from a
// number vLLM itself will never use.
func TestGetFlagValue_InvalidValueErrors(t *testing.T) {
	for name, args := range map[string][]string{
		"invalid then valid": {tensorParallelSizeFlag, "not-a-number", tensorParallelSizeFlag, "4"},
		"valid then invalid": {tensorParallelSizeFlag, "4", tensorParallelSizeFlag, "not-a-number"},
		"only invalid":       {tensorParallelSizeFlag, "not-a-number"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := getFlagValue(getExpandedCommandLine(vllmContainer(args...)), tensorParallelSizeFlag)
			if err == nil {
				t.Fatalf("getFlagValue(%q) = nil error, want an error -- vLLM would refuse to start on this value", args)
			}
		})
	}
}

// TestVLLMBackend_UpdateContainer_RejectsInvalidParallelismValue proves the
// error from a malformed integer flag reaches and halts UpdateContainer, the
// same way TestVLLMBackend_UpdateContainer_RejectsNonPositiveParallelismSizes
// proves it for a non-positive one.
func TestVLLMBackend_UpdateContainer_RejectsInvalidParallelismValue(t *testing.T) {
	backend := &VLLMBackend{}
	container := &corev1.Container{
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm", tensorParallelSizeFlag, "4", tensorParallelSizeFlag, "not-a-number"},
	}
	err := backend.UpdateContainer(container, 2, RoleLeader, betaComponent(t, &v1alpha1.DynamoComponentDeploymentSharedSpec{}), "test-service", &GroveMultinodeDeployer{}, staticContainerGPUCount(8))
	if err == nil {
		t.Fatal("UpdateContainer() = nil error, want an error for an unparseable --tensor-parallel-size value")
	}
}

// TestNormalizeVLLMFlags_ShortAliasIsNotSubstringMatched guards the obvious wrong
// implementation: rewriting by prefix or substring rather than by whole token. "-dpb" must not
// be read as "-dp" with a stray "b", and a longer flag that merely starts with a short alias
// must be left alone.
func TestNormalizeVLLMFlags_ShortAliasIsNotSubstringMatched(t *testing.T) {
	got := normalizeVLLMFlags([]string{"-dpb", "ray", "-dpx", "1", "--dp", "2"})
	want := []string{dataParallelBackendFlag, "ray", "-dpx", "1", "--dp", "2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d = %q, want %q (full: %q)", i, got[i], want[i], got)
		}
	}
}
