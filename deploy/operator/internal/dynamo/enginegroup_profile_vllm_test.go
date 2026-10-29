/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestResolveVLLMElasticEPProfile(t *testing.T) {
	t.Log("describe native and Dynamo exec-form launches with equivalent external DP ownership")
	tests := []struct {
		name    string
		command []string
		args    []string
		tp      int64
		uni     bool
		auto    bool
	}{
		{name: "native CLI", command: []string{"/usr/local/bin/vllm", "serve", "model"}},
		{name: "native Python module", command: []string{"python3", "-m", "vllm.entrypoints.openai.api_server"}},
		{name: "Dynamo Python module", command: []string{"python", "-m", "dynamo.vllm"}},
		{name: "TP1 single-GPU allocation", command: []string{"vllm", "serve", "model"}, tp: 1},
		{name: "TP1 local uni executor", command: []string{"vllm", "serve", "model"}, tp: 1, uni: true},
		{name: "native automatic bootstrap", command: []string{"vllm", "serve", "model"}, auto: true},
		{name: "Dynamo automatic bootstrap", command: []string{"python3", "-m", "dynamo.vllm"}, auto: true},
		{
			name: "split Python module", command: []string{"python3"},
			args: append([]string{"-m", "vllm.entrypoints.openai.api_server"}, newTestVLLMProfileGeometrySource().Args...),
		},
		{
			name: "documented short aliases", command: []string{"vllm", "serve", "model"},
			args: []string{
				"-tp", "4", "-pp", "1", "-dp", "8", "-dpl", "1", "-dpn", "0", "-dpe",
				"-dpb", "mp", "--distributed-executor-backend", "mp", "--elastic-ep-max-dp-size", "16",
				"-ep", "--enable-eplb", "--enable-elastic-ep",
				"--eplb-config", `{"use_async":false,"communicator":"pynccl","num_redundant_experts":0}`,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build a concrete initial allocation declaration")
			source := newTestVLLMProfileGeometrySource()
			source.Command = test.command
			if test.args != nil {
				source.Args = test.args
			}
			if test.tp != 0 {
				source.MainContainerGPUs = test.tp
				source.Args = append(withoutTestVLLMProfileOptions(source.Args, "--tensor-parallel-size"), "--tensor-parallel-size="+strconv.FormatInt(test.tp, 10))
			}
			if test.uni {
				source.Args = append(withoutTestVLLMProfileOptions(source.Args, "--distributed-executor-backend"), "--distributed-executor-backend=uni")
			}
			if test.auto {
				source.Args = append(withoutTestVLLMProfileOptions(source.Args, "--data-parallel-size", "--data-parallel-rank"),
					"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size=8",
					"--data-parallel-rank="+engineGroupOrdinalRef, "--data-parallel-address="+engineGroupRendezvousRef)
			}

			t.Log("resolve the initial DP count separately from each Pod's fixed TP subgroup")
			profile, err := ResolveVLLMElasticEPProfile(source)
			require.NoError(t, err)

			t.Log("verify eight one-member allocations each own a complete fixed TP subgroup")
			assert.Equal(t, int32(8), profile.InitialReplicas)
			assert.Equal(t, int32(16), profile.MaximumReplicas)
			assert.Equal(t, "vllm", profile.Geometry.Backend)
			assert.Equal(t, source.MainContainerGPUs, profile.Geometry.GPUsPerReplica)
			assert.Equal(t, int32(1), profile.Geometry.PodsPerReplica)
			assert.Equal(t, enginegroup.CapacityRoleGeometry{
				Name: "main", Class: enginegroup.CapacityRoleClassRankOwningCapacity,
				EngineGPUsPerPod: source.MainContainerGPUs, DedicatedGPUs: true,
			}, profile.Geometry.CapacityRole)
			assert.Regexp(t, `^sha256:[0-9a-f]{64}$`, profile.Geometry.Fingerprint)
		})
	}
}

func TestVLLMProfilePackedAllocations(t *testing.T) {
	for _, test := range []struct {
		name             string
		tp, width        int64
		initial, maximum int32
	}{
		{name: "two TP1 members", tp: 1, width: 2, initial: 1, maximum: 2},
		{name: "four TP1 members", tp: 1, width: 4, initial: 2, maximum: 3},
		{name: "two TP2 members", tp: 2, width: 2, initial: 1, maximum: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("declare native automatic bootstrap with a complete Pod-local engine range")
			source := newTestVLLMProfileGeometrySource()
			source.InitialReplicas, source.MainContainerGPUs = test.initial, test.tp*test.width
			source.Args = append(withoutTestVLLMProfileOptions(source.Args,
				"--tensor-parallel-size", "--data-parallel-size", "--data-parallel-size-local",
				"--data-parallel-rank", "--data-parallel-external-lb", "--elastic-ep-max-dp-size"),
				"--elastic-ep-auto-bootstrap", "--elastic-ep-allocation-index="+engineGroupOrdinalRef,
				fmt.Sprintf("--tensor-parallel-size=%d", test.tp), fmt.Sprintf("--data-parallel-size-local=%d", test.width),
				fmt.Sprintf("--elastic-ep-initial-dp-size=%d", int64(test.initial)*test.width),
				fmt.Sprintf("--elastic-ep-max-dp-size=%d", int64(test.maximum)*test.width))

			t.Log("resolve allocation counts independently from physical GPUs and native membership")
			profile, err := ResolveVLLMElasticEPProfile(source)
			require.NoError(t, err)
			assert.Equal(t, test.initial, profile.InitialReplicas)
			assert.Equal(t, test.maximum, profile.MaximumReplicas)
			assert.Equal(t, int32(test.width), profile.NativeMembersPerReplica)
			assert.Equal(t, int32(1), profile.Geometry.PodsPerReplica)
			assert.Equal(t, test.tp*test.width, profile.Geometry.GPUsPerReplica)

			t.Log("reject a reserved maximum that cuts an indivisible allocation")
			source.Args = append(withoutTestVLLMProfileOptions(source.Args, "--elastic-ep-max-dp-size"),
				fmt.Sprintf("--elastic-ep-max-dp-size=%d", int64(test.maximum)*test.width+1))
			_, err = ResolveVLLMElasticEPProfile(source)
			require.ErrorContains(t, err, "complete")
		})
	}
}

func TestVLLMProfileEffectiveOwnership(t *testing.T) {
	t.Log("exercise EngineArgs ownership, including implicit external and hybrid-to-external modes")
	tests := []struct {
		name    string
		remove  []string
		append  []string
		wantErr string
	}{
		{name: "explicit rank implies external", remove: []string{"--data-parallel-external-lb"}},
		{name: "external mode defaults local DP to one", remove: []string{"--data-parallel-size-local"}},
		{
			name:   "start rank and explicit local DP infer external",
			remove: []string{"--data-parallel-external-lb", "--data-parallel-rank"},
			append: []string{"--data-parallel-start-rank=3"},
		},
		{
			name:   "hybrid with width one switches to external",
			remove: []string{"--data-parallel-external-lb", "--data-parallel-rank"},
			append: []string{"--data-parallel-hybrid-lb"},
		},
		{
			name:    "local DP alone still manages global membership",
			remove:  []string{"--data-parallel-external-lb", "--data-parallel-rank"},
			wantErr: "local DP=1 alone does not establish external ownership",
		},
		{
			name: "external flag still needs a rank", remove: []string{"--data-parallel-rank"},
			wantErr: "external LB requires an explicit data-parallel rank",
		},
		{
			name:   "start rank without explicit local size is not external",
			remove: []string{"--data-parallel-external-lb", "--data-parallel-rank", "--data-parallel-size-local"},
			append: []string{"--data-parallel-start-rank=0"}, wantErr: "internal LB manages the global DP set",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("express the same allocation through the selected DP launch mode")
			source := newTestVLLMProfileGeometrySource()
			source.Args = withoutTestVLLMProfileOptions(source.Args, test.remove...)
			source.Args = append(source.Args, test.append...)

			t.Log("resolve effective ownership rather than treating local DP as proof")
			profile, err := ResolveVLLMElasticEPProfile(source)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, int64(4), profile.Geometry.GPUsPerReplica)
		})
	}
}

func TestVLLMProfileIdentity(t *testing.T) {
	t.Log("resolve a canonical fixed-TP profile and retain its immutable identity")
	baseline, err := ResolveVLLMElasticEPProfile(newTestVLLMProfileGeometrySource())
	require.NoError(t, err)
	tests := []struct {
		name          string
		replace       []string
		remove        []string
		initial       int32
		gpus          int64
		workload      string
		deriveInitial bool
		wantDifferent bool
	}{
		{name: "derive initial size from launch", deriveInitial: true},
		{name: "smaller initial world under same reservation", replace: []string{"--data-parallel-size=4"}, initial: 4},
		{name: "another initial rank", replace: []string{"--data-parallel-rank=6"}},
		{name: "rank-implied external mode", remove: []string{"--data-parallel-external-lb"}},
		{name: "implicit local width", remove: []string{"--data-parallel-size-local"}},
		{name: "documented mp DP default", remove: []string{"--data-parallel-backend"}},
		{name: "normalized underscore spelling", replace: []string{"--tensor_parallel_size=4"}, remove: []string{"--tensor-parallel-size"}},
		{name: "canonical EPLB JSON", replace: []string{`--eplb-config={"num_redundant_experts":0,"communicator":"pynccl","use_async":false}`}},
		{name: "different fixed TP", replace: []string{"--tensor-parallel-size=8"}, gpus: 8, wantDifferent: true},
		{name: "different reserved maximum", replace: []string{"--elastic-ep-max-dp-size=32"}, wantDifferent: true},
		{name: "different expert storage", replace: []string{`--eplb-config={"num_redundant_experts":8,"communicator":"pynccl","use_async":false}`}, wantDifferent: true},
		{name: "different communication backend", replace: []string{"--all2all-backend=deepep_low_latency"}, wantDifferent: true},
		{name: "different immutable workload", workload: "sha256:other-image-model", wantDifferent: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("change syntax, launch identity, or an immutable profile dimension")
			source := newTestVLLMProfileGeometrySource()
			source.Args = withoutTestVLLMProfileOptions(source.Args, test.remove...)
			for _, argument := range test.replace {
				name, _, _ := strings.Cut(argument, "=")
				source.Args = append(withoutTestVLLMProfileOptions(source.Args, name), argument)
			}
			if test.initial != 0 {
				source.InitialReplicas = test.initial
			}
			if test.deriveInitial {
				source.InitialReplicas = 0
			}
			if test.gpus != 0 {
				source.MainContainerGPUs = test.gpus
			}
			if test.workload != "" {
				source.WorkloadRevisionDigest = test.workload
			}

			t.Log("compare canonical identity independently of initial and live replica counts")
			profile, err := ResolveVLLMElasticEPProfile(source)
			require.NoError(t, err)
			if test.wantDifferent {
				assert.NotEqual(t, baseline.Geometry.Fingerprint, profile.Geometry.Fingerprint)
			} else {
				assert.Equal(t, baseline.Geometry.Fingerprint, profile.Geometry.Fingerprint)
			}
		})
	}
}

func TestVLLMProfileRejectsUnsupportedSources(t *testing.T) {
	t.Log("enumerate declarations whose static launch or physical mapping cannot be proven")
	tests := []struct {
		name    string
		replace []string
		remove  []string
		append  []string
		command []string
		gpus    int64
		shared  bool
		wantErr string
	}{
		{name: "Ray executor", replace: []string{"--distributed-executor-backend=ray"}, wantErr: "Pod-local mp executor"},
		{name: "Ray DP backend", replace: []string{"--data-parallel-backend=ray"}, wantErr: "mp DP backend"},
		{name: "external launcher", replace: []string{"--distributed-executor-backend=external_launcher"}, wantErr: "Pod-local mp executor"},
		{name: "auto executor can select Ray", remove: []string{"--distributed-executor-backend"}, wantErr: "explicit Pod-local"},
		{name: "uni cannot host TP4", replace: []string{"--distributed-executor-backend=uni"}, wantErr: "Pod-local mp executor"},
		{name: "packed without native bootstrap", replace: []string{"--data-parallel-size-local=2"}, wantErr: "native automatic bootstrap"},
		{name: "unclaimed GPUs", gpus: 8, wantErr: "TP times local DP"},
		{name: "multi-Pod TP allocation", gpus: 2, wantErr: "TP times local DP"},
		{name: "shared GPU allocation", shared: true, wantErr: "must be dedicated"},
		{name: "pipeline parallelism", append: []string{"--pipeline-parallel-size=2"}, wantErr: "PP=CP=1"},
		{name: "prefill CP", append: []string{"--prefill-context-parallel-size=2"}, wantErr: "PP=CP=1"},
		{name: "decode CP", append: []string{"--decode-context-parallel-size=2"}, wantErr: "PP=CP=1"},
		{name: "multiple local nodes", append: []string{"--nnodes=2"}, wantErr: "one local node"},
		{name: "multiple API processes", append: []string{"--api-server-count=2"}, wantErr: "one API server"},
		{name: "multi-port supervisor", append: []string{"--data-parallel-multi-port-external-lb"}, wantErr: "outside this profile"},
		{name: "fault tolerance is another protocol", append: []string{"--enable-fault-tolerance"}, wantErr: "outside this profile"},
		{name: "headless process", append: []string{"--headless"}, wantErr: "outside this profile"},
		{name: "device selection changes visible capacity", append: []string{"--device-ids=0,1"}, wantErr: "outside this profile"},
		{name: "no Elastic EP", remove: []string{"--enable-elastic-ep"}, wantErr: "--enable-elastic-ep is required"},
		{name: "no EPLB", remove: []string{"--enable-eplb"}, wantErr: "--enable-eplb is required"},
		{name: "no expert parallelism", remove: []string{"--enable-expert-parallel"}, wantErr: "--enable-expert-parallel is required"},
		{name: "initial DP fallback", replace: []string{"--data-parallel-size=1"}, wantErr: "initial DP > 1"},
		{name: "missing DP cannot use target as geometry", remove: []string{"--data-parallel-size"}, wantErr: "--data-parallel-size must be explicit"},
		{name: "missing reserved maximum", remove: []string{"--elastic-ep-max-dp-size"}, wantErr: "--elastic-ep-max-dp-size must be explicit"},
		{name: "automatic bootstrap without seed", remove: []string{"--data-parallel-size"}, append: []string{"--elastic-ep-auto-bootstrap"}, wantErr: "requires --elastic-ep-initial-dp-size"},
		{name: "automatic bootstrap with live DP target", append: []string{"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size=8"}, wantErr: "cannot include --data-parallel-size"},
		{name: "rank outside initial world", replace: []string{"--data-parallel-rank=8"}, wantErr: "must be in [0, 8)"},
		{name: "reserved maximum below initial", replace: []string{"--elastic-ep-max-dp-size=4"}, wantErr: "maximum must cover initial"},
		{name: "target outside API range", replace: []string{"--elastic-ep-max-dp-size=2147483648"}, wantErr: "replica range"},
		{name: "zero TP", replace: []string{"--tensor-parallel-size=0"}, wantErr: "invalid vLLM option"},
		{name: "numeric overflow", replace: []string{"--tensor-parallel-size=9223372036854775808"}, wantErr: "invalid vLLM option"},
		{name: "both LB modes", append: []string{"--data-parallel-hybrid-lb"}, wantErr: "cannot both be enabled"},
		{name: "implicit external plus hybrid", remove: []string{"--data-parallel-external-lb"}, append: []string{"--data-parallel-hybrid-lb"}, wantErr: "cannot both be enabled"},
		{name: "rank and start rank conflict", append: []string{"--data-parallel-start-rank=0"}, wantErr: "cannot both be enabled"},
		{name: "external YAML", append: []string{"--config=engine.yaml"}, wantErr: "must be resolved before"},
		{name: "abbreviated YAML", append: []string{"--conf=engine.yaml"}, wantErr: "must be resolved before"},
		{name: "nested EPLB override", append: []string{"--eplb-config.num_redundant_experts=8"}, wantErr: "must be resolved before"},
		{name: "invalid EPLB object", replace: []string{"--eplb-config=null"}, wantErr: "must be a JSON object"},
		{name: "abbreviated DP option", remove: []string{"--data-parallel-size"}, append: []string{"--data-parallel-siz=8"}, wantErr: "not an exact"},
		{name: "duplicate aliases", append: []string{"-tp=4"}, wantErr: "specified more than once"},
		{name: "Boolean negation", remove: []string{"--enable-elastic-ep"}, append: []string{"--disable-elastic-ep"}, wantErr: "--enable-elastic-ep is required"},
		{name: "Boolean conflict", append: []string{"--no-enable-elastic-ep"}, wantErr: "specified more than once"},
		{name: "Boolean with a value", remove: []string{"--enable-elastic-ep"}, append: []string{"--enable-elastic-ep=true"}, wantErr: "does not accept a value"},
		{name: "expanded geometry", replace: []string{"--data-parallel-size=$(DP_SIZE)"}, wantErr: "literal value"},
		{name: "option-bearing expansion", append: []string{"$(ENGINE_FLAGS)"}, wantErr: "expand into geometry"},
		{name: "shell entrypoint", command: []string{"bash", "-c", "vllm serve model"}, wantErr: "shell-or-wrapper"},
		{name: "opaque launcher", command: []string{"/scripts/start-engine"}, wantErr: "expected vllm serve"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("modify one supported declaration into the unsupported shape")
			source := newTestVLLMProfileGeometrySource()
			source.Args = withoutTestVLLMProfileOptions(source.Args, test.remove...)
			for _, argument := range test.replace {
				name, _, _ := strings.Cut(argument, "=")
				source.Args = append(withoutTestVLLMProfileOptions(source.Args, name), argument)
			}
			source.Args = append(source.Args, test.append...)
			if test.command != nil {
				source.Command = test.command
			}
			if test.gpus != 0 {
				source.MainContainerGPUs = test.gpus
			}
			if test.shared {
				source.DedicatedMainGPUAllocation = false
			}

			t.Log("reject rather than constructing a speculative allocation mapping")
			_, err := ResolveVLLMElasticEPProfile(source)
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

func TestVLLMProfileEnvironment(t *testing.T) {
	t.Log("exercise native environment fallback and unresolved provider inputs")
	tests := []struct {
		name    string
		dp      string
		env     []corev1.EnvVar
		envFrom []corev1.EnvFromSource
		wantErr string
	}{
		{
			name: "explicit DP greater than one bypasses native DP environment",
			env:  []corev1.EnvVar{{Name: "VLLM_DP_SIZE", Value: "99"}, {Name: "VLLM_DP_RANK", Value: "98"}},
		},
		{
			name: "explicit DP one still activates fallback", dp: "1",
			env: []corev1.EnvVar{{Name: "VLLM_DP_SIZE", Value: "8"}}, wantErr: "initial DP > 1",
		},
		{
			name: "missing global DP cannot rely on environment", dp: "omit",
			env: []corev1.EnvVar{{Name: "VLLM_DP_SIZE", Value: "8"}}, wantErr: "must be explicit",
		},
		{
			name: "opaque environment import", envFrom: []corev1.EnvFromSource{{
				ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "engine"},
			}}, wantErr: "envFrom must be resolved",
		},
		{
			name: "unresolved native rank", env: []corev1.EnvVar{{Name: "VLLM_DP_RANK", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
			}}}, wantErr: "must be resolved to a literal",
		},
		{name: "expanded native DP", env: []corev1.EnvVar{{Name: "VLLM_DP_SIZE", Value: "$(DP)"}}, wantErr: "must be resolved to a literal"},
		{name: "joiner is not an initial launch", env: []corev1.EnvVar{{Name: "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", Value: "1"}}, wantErr: "conflicts with the initial"},
		{name: "Rust frontend does not expose the protocol", env: []corev1.EnvVar{{Name: "VLLM_USE_RUST_FRONTEND", Value: "1"}}, wantErr: "conflicts with the initial"},
		{name: "in-process engine mode", env: []corev1.EnvVar{{Name: "VLLM_ENABLE_V1_MULTIPROCESSING", Value: "0"}}, wantErr: "conflicts with the initial"},
		{name: "native request drain enabled", env: []corev1.EnvVar{{Name: vllmElasticEPDrainEnvVar, Value: "1"}}},
		{name: "native request drain disabled", env: []corev1.EnvVar{{Name: vllmElasticEPDrainEnvVar, Value: "0"}}, wantErr: "conflicts with the initial"},
		{name: "native request drain unresolved", env: []corev1.EnvVar{{Name: vllmElasticEPDrainEnvVar, ValueFrom: &corev1.EnvVarSource{
			ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "engine", Key: "drain"},
		}}}, wantErr: "must be resolved to a literal"},
		{name: "native request drain expanded", env: []corev1.EnvVar{{Name: vllmElasticEPDrainEnvVar, Value: "$(DRAIN)"}}, wantErr: "must be resolved to a literal"},
		{name: "unrelated downward API is allowed", env: []corev1.EnvVar{{Name: "POD_IP", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}}}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("provide the effective environment independently of command arguments")
			source := newTestVLLMProfileGeometrySource()
			source.Env, source.EnvFrom = test.env, test.envFrom
			if test.dp != "" {
				source.Args = withoutTestVLLMProfileOptions(source.Args, "--data-parallel-size")
				if test.dp != "omit" {
					source.Args = append(source.Args, "--data-parallel-size="+test.dp)
				}
			}

			t.Log("never substitute requested replicas for opaque effective engine geometry")
			profile, err := ResolveVLLMElasticEPProfile(source)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				assert.ErrorIs(t, err, enginegroup.ErrUnsupportedProfile)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, int32(8), profile.InitialReplicas)
		})
	}
}

func TestVLLMProfileCreationSeed(t *testing.T) {
	t.Log("distinguish immutable launch size from a caller-provided initial allocation count")
	tests := []struct {
		name    string
		initial int32
		wantErr string
	}{
		{name: "derived", initial: 0},
		{name: "matching", initial: 8},
		{name: "negative", initial: -1, wantErr: "must not be negative"},
		{name: "conflicting", initial: 4, wantErr: "conflict with vLLM launch DP size 8"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("supply the creation seed without changing the engine's launch declaration")
			source := newTestVLLMProfileGeometrySource()
			source.InitialReplicas = test.initial

			t.Log("derive or check the seed, never reinterpret it as a live resize")
			profile, err := ResolveVLLMElasticEPProfile(source)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, int32(8), profile.InitialReplicas)
		})
	}
}

func TestVLLMProfileDoesNotMutateInputs(t *testing.T) {
	t.Log("snapshot a source containing caller-owned argv and environment slices")
	source := newTestVLLMProfileGeometrySource()
	source.Env = []corev1.EnvVar{{Name: "VLLM_DP_SIZE", Value: "99"}}
	before, err := json.Marshal(source)
	require.NoError(t, err)

	t.Log("resolve the profile without rewriting commands or native environment")
	_, err = ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	after, err := json.Marshal(source)
	require.NoError(t, err)
	assert.Equal(t, string(before), string(after))
}

func newTestVLLMProfileGeometrySource() VLLMProfileGeometrySource {
	return VLLMProfileGeometrySource{
		Command: []string{"vllm", "serve", "model"},
		Args: []string{
			"--tensor-parallel-size=4", "--data-parallel-size=8", "--data-parallel-size-local=1",
			"--data-parallel-rank=0", "--data-parallel-external-lb", "--data-parallel-backend=mp",
			"--distributed-executor-backend=mp", "--elastic-ep-max-dp-size=16",
			"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep",
			`--eplb-config={"use_async":false,"communicator":"pynccl","num_redundant_experts":0}`,
		},
		InitialReplicas: 8, MainContainerGPUs: 4, DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest: "sha256:immutable-image-model",
	}
}

func withoutTestVLLMProfileOptions(args []string, names ...string) []string {
	var result []string
	for _, token := range args {
		name, _, _ := strings.Cut(token, "=")
		remove := false
		for _, candidate := range names {
			remove = remove || name == candidate
		}
		if !remove {
			result = append(result, token)
		}
	}
	return result
}
