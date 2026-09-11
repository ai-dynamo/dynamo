/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestValidateCheckpointCompatibility(t *testing.T) {
	tests := []struct {
		name         string
		experimental *nvidiacomv1beta1.ExperimentalSpec
		wantErrs     []string
	}{
		{name: "no experimental features"},
		{
			name: "no checkpoint configuration",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
				Failover:         &nvidiacomv1beta1.FailoverSpec{},
			},
		},
		{
			name: "disabled checkpoint ignores incompatible settings",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
				Failover:         &nvidiacomv1beta1.FailoverSpec{},
			},
		},
		{
			name: "enabled checkpoint with intra-pod GMS",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
			},
		},
		{
			name: "enabled checkpoint with inter-pod GMS and failover",
			experimental: &nvidiacomv1beta1.ExperimentalSpec{
				Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeInterPod},
				Failover:         &nvidiacomv1beta1.FailoverSpec{},
			},
			wantErrs: []string{
				checkpointInterPodCompatibilityMessage,
				checkpointFailoverCompatibilityMessage,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			violations := ValidateCheckpointCompatibility(test.experimental)
			var gotErrs []string
			for _, violation := range violations {
				gotErrs = append(gotErrs, violation.Error())
			}
			assert.Equal(t, test.wantErrs, gotErrs)
		})
	}
}

func TestComputeSnapshotCompatibilityHashIsPortableAcrossGraphIdentity(t *testing.T) {
	t.Log("Given equivalent capture targets rendered for two different DGD identities")
	first := snapshotCompatibilityTestPodTemplate("capture-dgd", "capture-ns", "worker-a")
	second := snapshotCompatibilityTestPodTemplate("restore-dgd", "restore-ns", "worker-b")
	first.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "network"}, {Name: "accelerator"}}
	second.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "accelerator"}, {Name: "network"}}
	first.Spec.Volumes = append(first.Spec.Volumes, corev1.Volume{Name: "helper-only"})
	first.Spec.Containers = append(first.Spec.Containers, corev1.Container{
		Name:         "checkpoint-helper",
		Image:        "helper:latest",
		VolumeMounts: []corev1.VolumeMount{{Name: "helper-only", MountPath: "/helper"}},
	})

	t.Log("When their snapshot compatibility hashes are computed")
	firstHash, err := ComputeSnapshotCompatibilityHash(&first, "main", "vllm", "disabled")
	require.NoError(t, err)
	secondHash, err := ComputeSnapshotCompatibilityHash(&second, "main", "vllm", "disabled")
	require.NoError(t, err)

	t.Log("Then graph, Pod, worker-generation, and helper-sidecar identity do not make the artifact incompatible")
	assert.Equal(t, firstHash, secondHash)
	assert.Len(t, firstHash, 64)
}

func TestComputeSnapshotCompatibilityHashRejectsProcessContractChanges(t *testing.T) {
	base := snapshotCompatibilityTestPodTemplate("capture-dgd", "capture-ns", "worker-a")
	baseHash, err := ComputeSnapshotCompatibilityHash(&base, "main", "vllm", "disabled")
	require.NoError(t, err)

	tests := []struct {
		name    string
		mutate  func(*corev1.PodTemplateSpec)
		backend string
		gmsMode string
	}{
		{
			name: "image",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.Containers[0].Image = "worker:2.0"
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name: "engine arguments",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.Containers[0].Args = append(template.Spec.Containers[0].Args, "--tensor-parallel-size=2")
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name: "model environment",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.Containers[0].Env = append(template.Spec.Containers[0].Env, corev1.EnvVar{Name: "MODEL_PATH", Value: "/models/other"})
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name: "GPU node selector",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.product": "H100"}
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name: "DRA resource claim",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.ResourceClaims = []corev1.PodResourceClaim{{Name: "accelerator"}}
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name: "PVC claim identity",
			mutate: func(template *corev1.PodTemplateSpec) {
				template.Spec.Containers[0].VolumeMounts = append(
					template.Spec.Containers[0].VolumeMounts,
					corev1.VolumeMount{Name: "model-cache", MountPath: "/models"},
				)
				template.Spec.Volumes = append(template.Spec.Volumes, corev1.Volume{
					Name: "model-cache",
					VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: "capture-dgd-model-cache",
					}},
				})
			},
			backend: "vllm",
			gmsMode: "disabled",
		},
		{
			name:    "backend",
			mutate:  func(*corev1.PodTemplateSpec) {},
			backend: "sglang",
			gmsMode: "disabled",
		},
		{
			name:    "GMS topology",
			mutate:  func(*corev1.PodTemplateSpec) {},
			backend: "vllm",
			gmsMode: "IntraPod",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			changed := base.DeepCopy()
			test.mutate(changed)
			changedHash, err := ComputeSnapshotCompatibilityHash(changed, "main", test.backend, test.gmsMode)
			require.NoError(t, err)
			assert.NotEqual(t, baseHash, changedHash)
		})
	}
}

func TestComputeSnapshotCompatibilityHashPreservesEnvironmentOrder(t *testing.T) {
	ordered := snapshotCompatibilityTestPodTemplate("capture-dgd", "capture-ns", "worker-a")
	ordered.Spec.Containers[0].Env = []corev1.EnvVar{
		{Name: "MODEL_ROOT", Value: "/models"},
		{Name: "MODEL_PATH", Value: "$(MODEL_ROOT)/model"},
		{Name: "DYN_NAMESPACE", Value: "capture-runtime"},
	}
	reordered := ordered.DeepCopy()
	reordered.Spec.Containers[0].Env[0], reordered.Spec.Containers[0].Env[1] =
		reordered.Spec.Containers[0].Env[1], reordered.Spec.Containers[0].Env[0]

	orderedHash, err := ComputeSnapshotCompatibilityHash(&ordered, "main", "vllm", "disabled")
	require.NoError(t, err)
	reorderedHash, err := ComputeSnapshotCompatibilityHash(reordered, "main", "vllm", "disabled")
	require.NoError(t, err)

	assert.NotEqual(t, orderedHash, reorderedHash,
		"Kubernetes expands $(VAR_NAME) from earlier entries, so environment order is part of the process contract")
}

func TestSnapshotRestoreEnvironmentNamesMatchPythonRuntime(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	require.True(t, ok)
	constantsPath := filepath.Join(
		filepath.Dir(thisFile),
		"../../../../components/src/dynamo/common/snapshot/constants.py",
	)
	contents, err := os.ReadFile(constantsPath)
	if os.IsNotExist(err) && os.Getenv("DYNAMO_REQUIRE_SNAPSHOT_ENV_PARITY") == "" {
		t.Skip("Python snapshot constants are not present in the operator-only build context")
	}
	require.NoError(t, err)

	pythonNames := map[string]struct{}{}
	quotedName := regexp.MustCompile(`"([A-Z][A-Z0-9_]*)"`)
	for _, variable := range []string{
		"KUBERNETES_REQUIRED_ENV_NAMES",
		"KUBERNETES_OPTIONAL_ENV_NAMES",
		"RESTORE_RUNTIME_ENV_NAMES",
	} {
		assignment := regexp.MustCompile(`(?ms)^` + regexp.QuoteMeta(variable) + `\s*=\s*\{(.*?)\}`)
		match := assignment.FindSubmatch(contents)
		require.Len(t, match, 2, "find Python assignment for %s", variable)
		for _, name := range quotedName.FindAllSubmatch(match[1], -1) {
			pythonNames[string(name[1])] = struct{}{}
		}
	}

	assert.Equal(t, pythonNames, snapshotRestoreEnvironmentNames,
		"Go compatibility filtering must track the Python restore-context allowlist")
}

func snapshotCompatibilityTestPodTemplate(dgdName, namespace, workerSuffix string) corev1.PodTemplateSpec {
	return corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "main",
				Image:   "worker:1.0",
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Args:    []string{"--model", "/models/model"},
				Env: []corev1.EnvVar{
					{Name: "MODEL_PATH", Value: "/models/model"},
					{Name: "DYN_PARENT_DGD_K8S_NAME", Value: dgdName},
					{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: namespace},
					{Name: "DYN_NAMESPACE", Value: namespace + "-runtime"},
					{Name: "DYN_NAMESPACE_WORKER_SUFFIX", Value: workerSuffix},
					{Name: "POD_NAME", Value: dgdName + "-pod"},
				},
			}},
		},
	}
}
