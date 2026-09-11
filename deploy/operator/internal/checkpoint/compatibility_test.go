/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
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
