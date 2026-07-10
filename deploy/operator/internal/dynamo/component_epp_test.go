/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// TestEPPDefaults_GetBaseContainerSetsResourceRequests verifies that the EPP
// base container carries default resource requests. Without requests the pod is
// BestEffort QoS and is the first workload the kubelet evicts under node
// pressure (e.g. DiskPressure), which stalls the whole deployment because the
// InferencePool loses its endpoint picker.
func TestEPPDefaults_GetBaseContainerSetsResourceRequests(t *testing.T) {
	container, err := NewEPPDefaults().GetBaseContainer(ComponentContext{
		numberOfNodes:                  1,
		ParentGraphDeploymentName:      "qwen",
		ParentGraphDeploymentNamespace: "default",
		DynamoNamespace:                "default-qwen",
		ComponentType:                  commonconsts.ComponentTypeEPP,
	})
	require.NoError(t, err)

	requests := container.Resources.Requests
	require.NotEmpty(t, requests, "EPP container must set resource requests so it is not BestEffort QoS")

	cpu := requests[corev1.ResourceCPU]
	mem := requests[corev1.ResourceMemory]
	storage := requests[corev1.ResourceEphemeralStorage]

	assert.False(t, cpu.IsZero(), "EPP must request CPU")
	assert.False(t, mem.IsZero(), "EPP must request memory")
	assert.False(t, storage.IsZero(), "EPP must request ephemeral-storage to lower DiskPressure eviction ranking")

	assert.Equal(t, resource.MustParse("100m"), cpu)
	assert.Equal(t, resource.MustParse("256Mi"), mem)
	assert.Equal(t, resource.MustParse("1Gi"), storage)
}

// TestEPPDefaults_GetBaseContainerUsesSaneLogDefault verifies that the EPP
// default RUST_LOG level is not debug/trace. Verbose Rust logging streams to
// the container log, counts against the pod's ephemeral-storage request, and
// can trigger DiskPressure eviction. It must match the Rust EPP's documented
// default (info) and stay overridable per deployment.
func TestEPPDefaults_GetBaseContainerUsesSaneLogDefault(t *testing.T) {
	container, err := NewEPPDefaults().GetBaseContainer(ComponentContext{
		numberOfNodes:                  1,
		ParentGraphDeploymentName:      "qwen",
		ParentGraphDeploymentNamespace: "default",
		DynamoNamespace:                "default-qwen",
		ComponentType:                  commonconsts.ComponentTypeEPP,
	})
	require.NoError(t, err)

	var rustLog string
	found := false
	for _, env := range container.Env {
		if env.Name == "RUST_LOG" {
			rustLog = env.Value
			found = true
			break
		}
	}
	require.True(t, found, "EPP container must set a default RUST_LOG")
	assert.Equal(t, "info", rustLog)
	assert.NotContains(t, rustLog, "trace", "EPP must not default to trace-level logging")
	assert.NotContains(t, rustLog, "debug", "EPP must not default to debug-level logging")
}
