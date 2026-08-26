/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPopulateComponentGPUCounts(t *testing.T) {
	t.Log("Build scalar and single-node DRA worker components")
	scalarWorker := nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "scalar-worker",
		ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: commonconsts.MainContainerName,
			Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
				corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("4"),
			}},
		}}}},
	}
	draWorker := testDRAClaimComponent("gpu-template", true)
	draWorker.ComponentName = "dra-worker"
	draWorker.ComponentType = nvidiacomv1beta1.ComponentTypeDecode
	draWorker.Multinode = nil
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{scalarWorker, draWorker},
		},
	}

	t.Log("Expose the DRA template and DeviceClass through the cached reader")
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu.nvidia.com"}},
		&resourcev1.ResourceClaimTemplate{
			ObjectMeta: metav1.ObjectMeta{Name: "gpu-template", Namespace: "default"},
			Spec: resourcev1.ResourceClaimTemplateSpec{Spec: resourcev1.ResourceClaimSpec{
				Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
					Name: "gpu",
					Exactly: &resourcev1.ExactDeviceRequest{
						DeviceClassName: "gpu.nvidia.com",
						AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
						Count:           2,
					},
				}}},
			}},
		},
	).Build()
	statuses := map[string]nvidiacomv1beta1.ComponentReplicaStatus{
		"scalar-worker": {},
		"dra-worker":    {},
	}

	t.Log("Publish the same per-Pod count shape for scalar and DRA workers")
	require.NoError(t, populateComponentGPUCounts(t.Context(), reader, dgd, statuses))
	require.NotNil(t, statuses["scalar-worker"].GPUCountPerPod)
	require.NotNil(t, statuses["dra-worker"].GPUCountPerPod)
	assert.Equal(t, int64(4), *statuses["scalar-worker"].GPUCountPerPod)
	assert.Equal(t, int64(2), *statuses["dra-worker"].GPUCountPerPod)
}

func TestPopulateComponentGPUCountsClearsStaleCountOnResolutionError(t *testing.T) {
	t.Log("Build a valid DRA worker and dependency set")
	draWorker := testDRAClaimComponent("gpu-template", true)
	draWorker.ComponentName = "dra-worker"
	draWorker.ComponentType = nvidiacomv1beta1.ComponentTypeDecode
	draWorker.Multinode = nil
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{draWorker},
		},
	}
	statuses := map[string]nvidiacomv1beta1.ComponentReplicaStatus{
		"dra-worker": {},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	claimTemplate := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-template", Namespace: "default"},
		Spec: resourcev1.ResourceClaimTemplateSpec{Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
				Name: "gpu",
				Exactly: &resourcev1.ExactDeviceRequest{
					DeviceClassName: "gpu.nvidia.com",
					AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
					Count:           2,
				},
			}}},
		}},
	}
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu.nvidia.com"}},
		claimTemplate,
	).Build()

	t.Log("Publish the valid count before the dependency disappears")
	require.NoError(t, populateComponentGPUCounts(t.Context(), reader, dgd, statuses))
	require.NotNil(t, statuses["dra-worker"].GPUCountPerPod)
	assert.Equal(t, int64(2), *statuses["dra-worker"].GPUCountPerPod)
	require.NoError(t, reader.Delete(t.Context(), claimTemplate))

	t.Log("Fail the next resolution without leaving the old count available to Planner")
	err := populateComponentGPUCounts(t.Context(), reader, dgd, statuses)
	require.ErrorContains(t, err, "gpu-template")
	assert.Nil(t, statuses["dra-worker"].GPUCountPerPod)
}
