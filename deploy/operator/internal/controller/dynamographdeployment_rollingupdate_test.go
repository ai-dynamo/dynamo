/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"context"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

const (
	testOldWorkerHash = "oldhash1"
	testNewWorkerHash = "newhash2"
)

// createTestDGD creates a DynamoGraphDeployment for testing with the given services
func createTestDGD(name string, services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1alpha1.DynamoGraphDeployment {
	return &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: services,
		},
	}
}

// createTestReconciler creates a DynamoGraphDeploymentReconciler for testing
func createTestReconciler(objs ...runtime.Object) *DynamoGraphDeploymentReconciler {
	scheme := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(objs...).
		Build()

	return &DynamoGraphDeploymentReconciler{
		Client:   fakeClient,
		Recorder: record.NewFakeRecorder(10),
	}
}

func TestShouldTriggerRollingUpdate(t *testing.T) {
	tests := []struct {
		name         string
		services     map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		existingHash string // empty means no annotation, "compute" means compute from services
		expected     bool
	}{
		{
			name: "new deployment - no hash annotation",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
				},
			},
			existingHash: "",
			expected:     false,
		},
		{
			name: "hash unchanged - matches current spec",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
				},
			},
			existingHash: "compute",
			expected:     false,
		},
		{
			name: "hash changed - differs from current spec",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "FOO", Value: "new-value"}},
				},
			},
			existingHash: "old-hash-12345678",
			expected:     true,
		},
		{
			name: "frontend-only change - hash unchanged",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Envs:          []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "changed"}},
				},
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "WORKER_VAR", Value: "unchanged"}},
				},
			},
			existingHash: "compute",
			expected:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", tt.services)

			if tt.existingHash == "compute" {
				hash := dynamo.ComputeDGDWorkersSpecHash(dgd)
				dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: hash}
			} else if tt.existingHash != "" {
				dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: tt.existingHash}
			}

			r := createTestReconciler(dgd)
			result := r.shouldTriggerRollingUpdate(dgd)

			if result != tt.expected {
				t.Errorf("shouldTriggerRollingUpdate() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestInitializeWorkerHashIfNeeded_FirstDeploy(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
		},
	})

	// Create reconciler with DGD already in the fake client (simulates existing resource)
	r := createTestReconciler(dgd)
	ctx := context.Background()

	// Initialize the hash
	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// Verify the hash was set
	hash := r.getCurrentWorkerHash(dgd)
	assert.NotEmpty(t, hash, "Hash should be set after initialization")

	// Verify the hash is correct
	expectedHash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	assert.Equal(t, expectedHash, hash, "Hash should match computed value")
}

func TestInitializeWorkerHashIfNeeded_AlreadyInitialized(t *testing.T) {
	existingHash := "existing-hash"
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: existingHash,
	}

	// Create reconciler with DGD already in the fake client
	r := createTestReconciler(dgd)
	ctx := context.Background()

	// Initialize should be a no-op
	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// Verify the hash was NOT changed
	hash := r.getCurrentWorkerHash(dgd)
	assert.Equal(t, existingHash, hash, "Hash should not change when already initialized")
}

func TestSupportsManagedRollingUpdate(t *testing.T) {
	tests := []struct {
		name     string
		services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		expected bool
	}{
		{
			name: "standard single-node deployment",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			},
			expected: true,
		},
		{
			name: "multinode deployment",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 4},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", tt.services)
			r := createTestReconciler(dgd)

			result := r.supportsManagedRollingUpdate(dgd)
			if result != tt.expected {
				t.Errorf("isUnsupportedRollingUpdatePathway() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestWorkerHashChanges_OnlyWhenWorkerSpecChanges(t *testing.T) {
	// Test that hash only changes when worker specs change, not frontend specs
	workerEnvs := []corev1.EnvVar{{Name: "WORKER_VAR", Value: "value1"}}
	frontendEnvs := []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "value1"}}

	dgd1 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: workerEnvs},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: frontendEnvs},
	})

	hash1 := dynamo.ComputeDGDWorkersSpecHash(dgd1)

	// Change only frontend envs
	dgd2 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: workerEnvs},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "changed"}}},
	})

	hash2 := dynamo.ComputeDGDWorkersSpecHash(dgd2)
	assert.Equal(t, hash1, hash2, "Hash should not change when only frontend changes")

	// Change worker envs
	dgd3 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "WORKER_VAR", Value: "changed"}}},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: frontendEnvs},
	})

	hash3 := dynamo.ComputeDGDWorkersSpecHash(dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when worker specs change")
}

func TestWorkerHashChanges_PrefillAndDecode(t *testing.T) {
	// Test that prefill and decode component types are also considered workers
	dgd1 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash1 := dynamo.ComputeDGDWorkersSpecHash(dgd1)
	assert.NotEmpty(t, hash1, "Hash should be computed for prefill/decode")

	// Change prefill spec
	dgd2 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash2 := dynamo.ComputeDGDWorkersSpecHash(dgd2)
	assert.NotEqual(t, hash1, hash2, "Hash should change when prefill specs change")

	// Change decode spec
	dgd3 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
	})

	hash3 := dynamo.ComputeDGDWorkersSpecHash(dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when decode specs change")
}

func TestGetOrCreateRollingUpdateStatus(t *testing.T) {
	tests := []struct {
		name           string
		existingStatus *nvidiacomv1alpha1.RollingUpdateStatus
		expectedPhase  nvidiacomv1alpha1.RollingUpdatePhase
	}{
		{
			name:           "creates new status when nil",
			existingStatus: nil,
			expectedPhase:  nvidiacomv1alpha1.RollingUpdatePhaseNone,
		},
		{
			name: "returns existing status",
			existingStatus: &nvidiacomv1alpha1.RollingUpdateStatus{
				Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
			},
			expectedPhase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.RollingUpdate = tt.existingStatus

			r := createTestReconciler(dgd)
			status := r.getOrCreateRollingUpdateStatus(dgd)

			assert.NotNil(t, status)
			assert.Equal(t, tt.expectedPhase, status.Phase)
		})
	}
}

func TestIsRollingUpdateInProgress(t *testing.T) {
	tests := []struct {
		name     string
		status   *nvidiacomv1alpha1.RollingUpdateStatus
		expected bool
	}{
		{
			name:     "nil status - not in progress",
			status:   nil,
			expected: false,
		},
		{
			name:     "phase none - not in progress",
			status:   &nvidiacomv1alpha1.RollingUpdateStatus{Phase: nvidiacomv1alpha1.RollingUpdatePhaseNone},
			expected: false,
		},
		{
			name:     "phase pending - in progress",
			status:   &nvidiacomv1alpha1.RollingUpdateStatus{Phase: nvidiacomv1alpha1.RollingUpdatePhasePending},
			expected: true,
		},
		{
			name:     "phase in progress - in progress",
			status:   &nvidiacomv1alpha1.RollingUpdateStatus{Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress},
			expected: true,
		},
		{
			name:     "phase completed - not in progress",
			status:   &nvidiacomv1alpha1.RollingUpdateStatus{Phase: nvidiacomv1alpha1.RollingUpdatePhaseCompleted},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.RollingUpdate = tt.status

			r := createTestReconciler(dgd)
			result := r.isRollingUpdateInProgress(dgd)

			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetDesiredWorkerReplicas(t *testing.T) {
	tests := []struct {
		name     string
		services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		expected int32
	}{
		{
			name: "single worker with replicas",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
				},
			},
			expected: 3,
		},
		{
			name: "single worker without replicas defaults to 1",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
				},
			},
			expected: 1,
		},
		{
			name: "multiple workers",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"prefill": {
					ComponentType: consts.ComponentTypePrefill,
					Replicas:      ptr.To(int32(2)),
				},
				"decode": {
					ComponentType: consts.ComponentTypeDecode,
					Replicas:      ptr.To(int32(4)),
				},
			},
			expected: 6,
		},
		{
			name: "workers and frontend - only counts workers",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(2)),
				},
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
				},
			},
			expected: 3,
		},
		{
			name:     "no workers",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", tt.services)
			r := createTestReconciler(dgd)

			result := r.getDesiredWorkerReplicas(dgd)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDeleteOldWorkerDCDs(t *testing.T) {
	newWorkerHash := testNewWorkerHash

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	// Create DCD with old worker hash
	oldDCD1 := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-oldhash1",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	// Create DCD with new worker hash (should not be deleted)
	newDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-newhash2",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	r := createTestReconciler(dgd, oldDCD1, newDCD)
	ctx := context.Background()

	// Delete old worker DCDs
	err := r.deleteOldWorkerDCDs(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Verify old DCD is deleted
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	err = r.List(ctx, dcdList)
	require.NoError(t, err)

	// Should only have the new DCD remaining
	assert.Len(t, dcdList.Items, 1)
	assert.Equal(t, "test-dgd-worker-newhash2", dcdList.Items[0].Name)
}

func TestDeleteOldWorkerDCDs_NoDCDsToDelete(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	r := createTestReconciler(dgd)
	ctx := context.Background()

	// Delete old worker DCDs when there are none - should not error
	err := r.deleteOldWorkerDCDs(ctx, dgd, "somehash")
	require.NoError(t, err)
}

// createTestReconcilerWithStatus creates a reconciler with status subresource support.
func createTestReconcilerWithStatus(dgd *nvidiacomv1alpha1.DynamoGraphDeployment, objs ...runtime.Object) *DynamoGraphDeploymentReconciler {
	scheme := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	allObjs := append([]runtime.Object{dgd}, objs...)

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(allObjs...).
		WithStatusSubresource(&nvidiacomv1alpha1.DynamoGraphDeployment{}).
		Build()

	return &DynamoGraphDeploymentReconciler{
		Client:   fakeClient,
		Recorder: record.NewFakeRecorder(10),
	}
}

func TestContinueRollingUpdate_UpdatedServicesPartialCompletion(t *testing.T) {
	oldWorkerHash := testOldWorkerHash
	newWorkerHash := testNewWorkerHash

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Replicas:      ptr.To(int32(2)),
		},
		"decode": {
			ComponentType: consts.ComponentTypeDecode,
			Replicas:      ptr.To(int32(3)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: oldWorkerHash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	// New DCDs: prefill fully ready, decode not ready yet
	newPrefillDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-prefill-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(2)),
			},
		},
	}

	newDecodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
				Replicas:      ptr.To(int32(3)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)), // Not yet fully ready
			},
		},
	}

	// Old DCDs: prefill gone, decode still has replicas
	oldDecodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode-" + oldWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          oldWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
				Replicas:      ptr.To(int32(3)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(2)), // Still has old replicas
			},
		},
	}

	r := createTestReconcilerWithStatus(dgd, newPrefillDCD, newDecodeDCD, oldDecodeDCD)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	require.NoError(t, err)

	// Prefill is updated (new ready >= desired, old gone), decode is not
	assert.Equal(t, []string{"prefill"}, rollingUpdateStatus.UpdatedServices)
	// Rolling update should remain in progress since not all services are updated
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
}

func TestContinueRollingUpdate_AggregateReadyButPerServiceNot(t *testing.T) {
	oldWorkerHash := testOldWorkerHash
	newWorkerHash := testNewWorkerHash

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Replicas:      ptr.To(int32(2)),
		},
		"decode": {
			ComponentType: consts.ComponentTypeDecode,
			Replicas:      ptr.To(int32(3)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: oldWorkerHash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	// New DCDs: prefill has excess ready replicas (5), decode has 0
	// Aggregate: 5 total new ready >= 5 desired, 0 old ready == 0
	// Per-service: prefill ready (5 >= 2), decode NOT ready (0 < 3)
	newPrefillDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-prefill-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(5)), // Excess ready replicas
			},
		},
	}

	newDecodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
				Replicas:      ptr.To(int32(3)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(0)), // No ready replicas
			},
		},
	}

	// No old DCDs — old workers are fully scaled down
	r := createTestReconcilerWithStatus(dgd, newPrefillDCD, newDecodeDCD)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	require.NoError(t, err)

	// Only prefill is updated; decode has 0 ready replicas
	assert.Equal(t, []string{"prefill"}, rollingUpdateStatus.UpdatedServices)
	// Rolling update must NOT complete — decode is not ready
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
}

func TestStartRollingUpdate_UpdatedServicesInitializedToNil(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(2)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}
	// Simulate a previous rolling update that had UpdatedServices populated
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase:           nvidiacomv1alpha1.RollingUpdatePhaseNone,
		UpdatedServices: []string{"worker"},
	}

	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.startRollingUpdate(ctx, dgd, rollingUpdateStatus, testNewWorkerHash)
	require.NoError(t, err)

	assert.Nil(t, rollingUpdateStatus.UpdatedServices)
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhasePending, rollingUpdateStatus.Phase)
}

func TestCompleteRollingUpdate_UpdatedServicesContainsAllWorkers(t *testing.T) {
	oldWorkerHash := testOldWorkerHash
	newWorkerHash := testNewWorkerHash

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"frontend": {
			ComponentType: consts.ComponentTypeFrontend,
			Replicas:      ptr.To(int32(1)),
		},
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Replicas:      ptr.To(int32(2)),
		},
		"decode": {
			ComponentType: consts.ComponentTypeDecode,
			Replicas:      ptr.To(int32(3)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: oldWorkerHash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	require.NoError(t, err)

	// Should contain all worker services (sorted), but not frontend
	assert.Equal(t, []string{"decode", "prefill"}, rollingUpdateStatus.UpdatedServices)
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseCompleted, rollingUpdateStatus.Phase)
	assert.NotNil(t, rollingUpdateStatus.EndTime)
}

func TestContinueRollingUpdate_AllServicesUpdated(t *testing.T) {
	oldWorkerHash := testOldWorkerHash
	newWorkerHash := testNewWorkerHash

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Replicas:      ptr.To(int32(2)),
		},
		"decode": {
			ComponentType: consts.ComponentTypeDecode,
			Replicas:      ptr.To(int32(3)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: oldWorkerHash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	// All new DCDs fully ready
	newPrefillDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-prefill-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(2)),
			},
		},
	}

	newDecodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
				Replicas:      ptr.To(int32(3)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(3)),
			},
		},
	}

	// No old DCDs (all scaled down and removed)

	r := createTestReconcilerWithStatus(dgd, newPrefillDCD, newDecodeDCD)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	require.NoError(t, err)

	// Rolling update should complete, and all services should be listed
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseCompleted, rollingUpdateStatus.Phase)
	assert.Equal(t, []string{"decode", "prefill"}, rollingUpdateStatus.UpdatedServices)
}

func TestGetWorkerInfoForWorkerHash(t *testing.T) {
	workerHash := "hash1234"

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill},
		"decode":  {ComponentType: consts.ComponentTypeDecode},
	})

	// Create DCDs for prefill and decode with different ready counts
	prefillDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-prefill-hash1234",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          workerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(2)),
			},
		},
	}

	decodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode-hash1234",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          workerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
				Replicas:      ptr.To(int32(3)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	}

	r := createTestReconciler(dgd, prefillDCD, decodeDCD)
	ctx := context.Background()

	status, err := r.getWorkerInfoForWorkerHash(ctx, dgd, workerHash)
	require.NoError(t, err)

	assert.Len(t, status.services, 2)
	assert.Equal(t, int32(2), status.services[consts.ComponentTypePrefill].readyReplicas)
	assert.Equal(t, int32(1), status.services[consts.ComponentTypeDecode].readyReplicas)
	assert.Equal(t, int32(3), status.totalReadyWorkers) // 2 + 1
}

func TestMergeWorkerServiceStatuses(t *testing.T) {
	tests := []struct {
		name              string
		serviceStatuses   map[string]nvidiacomv1alpha1.ServiceReplicaStatus
		oldWorkerStatuses map[string]nvidiacomv1alpha1.ServiceReplicaStatus
		expected          map[string]nvidiacomv1alpha1.ServiceReplicaStatus
	}{
		{
			name: "merges old and new for a single worker service",
			serviceStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-newhash1",
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-oldhash1",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          1,
					UpdatedReplicas:   0,
					ReadyReplicas:     ptr.To(int32(1)),
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
			expected: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-newhash1",
					ComponentNames:    []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:          3,
					UpdatedReplicas:   2, // Only new are "updated"
					ReadyReplicas:     ptr.To(int32(3)),
					AvailableReplicas: ptr.To(int32(3)),
				},
			},
		},
		{
			name: "no old statuses - no-op",
			serviceStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-prefill-newhash1",
					Replicas:      2,
					ReadyReplicas: ptr.To(int32(2)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{},
			expected: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-prefill-newhash1",
					Replicas:      2,
					ReadyReplicas: ptr.To(int32(2)),
				},
			},
		},
		{
			name:            "old exists but new doesn't yet",
			serviceStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{},
			oldWorkerStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-prefill-oldhash1",
					Replicas:      2,
					ReadyReplicas: ptr.To(int32(2)),
				},
			},
			expected: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{},
		},
		{
			name: "handles nil ReadyReplicas and AvailableReplicas on old",
			serviceStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-newhash1",
					Replicas:          2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-oldhash1",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          1,
					ReadyReplicas:     nil,
					AvailableReplicas: nil,
				},
			},
			expected: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentName:     "dgd-prefill-newhash1",
					ComponentNames:    []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:          3,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
		},
		{
			name: "frontend status untouched by merge",
			serviceStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"frontend": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-frontend",
					Replicas:      1,
					ReadyReplicas: ptr.To(int32(1)),
				},
				"prefill": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-prefill-newhash1",
					Replicas:      2,
					ReadyReplicas: ptr.To(int32(2)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentName:  "dgd-prefill-oldhash1",
					ComponentNames: []string{"dgd-prefill-oldhash1"},
					Replicas:       1,
					ReadyReplicas:  ptr.To(int32(1)),
				},
			},
			expected: map[string]nvidiacomv1alpha1.ServiceReplicaStatus{
				"frontend": {
					ComponentKind: "Deployment",
					ComponentName: "dgd-frontend",
					Replicas:      1,
					ReadyReplicas: ptr.To(int32(1)),
				},
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentName:  "dgd-prefill-newhash1",
					ComponentNames: []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:       3,
					ReadyReplicas:  ptr.To(int32(3)),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mergeWorkerServiceStatuses(tt.serviceStatuses, tt.oldWorkerStatuses)
			assert.Equal(t, tt.expected, tt.serviceStatuses)
		})
	}
}

func TestAggregateOldWorkerServiceStatuses(t *testing.T) {
	t.Run("old DCD exists with status", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"prefill": {
				ComponentType: consts.ComponentTypePrefill,
				Replicas:      ptr.To(int32(2)),
			},
		})

		oldDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-prefill-oldhash1",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypePrefill,
					ServiceName:   "prefill",
					Replicas:      ptr.To(int32(1)),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					ComponentKind:   "Deployment",
					ComponentName:   "test-dgd-prefill-oldhash1",
					Replicas:        1,
					UpdatedReplicas: 0,
					ReadyReplicas:   ptr.To(int32(1)),
				},
			},
		}

		r := createTestReconciler(dgd, oldDCD)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     testNewWorkerHash,
			OldWorkerReplicas: map[string]int32{"prefill": 1},
			NewWorkerReplicas: map[string]int32{"prefill": 2},
		}

		statuses, err := r.aggregateOldWorkerServiceStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Len(t, statuses, 1)
		assert.Equal(t, "test-dgd-prefill-oldhash1", statuses["prefill"].ComponentName)
		assert.Equal(t, []string{"test-dgd-prefill-oldhash1"}, statuses["prefill"].ComponentNames)
		assert.Equal(t, int32(1), statuses["prefill"].Replicas)
		assert.Equal(t, ptr.To(int32(1)), statuses["prefill"].ReadyReplicas)
	})

	t.Run("old DCD not found - skips gracefully", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"prefill": {
				ComponentType: consts.ComponentTypePrefill,
				Replicas:      ptr.To(int32(2)),
			},
		})

		r := createTestReconciler(dgd)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     testNewWorkerHash,
			OldWorkerReplicas: map[string]int32{"prefill": 1},
			NewWorkerReplicas: map[string]int32{"prefill": 2},
		}

		statuses, err := r.aggregateOldWorkerServiceStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Empty(t, statuses)
	})
}

func TestGetExistingRestartAnnotationsDCD(t *testing.T) {
	t.Run("worker DCD with hash suffix - finds annotation", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {
				ComponentType: consts.ComponentTypeFrontend,
			},
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		// Annotation hash can differ from computed hash — function uses computed hash
		computedHash := dynamo.ComputeDGDWorkersSpecHash(dgd)
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: "oldhash",
		}

		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-frontend",
				Namespace: "default",
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					Annotations: map[string]string{
						consts.RestartAnnotation: "2025-01-01T00:00:00Z",
					},
				},
			},
		}

		workerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-" + computedHash,
				Namespace: "default",
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					Annotations: map[string]string{
						consts.RestartAnnotation: "2025-01-01T00:00:00Z",
					},
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD, workerDCD)
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["worker"])
	})

	t.Run("worker DCD not found during rolling update - gracefully skips", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {
				ComponentType: consts.ComponentTypeFrontend,
			},
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
		}

		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-frontend",
				Namespace: "default",
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					Annotations: map[string]string{
						consts.RestartAnnotation: "2025-01-01T00:00:00Z",
					},
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD)
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
		_, hasWorker := annotations["worker"]
		assert.False(t, hasWorker, "worker annotation should not be present when DCD doesn't exist")
	})

	t.Run("non-worker without hash suffix - found normally", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {
				ComponentType: consts.ComponentTypeFrontend,
			},
		})
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
		}

		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-frontend",
				Namespace: "default",
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					Annotations: map[string]string{
						consts.RestartAnnotation: "2025-01-01T00:00:00Z",
					},
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD)
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
	})
}

func TestCheckComponentServiceFullyUpdated(t *testing.T) {
	t.Run("worker with hash suffix - finds DCD", func(t *testing.T) {
		workerHash := "abc12345"
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: workerHash + "fullhashextra",
		}

		workerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-dgd-worker-" + workerHash + "fullhashextra",
				Namespace:  "default",
				Generation: 1,
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				ObservedGeneration: 1,
				Conditions: []metav1.Condition{
					{
						Type:   nvidiacomv1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
						Status: metav1.ConditionTrue,
					},
				},
			},
		}

		r := createTestReconciler(dgd, workerDCD)
		ctx := context.Background()

		isReady, reason := r.checkComponentServiceFullyUpdated(ctx, dgd, "worker")
		assert.True(t, isReady, "worker DCD should be ready")
		assert.Empty(t, reason)
	})

	t.Run("non-worker without hash suffix - finds DCD", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {
				ComponentType: consts.ComponentTypeFrontend,
			},
		})

		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-dgd-frontend",
				Namespace:  "default",
				Generation: 1,
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				ObservedGeneration: 1,
				Conditions: []metav1.Condition{
					{
						Type:   nvidiacomv1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
						Status: metav1.ConditionTrue,
					},
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD)
		ctx := context.Background()

		isReady, reason := r.checkComponentServiceFullyUpdated(ctx, dgd, "frontend")
		assert.True(t, isReady, "frontend DCD should be ready")
		assert.Empty(t, reason)
	})

	t.Run("worker without hash annotation - falls back to non-hash name", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		// No worker hash annotation

		workerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-dgd-worker",
				Namespace:  "default",
				Generation: 1,
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				ObservedGeneration: 1,
				Conditions: []metav1.Condition{
					{
						Type:   nvidiacomv1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
						Status: metav1.ConditionTrue,
					},
				},
			},
		}

		r := createTestReconciler(dgd, workerDCD)
		ctx := context.Background()

		isReady, reason := r.checkComponentServiceFullyUpdated(ctx, dgd, "worker")
		assert.True(t, isReady, "worker DCD should be ready via fallback")
		assert.Empty(t, reason)
	})
}

func TestInitializeWorkerHashIfNeeded_LegacyDCDsMigration(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
		},
	})

	// Create a legacy worker DCD: has DGD name label but NO worker hash label.
	// This simulates a DCD created by a pre-rolling-update operator version.
	legacyWorkerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				// Note: No KubeLabelDynamoWorkerHash label
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
			},
		},
	}

	r := createTestReconciler(dgd, legacyWorkerDCD)
	ctx := context.Background()

	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// DGD annotation should be set to the legacy sentinel, NOT the computed hash
	hash := r.getCurrentWorkerHash(dgd)
	assert.Equal(t, consts.LegacyWorkerHash, hash, "Hash should be legacy sentinel after migration")

	// Legacy DCD should now have the worker hash label backfilled
	updatedDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker", Namespace: "default"}, updatedDCD)
	require.NoError(t, err)
	assert.Equal(t, consts.LegacyWorkerHash, updatedDCD.Labels[consts.KubeLabelDynamoWorkerHash],
		"Legacy DCD should have worker hash label backfilled")
}

func TestInitializeWorkerHashIfNeeded_LegacyMultipleWorkers(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
		},
		"decode": {
			ComponentType: consts.ComponentTypeDecode,
		},
		"frontend": {
			ComponentType: consts.ComponentTypeFrontend,
		},
	})

	// Legacy worker DCDs (no hash label)
	legacyPrefillDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-prefill",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
			},
		},
	}

	legacyDecodeDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-decode",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeDecode,
				ServiceName:   "decode",
			},
		},
	}

	// Frontend DCD (not a worker, should not be touched)
	frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-frontend",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeFrontend,
				ServiceName:   "frontend",
			},
		},
	}

	r := createTestReconciler(dgd, legacyPrefillDCD, legacyDecodeDCD, frontendDCD)
	ctx := context.Background()

	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// DGD should have legacy sentinel hash
	assert.Equal(t, consts.LegacyWorkerHash, r.getCurrentWorkerHash(dgd))

	// Both worker DCDs should have hash label backfilled
	for _, name := range []string{"test-dgd-prefill", "test-dgd-decode"} {
		dcd := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err = r.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, dcd)
		require.NoError(t, err)
		assert.Equal(t, consts.LegacyWorkerHash, dcd.Labels[consts.KubeLabelDynamoWorkerHash],
			"Worker DCD %s should have legacy hash label", name)
	}

	// Frontend should NOT have hash label
	fe := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-frontend", Namespace: "default"}, fe)
	require.NoError(t, err)
	assert.Empty(t, fe.Labels[consts.KubeLabelDynamoWorkerHash],
		"Frontend DCD should not have worker hash label")
}

func TestFindLegacyWorkerDCDs(t *testing.T) {
	t.Run("finds worker DCDs without hash label", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		}

		r := createTestReconciler(dgd, legacyDCD)
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "test-dgd-worker", result[0].Name)
	})

	t.Run("ignores non-worker DCDs", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {ComponentType: consts.ComponentTypeFrontend},
		})

		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-frontend",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD)
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("ignores DCDs that already have hash label", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		hashedDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-abc12345",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          "abc12345",
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		}

		r := createTestReconciler(dgd, hashedDCD)
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("ignores DCDs from other DGDs", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		otherDGDWorkerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "other-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "other-dgd",
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		}

		r := createTestReconciler(dgd, otherDGDWorkerDCD)
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("no DCDs at all", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		r := createTestReconciler(dgd)
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})
}

func TestListOldWorkerDCDs(t *testing.T) {
	t.Run("finds legacy DCDs as old", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		// Legacy DCD with backfilled "legacy" hash label
		legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
				},
			},
		}

		r := createTestReconciler(dgd, legacyDCD)
		ctx := context.Background()

		result, err := r.listOldWorkerDCDs(ctx, dgd, "newhash1")
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "test-dgd-worker", result[0].Name)
	})

	t.Run("excludes current hash DCDs", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		currentDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-abc12345",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          "abc12345",
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
				},
			},
		}

		r := createTestReconciler(dgd, currentDCD)
		ctx := context.Background()

		result, err := r.listOldWorkerDCDs(ctx, dgd, "abc12345")
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("excludes non-worker DCDs", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {ComponentType: consts.ComponentTypeFrontend},
			"worker":   {ComponentType: consts.ComponentTypeWorker},
		})

		// A frontend DCD with non-matching hash (should be excluded as non-worker)
		frontendDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-frontend",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
					ServiceName:   "frontend",
				},
			},
		}

		workerDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-oldhash1",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          testOldWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
				},
			},
		}

		r := createTestReconciler(dgd, frontendDCD, workerDCD)
		ctx := context.Background()

		result, err := r.listOldWorkerDCDs(ctx, dgd, testNewWorkerHash)
		require.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "test-dgd-worker-oldhash1", result[0].Name)
	})
}

func TestScaleOldWorkerDCDs_LegacyDCDs(t *testing.T) {
	t.Run("scales legacy-named DCD via label lookup", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      ptr.To(int32(3)),
			},
		})

		// Legacy DCD with backfilled hash label but old-style name (no hash suffix)
		legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
					Replicas:      ptr.To(int32(3)),
				},
			},
		}

		r := createTestReconciler(dgd, legacyDCD)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     "newhash1",
			OldWorkerReplicas: map[string]int32{"worker": 1},
			NewWorkerReplicas: map[string]int32{"worker": 3},
		}

		err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		// Verify the legacy DCD was scaled down
		updatedDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker", Namespace: "default"}, updatedDCD)
		require.NoError(t, err)
		assert.Equal(t, int32(1), *updatedDCD.Spec.Replicas, "Legacy DCD should be scaled to 1")
	})

	t.Run("no-op when rolling update not in progress", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		r := createTestReconciler(dgd)
		ctx := context.Background()

		// Empty OldWorkerReplicas = not in progress
		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     "samehash",
			OldWorkerReplicas: map[string]int32{},
			NewWorkerReplicas: map[string]int32{},
		}

		err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)
	})

	t.Run("skips when replicas already at desired value", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      ptr.To(int32(3)),
			},
		})

		legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
					Replicas:      ptr.To(int32(1)),
				},
			},
		}

		r := createTestReconciler(dgd, legacyDCD)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     "newhash1",
			OldWorkerReplicas: map[string]int32{"worker": 1},
			NewWorkerReplicas: map[string]int32{"worker": 3},
		}

		err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		// Replicas should remain at 1 (no patch needed)
		updatedDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker", Namespace: "default"}, updatedDCD)
		require.NoError(t, err)
		assert.Equal(t, int32(1), *updatedDCD.Spec.Replicas)
	})
}

func TestAggregateOldWorkerServiceStatuses_LegacyDCDs(t *testing.T) {
	t.Run("aggregates status from legacy-named DCD", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      ptr.To(int32(3)),
			},
		})

		// Legacy DCD with old-style name but backfilled hash label
		legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeWorker,
					ServiceName:   "worker",
					Replicas:      ptr.To(int32(2)),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					ComponentKind: "Deployment",
					ComponentName: "test-dgd-worker",
					Replicas:      2,
					ReadyReplicas: ptr.To(int32(2)),
				},
			},
		}

		r := createTestReconciler(dgd, legacyDCD)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     "newhash1",
			OldWorkerReplicas: map[string]int32{"worker": 2},
			NewWorkerReplicas: map[string]int32{"worker": 3},
		}

		statuses, err := r.aggregateOldWorkerServiceStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Len(t, statuses, 1)
		assert.Equal(t, "test-dgd-worker", statuses["worker"].ComponentName)
		assert.Equal(t, []string{"test-dgd-worker"}, statuses["worker"].ComponentNames)
		assert.Equal(t, int32(2), statuses["worker"].Replicas)
		assert.Equal(t, ptr.To(int32(2)), statuses["worker"].ReadyReplicas)
	})

	t.Run("no legacy DCDs found - returns empty", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})

		r := createTestReconciler(dgd)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:     "newhash1",
			OldWorkerReplicas: map[string]int32{"worker": 1},
			NewWorkerReplicas: map[string]int32{"worker": 1},
		}

		statuses, err := r.aggregateOldWorkerServiceStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)
		assert.Empty(t, statuses)
	})
}

func TestDeleteOldWorkerDCDs_LegacyDCDs(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	// Legacy DCD with backfilled hash label
	legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	// New DCD with real hash (should NOT be deleted)
	newDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-abc12345",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "abc12345",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	r := createTestReconciler(dgd, legacyDCD, newDCD)
	ctx := context.Background()

	err := r.deleteOldWorkerDCDs(ctx, dgd, "abc12345")
	require.NoError(t, err)

	// Verify legacy DCD is deleted and new DCD remains
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	err = r.List(ctx, dcdList)
	require.NoError(t, err)

	assert.Len(t, dcdList.Items, 1)
	assert.Equal(t, "test-dgd-worker-abc12345", dcdList.Items[0].Name)
}

func TestDeleteOldWorkerDCDs_MultipleGenerations(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	// Generation A (legacy)
	legacyDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          consts.LegacyWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	// Generation B (intermediate)
	genBDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashbbbb",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashbbbb",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	// Generation C (current)
	currentDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashcccc",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashcccc",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}

	r := createTestReconciler(dgd, legacyDCD, genBDCD, currentDCD)
	ctx := context.Background()

	err := r.deleteOldWorkerDCDs(ctx, dgd, "hashcccc")
	require.NoError(t, err)

	// Verify both old generations are deleted, only current remains
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	err = r.List(ctx, dcdList)
	require.NoError(t, err)

	assert.Len(t, dcdList.Items, 1)
	assert.Equal(t, "test-dgd-worker-hashcccc", dcdList.Items[0].Name)
}

func TestListOldWorkerDCDs_ExcludesCurrentHash(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	// Generation A
	genADCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashaaaa",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashaaaa",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
			},
		},
	}

	// Generation B
	genBDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashbbbb",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashbbbb",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
			},
		},
	}

	// Generation C (current)
	genCDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashcccc",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashcccc",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
			},
		},
	}

	r := createTestReconciler(dgd, genADCD, genBDCD, genCDCD)
	ctx := context.Background()

	result, err := r.listOldWorkerDCDs(ctx, dgd, "hashcccc")
	require.NoError(t, err)
	assert.Len(t, result, 2)

	names := []string{result[0].Name, result[1].Name}
	sort.Strings(names)
	assert.Equal(t, []string{"test-dgd-worker-hashaaaa", "test-dgd-worker-hashbbbb"}, names)
}

func TestScaleOldWorkerDCDs_MultipleOldGenerations(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(4)),
		},
	})

	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-1 * 60 * 1e9)) // 1 minute earlier

	// Generation A (oldest)
	genADCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-dgd-worker-hashaaaa",
			Namespace:         "default",
			CreationTimestamp: earlier,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashaaaa",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(2)),
			},
		},
	}

	// Generation B (newer old)
	genBDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-dgd-worker-hashbbbb",
			Namespace:         "default",
			CreationTimestamp: now,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashbbbb",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(2)),
			},
		},
	}

	r := createTestReconciler(dgd, genADCD, genBDCD)
	ctx := context.Background()

	// oldNeeded = 2: newest old (B) should get 2, oldest (A) should get 0
	rollingUpdateCtx := dynamo.RollingUpdateContext{
		NewWorkerHash:     "hashcccc",
		OldWorkerReplicas: map[string]int32{"worker": 2},
		NewWorkerReplicas: map[string]int32{"worker": 4},
	}

	err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
	require.NoError(t, err)

	// Newest old (B) should keep replicas (up to 2)
	updatedB := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashbbbb", Namespace: "default"}, updatedB)
	require.NoError(t, err)
	assert.Equal(t, int32(2), *updatedB.Spec.Replicas, "Newest old DCD should have 2 replicas")

	// Oldest (A) should be drained to 0
	updatedA := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashaaaa", Namespace: "default"}, updatedA)
	require.NoError(t, err)
	assert.Equal(t, int32(0), *updatedA.Spec.Replicas, "Oldest old DCD should be drained to 0")
}

func TestAggregateOldWorkerServiceStatuses_MultipleOldGenerations(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(4)),
		},
	})

	// Generation A
	genADCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashaaaa",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashaaaa",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(1)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ComponentKind: "Deployment",
				ComponentName: "test-dgd-worker-hashaaaa",
				Replicas:      1,
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	}

	// Generation B
	genBDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashbbbb",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashbbbb",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ComponentKind: "Deployment",
				ComponentName: "test-dgd-worker-hashbbbb",
				Replicas:      2,
				ReadyReplicas: ptr.To(int32(2)),
			},
		},
	}

	r := createTestReconciler(dgd, genADCD, genBDCD)
	ctx := context.Background()

	rollingUpdateCtx := dynamo.RollingUpdateContext{
		NewWorkerHash:     "hashcccc",
		OldWorkerReplicas: map[string]int32{"worker": 3},
		NewWorkerReplicas: map[string]int32{"worker": 4},
	}

	statuses, err := r.aggregateOldWorkerServiceStatuses(ctx, dgd, rollingUpdateCtx)
	require.NoError(t, err)

	assert.Len(t, statuses, 1)
	// Replicas should be summed across both old generations
	assert.Equal(t, int32(3), statuses["worker"].Replicas)
	assert.Equal(t, ptr.To(int32(3)), statuses["worker"].ReadyReplicas)
	// ComponentNames should include both old DCDs
	assert.Len(t, statuses["worker"].ComponentNames, 2)
}

func TestContinueRollingUpdate_CascadingSpecChange(t *testing.T) {
	// Scenario: A→B rolling update in progress, spec changes to C.
	// B DCDs should be treated as old alongside A DCDs.
	newWorkerHash := "hashcccc"

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(2)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: "hashaaaa",
	}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	// Generation A (old)
	genADCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashaaaa",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashaaaa",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(1)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	}

	// Generation B (intermediate, now also old)
	genBDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-hashbbbb",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          "hashbbbb",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(1)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	}

	// Generation C (new, not yet ready)
	genCDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-" + newWorkerHash[:8],
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newWorkerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
				Replicas:      ptr.To(int32(2)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(0)),
			},
		},
	}

	r := createTestReconcilerWithStatus(dgd, genADCD, genBDCD, genCDCD)
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	require.NoError(t, err)

	// Both A and B have ready replicas, C has 0 — rolling update not complete
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
	assert.Empty(t, rollingUpdateStatus.UpdatedServices, "No services should be fully updated yet")
}

func TestResolveRollingUpdateParams(t *testing.T) {
	tests := []struct {
		name            string
		annotations     map[string]string
		desiredReplicas int32
		expectedSurge   int32
		expectedUnavail int32
	}{
		{
			name:            "defaults - no annotations - 25%/25% of 4 = 1/1",
			annotations:     nil,
			desiredReplicas: 4,
			expectedSurge:   1,
			expectedUnavail: 1,
		},
		{
			name: "absolute maxSurge overrides default",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge: "2",
			},
			desiredReplicas: 4,
			expectedSurge:   2,
			expectedUnavail: 1,
		},
		{
			name: "absolute maxUnavailable overrides default",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
			},
			desiredReplicas: 4,
			expectedSurge:   1,
			expectedUnavail: 0,
		},
		{
			name: "percentage maxSurge - 50% of 4 = 2",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge: "50%",
			},
			desiredReplicas: 4,
			expectedSurge:   2,
			expectedUnavail: 1,
		},
		{
			name: "percentage maxUnavailable - 50% of 4 = 2",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "50%",
			},
			desiredReplicas: 4,
			expectedSurge:   1,
			expectedUnavail: 2,
		},
		{
			name: "both annotations set with percentages",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "50%",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "25%",
			},
			desiredReplicas: 4,
			expectedSurge:   2,
			expectedUnavail: 1,
		},
		{
			name: "both zero - force surge to 1 for progress",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "0",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
			},
			desiredReplicas: 4,
			expectedSurge:   1,
			expectedUnavail: 0,
		},
		{
			name: "maxSurge 0 with maxUnavailable 1 - allowed",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "0",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "1",
			},
			desiredReplicas: 4,
			expectedSurge:   0,
			expectedUnavail: 1,
		},
		{
			name: "percentage surge rounds up - 34% of 3 rounds up to 2",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge: "34%",
			},
			desiredReplicas: 3,
			expectedSurge:   2,
			expectedUnavail: 0,
		},
		{
			name: "percentage unavailable rounds down - 34% of 3 rounds down to 1",
			annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "34%",
			},
			desiredReplicas: 3,
			expectedSurge:   1,
			expectedUnavail: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			surge, unavail := resolveRollingUpdateParams(tt.annotations, tt.desiredReplicas)
			assert.Equal(t, tt.expectedSurge, surge, "maxSurge")
			assert.Equal(t, tt.expectedUnavail, unavail, "maxUnavailable")
		})
	}
}

// --- reconcileRollingUpdate state machine tests ---

func TestReconcileRollingUpdate_NoChange(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	hash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: hash}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseCompleted,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Phase should stay Completed — no spec change
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_SpecChangeStartsRollout(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "stale000"}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseCompleted,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should transition to Pending (new rollout started)
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
	assert.NotNil(t, dgd.Status.RollingUpdate.StartTime)
}

func TestReconcileRollingUpdate_PendingToInProgress(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhasePending,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseInProgress, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_StuckDetection(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	hash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	// Hash matches current but phase is InProgress — stuck
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: hash}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should auto-complete
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_NewRollingUpdate(t *testing.T) {
	newHash := "newhash1"
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseCompleted,
	}

	// Create a DCD with the new hash that has ready replicas — stale annotation scenario
	newDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-" + newHash,
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				ServiceName:   "worker",
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	}

	r := createTestReconcilerWithStatus(dgd, newDCD)

	// When computed hash != current hash and no DCDs exist with computed hash, start rollout.
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should start a new rolling update (Pending) since computed hash DCDs don't exist
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_NonePhaseStartsRollout(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseNone,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1alpha1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
	assert.NotNil(t, dgd.Status.RollingUpdate.StartTime)
	assert.Nil(t, dgd.Status.RollingUpdate.UpdatedServices)
}
