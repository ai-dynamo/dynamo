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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// createTestDGD creates a DynamoGraphDeployment for testing with the given services
func createTestDGD(name, namespace string, services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1alpha1.DynamoGraphDeployment {
	return &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
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
			dgd := createTestDGD("test-dgd", "default", tt.services)

			if tt.existingHash == "compute" {
				hash := dynamo.ComputeWorkerSpecHash(dgd)
				dgd.Annotations = map[string]string{consts.AnnotationActiveWorkerHash: hash}
			} else if tt.existingHash != "" {
				dgd.Annotations = map[string]string{consts.AnnotationActiveWorkerHash: tt.existingHash}
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
	dgd := createTestDGD("test-dgd", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
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
	hash := r.getCurrentActiveWorkerHash(dgd)
	assert.NotEmpty(t, hash, "Hash should be set after initialization")

	// Verify the hash is correct
	expectedHash := dynamo.ComputeWorkerSpecHash(dgd)
	assert.Equal(t, expectedHash, hash, "Hash should match computed value")
}

func TestInitializeWorkerHashIfNeeded_AlreadyInitialized(t *testing.T) {
	existingHash := "existing-hash"
	dgd := createTestDGD("test-dgd", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationActiveWorkerHash: existingHash,
	}

	// Create reconciler with DGD already in the fake client
	r := createTestReconciler(dgd)
	ctx := context.Background()

	// Initialize should be a no-op
	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// Verify the hash was NOT changed
	hash := r.getCurrentActiveWorkerHash(dgd)
	assert.Equal(t, existingHash, hash, "Hash should not change when already initialized")
}

func TestIsUnsupportedRollingUpdatePathway(t *testing.T) {
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
			expected: false,
		},
		{
			name: "multinode deployment",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 4},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", "default", tt.services)
			r := createTestReconciler(dgd)

			result := r.isUnsupportedRollingUpdatePathway(dgd)
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

	dgd1 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: workerEnvs},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: frontendEnvs},
	})

	hash1 := dynamo.ComputeWorkerSpecHash(dgd1)

	// Change only frontend envs
	dgd2 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: workerEnvs},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "changed"}}},
	})

	hash2 := dynamo.ComputeWorkerSpecHash(dgd2)
	assert.Equal(t, hash1, hash2, "Hash should not change when only frontend changes")

	// Change worker envs
	dgd3 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "WORKER_VAR", Value: "changed"}}},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: frontendEnvs},
	})

	hash3 := dynamo.ComputeWorkerSpecHash(dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when worker specs change")
}

func TestWorkerHashChanges_PrefillAndDecode(t *testing.T) {
	// Test that prefill and decode component types are also considered workers
	dgd1 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash1 := dynamo.ComputeWorkerSpecHash(dgd1)
	assert.NotEmpty(t, hash1, "Hash should be computed for prefill/decode")

	// Change prefill spec
	dgd2 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash2 := dynamo.ComputeWorkerSpecHash(dgd2)
	assert.NotEqual(t, hash1, hash2, "Hash should change when prefill specs change")

	// Change decode spec
	dgd3 := createTestDGD("test", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
	})

	hash3 := dynamo.ComputeWorkerSpecHash(dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when decode specs change")
}

func TestGetOrCreateRolloutStatus(t *testing.T) {
	tests := []struct {
		name           string
		existingStatus *nvidiacomv1alpha1.RolloutStatus
		expectedPhase  nvidiacomv1alpha1.RolloutPhase
	}{
		{
			name:           "creates new status when nil",
			existingStatus: nil,
			expectedPhase:  nvidiacomv1alpha1.RolloutPhaseNone,
		},
		{
			name: "returns existing status",
			existingStatus: &nvidiacomv1alpha1.RolloutStatus{
				Phase: nvidiacomv1alpha1.RolloutPhaseInProgress,
			},
			expectedPhase: nvidiacomv1alpha1.RolloutPhaseInProgress,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.Rollout = tt.existingStatus

			r := createTestReconciler(dgd)
			status := r.getOrCreateRolloutStatus(dgd)

			assert.NotNil(t, status)
			assert.Equal(t, tt.expectedPhase, status.Phase)
		})
	}
}

func TestIsRollingUpdateInProgress(t *testing.T) {
	tests := []struct {
		name     string
		status   *nvidiacomv1alpha1.RolloutStatus
		expected bool
	}{
		{
			name:     "nil status - not in progress",
			status:   nil,
			expected: false,
		},
		{
			name:     "phase none - not in progress",
			status:   &nvidiacomv1alpha1.RolloutStatus{Phase: nvidiacomv1alpha1.RolloutPhaseNone},
			expected: false,
		},
		{
			name:     "phase pending - in progress",
			status:   &nvidiacomv1alpha1.RolloutStatus{Phase: nvidiacomv1alpha1.RolloutPhasePending},
			expected: true,
		},
		{
			name:     "phase in progress - in progress",
			status:   &nvidiacomv1alpha1.RolloutStatus{Phase: nvidiacomv1alpha1.RolloutPhaseInProgress},
			expected: true,
		},
		{
			name:     "phase completed - not in progress",
			status:   &nvidiacomv1alpha1.RolloutStatus{Phase: nvidiacomv1alpha1.RolloutPhaseCompleted},
			expected: false,
		},
		{
			name:     "phase failed - not in progress",
			status:   &nvidiacomv1alpha1.RolloutStatus{Phase: nvidiacomv1alpha1.RolloutPhaseFailed},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.Rollout = tt.status

			r := createTestReconciler(dgd)
			result := r.isRollingUpdateInProgress(dgd)

			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestClearRolloutStatus(t *testing.T) {
	dgd := createTestDGD("test-dgd", "default", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Status.Rollout = &nvidiacomv1alpha1.RolloutStatus{
		Phase:            nvidiacomv1alpha1.RolloutPhaseCompleted,
		TrafficWeightOld: 0,
		TrafficWeightNew: 100,
	}

	r := createTestReconciler(dgd)
	r.clearRolloutStatus(dgd)

	assert.NotNil(t, dgd.Status.Rollout)
	assert.Equal(t, nvidiacomv1alpha1.RolloutPhaseNone, dgd.Status.Rollout.Phase)
	assert.Zero(t, dgd.Status.Rollout.TrafficWeightOld)
	assert.Zero(t, dgd.Status.Rollout.TrafficWeightNew)
}

func TestGetWorkerServices(t *testing.T) {
	tests := []struct {
		name     string
		services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		expected []string
	}{
		{
			name: "single worker",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			},
			expected: []string{"worker"},
		},
		{
			name: "prefill and decode",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"prefill": {ComponentType: consts.ComponentTypePrefill},
				"decode":  {ComponentType: consts.ComponentTypeDecode},
			},
			expected: []string{"prefill", "decode"},
		},
		{
			name: "mixed services - only workers returned",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {ComponentType: consts.ComponentTypeFrontend},
				"worker":   {ComponentType: consts.ComponentTypeWorker},
				"planner":  {ComponentType: consts.ComponentTypePlanner},
			},
			expected: []string{"worker"},
		},
		{
			name: "no workers",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {ComponentType: consts.ComponentTypeFrontend},
			},
			expected: []string{},
		},
		{
			name: "all worker types",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker":  {ComponentType: consts.ComponentTypeWorker},
				"prefill": {ComponentType: consts.ComponentTypePrefill},
				"decode":  {ComponentType: consts.ComponentTypeDecode},
			},
			expected: []string{"worker", "prefill", "decode"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", "default", tt.services)
			r := createTestReconciler(dgd)

			result := r.getWorkerServices(dgd)

			// Sort both slices for comparison since map iteration order is not guaranteed
			assert.ElementsMatch(t, tt.expected, result)
		})
	}
}
