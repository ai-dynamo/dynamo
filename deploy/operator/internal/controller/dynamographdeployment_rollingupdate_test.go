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
	"errors"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

const (
	testOldWorkerHash = "oldhash1"
	testNewWorkerHash = "newhash2"
)

// createTestDGD creates a DynamoGraphDeployment for testing with the given services
func createTestDGD(name string, services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec) *nvidiacomv1beta1.DynamoGraphDeployment {
	return mustBetaDGD(&nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: services,
		},
	})
}

type testReconcilerOption func(*fake.ClientBuilder)

// withObjects seeds the fake client with additional runtime objects beyond the DGD.
func withObjects(objs ...runtime.Object) testReconcilerOption {
	return func(b *fake.ClientBuilder) {
		b.WithRuntimeObjects(objs...)
	}
}

// withInterceptor routes all client method calls through the supplied
// interceptor.Funcs, letting tests inject API errors on specific code paths.
func withInterceptor(funcs interceptor.Funcs) testReconcilerOption {
	return func(b *fake.ClientBuilder) {
		b.WithInterceptorFuncs(funcs)
	}
}

func createTestReconcilerWithStatus(dgd *nvidiacomv1beta1.DynamoGraphDeployment, opts ...testReconcilerOption) *DynamoGraphDeploymentReconciler {
	scheme := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(scheme)
	_ = nvidiacomv1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)

	builder := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(dgd).
		WithIndex(&corev1.Pod{}, dgdComponentPodIndex, dgdComponentPodIndexValues).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphDeployment{})
	for _, opt := range opts {
		opt(builder)
	}

	return &DynamoGraphDeploymentReconciler{
		Client:        builder.Build(),
		Recorder:      record.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commonController.RuntimeConfig{},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
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
			name: "unversioned legacy alpha hash - compatible migration does not trigger rollout",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
					Resources: &nvidiacomv1alpha1.Resources{
						Requests: &nvidiacomv1alpha1.ResourceItem{CPU: "1"},
					},
				},
			},
			existingHash: "legacy-compute",
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
				hash := legacyDGDWorkersSpecHash(t, dgd)
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash:   hash,
					consts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
				}
			} else if tt.existingHash == "legacy-compute" {
				hash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
				require.NoError(t, err)
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash: hash,
				}
			} else if tt.existingHash != "" {
				dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: tt.existingHash}
			}

			r := createTestReconcilerWithStatus(dgd)
			result, err := r.shouldTriggerRollingUpdate(dgd)
			require.NoError(t, err)

			if result != tt.expected {
				t.Errorf("shouldTriggerRollingUpdate() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestShouldTriggerRollingUpdate_IgnoresReplicaChanges(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
		},
	})
	legacyHash := legacyDGDWorkersSpecHash(t, dgd)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   legacyHash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}

	dgd.Spec.Components[0].Replicas = ptr.To(int32(10))

	r := createTestReconcilerWithStatus(dgd)
	desired, err := r.desiredWorkerHashes(dgd)
	require.NoError(t, err)
	assert.Equal(t, legacyHash, desired.v1)
	assert.Equal(t, v2Hash, desired.v2)

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	assert.False(t, trigger)
}

func TestCanonicalWorkerHashLifecycle_FirstDeploySpecChangeAndCompletion(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
			Resources: &nvidiacomv1alpha1.Resources{
				Requests: &nvidiacomv1alpha1.ResourceItem{CPU: "1"},
			},
		},
	})

	// Create reconciler with DGD already in the fake client (simulates existing resource)
	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	// Initialize the hash
	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// Verify only the v2 hash was set.
	hash := r.getCurrentWorkerHashV2(dgd)
	assert.NotEmpty(t, hash, "Hash should be set after initialization")
	assert.NotContains(t, dgd.Annotations, consts.AnnotationCurrentWorkerHash)

	// Fresh deployments store one canonical v2 hash.
	expectedV2Hash := betaDGDWorkersSpecHash(t, dgd)
	assert.Equal(t, expectedV2Hash, hash)

	// A later worker change rolls directly from one canonical v2 generation to the next.
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = append(
		dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "NEW_WORKER_SETTING", Value: "true"},
	)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	newLegacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	require.NotEqual(t, expectedV2Hash, newV2Hash)
	require.NotEqual(t, newLegacyHash, newV2Hash)

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.True(t, trigger)

	rollingCtx, err := r.buildRollingUpdateContext(ctx, dgd)
	require.NoError(t, err)
	require.Equal(t, newV2Hash, rollingCtx.NewWorkerHash)
	require.NotEqual(t, newLegacyHash, rollingCtx.NewWorkerHash)

	require.NoError(t, r.completeRollingUpdate(ctx, dgd, newV2Hash))
	require.NotContains(t, dgd.Annotations, consts.AnnotationCurrentWorkerHash)
	require.Equal(t, newV2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
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
	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	// Initialize should be a no-op
	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// Verify the hash was NOT changed
	hash := r.getCurrentWorkerHash(dgd)
	assert.Equal(t, existingHash, hash, "Hash should not change when already initialized")
}

func TestInitializeWorkerHashIfNeeded_PreservesLegacyAlphaHash(t *testing.T) {
	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			Annotations: map[string]string{
				consts.AnnotationCurrentWorkerHash: "old-alpha-hash",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
					Resources: &nvidiacomv1alpha1.Resources{
						Requests: &nvidiacomv1alpha1.ResourceItem{CPU: "1"},
					},
				},
			},
		},
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, alpha.ConvertTo(dgd))
	legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, legacyHash, v2Hash)
	if dgd.Annotations == nil {
		dgd.Annotations = map[string]string{}
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = legacyHash

	r := createTestReconcilerWithStatus(dgd)
	err = r.initializeWorkerHashIfNeeded(context.Background(), dgd)
	require.NoError(t, err)

	assert.Equal(t, legacyHash, r.getCurrentWorkerHash(dgd))
	assert.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	assert.False(t, trigger)

	ctx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, legacyHash, ctx.NewWorkerHash)
	assert.False(t, ctx.InProgress())
	assert.NotEqual(t, v2Hash, ctx.NewWorkerHash)
}

func TestLegacyAlphaHashCompatibility_NoOpUpgradeUsesExistingWorkerGeneration(t *testing.T) {
	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "qwen",
			Namespace: "default",
			Annotations: map[string]string{
				consts.AnnotationCurrentWorkerHash: "old-alpha-hash",
			},
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"VllmDecodeWorker": {
					ComponentType:    consts.ComponentTypeWorker,
					SubComponentType: consts.ComponentTypeDecode,
					Envs:             []corev1.EnvVar{{Name: "MODEL_PATH", Value: "Qwen/Qwen3-0.6B"}},
					Resources: &nvidiacomv1alpha1.Resources{
						Requests: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
					},
				},
			},
		},
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, alpha.ConvertTo(dgd))
	legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, legacyHash, v2Hash)
	if dgd.Annotations == nil {
		dgd.Annotations = map[string]string{}
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = legacyHash

	r := createTestReconcilerWithStatus(dgd)
	require.NoError(t, r.initializeWorkerHashIfNeeded(context.Background(), dgd))

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.False(t, trigger)

	rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.Equal(t, legacyHash, rollingCtx.NewWorkerHash)
	require.False(t, rollingCtx.InProgress())

	dcds, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, rollingCtx)
	require.NoError(t, err)
	require.Equal(t, "qwen-vllmdecodeworker-"+legacyHash, dcds["VllmDecodeWorker"].Name)
	require.NotEqual(t, "qwen-vllmdecodeworker-"+v2Hash, dcds["VllmDecodeWorker"].Name)
}

func TestLegacyAlphaHashCompatibility_WorkerSpecChangeUsesNewV2Generation(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
			Resources: &nvidiacomv1alpha1.Resources{
				Requests: &nvidiacomv1alpha1.ResourceItem{CPU: "1"},
			},
		},
	})
	legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, legacyHash, v2Hash)
	if dgd.Annotations == nil {
		dgd.Annotations = map[string]string{}
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = legacyHash

	r := createTestReconcilerWithStatus(dgd)
	require.NoError(t, r.initializeWorkerHashIfNeeded(context.Background(), dgd))
	require.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = append(
		dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "NEW_WORKER_SETTING", Value: "true"},
	)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	newLegacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	require.NotEqual(t, v2Hash, newV2Hash)
	require.NotEqual(t, legacyHash, newLegacyHash)

	require.NoError(t, r.migrateCurrentWorkerHashIfNeeded(context.Background(), dgd))

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.True(t, trigger)

	rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.Equal(t, newV2Hash, rollingCtx.NewWorkerHash)
	require.NotEqual(t, newLegacyHash, rollingCtx.NewWorkerHash)
}

func TestLegacyAlphaHashCompatibility_V2OnlyChangeUsesNewV2Generation(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
		},
	})
	dgd.Spec.BackendFramework = "vllm"
	legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   legacyHash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}

	r := createTestReconcilerWithStatus(dgd)
	dgd.Spec.BackendFramework = "sglang"

	newLegacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.Equal(t, legacyHash, newLegacyHash)
	require.NotEqual(t, v2Hash, newV2Hash)

	require.NoError(t, r.migrateCurrentWorkerHashIfNeeded(context.Background(), dgd))
	require.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.True(t, trigger)

	rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.Equal(t, newV2Hash, rollingCtx.NewWorkerHash)
	require.NotEqual(t, newLegacyHash, rollingCtx.NewWorkerHash)

	require.NoError(t, r.completeRollingUpdate(context.Background(), dgd, newV2Hash))
	require.Empty(t, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, newV2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
}

func TestUnsupportedPathwayMigratesV1OnlyAndKeepsV2OnlyGeneration(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
			Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
		},
	})
	dgd.Spec.BackendFramework = "vllm"
	legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: legacyHash,
	}

	r := createTestReconcilerWithStatus(dgd)
	require.False(t, r.supportsManagedRollingUpdate(dgd))

	require.NoError(t, r.migrateCurrentWorkerHashIfNeeded(context.Background(), dgd))
	require.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.False(t, trigger)

	dgd.Spec.BackendFramework = "sglang"

	newLegacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.Equal(t, legacyHash, newLegacyHash)
	require.NotEqual(t, v2Hash, newV2Hash)

	require.NoError(t, r.migrateCurrentWorkerHashIfNeeded(context.Background(), dgd))
	require.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	desired, err := r.desiredWorkerHashes(dgd)
	require.NoError(t, err)
	completed := r.workerHashesForUnsupportedPathway(dgd, desired)
	require.Empty(t, completed.v1)
	require.Equal(t, newV2Hash, completed.v2)

	r.setCurrentWorkerHashes(dgd, completed)
	rollingCtx, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	require.Equal(t, newV2Hash, rollingCtx.NewWorkerHash)
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
			r := createTestReconcilerWithStatus(dgd)

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

	hash1 := betaDGDWorkersSpecHash(t, dgd1)

	// Change only frontend envs
	dgd2 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: workerEnvs},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: []corev1.EnvVar{{Name: "FRONTEND_VAR", Value: "changed"}}},
	})

	hash2 := betaDGDWorkersSpecHash(t, dgd2)
	assert.Equal(t, hash1, hash2, "Hash should not change when only frontend changes")

	// Change worker envs
	dgd3 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker":   {ComponentType: consts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "WORKER_VAR", Value: "changed"}}},
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Envs: frontendEnvs},
	})

	hash3 := betaDGDWorkersSpecHash(t, dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when worker specs change")
}

func TestWorkerHashChanges_PrefillAndDecode(t *testing.T) {
	// Test that prefill and decode component types are also considered workers
	dgd1 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash1 := betaDGDWorkersSpecHash(t, dgd1)
	assert.NotEmpty(t, hash1, "Hash should be computed for prefill/decode")

	// Change prefill spec
	dgd2 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
	})

	hash2 := betaDGDWorkersSpecHash(t, dgd2)
	assert.NotEqual(t, hash1, hash2, "Hash should change when prefill specs change")

	// Change decode spec
	dgd3 := createTestDGD("test", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v1"}}},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "VAR", Value: "v2"}}},
	})

	hash3 := betaDGDWorkersSpecHash(t, dgd3)
	assert.NotEqual(t, hash1, hash3, "Hash should change when decode specs change")
}

func TestGetOrCreateRollingUpdateStatus(t *testing.T) {
	tests := []struct {
		name           string
		existingStatus *nvidiacomv1beta1.RollingUpdateStatus
		expectedPhase  nvidiacomv1beta1.RollingUpdatePhase
	}{
		{
			name:           "creates new status when nil",
			existingStatus: nil,
			expectedPhase:  nvidiacomv1beta1.RollingUpdatePhaseNone,
		},
		{
			name: "returns existing status",
			existingStatus: &nvidiacomv1beta1.RollingUpdateStatus{
				Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
			},
			expectedPhase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.RollingUpdate = tt.existingStatus

			r := createTestReconcilerWithStatus(dgd)
			status := r.getOrCreateRollingUpdateStatus(dgd)

			assert.NotNil(t, status)
			assert.Equal(t, tt.expectedPhase, status.Phase)
		})
	}
}

func TestIsRollingUpdateInProgress(t *testing.T) {
	tests := []struct {
		name     string
		status   *nvidiacomv1beta1.RollingUpdateStatus
		expected bool
	}{
		{
			name:     "nil status - not in progress",
			status:   nil,
			expected: false,
		},
		{
			name:     "phase none - not in progress",
			status:   &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseNone},
			expected: false,
		},
		{
			name:     "phase pending - in progress",
			status:   &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhasePending},
			expected: true,
		},
		{
			name:     "phase in progress - in progress",
			status:   &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress},
			expected: true,
		},
		{
			name:     "phase completed - not in progress",
			status:   &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Status.RollingUpdate = tt.status

			r := createTestReconcilerWithStatus(dgd)
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
			r := createTestReconcilerWithStatus(dgd)

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
	oldDCD1 := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Create DCD with new worker hash (should not be deleted)
	newDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD1, newDCD))
	ctx := context.Background()

	// Delete old worker DCDs
	err := r.deleteOldWorkerDCDs(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Verify old DCD is deleted
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
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

	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	// Delete old worker DCDs when there are none - should not error
	err := r.deleteOldWorkerDCDs(ctx, dgd, "somehash")
	require.NoError(t, err)
}

func TestContinueRollingUpdate_UpdatedComponentsPartialCompletion(t *testing.T) {
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
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	// New DCDs: prefill fully ready, decode not ready yet
	newPrefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	newDecodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Old DCDs: prefill gone, decode still has replicas
	oldDecodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(newPrefillDCD, newDecodeDCD, oldDecodeDCD))
	ctx := context.Background()

	rollingUpdateStatus := dgd.Status.RollingUpdate
	err := r.continueRollingUpdate(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Prefill is updated (new ready >= desired, old gone), decode is not
	assert.Equal(t, []string{"prefill"}, rollingUpdateStatus.UpdatedComponents)
	// Rolling update should remain in progress since not all services are updated
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
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
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	// New DCDs: prefill has excess ready replicas (5), decode has 0
	// Aggregate: 5 total new ready >= 5 desired, 0 old ready == 0
	// Per-service: prefill ready (5 >= 2), decode NOT ready (0 < 3)
	newPrefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	newDecodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// No old DCDs — old workers are fully scaled down
	r := createTestReconcilerWithStatus(dgd, withObjects(newPrefillDCD, newDecodeDCD))
	ctx := context.Background()

	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	err := r.continueRollingUpdate(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Only prefill is updated; decode has 0 ready replicas
	assert.Equal(t, []string{"prefill"}, rollingUpdateStatus.UpdatedComponents)
	// Rolling update must NOT complete — decode is not ready
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
}

func TestStartRollingUpdate_UpdatedComponentsInitializedToNil(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(2)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}
	// Simulate a previous rolling update that had UpdatedComponents populated
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase:             nvidiacomv1beta1.RollingUpdatePhaseNone,
		UpdatedComponents: []string{"worker"},
	}

	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	err := r.startRollingUpdate(ctx, dgd, testNewWorkerHash)
	require.NoError(t, err)

	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	assert.Nil(t, rollingUpdateStatus.UpdatedComponents)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, rollingUpdateStatus.Phase)
}

func TestCompleteRollingUpdate_UpdatedComponentsContainsAllWorkers(t *testing.T) {
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
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	r := createTestReconcilerWithStatus(dgd)
	ctx := context.Background()

	err := r.completeRollingUpdate(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Check dgd.Status.RollingUpdate directly because r.Update() inside completeRollingUpdate
	// decodes the API server response back into dgd, and status is re-set after the update.
	assert.Equal(t, []string{"decode", "prefill"}, dgd.Status.RollingUpdate.UpdatedComponents)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
	assert.NotNil(t, dgd.Status.RollingUpdate.EndTime)
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
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	// All new DCDs fully ready
	newPrefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	newDecodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// No old DCDs (all scaled down and removed)

	r := createTestReconcilerWithStatus(dgd, withObjects(newPrefillDCD, newDecodeDCD))
	ctx := context.Background()

	err := r.continueRollingUpdate(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Rolling update should complete, and all services should be listed.
	// Check dgd.Status.RollingUpdate directly because r.Update() inside completeRollingUpdate
	// decodes the API server response back into dgd, and status is re-set after the update.
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
	assert.Equal(t, []string{"decode", "prefill"}, dgd.Status.RollingUpdate.UpdatedComponents)
}

func TestReconcileRollingUpdate_RecreateAtZeroWaitsForOldPodTermination(t *testing.T) {
	const oldWorkerHash = "oldhash0"

	tests := []struct {
		name         string
		initialPhase nvidiacomv1beta1.RollingUpdatePhase
	}{
		{
			name:         "in-progress rollout",
			initialPhase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
		},
		{
			name:         "stale-annotation recovery",
			initialPhase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(0)),
					Envs:          []corev1.EnvVar{{Name: "REVISION", Value: "v2"}},
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
					},
				},
			})
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: oldWorkerHash}
			dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
				Phase: tt.initialPhase,
			}
			newWorkerHash := betaDGDWorkersSpecHash(t, dgd)
			require.NotEqual(t, oldWorkerHash, newWorkerHash)

			makeZeroReplicaDCD := func(name, workerHash string) *nvidiacomv1beta1.DynamoComponentDeployment {
				return &nvidiacomv1beta1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       name,
						Namespace:  dgd.Namespace,
						Generation: 1,
						Labels: map[string]string{
							consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
							consts.KubeLabelDynamoWorkerHash:          workerHash,
						},
					},
					Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
							ComponentName: "worker",
							ComponentType: consts.ComponentTypeWorker,
							Replicas:      ptr.To(int32(0)),
						},
					},
					Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Component: &nvidiacomv1beta1.ComponentReplicaStatus{
							Replicas:      0,
							ReadyReplicas: ptr.To(int32(0)),
						},
					},
				}
			}
			oldDCD := makeZeroReplicaDCD(
				dynamo.GetDCDResourceName(dgd, "worker", oldWorkerHash),
				oldWorkerHash,
			)
			newDCD := makeZeroReplicaDCD(
				dynamo.GetDCDResourceName(dgd, "worker", newWorkerHash),
				newWorkerHash,
			)
			oldPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "terminating-old-worker",
					Namespace:         dgd.Namespace,
					DeletionTimestamp: ptr.To(metav1.Now()),
					Finalizers:        []string{"test.example/finalizer"},
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
						consts.KubeLabelDynamoComponent:           "worker",
						consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
						consts.KubeLabelDynamoWorkerHash:          oldWorkerHash,
						consts.KubeLabelDynamoSelector:            oldDCD.Name,
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			}

			r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD, newDCD, oldPod))
			ctx := context.Background()

			// Zero ready replicas satisfy the status-only completion check, but the
			// running old Pod must keep both the rollout and its hash in place.
			require.NoError(t, r.reconcileRollingUpdate(ctx, dgd))
			assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, dgd.Status.RollingUpdate.Phase)
			assert.Equal(t, oldWorkerHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldDCD), &nvidiacomv1beta1.DynamoComponentDeployment{}))

			// Once the old Pod reaches a terminal phase, the state machine may
			// complete the rollout and retire the old generation.
			cachedPod := &corev1.Pod{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(oldPod), cachedPod))
			cachedPod.Status.Phase = corev1.PodSucceeded
			require.NoError(t, r.Status().Update(ctx, cachedPod))

			require.NoError(t, r.reconcileRollingUpdate(ctx, dgd))
			assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
			assert.True(t, r.currentWorkerHashes(dgd).contains(newWorkerHash))
			err := r.Get(ctx, client.ObjectKeyFromObject(oldDCD), &nvidiacomv1beta1.DynamoComponentDeployment{})
			assert.True(t, apierrors.IsNotFound(err))

			// Replicas are excluded from the worker hash. A later scale-up
			// therefore stays on the completed generation.
			dgd.Spec.Components[0].Replicas = ptr.To(int32(1))
			trigger, err := r.shouldTriggerRollingUpdate(dgd)
			require.NoError(t, err)
			assert.False(t, trigger)
			require.NoError(t, r.reconcileRollingUpdate(ctx, dgd))
			assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
			assert.True(t, r.currentWorkerHashes(dgd).contains(newWorkerHash))
		})
	}
}

func TestGetWorkerInfoForWorkerHash(t *testing.T) {
	workerHash := "hash1234"

	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill},
		"decode":  {ComponentType: consts.ComponentTypeDecode},
	})

	// Create DCDs for prefill and decode with different ready counts
	prefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	decodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(prefillDCD, decodeDCD))
	ctx := context.Background()

	status, err := r.getWorkerInfoForWorkerHash(ctx, dgd, workerHash)
	require.NoError(t, err)

	assert.Len(t, status.components, 2)
	assert.Equal(t, int32(2), status.components[consts.ComponentTypePrefill].readyReplicas)
	assert.Equal(t, int32(1), status.components[consts.ComponentTypeDecode].readyReplicas)
	assert.Equal(t, int32(3), status.totalReadyWorkers) // 2 + 1
}

func TestMergeWorkerComponentStatuses(t *testing.T) {
	tests := []struct {
		name              string
		componentStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus
		oldWorkerStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus
		expected          map[string]nvidiacomv1beta1.ComponentReplicaStatus
	}{
		{
			name: "merges old and new for a single worker service",
			componentStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-newhash1"},
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
					RuntimeNamespace:  "dynamo-newhash1",
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          1,
					UpdatedReplicas:   0,
					ReadyReplicas:     ptr.To(int32(1)),
					AvailableReplicas: ptr.To(int32(1)),
					RuntimeNamespace:  "dynamo-oldhash1",
				},
			},
			expected: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:          3,
					UpdatedReplicas:   2, // Only new are "updated"
					ReadyReplicas:     ptr.To(int32(3)),
					AvailableReplicas: ptr.To(int32(3)),
					RuntimeNamespace:  "dynamo-oldhash1",
				},
			},
		},
		{
			name: "no old statuses - no-op",
			componentStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-prefill-newhash1"},
					Replicas:       2,
					ReadyReplicas:  ptr.To(int32(2)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{},
			expected: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-prefill-newhash1"},
					Replicas:       2,
					ReadyReplicas:  ptr.To(int32(2)),
				},
			},
		},
		{
			name:              "old exists but new doesn't yet",
			componentStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{},
			oldWorkerStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
					RuntimeNamespace:  "dynamo-oldhash1",
				},
			},
			expected: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          2,
					UpdatedReplicas:   0,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
					RuntimeNamespace:  "dynamo-oldhash1",
				},
			},
		},
		{
			name: "handles nil ReadyReplicas and AvailableReplicas on old",
			componentStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-newhash1"},
					Replicas:          2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-oldhash1"},
					Replicas:          1,
					ReadyReplicas:     nil,
					AvailableReplicas: nil,
				},
			},
			expected: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:     "Deployment",
					ComponentNames:    []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:          3,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(1)),
				},
			},
		},
		{
			name: "frontend status untouched by merge",
			componentStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"frontend": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-frontend"},
					Replicas:       1,
					ReadyReplicas:  ptr.To(int32(1)),
				},
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-prefill-newhash1"},
					Replicas:       2,
					ReadyReplicas:  ptr.To(int32(2)),
				},
			},
			oldWorkerStatuses: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-prefill-oldhash1"},
					Replicas:       1,
					ReadyReplicas:  ptr.To(int32(1)),
				},
			},
			expected: map[string]nvidiacomv1beta1.ComponentReplicaStatus{
				"frontend": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-frontend"},
					Replicas:       1,
					ReadyReplicas:  ptr.To(int32(1)),
				},
				"prefill": {
					ComponentKind:  "Deployment",
					ComponentNames: []string{"dgd-prefill-newhash1", "dgd-prefill-oldhash1"},
					Replicas:       3,
					ReadyReplicas:  ptr.To(int32(3)),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mergeWorkerComponentStatuses(tt.componentStatuses, tt.oldWorkerStatuses)
			assert.Equal(t, tt.expected, tt.componentStatuses)
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

		oldDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-prefill-oldhash1",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoNamespace:           "base",
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
					ComponentKind:    "Deployment",
					ComponentNames:   []string{"test-dgd-prefill-oldhash1"},
					RuntimeNamespace: "base-oldhash1",
					Replicas:         1,
					UpdatedReplicas:  0,
					ReadyReplicas:    ptr.To(int32(1)),
				},
			},
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD))
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      testNewWorkerHash,
			OldWorkerReplicaTargetsByComponent: map[string]int32{"prefill": 1},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"prefill": 2},
		}

		statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Len(t, statuses, 1)
		assert.Equal(t, []string{"test-dgd-prefill-oldhash1"}, statuses["prefill"].ComponentNames)
		assert.Equal(t, int32(1), statuses["prefill"].Replicas)
		assert.Equal(t, ptr.To(int32(1)), statuses["prefill"].ReadyReplicas)
		assert.Equal(t, "base-oldhash1", statuses["prefill"].RuntimeNamespace)
	})

	t.Run("multiple old generations selects newest nonzero target runtime namespace", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      ptr.To(int32(3)),
			},
		})

		now := metav1.Now()
		earlier := metav1.NewTime(now.Add(-1 * 60 * 1e9))

		oldestDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
					Replicas:      ptr.To(int32(0)),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					ComponentKind:    "Deployment",
					ComponentNames:   []string{"test-dgd-worker-hashaaaa"},
					RuntimeNamespace: "base-hashaaaa",
					Replicas:         0,
					ReadyReplicas:    ptr.To(int32(0)),
				},
			},
		})

		newerOldDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "test-dgd-worker-hashbbbb",
				Namespace:         "default",
				CreationTimestamp: now,
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoNamespace:           "base",
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
					ComponentKind:  "Deployment",
					ComponentNames: []string{"test-dgd-worker-hashbbbb"},
					Replicas:       2,
					ReadyReplicas:  ptr.To(int32(2)),
				},
			},
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(oldestDCD, newerOldDCD))
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "hashcccc",
			OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
			OldWorkerReplicaTargetsByDCD: map[string]int32{
				"test-dgd-worker-hashaaaa": 0,
				"test-dgd-worker-hashbbbb": 2,
			},
		}

		statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Equal(t, int32(2), statuses["worker"].Replicas)
		assert.Equal(t, ptr.To(int32(2)), statuses["worker"].ReadyReplicas)
		assert.Equal(t, "base-hashbbbb", statuses["worker"].RuntimeNamespace)

		rollingUpdateCtx.OldWorkerReplicaTargetsByDCD = map[string]int32{
			"test-dgd-worker-hashaaaa": 0,
			"test-dgd-worker-hashbbbb": 0,
		}
		statuses, err = r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)
		assert.Equal(t, "base-hashbbbb", statuses["worker"].RuntimeNamespace)
	})

	t.Run("old DCD not found - skips gracefully", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"prefill": {
				ComponentType: consts.ComponentTypePrefill,
				Replicas:      ptr.To(int32(2)),
			},
		})

		r := createTestReconcilerWithStatus(dgd)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      testNewWorkerHash,
			OldWorkerReplicaTargetsByComponent: map[string]int32{"prefill": 1},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"prefill": 2},
		}

		statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
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
		computedHash := betaDGDWorkersSpecHash(t, dgd)
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: "oldhash",
		}

		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD, workerDCD))
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["worker"])
	})

	t.Run("worker DCD with v2 hash suffix - finds annotation", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		legacyHash := legacyDGDWorkersSpecHash(t, dgd)
		v2Hash := betaDGDWorkersSpecHash(t, dgd)
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash:   legacyHash,
			consts.AnnotationCurrentWorkerHashV2: v2Hash,
		}

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-dgd-worker-" + v2Hash,
				Namespace: "default",
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					Annotations: map[string]string{
						consts.RestartAnnotation: "2025-01-01T00:00:00Z",
					},
				},
			},
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(workerDCD))
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

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

		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD))
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

		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD))
		ctx := context.Background()

		annotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
	})
}

func TestCheckComponentFullyUpdated(t *testing.T) {
	t.Run("worker with hash suffix - finds DCD", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		workerHash := legacyDGDWorkersSpecHash(t, dgd)
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash: workerHash,
		}

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-dgd-worker-" + workerHash,
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(workerDCD))
		ctx := context.Background()

		isReady, reason := r.checkComponentFullyUpdated(ctx, dgd, "worker")
		assert.True(t, isReady, "worker DCD should be ready")
		assert.Empty(t, reason)
	})

	t.Run("worker with v2 hash suffix - finds DCD", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: consts.ComponentTypeWorker,
			},
		})
		legacyHash := legacyDGDWorkersSpecHash(t, dgd)
		v2Hash := betaDGDWorkersSpecHash(t, dgd)
		dgd.Annotations = map[string]string{
			consts.AnnotationCurrentWorkerHash:   legacyHash,
			consts.AnnotationCurrentWorkerHashV2: v2Hash,
		}

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:       "test-dgd-worker-" + v2Hash,
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(workerDCD))
		ctx := context.Background()

		isReady, reason := r.checkComponentFullyUpdated(ctx, dgd, "worker")
		assert.True(t, isReady, "worker DCD should be ready")
		assert.Empty(t, reason)
	})

	t.Run("non-worker without hash suffix - finds DCD", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"frontend": {
				ComponentType: consts.ComponentTypeFrontend,
			},
		})

		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD))
		ctx := context.Background()

		isReady, reason := r.checkComponentFullyUpdated(ctx, dgd, "frontend")
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

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(workerDCD))
		ctx := context.Background()

		isReady, reason := r.checkComponentFullyUpdated(ctx, dgd, "worker")
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
	legacyWorkerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(legacyWorkerDCD))
	ctx := context.Background()

	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// DGD annotation should be set to the legacy sentinel, NOT the computed hash
	hash := r.getCurrentWorkerHash(dgd)
	assert.Equal(t, consts.LegacyWorkerHash, hash, "Hash should be legacy sentinel after migration")

	// Legacy DCD should now have the worker hash label backfilled
	updatedDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker", Namespace: "default"}, updatedDCD)
	require.NoError(t, err)
	assert.Equal(t, consts.LegacyWorkerHash, updatedDCD.Labels[consts.KubeLabelDynamoWorkerHash],
		"Legacy DCD should have worker hash label backfilled")

	desired, err := r.desiredWorkerHashes(dgd)
	require.NoError(t, err)
	require.NoError(t, r.completeRollingUpdate(ctx, dgd, desired.v1))

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.False(t, trigger)
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
	legacyPrefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	legacyDecodeDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Frontend DCD (not a worker, should not be touched)
	frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(legacyPrefillDCD, legacyDecodeDCD, frontendDCD))
	ctx := context.Background()

	err := r.initializeWorkerHashIfNeeded(ctx, dgd)
	require.NoError(t, err)

	// DGD should have legacy sentinel hash
	assert.Equal(t, consts.LegacyWorkerHash, r.getCurrentWorkerHash(dgd))

	// Both worker DCDs should have hash label backfilled
	for _, name := range []string{"test-dgd-prefill", "test-dgd-decode"} {
		dcd := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
		err = r.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, dcd)
		require.NoError(t, err)
		assert.Equal(t, consts.LegacyWorkerHash, dcd.Labels[consts.KubeLabelDynamoWorkerHash],
			"Worker DCD %s should have legacy hash label", name)
	}

	// Frontend should NOT have hash label
	fe := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
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

		legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD))
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

		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD))
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("ignores DCDs that already have hash label", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		hashedDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(hashedDCD))
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("ignores DCDs from other DGDs", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		otherDGDWorkerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(otherDGDWorkerDCD))
		ctx := context.Background()

		result, err := r.findLegacyWorkerDCDs(ctx, dgd)
		require.NoError(t, err)
		assert.Empty(t, result)
	})

	t.Run("no DCDs at all", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		r := createTestReconcilerWithStatus(dgd)
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
		legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD))
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

		currentDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(currentDCD))
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
		frontendDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		workerDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(frontendDCD, workerDCD))
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
		legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD))
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "newhash1",
			OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
			OldWorkerReplicaTargetsByDCD:       map[string]int32{"test-dgd-worker": 1},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 3},
		}

		err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		// Verify the legacy DCD was scaled down
		updatedDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
		err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker", Namespace: "default"}, updatedDCD)
		require.NoError(t, err)
		assert.Equal(t, int32(1), *updatedDCD.Spec.Replicas, "Legacy DCD should be scaled to 1")
	})

	t.Run("no-op when rolling update not in progress", func(t *testing.T) {
		dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker},
		})

		r := createTestReconcilerWithStatus(dgd)
		ctx := context.Background()

		// Empty OldWorkerComponentReplicas = not in progress
		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "samehash",
			OldWorkerReplicaTargetsByComponent: map[string]int32{},
			OldWorkerReplicaTargetsByDCD:       map[string]int32{},
			NewWorkerReplicaTargetsByComponent: map[string]int32{},
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

		legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD))
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "newhash1",
			OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
			OldWorkerReplicaTargetsByDCD:       map[string]int32{"test-dgd-worker": 1},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 3},
		}

		err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		// Replicas should remain at 1 (no patch needed)
		updatedDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
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
		legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
					ComponentKind:  "Deployment",
					ComponentNames: []string{"test-dgd-worker"},
					Replicas:       2,
					ReadyReplicas:  ptr.To(int32(2)),
				},
			},
		})

		r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD))
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "newhash1",
			OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 3},
		}

		statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)

		assert.Len(t, statuses, 1)
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

		r := createTestReconcilerWithStatus(dgd)
		ctx := context.Background()

		rollingUpdateCtx := dynamo.RollingUpdateContext{
			NewWorkerHash:                      "newhash1",
			OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
			NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
		}

		statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		require.NoError(t, err)
		assert.Empty(t, statuses)
	})
}

func TestDeleteOldWorkerDCDs_LegacyDCDs(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})

	// Legacy DCD with backfilled hash label
	legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// New DCD with real hash (should NOT be deleted)
	newDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD, newDCD))
	ctx := context.Background()

	err := r.deleteOldWorkerDCDs(ctx, dgd, "abc12345")
	require.NoError(t, err)

	// Verify legacy DCD is deleted and new DCD remains
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
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
	legacyDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation B (intermediate)
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation C (current)
	currentDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(legacyDCD, genBDCD, currentDCD))
	ctx := context.Background()

	err := r.deleteOldWorkerDCDs(ctx, dgd, "hashcccc")
	require.NoError(t, err)

	// Verify both old generations are deleted, only current remains
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
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
	genADCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation B
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation C (current)
	genCDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(genADCD, genBDCD, genCDCD))
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
	genADCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation B (newer old)
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(genADCD, genBDCD))
	ctx := context.Background()

	// oldNeeded = 2: newest old (B) should get 2, oldest (A) should get 0
	rollingUpdateCtx := dynamo.RollingUpdateContext{
		NewWorkerHash:                      "hashcccc",
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
		OldWorkerReplicaTargetsByDCD: map[string]int32{
			"test-dgd-worker-hashaaaa": 0,
			"test-dgd-worker-hashbbbb": 2,
		},
		NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 4},
	}

	err := r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
	require.NoError(t, err)

	// Newest old (B) should keep replicas (up to 2)
	updatedB := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashbbbb", Namespace: "default"}, updatedB)
	require.NoError(t, err)
	assert.Equal(t, int32(2), *updatedB.Spec.Replicas, "Newest old DCD should have 2 replicas")

	// Oldest (A) should be drained to 0
	updatedA := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashaaaa", Namespace: "default"}, updatedA)
	require.NoError(t, err)
	assert.Equal(t, int32(0), *updatedA.Spec.Replicas, "Oldest old DCD should be drained to 0")
}

func TestAllocateOldWorkerDCDReplicas(t *testing.T) {
	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-1 * 60 * 1e9))

	dcd := func(name string, createdAt metav1.Time, spec, available int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		return &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:              name,
				CreationTimestamp: createdAt,
			},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(spec),
				},
			},
			Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{
				Component: &nvidiacomv1beta1.ComponentReplicaStatus{
					Replicas:          spec,
					AvailableReplicas: ptr.To(available),
				},
			},
		}
	}

	tests := []struct {
		name      string
		oldTarget int32
		dcds      []*nvidiacomv1beta1.DynamoComponentDeployment
		want      map[string]int32
	}{
		{
			name:      "overlapping update keeps healthy original and drops unavailable intermediate",
			oldTarget: 15,
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				dcd("test-dgd-worker-hashaaaa", earlier, 15, 15),
				dcd("test-dgd-worker-hashbbbb", now, 10, 0),
			},
			want: map[string]int32{
				"test-dgd-worker-hashaaaa": 15,
				"test-dgd-worker-hashbbbb": 0,
			},
		},
		{
			name:      "available surplus removes replicas from oldest generation first",
			oldTarget: 3,
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				dcd("test-dgd-worker-hashaaaa", earlier, 3, 3),
				dcd("test-dgd-worker-hashbbbb", now, 1, 1),
			},
			want: map[string]int32{
				"test-dgd-worker-hashaaaa": 2,
				"test-dgd-worker-hashbbbb": 1,
			},
		},
		{
			name:      "degraded original fills remaining target from newest old generation",
			oldTarget: 15,
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				dcd("test-dgd-worker-hashaaaa", earlier, 15, 12),
				dcd("test-dgd-worker-hashbbbb", now, 10, 0),
			},
			want: map[string]int32{
				"test-dgd-worker-hashaaaa": 12,
				"test-dgd-worker-hashbbbb": 3,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := allocateOldWorkerDCDReplicas(tt.dcds, tt.oldTarget)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestScaleOldWorkerDCDs_MultipleOldGenerationsPreservesAvailableReplicas(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(20)),
		},
	})

	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-1 * 60 * 1e9)) // 1 minute earlier

	// Generation A (oldest): healthy and serving the minAvailable budget.
	genADCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
				Replicas:      ptr.To(int32(15)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				Replicas:          15,
				AvailableReplicas: ptr.To(int32(15)),
			},
		},
	})

	// Generation B (newer old): spec consumes rollout budget but has no serving replicas.
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
				Replicas:      ptr.To(int32(10)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				Replicas:          10,
				AvailableReplicas: ptr.To(int32(0)),
			},
		},
	})

	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: "hashbbbb",
	}

	r := createTestReconcilerWithStatus(dgd, withObjects(genADCD, genBDCD))
	ctx := context.Background()

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(15), rollingUpdateCtx.OldWorkerReplicaTargetsByComponent["worker"])
	assert.Equal(t, int32(15), rollingUpdateCtx.OldWorkerReplicaTargetsByDCD["test-dgd-worker-hashaaaa"])
	assert.Equal(t, int32(0), rollingUpdateCtx.OldWorkerReplicaTargetsByDCD["test-dgd-worker-hashbbbb"])
	assert.Equal(t, int32(0), rollingUpdateCtx.NewWorkerReplicaTargetsByComponent["worker"])

	err = r.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx)
	require.NoError(t, err)

	updatedA := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashaaaa", Namespace: "default"}, updatedA)
	require.NoError(t, err)
	assert.Equal(t, int32(15), *updatedA.Spec.Replicas, "Healthy old DCD should continue serving minAvailable")

	updatedB := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{})
	err = r.Get(ctx, types.NamespacedName{Name: "test-dgd-worker-hashbbbb", Namespace: "default"}, updatedB)
	require.NoError(t, err)
	assert.Equal(t, int32(0), *updatedB.Spec.Replicas, "Unavailable newer old DCD should be drained first")
}

func TestAggregateOldWorkerServiceStatuses_MultipleOldGenerations(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(4)),
		},
	})

	// Generation A
	genADCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
				ComponentKind:  "Deployment",
				ComponentNames: []string{"test-dgd-worker-hashaaaa"},
				Replicas:       1,
				ReadyReplicas:  ptr.To(int32(1)),
			},
		},
	})

	// Generation B
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
				ComponentKind:  "Deployment",
				ComponentNames: []string{"test-dgd-worker-hashbbbb"},
				Replicas:       2,
				ReadyReplicas:  ptr.To(int32(2)),
			},
		},
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(genADCD, genBDCD))
	ctx := context.Background()

	rollingUpdateCtx := dynamo.RollingUpdateContext{
		NewWorkerHash:                      "hashcccc",
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 3},
		NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 4},
	}

	statuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
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
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	// Generation A (old)
	genADCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation B (intermediate, now also old)
	genBDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	// Generation C (new, not yet ready)
	genCDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(genADCD, genBDCD, genCDCD))
	ctx := context.Background()

	err := r.continueRollingUpdate(ctx, dgd, newWorkerHash)
	require.NoError(t, err)

	// Both A and B have ready replicas, C has 0 — rolling update not complete
	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, rollingUpdateStatus.Phase)
	assert.Empty(t, rollingUpdateStatus.UpdatedComponents, "No services should be fully updated yet")
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

func TestOldWorkerDCDsAtZero(t *testing.T) {
	newDCD := func(specReplicas int32, generation, observedGeneration int64, statusReplicas *int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{Generation: generation},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					Replicas: ptr.To(specReplicas),
				},
			},
			Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{
				ObservedGeneration: observedGeneration,
			},
		}
		if statusReplicas != nil {
			dcd.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{Replicas: *statusReplicas}
		}
		return dcd
	}

	tests := []struct {
		name string
		dcds []*nvidiacomv1beta1.DynamoComponentDeployment
		want bool
	}{
		{
			name: "no old generations",
			want: true,
		},
		{
			name: "old desired replicas have not been drained",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(1, 1, 1, ptr.To(int32(1))),
			},
			want: false,
		},
		{
			name: "default desired replica has not been drained",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				{
					ObjectMeta: metav1.ObjectMeta{Generation: 1},
					Status: nvidiacomv1beta1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Component:          &nvidiacomv1beta1.ComponentReplicaStatus{},
					},
				},
			},
			want: false,
		},
		{
			name: "scale-down has not been observed",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(0, 2, 1, ptr.To(int32(0))),
			},
			want: false,
		},
		{
			name: "replica status has not been reported",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(0, 2, 2, nil),
			},
			want: false,
		},
		{
			name: "old non-terminated replicas remain",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(0, 2, 2, ptr.To(int32(1))),
			},
			want: false,
		},
		{
			name: "all old generations are observed at zero",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(0, 2, 2, ptr.To(int32(0))),
				newDCD(0, 4, 4, ptr.To(int32(0))),
			},
			want: true,
		},
		{
			name: "one remaining old generation blocks cutover",
			dcds: []*nvidiacomv1beta1.DynamoComponentDeployment{
				newDCD(0, 2, 2, ptr.To(int32(0))),
				newDCD(0, 4, 4, ptr.To(int32(1))),
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, oldWorkerDCDsAtZero(tt.dcds))
		})
	}
}

func TestOldWorkerPodsTerminated(t *testing.T) {
	oldDCDs := []*nvidiacomv1beta1.DynamoComponentDeployment{
		{ObjectMeta: metav1.ObjectMeta{Name: "old-worker-a"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "old-worker-b"}},
	}
	newPod := func(name, selector string, phase corev1.PodPhase) corev1.Pod {
		return corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
				Labels: map[string]string{
					consts.KubeLabelDynamoSelector: selector,
				},
			},
			Status: corev1.PodStatus{Phase: phase},
		}
	}

	terminating := newPod("terminating", "old-worker-a", corev1.PodRunning)
	terminating.DeletionTimestamp = ptr.To(metav1.Now())

	tests := []struct {
		name string
		pods []corev1.Pod
		want bool
	}{
		{
			name: "no pods",
			want: true,
		},
		{
			name: "running old pod blocks",
			pods: []corev1.Pod{newPod("running", "old-worker-a", corev1.PodRunning)},
			want: false,
		},
		{
			name: "terminating running old pod blocks",
			pods: []corev1.Pod{terminating},
			want: false,
		},
		{
			name: "pending old pod blocks",
			pods: []corev1.Pod{newPod("pending", "old-worker-a", corev1.PodPending)},
			want: false,
		},
		{
			name: "unknown old pod blocks",
			pods: []corev1.Pod{newPod("unknown", "old-worker-a", corev1.PodUnknown)},
			want: false,
		},
		{
			name: "terminal old pods do not block",
			pods: []corev1.Pod{
				newPod("failed", "old-worker-a", corev1.PodFailed),
				newPod("succeeded", "old-worker-b", corev1.PodSucceeded),
			},
			want: true,
		},
		{
			name: "non-terminal new generation pod does not block",
			pods: []corev1.Pod{newPod("new", "new-worker", corev1.PodRunning)},
			want: true,
		},
		{
			name: "pod without a workload selector does not block",
			pods: []corev1.Pod{newPod("checkpoint", "", corev1.PodRunning)},
			want: true,
		},
		{
			name: "one non-terminal old generation blocks",
			pods: []corev1.Pod{
				newPod("failed", "old-worker-a", corev1.PodFailed),
				newPod("running", "old-worker-b", corev1.PodRunning),
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, oldWorkerPodsTerminated(oldDCDs, tt.pods))
		})
	}
}

func TestDGDComponentPodIndexValues(t *testing.T) {
	tests := []struct {
		name string
		obj  client.Object
		want []string
	}{
		{
			name: "DGD component pod",
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
				consts.KubeLabelDynamoComponent:           "decode",
			}}},
			want: []string{"graph/decode"},
		},
		{
			name: "missing DGD label",
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				consts.KubeLabelDynamoComponent: "decode",
			}}},
		},
		{
			name: "missing component label",
			obj: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
			}}},
		},
		{
			name: "non-Pod object",
			obj:  &corev1.ConfigMap{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, dgdComponentPodIndexValues(tt.obj))
		})
	}
}

func TestListDGDComponentPodsUsesCompositeIndex(t *testing.T) {
	const (
		namespace = "inference"
		dgdName   = "graph"
		component = "decode"
	)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: dgdName, Namespace: namespace},
	}
	newPod := func(namespace, name, graph, component string) *corev1.Pod {
		return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: graph,
				consts.KubeLabelDynamoComponent:           component,
			},
		}}
	}

	reconciler := createTestReconcilerWithStatus(
		dgd,
		withObjects(
			newPod(namespace, "matching-a", dgdName, component),
			newPod(namespace, "matching-b", dgdName, component),
			newPod(namespace, "other-component", dgdName, "prefill"),
			newPod(namespace, "other-dgd", "other-graph", component),
			newPod("other-namespace", "other-namespace", dgdName, component),
		),
	)

	pods, err := reconciler.listDGDComponentPods(context.Background(), dgd, component)
	require.NoError(t, err)

	names := make([]string, 0, len(pods))
	for i := range pods {
		names = append(names, pods[i].Name)
	}
	sort.Strings(names)
	assert.Equal(t, []string{"matching-a", "matching-b"}, names)
}

func TestDGDWorkerPodEventPredicate(t *testing.T) {
	newPod := func(phase corev1.PodPhase) *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "worker",
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoComponent:           "worker",
					consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
					consts.KubeLabelDynamoSelector:            "old-worker",
				},
			},
			Status: corev1.PodStatus{Phase: phase},
		}
	}

	pred := dgdWorkerPodEventPredicate()
	running := newPod(corev1.PodRunning)
	failed := newPod(corev1.PodFailed)
	succeeded := newPod(corev1.PodSucceeded)
	deleting := running.DeepCopy()
	deleting.DeletionTimestamp = ptr.To(metav1.Now())
	unrelated := running.DeepCopy()
	delete(unrelated.Labels, consts.KubeLabelDynamoSelector)
	frontend := running.DeepCopy()
	frontend.Labels[consts.KubeLabelDynamoComponentType] = consts.ComponentTypeFrontend
	moved := running.DeepCopy()
	moved.Labels[consts.KubeLabelDynamoSelector] = "another-old-worker"

	assert.True(t, pred.Create(event.CreateEvent{Object: running}), "pod creation changes drain membership")
	assert.False(t, pred.Create(event.CreateEvent{Object: frontend}), "non-worker pods must be ignored")
	assert.True(t, pred.Delete(event.DeleteEvent{Object: running}), "pod deletion can unblock a drain")
	assert.False(t, pred.Delete(event.DeleteEvent{Object: unrelated}), "unmanaged pods must be ignored")
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: running, ObjectNew: failed}), "transition to Failed can unblock a drain")
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: running, ObjectNew: succeeded}), "transition to Succeeded can unblock a drain")
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: running, ObjectNew: deleting}), "deletion timestamp alone does not make a pod terminal")
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: failed, ObjectNew: succeeded}), "terminal-to-terminal transitions are irrelevant")
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: running, ObjectNew: moved}), "selector changes move the pod between drain memberships")
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: unrelated, ObjectNew: running}), "becoming a managed worker pod changes drain membership")
	assert.False(t, pred.Generic(event.GenericEvent{Object: running}))
}

func TestMapDGDWorkerPodToRequests(t *testing.T) {
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name:      "worker",
		Namespace: "default",
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
			consts.KubeLabelDynamoComponent:           "worker",
			consts.KubeLabelDynamoComponentType:       consts.ComponentTypeWorker,
			consts.KubeLabelDynamoSelector:            "old-worker",
		},
	}}

	requests := mapDGDWorkerPodToRequests(context.Background(), pod)
	require.Len(t, requests, 1)
	assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "test-dgd"}, requests[0].NamespacedName)

	delete(pod.Labels, consts.KubeLabelDynamoGraphDeploymentName)
	assert.Empty(t, mapDGDWorkerPodToRequests(context.Background(), pod))
}

// --- reconcileRollingUpdate state machine tests ---

func TestReconcileRollingUpdate_NoChange(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	hash := legacyDGDWorkersSpecHash(t, dgd)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   hash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Phase should stay Completed — no spec change
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_SpecChangeStartsRollout(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "stale000"}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should transition to Pending (new rollout started)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
	assert.NotNil(t, dgd.Status.RollingUpdate.StartTime)
}

func TestReconcileRollingUpdate_PendingToInProgress(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhasePending,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_StuckDetection(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	hash := legacyDGDWorkersSpecHash(t, dgd)
	// Hash matches current but phase is InProgress — stuck
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   hash,
		consts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
	}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should auto-complete
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_NewRollingUpdate(t *testing.T) {
	newHash := "newhash1"
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
	}

	// Create a DCD with the new hash that has ready replicas — stale annotation scenario
	newDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
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
	})

	r := createTestReconcilerWithStatus(dgd, withObjects(newDCD))

	// When computed hash != current hash and no DCDs exist with computed hash, start rollout.
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	// Should start a new rolling update (Pending) since computed hash DCDs don't exist
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_StaleAnnotationRequiresAllNewWorkersReady(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill},
		"decode":  {ComponentType: consts.ComponentTypeDecode},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: testOldWorkerHash}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
	}
	newHash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, testOldWorkerHash, newHash)

	newPrefillDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.GetDCDResourceName(dgd, "prefill", newHash),
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          newHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				ReadyReplicas: ptr.To(int32(1)),
			},
		},
	})
	r := createTestReconcilerWithStatus(dgd, withObjects(newPrefillDCD))

	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)

	assert.Equal(t, testOldWorkerHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_StaleAnnotationUpdatesAfterAllNewWorkersReady(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill},
		"decode":  {ComponentType: consts.ComponentTypeDecode},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: testOldWorkerHash}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
	}
	newHash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, testOldWorkerHash, newHash)

	makeReadyDCD := func(componentName, componentType string) *nvidiacomv1beta1.DynamoComponentDeployment {
		return betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      dynamo.GetDCDResourceName(dgd, componentName, newHash),
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
					consts.KubeLabelDynamoWorkerHash:          newHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: componentType,
					ServiceName:   componentName,
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					ReadyReplicas: ptr.To(int32(1)),
				},
			},
		})
	}
	r := createTestReconcilerWithStatus(
		dgd,
		withObjects(
			makeReadyDCD("prefill", consts.ComponentTypePrefill),
			makeReadyDCD("decode", consts.ComponentTypeDecode),
		),
	)

	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)

	assert.NotContains(t, dgd.Annotations, consts.AnnotationCurrentWorkerHash)
	assert.Equal(t, newHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestReconcileRollingUpdate_NonePhaseStartsRollout(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: "oldhash0"}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseNone,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhasePending, dgd.Status.RollingUpdate.Phase)
	assert.NotNil(t, dgd.Status.RollingUpdate.StartTime)
	assert.Nil(t, dgd.Status.RollingUpdate.UpdatedComponents)
}

func TestReconcileRollingUpdate_StuckDetection_CompletesViaCompleteRollingUpdate(t *testing.T) {
	// Stuck case: hashes match but phase is InProgress (e.g., operator restarted between
	// annotation write and status persistence). Should call completeRollingUpdate which
	// cleans up old DCDs, updates annotation, and sets Completed.
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill},
		"decode":  {ComponentType: consts.ComponentTypeDecode},
	})
	legacyHash := legacyDGDWorkersSpecHash(t, dgd)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   legacyHash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
		Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
	}

	r := createTestReconcilerWithStatus(dgd)
	err := r.reconcileRollingUpdate(context.Background(), dgd)
	require.NoError(t, err)

	// Phase should be Completed
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
	// EndTime should be set
	assert.NotNil(t, dgd.Status.RollingUpdate.EndTime)
	// UpdatedComponents should contain all worker services
	assert.Contains(t, dgd.Status.RollingUpdate.UpdatedComponents, "prefill")
	assert.Contains(t, dgd.Status.RollingUpdate.UpdatedComponents, "decode")
	// Completion records both active compatibility hashes.
	assert.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	assert.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
}

func TestBuildRollingUpdateContext(t *testing.T) {
	makeOldDCD := func(dgdName, serviceName, componentType, workerHash string, specReplicas, statusReplicas, availableReplicas int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		if workerHash == "" {
			workerHash = testOldWorkerHash
		}
		return betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      dgdName + "-" + serviceName + "-" + workerHash[:8],
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgdName,
					consts.KubeLabelDynamoWorkerHash:          workerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: componentType,
					ServiceName:   serviceName,
					Replicas:      ptr.To(specReplicas),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					Replicas:          statusReplicas,
					AvailableReplicas: ptr.To(availableReplicas),
				},
			},
		})
	}

	makeNewDCD := func(dgdName, serviceName, componentType, workerHash string, specReplicas, statusReplicas, availableReplicas int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		return betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      dgdName + "-" + serviceName + "-" + workerHash[:8],
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgdName,
					consts.KubeLabelDynamoWorkerHash:          workerHash,
				},
			},
			Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: componentType,
					ServiceName:   serviceName,
					Replicas:      ptr.To(specReplicas),
				},
			},
			Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
				Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
					Replicas:          statusReplicas,
					AvailableReplicas: ptr.To(availableReplicas),
				},
			},
		})
	}
	makeWorkerPod := func(name, dgdName, componentName, componentType, workerHash, selector string, phase corev1.PodPhase) *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "default",
				Labels: map[string]string{
					consts.KubeLabelDynamoGraphDeploymentName: dgdName,
					consts.KubeLabelDynamoComponent:           componentName,
					consts.KubeLabelDynamoComponentType:       componentType,
					consts.KubeLabelDynamoWorkerHash:          workerHash,
					consts.KubeLabelDynamoSelector:            selector,
				},
			},
			Status: corev1.PodStatus{Phase: phase},
		}
	}
	makeDefaultReplicaOldDCD := func(dgdName, serviceName, componentType, workerHash string, statusReplicas, availableReplicas int32) *nvidiacomv1beta1.DynamoComponentDeployment {
		dcd := makeOldDCD(dgdName, serviceName, componentType, workerHash, 1, statusReplicas, availableReplicas)
		dcd.Spec.Replicas = nil
		return dcd
	}

	tests := []struct {
		name                           string
		services                       map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		dgdSpecAnnotations             map[string]string
		preserveAlphaComponentMetadata bool
		oldDCDs                        func(newHash string) []runtime.Object
		newDCDs                        func(newHash string) []runtime.Object
		pods                           func(newHash string) []runtime.Object
		expectedOld                    map[string]int32
		expectedNew                    map[string]int32
	}{
		{
			name: "normal rollout start - all old healthy, no new pods yet",
			// desired=10, maxSurge=0, maxUnavailable=2
			// old: spec=10, available=10, actual=10 | new: spec=0, available=0, actual=0
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "0",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "2",
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 10, 10, 10),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 8},
			expectedNew: map[string]int32{"worker": 0}, // can't surge yet, need to wait for old replicas to be terminated
		},
		{
			name: "default old replica counts as one for maxUnavailable zero",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Annotations: map[string]string{
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeDefaultReplicaOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 1, 1),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 1},
			expectedNew: map[string]int32{"worker": 1},
		},
		{
			name: "surge budget uses spec not actual",
			// desired=10, maxSurge=0, maxUnavailable=2
			// old: spec=8, available=8, actual=9 | new: spec=0, available=0, actual=0
			// Surge budget = 10+0-oldSpec(8)-newSpec(0) = 2 (actual is irrelevant for surge)
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "0",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "2",
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 8, 9, 8),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 8},
			expectedNew: map[string]int32{"worker": 2}, // budget from Spec: 10+0-8-0=2
		},
		{
			name: "old fleet degraded - should protect healthy old pods",
			// desired=10, maxSurge=3, maxUnavailable=2
			// old: spec=8, available=4, actual=8 | new: spec=3, available=3, actual=3
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
					// Default annotations: 25% surge, 25% unavailable
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 8, 8, 4),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "worker", consts.ComponentTypeWorker, newHash, 3, 3, 3),
				}
			},
			expectedOld: map[string]int32{"worker": 5},
			expectedNew: map[string]int32{"worker": 5},
		},
		{
			name: "some unhealthy old pods - cleanup frees resources",
			// desired=10, maxSurge=3, maxUnavailable=2
			// old: spec=10, available=6, actual=10 | new: spec=0, available=0, actual=0
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 10, 10, 6),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 8},
			expectedNew: map[string]int32{"worker": 3},
		},
		{
			name: "terminating pods - surge uses spec, scheduler enforces resources",
			// desired=10, maxSurge=3, maxUnavailable=2
			// old: spec=5, available=5, actual=8 (3 Terminating, still holding GPUs)
			// new: spec=5, available=5, actual=5
			// Surge budget uses Spec: 10+3-5-5=3, newTarget=min(10,5+3)=8
			// Scheduler prevents new pods from scheduling until GPUs are freed.
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 5, 8, 5),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "worker", consts.ComponentTypeWorker, newHash, 5, 5, 5),
				}
			},
			expectedOld: map[string]int32{"worker": 3},
			expectedNew: map[string]int32{"worker": 8},
		},
		{
			name: "multi-service independence - prefill done, decode mid-rollout",
			// prefill: desired=4, maxSurge=1, maxUnavailable=1
			//   old: spec=0, available=0, actual=0 | new: spec=4, available=4, actual=4
			// decode: desired=8, maxSurge=2, maxUnavailable=2
			//   old: spec=6, available=6, actual=6 | new: spec=2, available=0, actual=2
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"prefill": {
					ComponentType: consts.ComponentTypePrefill,
					Replicas:      ptr.To(int32(4)),
				},
				"decode": {
					ComponentType: consts.ComponentTypeDecode,
					Replicas:      ptr.To(int32(8)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "prefill", consts.ComponentTypePrefill, "", 0, 0, 0),
					makeOldDCD("test-dgd", "decode", consts.ComponentTypeDecode, "", 6, 6, 6),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "prefill", consts.ComponentTypePrefill, newHash, 4, 4, 4),
					makeNewDCD("test-dgd", "decode", consts.ComponentTypeDecode, newHash, 2, 2, 0),
				}
			},
			expectedOld: map[string]int32{"prefill": 0, "decode": 6},
			expectedNew: map[string]int32{"prefill": 4, "decode": 4},
		},
		{
			name: "new pods unavailable - hold old until new becomes Ready",
			// desired=10, maxSurge=3, maxUnavailable=2
			// old: spec=10, available=6, actual=10 | new: spec=4, available=0, actual=4
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 10, 10, 6),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "worker", consts.ComponentTypeWorker, newHash, 4, 4, 0),
				}
			},
			expectedOld: map[string]int32{"worker": 8}, // newUnavailable shrinks scale-down budget; hold unhealthy old
			expectedNew: map[string]int32{"worker": 4}, // no surge: actual already at desired+maxSurge-1
		},
		{
			name: "multiple old generations - aggregate state across hashes",
			// desired=10, maxSurge=3, maxUnavailable=2
			// old (2 gens, 4+4): spec=8, available=8, actual=8 | new: spec=2, available=2, actual=2
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(10)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "oldhash0", 4, 4, 4),
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, testOldWorkerHash, 4, 4, 4),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "worker", consts.ComponentTypeWorker, newHash, 2, 2, 2),
				}
			},
			expectedOld: map[string]int32{"worker": 6}, // aggregated across both old gens
			expectedNew: map[string]int32{"worker": 5},
		},
		{
			name: "recreate drains old generation before starting new generation",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy:                    string(common.DeploymentStrategyRecreate),
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "100%",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 3, 3, 3),
				}
			},
			newDCDs: func(newHash string) []runtime.Object {
				return []runtime.Object{
					makeNewDCD("test-dgd", "worker", consts.ComponentTypeWorker, newHash, 2, 2, 2),
				}
			},
			expectedOld: map[string]int32{"worker": 0},
			expectedNew: map[string]int32{"worker": 0},
		},
		{
			name: "recreate starts new generation after old generation reaches zero",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 0, 0, 0),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 0},
			expectedNew: map[string]int32{"worker": 3},
		},
		{
			name: "recreate waits for terminating old pod after DCD reaches zero",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
					},
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 0, 0, 0),
				}
			},
			pods: func(_ string) []runtime.Object {
				pod := makeWorkerPod(
					"terminating-old-worker",
					"test-dgd",
					"worker",
					consts.ComponentTypeWorker,
					testOldWorkerHash,
					"test-dgd-worker-"+testOldWorkerHash[:8],
					corev1.PodRunning,
				)
				pod.DeletionTimestamp = ptr.To(metav1.Now())
				pod.Finalizers = []string{"test.example/finalizer"}
				return []runtime.Object{pod}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 0},
			expectedNew: map[string]int32{"worker": 0},
		},
		{
			name: "recreate inherited from DGD spec annotations drains old generation",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
				},
			},
			dgdSpecAnnotations: map[string]string{
				KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 3, 3, 3),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 0},
			expectedNew: map[string]int32{"worker": 0},
		},
		{
			name: "recreate preserved from v1alpha1 component annotations drains old generation",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
					},
				},
			},
			preserveAlphaComponentMetadata: true,
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "worker", consts.ComponentTypeWorker, "", 3, 3, 3),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"worker": 0},
			expectedNew: map[string]int32{"worker": 0},
		},
		{
			name: "recreate and rolling update components progress independently",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"prefill": {
					ComponentType: consts.ComponentTypePrefill,
					Replicas:      ptr.To(int32(2)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
					},
				},
				"decode": {
					ComponentType: consts.ComponentTypeDecode,
					Replicas:      ptr.To(int32(4)),
				},
			},
			oldDCDs: func(_ string) []runtime.Object {
				return []runtime.Object{
					makeOldDCD("test-dgd", "prefill", consts.ComponentTypePrefill, "", 2, 2, 2),
					makeOldDCD("test-dgd", "decode", consts.ComponentTypeDecode, "", 4, 4, 4),
				}
			},
			newDCDs:     func(_ string) []runtime.Object { return nil },
			expectedOld: map[string]int32{"prefill": 0, "decode": 3},
			expectedNew: map[string]int32{"prefill": 0, "decode": 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var dgd *nvidiacomv1beta1.DynamoGraphDeployment
			if tt.preserveAlphaComponentMetadata {
				dgd = &nvidiacomv1beta1.DynamoGraphDeployment{}
				err := (&nvidiacomv1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
					Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
						Services: tt.services,
					},
				}).ConvertTo(dgd)
				require.NoError(t, err)
				require.Len(t, dgd.Spec.Components, 1)
				require.Empty(t, dynamo.GetPodTemplateAnnotations(&dgd.Spec.Components[0])[KubeAnnotationDeploymentStrategy],
					"test setup must keep the v1alpha1 annotation in compatibility metadata")
			} else {
				dgd = createTestDGD("test-dgd", tt.services)
			}

			dgd.Spec.Annotations = tt.dgdSpecAnnotations
			if dgd.Annotations == nil {
				dgd.Annotations = make(map[string]string)
			}
			dgd.Annotations[consts.AnnotationCurrentWorkerHash] = testOldWorkerHash

			// Compute the actual new DCD label hash from the DGD spec.
			newHash := betaDGDWorkersSpecHash(t, dgd)
			require.NotEqual(t, testOldWorkerHash, newHash, "test setup: computed hash must differ from old hash")

			// Collect all mock objects
			var objs []runtime.Object
			if tt.oldDCDs != nil {
				objs = append(objs, tt.oldDCDs(newHash)...)
			}
			if tt.newDCDs != nil {
				objs = append(objs, tt.newDCDs(newHash)...)
			}
			if tt.pods != nil {
				objs = append(objs, tt.pods(newHash)...)
			}

			r := createTestReconcilerWithStatus(dgd, withObjects(objs...))
			ctx := context.Background()

			result, err := r.buildRollingUpdateContext(ctx, dgd)
			assert.NoError(t, err)

			assert.Equal(t, newHash, result.NewWorkerHash)
			for svc, expectedOld := range tt.expectedOld {
				assert.Equal(t, expectedOld, result.OldWorkerReplicaTargetsByComponent[svc],
					"old replicas for service %s", svc)
			}
			for svc, expectedNew := range tt.expectedNew {
				assert.Equal(t, expectedNew, result.NewWorkerReplicaTargetsByComponent[svc],
					"new replicas for service %s", svc)
			}
		})
	}
}

func TestBuildRollingUpdateContext_NoNewDCDExists(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(10)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}

	oldDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-" + testOldWorkerHash[:8],
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
				Replicas:      ptr.To(int32(10)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				Replicas:          10,
				AvailableReplicas: ptr.To(int32(10)),
			},
		},
	})

	newHash := betaDGDWorkersSpecHash(t, dgd)
	assert.NotEqual(t, testOldWorkerHash, newHash, "test setup: computed hash must differ from old hash")

	r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD))
	ctx := context.Background()

	result, err := r.buildRollingUpdateContext(ctx, dgd)

	assert.NoError(t, err, "IsNotFound on the new-hash DCD must not produce an error")
	assert.Equal(t, newHash, result.NewWorkerHash)
	// Math runs with newState={0,0,0}: drain old to minAvailable, surge new from zero.
	assert.Equal(t, int32(8), result.OldWorkerReplicaTargetsByComponent["worker"])
	assert.Equal(t, int32(3), result.NewWorkerReplicaTargetsByComponent["worker"])
}

func TestBuildRollingUpdateContext_ListOldDCDsError(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(10)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}

	assert.NotEqual(t, testOldWorkerHash, betaDGDWorkersSpecHash(t, dgd),
		"test setup: computed hash must differ so we proceed past the early-return")

	injectedErr := errors.New("simulated apiserver list failure")
	funcs := interceptor.Funcs{
		List: func(_ context.Context, _ client.WithWatch, _ client.ObjectList, _ ...client.ListOption) error {
			return injectedErr
		},
	}
	r := createTestReconcilerWithStatus(dgd, withInterceptor(funcs))
	ctx := context.Background()

	_, err := r.buildRollingUpdateContext(ctx, dgd)

	assert.Error(t, err)
	assert.ErrorIs(t, err, injectedErr, "List error must be wrapped and propagated, not swallowed")
	assert.Contains(t, err.Error(), "failed to get old worker component states",
		"error must originate from the old-DCD List path, not some other call")
}

func TestBuildRollingUpdateContext_ListPodsError(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(3)),
			Annotations: map[string]string{
				KubeAnnotationDeploymentStrategy: string(common.DeploymentStrategyRecreate),
			},
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}
	oldDCD := betaDCD(t, &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd-worker-" + testOldWorkerHash[:8],
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
				Replicas:      ptr.To(int32(0)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{Replicas: 0},
		},
	})

	injectedErr := errors.New("simulated pod list failure")
	funcs := interceptor.Funcs{
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, ok := list.(*corev1.PodList); ok {
				return injectedErr
			}
			return c.List(ctx, list, opts...)
		},
	}
	r := createTestReconcilerWithStatus(dgd, withObjects(oldDCD), withInterceptor(funcs))

	_, err := r.buildRollingUpdateContext(context.Background(), dgd)

	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)
	assert.Contains(t, err.Error(), "failed to list pods for DGD test-dgd component worker")
}

func TestBuildRollingUpdateContext_GetNewDCDError(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(10)),
		},
	})
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash: testOldWorkerHash,
	}

	require.NotEqual(t, testOldWorkerHash, legacyDGDWorkersSpecHash(t, dgd),
		"test setup: computed hash must differ so we proceed past the early-return")

	injectedErr := errors.New("simulated apiserver get failure")
	funcs := interceptor.Funcs{
		Get: func(_ context.Context, _ client.WithWatch, _ client.ObjectKey, _ client.Object, _ ...client.GetOption) error {
			return injectedErr
		},
	}
	r := createTestReconcilerWithStatus(dgd, withInterceptor(funcs))
	ctx := context.Background()

	_, err := r.buildRollingUpdateContext(ctx, dgd)

	assert.Error(t, err)
	assert.ErrorIs(t, err, injectedErr, "non-NotFound Get error must be wrapped and propagated")
	assert.Contains(t, err.Error(), "failed to get new worker DCD",
		"error must originate from the new-DCD Get path, not some other call")
}
