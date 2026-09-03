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
	"time"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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

// withObjects seeds the rollout test client with runtime objects beyond the DGD.
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

func createTestDGDReconcilerWithStatus(dgd *nvidiacomv1beta1.DynamoGraphDeployment, opts ...testReconcilerOption) *DynamoGraphDeploymentReconciler {
	scheme := runtime.NewScheme()
	_ = nvidiacomv1alpha1.AddToScheme(scheme)
	_ = nvidiacomv1beta1.AddToScheme(scheme)
	_ = grovev1alpha1.AddToScheme(scheme)
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
		Recorder:      events.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commonController.RuntimeConfig{},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
	}
}

func createTestReconcilerWithStatus(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	opts ...testReconcilerOption,
) *dgdWorkerRolloutReconciler {
	reconciler := createTestDGDReconcilerWithStatus(dgd, opts...)
	return newDGDWorkerRolloutReconciler(reconciler.Client, reconciler.Recorder)
}

func newTestComponentWorkloadsReconciler(
	rollout *dgdWorkerRolloutReconciler,
) *componentWorkloadsReconciler {
	return newComponentWorkloadsReconciler(rollout.Client, rollout.GetRecorder(), rollout)
}

func TestGroveWorkerHashSuffixMigration(t *testing.T) {
	tests := []struct {
		name                    string
		existing                bool
		existingHash            string
		workerGenerationChanged bool
		wantSuffix              bool
	}{
		{name: "new PCS renders a suffix without a worker generation change", wantSuffix: true},
		{name: "legacy PCS with no generation change remains unsuffixed", existing: true},
		{name: "legacy PCS renders a suffix after a worker generation change", existing: true, workerGenerationChanged: true, wantSuffix: true},
		{name: "suffixed PCS continues rendering the suffix", existing: true, existingHash: "active", wantSuffix: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the existing worker PCS and worker-generation transition")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			var existing *grovev1alpha1.PodCliqueSet
			if tt.existing {
				existing = &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
					Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{Labels: map[string]string{consts.KubeLabelDynamoComponent: "worker"}}},
				}}}
				if tt.existingHash != "" {
					existing.Spec.Template.Cliques[0].Labels[consts.KubeLabelDynamoWorkerHash] = tt.existingHash
				}
			}

			t.Log("Verify suffix rendering from the worker generation")
			if got := shouldRenderGroveWorkerHashSuffix(dgd, existing, tt.workerGenerationChanged); got != tt.wantSuffix {
				t.Fatalf("shouldRenderGroveWorkerHashSuffix() = %t, want %t", got, tt.wantSuffix)
			}
		})
	}
}

func TestPlanUnsupportedWorkerHashTransitionIgnoresScaling(t *testing.T) {
	t.Log("Build an active worker generation and apply a replica-only change")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(1))},
	})
	activeHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: activeHash}
	dgd.GetComponentByName("worker").Replicas = ptr.To(int32(2))
	reconciler := createTestReconcilerWithStatus(dgd)

	t.Log("Plan the unsupported pathway transition")
	transition, err := reconciler.planUnsupportedWorkerHashTransition(dgd)
	require.NoError(t, err)

	t.Log("Verify scaling does not arm a worker generation migration")
	assert.False(t, transition.workerGenerationChanged)
	assert.False(t, transition.needsCommit())
}

func TestPlanUnsupportedWorkerHashTransitionDoesNotCommit(t *testing.T) {
	t.Log("Build a DGD with a persisted worker hash and a changed worker spec")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "old"}},
		},
	})
	currentHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}
	worker := dgd.GetComponentByName("worker")
	require.NotNil(t, worker)
	require.NotNil(t, worker.PodTemplate)
	require.NotEmpty(t, worker.PodTemplate.Spec.Containers)
	worker.PodTemplate.Spec.Containers[0].Env[0].Value = "new"
	reconciler := createTestReconcilerWithStatus(dgd)

	t.Log("Plan the unsupported worker hash transition")
	transition, err := reconciler.planUnsupportedWorkerHashTransition(dgd)
	require.NoError(t, err)

	t.Log("Verify planning detects the transition without mutating the DGD")
	require.True(t, transition.workerGenerationChanged)
	assert.Equal(t, currentHash, currentWorkerHashV2(dgd), "planning must not commit the DGD hash")
}

func TestGroveRenderDeploymentWorkerHashSuffix(t *testing.T) {
	tests := []struct {
		name             string
		workerHashSuffix bool
	}{
		{
			name:             "enabled",
			workerHashSuffix: true,
		},
		{
			name: "disabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD with the requested Grove worker suffix state")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})

			t.Log("Render the Grove deployment")
			rendered, err := groveRenderDeployment(dgd, nil, tt.workerHashSuffix)
			require.NoError(t, err)
			worker := rendered.GetComponentByName("worker")
			require.NotNil(t, worker)

			t.Log("Verify the rendered suffix and source DGD immutability")
			if tt.workerHashSuffix {
				wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
				require.NoError(t, err)
				require.NotNil(t, worker.PodTemplate)
				assert.Equal(t, wantHash, worker.PodTemplate.Labels[consts.KubeLabelDynamoWorkerHash])
			} else {
				assert.Nil(t, worker.PodTemplate)
			}
			assert.Nil(t, dgd.GetComponentByName("worker").PodTemplate)
		})
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
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash: legacyDGDWorkersSpecHash(t, dgd),
				}
			} else if tt.existingHash != "" {
				dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: tt.existingHash}
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
	desired, err := desiredWorkerHashes(dgd)
	require.NoError(t, err)
	assert.Equal(t, legacyHash, desired.v1)
	assert.Equal(t, v2Hash, desired.v2)

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	assert.False(t, trigger)
}

func TestShouldTriggerRollingUpdate_UsesResolvedRuntimeVersion(t *testing.T) {
	t.Log("define rollout decisions from current and desired runtime versions")
	tests := []struct {
		name            string
		image           string
		currentOverride string
		desiredOverride string
		wantTrigger     bool
	}{
		{
			name:            "triggers when the resolved override changes",
			image:           "registry.example/dynamo:custom",
			currentOverride: "1.5.0",
			desiredOverride: "1.5.1",
			wantTrigger:     true,
		},
		{
			name:            "does not trigger for an equivalent image-derived override",
			image:           "registry.example/dynamo:1.5.0",
			desiredOverride: "1.5.0",
			wantTrigger:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("create a worker DGD at the current runtime version")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker},
			})
			dgd.Spec.Components[0].RuntimeVersionOverride = tt.currentOverride
			dgd.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  consts.MainContainerName,
						Image: tt.image,
					}},
				},
			}

			t.Log("record the current v2 worker hash")
			dgd.Annotations = map[string]string{
				consts.AnnotationCurrentWorkerHashV2: betaDGDWorkersSpecHash(t, dgd),
			}

			t.Log("apply the desired runtime override")
			dgd.Spec.Components[0].RuntimeVersionOverride = tt.desiredOverride

			t.Log("evaluate whether the desired runtime requires rollout")
			r := createTestReconcilerWithStatus(dgd)
			trigger, err := r.shouldTriggerRollingUpdate(dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.wantTrigger, trigger)
		})
	}
}

func TestActiveWorkerHashCandidatesV2Only(t *testing.T) {
	t.Log("Build a normal v2-only generation")
	dgd := createTestDGD("test-dgd", nil)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHashV2: "v2",
	}

	t.Log("Verify candidate lookup cannot fall back to an empty legacy hash")
	assert.Equal(t, []string{"v2"}, activeWorkerHashCandidates(dgd, workerGenerationHashes{v2: "v2"}))
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
	legacyHash := legacyDGDWorkersSpecHash(t, dgd)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, legacyHash, v2Hash)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   legacyHash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}

	r := createTestReconcilerWithStatus(dgd)
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env = append(
		dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "NEW_WORKER_SETTING", Value: "true"},
	)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	newLegacyHash := legacyDGDWorkersSpecHash(t, dgd)
	require.NotEqual(t, v2Hash, newV2Hash)
	require.NotEqual(t, legacyHash, newLegacyHash)

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.True(t, trigger)
}

func TestLegacyAlphaHashCompatibility_V2OnlyChangeUsesNewV2Generation(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Envs:          []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
		},
	})
	dgd.Spec.BackendFramework = "vllm"
	legacyHash := legacyDGDWorkersSpecHash(t, dgd)
	v2Hash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   legacyHash,
		consts.AnnotationCurrentWorkerHashV2: v2Hash,
	}

	r := createTestReconcilerWithStatus(dgd)
	dgd.Spec.BackendFramework = "sglang"

	newLegacyHash := legacyDGDWorkersSpecHash(t, dgd)
	newV2Hash := betaDGDWorkersSpecHash(t, dgd)
	require.Equal(t, legacyHash, newLegacyHash)
	require.NotEqual(t, v2Hash, newV2Hash)

	require.NoError(t, r.migrateCurrentWorkerHashIfNeeded(context.Background(), dgd))
	require.Equal(t, legacyHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	require.Equal(t, v2Hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	trigger, err := r.shouldTriggerRollingUpdate(dgd)
	require.NoError(t, err)
	require.True(t, trigger)
}

func TestComponentProgram_SupportsManagedRollingUpdate(t *testing.T) {
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

			result := supportsManagedRollingUpdate(dgd)
			if result != tt.expected {
				t.Errorf("supportsManagedRollingUpdate() = %v, expected %v", result, tt.expected)
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
			status := r.getOrCreateRollingUpdateStatus(&dgd.Status)

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

			result := isRollingUpdateInProgress(&dgd.Status)

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

func TestComponentWorkloadsReconciler_GetExistingRestartAnnotationsDCD(t *testing.T) {
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
			consts.AnnotationCurrentWorkerHashV2: "oldhash",
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

		annotations, err := newTestComponentWorkloadsReconciler(r).getExistingRestartAnnotationsDCD(ctx, dgd, nil)
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

		annotations, err := newTestComponentWorkloadsReconciler(r).getExistingRestartAnnotationsDCD(ctx, dgd, nil)
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

		annotations, err := newTestComponentWorkloadsReconciler(r).getExistingRestartAnnotationsDCD(ctx, dgd, nil)
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

		annotations, err := newTestComponentWorkloadsReconciler(r).getExistingRestartAnnotationsDCD(ctx, dgd, nil)
		require.NoError(t, err)

		assert.Equal(t, "2025-01-01T00:00:00Z", annotations["frontend"])
	})
}

func TestComponentRestartProgressResolver_CheckComponentFullyUpdated(t *testing.T) {
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

		isReady, reason := newComponentRestartProgressResolver(r.Client).checkComponentFullyUpdated(ctx, dgd, "worker", nil)
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

		isReady, reason := newComponentRestartProgressResolver(r.Client).checkComponentFullyUpdated(ctx, dgd, "worker", nil)
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

		isReady, reason := newComponentRestartProgressResolver(r.Client).checkComponentFullyUpdated(ctx, dgd, "frontend", nil)
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

		isReady, reason := newComponentRestartProgressResolver(r.Client).checkComponentFullyUpdated(ctx, dgd, "worker", nil)
		assert.True(t, isReady, "worker DCD should be ready via fallback")
		assert.Empty(t, reason)
	})
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

func TestAllocateOldWorkerDCDReplicas(t *testing.T) {
	now := metav1.Now()
	earlier := metav1.NewTime(now.Add(-time.Minute))

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
			assert.Equal(t, tt.want, allocateOldWorkerDCDReplicas(tt.dcds, tt.oldTarget))
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

func TestManagedWorkerRolloutResolvesHashCohorts(t *testing.T) {
	const bridgeHash = "bridge-h"

	tests := []struct {
		name              string
		annotations       func(*nvidiacomv1beta1.DynamoGraphDeployment, string)
		observedSuffixes  func(string) []string
		mutate            func(*nvidiacomv1beta1.DynamoComponentDeployment)
		wantBridge        bool
		wantTargetPresent bool
		wantOld           int
	}{
		{
			name:              "canonical v2 target",
			observedSuffixes:  func(hash string) []string { return []string{hash} },
			wantTargetPresent: true,
		},
		{
			name: "persisted v1 bridge",
			annotations: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment, hash string) {
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash:   bridgeHash,
					consts.AnnotationCurrentWorkerHashV2: hash,
				}
			},
			observedSuffixes:  func(string) []string { return []string{bridgeHash} },
			wantBridge:        true,
			wantTargetPresent: true,
		},
		{
			name: "canonical v2 target wins over bridge",
			annotations: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment, hash string) {
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash:   bridgeHash,
					consts.AnnotationCurrentWorkerHashV2: hash,
				}
			},
			observedSuffixes:  func(hash string) []string { return []string{bridgeHash, hash} },
			wantTargetPresent: true,
			wantOld:           1,
		},
		{
			name: "v1 only is not a bridge",
			annotations: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment, _ string) {
				dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHash: bridgeHash}
			},
			observedSuffixes: func(string) []string { return []string{bridgeHash} },
			wantOld:          1,
		},
		{
			name: "v1 suffix with a different persisted v2 is old",
			annotations: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment, _ string) {
				dgd.Annotations = map[string]string{
					consts.AnnotationCurrentWorkerHash:   bridgeHash,
					consts.AnnotationCurrentWorkerHashV2: "previous-v2",
				}
			},
			observedSuffixes: func(string) []string { return []string{bridgeHash} },
			wantOld:          1,
		},
		{
			name:             "canonical name ignores mutable DCD evidence",
			observedSuffixes: func(hash string) []string { return []string{hash} },
			mutate: func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = "wrong-framework"
				dcd.Labels[consts.KubeLabelDynamoWorkerHash] = "wrong-label"
				dcd.Spec.PodTemplate.Labels[consts.KubeLabelDynamoWorkerHash] = "wrong-template-label"
				for i := range dcd.Spec.PodTemplate.Spec.Containers {
					for j := range dcd.Spec.PodTemplate.Spec.Containers[i].Env {
						if dcd.Spec.PodTemplate.Spec.Containers[i].Env[j].Name == "DYN_NAMESPACE_WORKER_SUFFIX" {
							dcd.Spec.PodTemplate.Spec.Containers[i].Env[j].Value = "wrong-worker-suffix"
						}
					}
				}
			},
			wantTargetPresent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(1))},
			})
			dgd.UID = types.UID("dgd-uid")
			desiredHash := betaDGDWorkersSpecHash(t, dgd)
			if tt.annotations != nil {
				tt.annotations(dgd, desiredHash)
			}

			objects := make([]runtime.Object, 0, len(tt.observedSuffixes(desiredHash)))
			for _, suffix := range tt.observedSuffixes(desiredHash) {
				dcd := managedInventoryWorkerDCD(t, dgd, suffix)
				if tt.mutate != nil {
					tt.mutate(dcd)
				}
				objects = append(objects, dcd)
			}
			reconciler := createTestReconcilerWithStatus(dgd, withObjects(objects...))

			plan, err := reconciler.buildManagedWorkerRollout(context.Background(), dgd)
			require.NoError(t, err)

			wantTarget := desiredHash
			if tt.wantBridge {
				wantTarget = bridgeHash
			}
			assert.Equal(t, wantTarget, plan.targetDCDSuffix)
			assert.Equal(t, tt.wantTargetPresent, !plan.targetPending())
			assert.Equal(t, tt.wantTargetPresent, plan.targetsByComponent["worker"] != nil)
			assert.Len(t, plan.oldDCDs, tt.wantOld)
		})
	}
}

func TestManagedWorkerRolloutCanonicalV2WinsPartialBridgeCohort(t *testing.T) {
	const bridgeHash = "bridge-h"
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {ComponentType: consts.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
		"decode":  {ComponentType: consts.ComponentTypeDecode, Replicas: ptr.To(int32(1))},
	})
	dgd.UID = types.UID("dgd-uid")
	desiredHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   bridgeHash,
		consts.AnnotationCurrentWorkerHashV2: desiredHash,
	}
	render := func(component, suffix string) *nvidiacomv1beta1.DynamoComponentDeployment {
		t.Helper()
		dcds, err := dynamo.GenerateDynamoComponentsDeployments(
			dgd, nil, nil, desiredHash, nil,
		)
		require.NoError(t, err)
		dcd := dcds[component].DeepCopy()
		dcd.Name = dynamo.GetDCDResourceName(dgd, component, suffix)
		dcd.OwnerReferences = []metav1.OwnerReference{
			*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
		}
		return dcd
	}
	target := render("prefill", desiredHash)
	old := render("decode", bridgeHash)
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(target, old))

	plan, err := reconciler.buildManagedWorkerRollout(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, desiredHash, plan.targetDCDSuffix)
	assert.True(t, plan.targetPending())
	assert.NotNil(t, plan.targetsByComponent["prefill"])
	assert.Nil(t, plan.targetsByComponent["decode"])
	assert.Len(t, plan.oldDCDs, 1)
}

func TestManagedWorkerRolloutProjectsSelectedBridgeHash(t *testing.T) {
	const bridgeHash = "bridge-h"
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(1))},
	})
	dgd.UID = types.UID("dgd-uid")
	desiredHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{
		consts.AnnotationCurrentWorkerHash:   bridgeHash,
		consts.AnnotationCurrentWorkerHashV2: desiredHash,
	}
	target := managedInventoryWorkerDCD(t, dgd, bridgeHash)
	setInventoryWorkerDCDReady(target)
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(target))

	plan, err := reconciler.buildManagedWorkerRollout(context.Background(), dgd)
	require.NoError(t, err)
	require.Equal(t, bridgeHash, plan.targetDCDSuffix)
	require.NoError(t, reconciler.advanceManagedWorkerRollout(context.Background(), dgd, &dgd.Status, plan))
	assert.Equal(t, bridgeHash, dgd.Annotations[consts.AnnotationCurrentWorkerHash])
	assert.Equal(t, desiredHash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
}

func TestManagedWorkerRolloutScalesObservedOldDCDDespiteDesiredHashLabel(t *testing.T) {
	ctx := context.Background()
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker, Replicas: ptr.To(int32(1))},
	})
	dgd.UID = types.UID("dgd-uid")
	old := managedInventoryWorkerDCD(t, dgd, "desired-hash")
	old.Labels[consts.KubeLabelDynamoWorkerHash] = "desired-hash"
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(old))

	targetReplicas := int32(0)
	rollout := managedWorkerRollout{
		oldDCDs: []managedOldWorkerDCD{{
			dcd:            *old.DeepCopy(),
			targetReplicas: &targetReplicas,
		}},
	}
	require.NoError(t, reconciler.scaleOldWorkerDCDs(ctx, rollout))
	assertWorkerDCDReplicas(t, reconciler, old, 0)
}

func TestManagedWorkerRolloutRejectsForeignCanonicalNameCollision(t *testing.T) {
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker},
	})
	dgd.UID = types.UID("dgd-uid")
	collision := managedInventoryWorkerDCD(t, dgd, betaDGDWorkersSpecHash(t, dgd))
	collision.OwnerReferences = nil
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(collision))

	_, err := reconciler.buildManagedWorkerRollout(context.Background(), dgd)
	var identityCollision *workerDCDIdentityCollisionError
	require.ErrorAs(t, err, &identityCollision)
	assert.Equal(t, "worker", identityCollision.component)
}

func renderManagedInventoryWorkerDCD(
	t *testing.T,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *nvidiacomv1beta1.DynamoComponentDeployment {
	t.Helper()
	hash := betaDGDWorkersSpecHash(t, dgd)
	dcds, err := dynamo.GenerateDynamoComponentsDeployments(
		dgd, nil, nil, hash, nil,
	)
	require.NoError(t, err)
	return dcds["worker"]
}

func managedInventoryWorkerDCD(
	t *testing.T,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	suffix string,
) *nvidiacomv1beta1.DynamoComponentDeployment {
	t.Helper()
	dcd := renderManagedInventoryWorkerDCD(t, dgd).DeepCopy()
	dcd.Name = dynamo.GetDCDResourceName(dgd, "worker", suffix)
	dcd.Labels[consts.KubeLabelDynamoWorkerHash] = suffix
	dcd.Spec.PodTemplate.Labels[consts.KubeLabelDynamoWorkerHash] = suffix
	for i := range dcd.Spec.PodTemplate.Spec.Containers {
		for j := range dcd.Spec.PodTemplate.Spec.Containers[i].Env {
			if dcd.Spec.PodTemplate.Spec.Containers[i].Env[j].Name == "DYN_NAMESPACE_WORKER_SUFFIX" {
				dcd.Spec.PodTemplate.Spec.Containers[i].Env[j].Value = suffix
			}
		}
	}
	dcd.OwnerReferences = []metav1.OwnerReference{
		*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
	}
	return dcd
}

func setInventoryWorkerDCDReady(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
	dcd.Generation = 1
	dcd.Status.ObservedGeneration = 1
	dcd.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{
		Replicas:          1,
		ReadyReplicas:     ptr.To(int32(1)),
		AvailableReplicas: ptr.To(int32(1)),
	}
}

func TestDeleteObservedOldWorkerDCDsUsesUIDPrecondition(t *testing.T) {
	ctx := context.Background()
	old := &nvidiacomv1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{
		Name:      "old-worker",
		Namespace: "default",
		UID:       types.UID("observed-uid"),
	}}
	var deleteUID *types.UID
	funcs := interceptor.Funcs{
		Delete: func(ctx context.Context, writer client.WithWatch, object client.Object, options ...client.DeleteOption) error {
			deleteUID = (&client.DeleteOptions{}).ApplyOptions(options).Preconditions.UID
			return writer.Delete(ctx, object, options...)
		},
	}
	reconciler := createTestReconcilerWithStatus(createTestDGD("test-dgd", nil), withObjects(old), withInterceptor(funcs))

	t.Log("Delete the observed old DCD")
	require.NoError(t, reconciler.deleteObservedOldWorkerDCDs(ctx, []managedOldWorkerDCD{{dcd: *old}}))

	t.Log("Verify the delete is constrained to the observed object UID")
	require.NotNil(t, deleteUID)
	assert.Equal(t, old.UID, *deleteUID)
}

func TestManagedWorkerRolloutAtoBtoAWaitsForOldDrainAndCacheDeletion(t *testing.T) {
	ctx := context.Background()
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(1)),
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "b"}},
		},
	})
	dgd.UID = types.UID("dgd-uid")
	old := managedInventoryWorkerDCD(t, dgd, "generation-b")
	setInventoryWorkerDCDReady(old)

	t.Log("Revert the DGD from B to A while its parent state already names A")
	dgd.GetComponentByName("worker").PodTemplate.Spec.Containers[0].Env[0].Value = "a"
	targetHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: targetHash}
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress}
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(old))

	t.Log("Keep serving generation B unchanged until target A is observed")
	first, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	assert.True(t, first.targetPending())
	require.NoError(t, reconciler.scaleOldWorkerDCDs(ctx, first))
	assertWorkerDCDReplicas(t, reconciler, old, 1)

	t.Log("Observe ready target A and scale B down through the rollout planner")
	target := managedInventoryWorkerDCD(t, dgd, targetHash)
	setInventoryWorkerDCDReady(target)
	require.NoError(t, reconciler.Create(ctx, target))
	second, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.False(t, second.targetPending())
	require.NoError(t, reconciler.scaleOldWorkerDCDs(ctx, second))
	assertWorkerDCDReplicas(t, reconciler, old, 0)

	t.Log("Report B drained, then delete only from the observed inventory")
	drainedOld := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(old), drainedOld))
	drainedOld.Status.ObservedGeneration = drainedOld.Generation
	drainedOld.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{Replicas: 0, ReadyReplicas: ptr.To(int32(0)), AvailableReplicas: ptr.To(int32(0))}
	require.NoError(t, reconciler.Update(ctx, drainedOld))
	third, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.advanceManagedWorkerRollout(ctx, dgd, &dgd.Status, third))
	assert.True(t, apierrors.IsNotFound(reconciler.Get(ctx, client.ObjectKeyFromObject(old), &nvidiacomv1beta1.DynamoComponentDeployment{})))
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseInProgress, dgd.Status.RollingUpdate.Phase)

	t.Log("Project completion only after the deletion is observed on the next inventory read")
	final, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.advanceManagedWorkerRollout(ctx, dgd, &dgd.Status, final))
	assert.Equal(t, nvidiacomv1beta1.RollingUpdatePhaseCompleted, dgd.Status.RollingUpdate.Phase)
}

func TestManagedWorkerRolloutAnnotationlessAtoBRollsOutSafely(t *testing.T) {
	ctx := context.Background()
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: consts.ComponentTypeWorker,
			Replicas:      ptr.To(int32(1)),
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "a"}},
		},
	})
	dgd.UID = types.UID("dgd-uid")
	old := managedInventoryWorkerDCD(t, dgd, "opaque-a")
	setInventoryWorkerDCDReady(old)
	reconciler := createTestReconcilerWithStatus(dgd, withObjects(old))

	dgd.GetComponentByName("worker").PodTemplate.Spec.Containers[0].Env[0].Value = "b"
	first, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	assert.True(t, first.targetPending())
	require.NoError(t, reconciler.scaleOldWorkerDCDs(ctx, first))
	assertWorkerDCDReplicas(t, reconciler, old, 1)

	target := managedInventoryWorkerDCD(t, dgd, betaDGDWorkersSpecHash(t, dgd))
	setInventoryWorkerDCDReady(target)
	require.NoError(t, reconciler.Create(ctx, target))
	second, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.False(t, second.targetPending())
	require.NoError(t, reconciler.scaleOldWorkerDCDs(ctx, second))
	assertWorkerDCDReplicas(t, reconciler, old, 0)

	drainedOld := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, reconciler.Get(ctx, client.ObjectKeyFromObject(old), drainedOld))
	drainedOld.Status.ObservedGeneration = drainedOld.Generation
	drainedOld.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{
		Replicas:          0,
		ReadyReplicas:     ptr.To(int32(0)),
		AvailableReplicas: ptr.To(int32(0)),
	}
	require.NoError(t, reconciler.Update(ctx, drainedOld))
	third, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.advanceManagedWorkerRollout(ctx, dgd, &dgd.Status, third))
	assert.True(t, apierrors.IsNotFound(reconciler.Get(ctx, client.ObjectKeyFromObject(old), &nvidiacomv1beta1.DynamoComponentDeployment{})))

	final, err := reconciler.buildManagedWorkerRollout(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.advanceManagedWorkerRollout(ctx, dgd, &dgd.Status, final))
	assert.Equal(t, betaDGDWorkersSpecHash(t, dgd), dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
}

func assertWorkerDCDReplicas(
	t *testing.T,
	reconciler *dgdWorkerRolloutReconciler,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	want int32,
) {
	t.Helper()
	persisted := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, reconciler.Get(context.Background(), client.ObjectKeyFromObject(dcd), persisted))
	require.NotNil(t, persisted.Spec.Replicas)
	assert.Equal(t, want, *persisted.Spec.Replicas)
}
