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

package controller

import (
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	t.Log("Build a ready frontend and its DGD")
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(1)),
				},
			},
		},
	})
	podClique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "graph-0-frontend",
			Namespace:  "default",
			Generation: 1,
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1},
		Status: grovev1alpha1.PodCliqueStatus{
			Replicas:           1,
			ReadyReplicas:      1,
			UpdatedReplicas:    1,
			ScheduledReplicas:  1,
			ObservedGeneration: ptr.To(int64(1)),
		},
	}

	t.Log("Configure the workload reconciler to record child reads after scaling")
	podCliqueReads := 0
	scaleClient := &recordingGroveScaleClient{}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, podClique).
		WithStatusSubresource(dgd, podClique).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				object client.Object,
				options ...client.GetOption,
			) error {
				if _, ok := object.(*grovev1alpha1.PodClique); ok {
					require.Len(t, scaleClient.updates, 1, "readiness must be observed after scaling")
					podCliqueReads++
				}
				return reader.Get(ctx, key, object, options...)
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		ScaleClient:   scaleClient,
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(string, string) ([]string, error) {
				return nil, nil
			},
		},
	}

	t.Log("Reconcile workloads and reuse the single child observation for readiness")
	result, err := reconciler.newGroveProgram().workloads.Reconcile(
		context.Background(),
		dgd,
		nil,
		nil,
	)

	t.Log("Verify scaling precedes one readiness observation")
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	assert.Equal(t, 1, podCliqueReads)
}

func TestGroveWorkloadsReconciler_DoesNotCommitWorkerHashWhenPodCliqueSetSyncFails(t *testing.T) {
	tests := []struct {
		name        string
		existingPCS bool
	}{
		{name: "stale PCS update", existingPCS: true},
		{name: "PCS create collision"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a changed worker DGD with its previously committed hash")
			dgd := createTestDGD("graph", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"prefill": {
					ComponentType: consts.ComponentTypePrefill,
					Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "old"}},
				},
			})
			currentHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}
			dgd.GetComponentByName("prefill").PodTemplate.Spec.Containers[0].Env[0].Value = "new"
			wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			require.NotEqual(t, currentHash, wantHash)

			var existingPCS *grovev1alpha1.PodCliqueSet
			if tt.existingPCS {
				existingPCS = &grovev1alpha1.PodCliqueSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
						Namespace: dgd.Namespace,
					},
					Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
							Labels: map[string]string{consts.KubeLabelDynamoComponent: "prefill"},
						}},
					}},
				}
			}

			t.Log("Inject the requested PCS write failure and construct the full workload reconciler")
			dgdUpdateCalls := 0
			builder := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(dgd).
				WithStatusSubresource(dgd).
				WithInterceptorFuncs(interceptor.Funcs{
					Create: func(
						_ context.Context,
						_ client.WithWatch,
						object client.Object,
						_ ...client.CreateOption,
					) error {
						if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
							return apierrors.NewAlreadyExists(
								schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
								object.GetName(),
							)
						}
						return nil
					},
					Update: func(
						_ context.Context,
						_ client.WithWatch,
						object client.Object,
						_ ...client.UpdateOption,
					) error {
						switch object.(type) {
						case *grovev1alpha1.PodCliqueSet:
							if tt.existingPCS {
								return apierrors.NewConflict(
									schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
									object.GetName(),
									errors.New("stale PodCliqueSet"),
								)
							}
						case *nvidiacomv1beta1.DynamoGraphDeployment:
							dgdUpdateCalls++
						}
						return nil
					},
				})
			if existingPCS != nil {
				builder.WithObjects(existingPCS)
			}
			kubeClient := builder.Build()
			workloads := newGroveWorkloadsReconciler(
				kubeClient,
				events.NewFakeRecorder(10),
				newDGDWorkerRolloutReconciler(kubeClient, nil),
				&configv1alpha1.OperatorConfiguration{},
				&commoncontroller.RuntimeConfig{},
				&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
				&recordingGroveScaleClient{},
			)

			t.Log("Reconcile the full workload transition")
			observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
			_, err = workloads.Reconcile(context.Background(), observedDGD, nil, nil)

			t.Log("Verify the failed PCS sync leaves the persisted DGD hash unchanged")
			require.Error(t, err)
			storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
			assert.Equal(t, currentHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
			assert.Zero(t, dgdUpdateCalls)
		})
	}
}

func TestGroveWorkloadsReconciler_RecoversWorkerHashCommitAfterPodCliqueSetSync(t *testing.T) {
	t.Log("Build a changed worker DGD and an existing legacy PCS")
	dgd := createTestDGD("graph", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "old"}},
		},
	})
	currentHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}
	dgd.GetComponentByName("prefill").PodTemplate.Spec.Containers[0].Env[0].Value = "new"
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	legacyPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
			Namespace: dgd.Namespace,
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
			Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
				Labels: map[string]string{consts.KubeLabelDynamoComponent: "prefill"},
			}},
		}},
	}

	t.Log("Inject a DGD update conflict after allowing the PCS sync to persist")
	failDGDUpdate := true
	pcsUpdateCalls := 0
	dgdUpdateCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, legacyPCS).
		WithStatusSubresource(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(
				ctx context.Context,
				writer client.WithWatch,
				object client.Object,
				options ...client.UpdateOption,
			) error {
				switch object.(type) {
				case *grovev1alpha1.PodCliqueSet:
					pcsUpdateCalls++
				case *nvidiacomv1beta1.DynamoGraphDeployment:
					dgdUpdateCalls++
					if failDGDUpdate {
						return apierrors.NewConflict(
							schema.GroupResource{Group: nvidiacomv1beta1.GroupVersion.Group, Resource: "dynamographdeployments"},
							object.GetName(),
							errors.New("stale DynamoGraphDeployment"),
						)
					}
				}
				return writer.Update(ctx, object, options...)
			},
		}).
		Build()
	workloads := newGroveWorkloadsReconciler(
		kubeClient,
		events.NewFakeRecorder(10),
		newDGDWorkerRolloutReconciler(kubeClient, nil),
		&configv1alpha1.OperatorConfiguration{},
		&commoncontroller.RuntimeConfig{},
		&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
		&recordingGroveScaleClient{},
	)

	t.Log("Persist the PCS suffix, then fail the DGD hash commit")
	observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
	_, err = workloads.Reconcile(context.Background(), observedDGD, nil, nil)
	require.Error(t, err)

	t.Log("Verify the durable state is a suffixed PCS with the previous DGD hash")
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(legacyPCS), storedPCS))
	clique := podCliqueSetCliqueForComponent(storedPCS, "prefill")
	require.NotNil(t, clique)
	assert.Equal(t, wantHash, clique.Labels[consts.KubeLabelDynamoWorkerHash])
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
	assert.Equal(t, currentHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Equal(t, 1, dgdUpdateCalls)

	t.Log("Reconcile from freshly read objects after the simulated controller restart")
	failDGDUpdate = false
	freshDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), freshDGD))
	workloads = newGroveWorkloadsReconciler(
		kubeClient,
		events.NewFakeRecorder(10),
		newDGDWorkerRolloutReconciler(kubeClient, nil),
		&configv1alpha1.OperatorConfiguration{},
		&commoncontroller.RuntimeConfig{},
		&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
		&recordingGroveScaleClient{},
	)
	_, err = workloads.Reconcile(context.Background(), freshDGD, nil, nil)
	require.NoError(t, err)

	t.Log("Verify the retry commits the target hash without rewriting the PCS")
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
	assert.Equal(t, wantHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Equal(t, 2, dgdUpdateCalls)

	t.Log("Verify the completed transition is idempotent")
	idempotentDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), idempotentDGD))
	_, err = workloads.Reconcile(context.Background(), idempotentDGD, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Equal(t, 2, dgdUpdateCalls)
}

func TestGroveWorkloadsReconciler_ReconcilePodCliqueSetRejectsStaleObservation(t *testing.T) {
	t.Log("Build a DGD and the exact PCS observation used for rendering")
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
	})
	existing := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
			Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "old"}},
		}},
	}
	t.Log("Inject an optimistic-update conflict for the stale PCS observation")
	updateCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(
				_ context.Context,
				_ client.WithWatch,
				object client.Object,
				_ ...client.UpdateOption,
			) error {
				updateCalls++
				assert.Equal(t, existing.ResourceVersion, object.GetResourceVersion())
				return apierrors.NewConflict(
					schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
					object.GetName(),
					errors.New("stale PodCliqueSet"),
				)
			},
		}).
		Build()
	observed := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(existing), observed))
	desired := observed.DeepCopy()
	desired.Spec.Template.Cliques[0].Name = "new"
	reconciler := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(kubeClient, nil)}

	t.Log("Reconcile the exact observation and surface the retryable conflict")
	_, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{
		existing: observed,
		desired:  desired,
	})

	t.Log("Verify the write attempted the original resource version and returned a conflict")
	require.Error(t, err)
	assert.True(t, apierrors.IsConflict(err))
	assert.Equal(t, 1, updateCalls)
}

func TestGroveWorkloadsReconciler_ReconcilePodCliqueSetReturnsCreateConflict(t *testing.T) {
	t.Log("Build a DGD and desired PCS with no observed PCS")
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid"},
	})
	desired := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
	}
	t.Log("Inject a concurrent PCS creation")
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithInterceptorFuncs(interceptor.Funcs{
			Create: func(
				_ context.Context,
				_ client.WithWatch,
				object client.Object,
				_ ...client.CreateOption,
			) error {
				return apierrors.NewAlreadyExists(
					schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
					object.GetName(),
				)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(kubeClient, nil)}

	t.Log("Reconcile the missing observation and surface the retryable creation collision")
	_, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{desired: desired})

	t.Log("Verify the creation collision is returned to the caller")
	require.Error(t, err)
	assert.True(t, apierrors.IsAlreadyExists(err))
}
