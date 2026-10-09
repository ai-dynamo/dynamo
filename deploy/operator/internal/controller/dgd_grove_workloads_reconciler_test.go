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
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/util/validation/field"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

const updatedWorkerVersion = "new"

func TestGroveReconcileRequestUsesDelegationPredicate(t *testing.T) {
	dgd := newLPXHandoffSource(t, "node-local-v2-hybrid")
	dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend",
		ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
	})
	req := groveReconcileRequest{
		DGD: dgd,
		IsDelegated: func(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
			return component.ComponentName == "lpx"
		},
	}

	t.Log("Select exactly the component managed by Grove")
	components := req.ManagedComponents()
	require.Len(t, components, 1)
	require.Equal(t, "frontend", components[0].ComponentName)
	require.False(t, components[0].IsLPX())
	require.Equal(t, []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{dgd.Spec.Components[0]}, req.DelegatedComponents())

	t.Log("Treat every component as managed when no delegation predicate is supplied")
	defaultReq := groveReconcileRequest{DGD: dgd}
	require.Equal(t, dgd.Spec.Components, defaultReq.ManagedComponents())
	require.Empty(t, defaultReq.DelegatedComponents())
}

func TestGroveProgram_CoherentScalingWaitsForCurrentGeneration(t *testing.T) {
	for _, test := range []struct {
		name                            string
		scalingGroup, completedProgress bool
		replicas                        int32
	}{
		{name: "standalone scale-up with missing progress", replicas: 3},
		{name: "standalone scale-down with completed old progress", replicas: 1, completedProgress: true},
		{name: "scaling group scale-up with completed old progress", replicas: 3, scalingGroup: true, completedProgress: true},
		{name: "scaling group scale-down with missing progress", replicas: 1, scalingGroup: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx := t.Context()
			t.Log("Observe a Coherent PCS and two serving replicas at generation 8")
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid", Generation: 1, Annotations: map[string]string{consts.KubeAnnotationGroveUpdateStrategy: string(grovev1alpha1.CoherentStrategy)}},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "frontend", ComponentType: nvidiacomv1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(2)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:old"}}}},
				}}},
			}
			if test.scalingGroup {
				dgd.Spec.Components[0].Experimental = &nvidiacomv1beta1.ExperimentalSpec{Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: ptr.To(true)}}
			}
			config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}}
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
			secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
			pcs, err := dynamo.GenerateGrovePodCliqueSet(ctx, dgd, nil, config, runtimeConfig, nil, secrets, nil, nil, false, nil)
			require.NoError(t, err)
			pcs.Generation = 8
			pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment"))}
			pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
			if test.completedProgress {
				pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now(), UpdateEndedAt: ptr.To(metav1.Now())}
			}
			hash, err := commoncontroller.GetSpecHash(pcs)
			require.NoError(t, err)
			pcs.Annotations = map[string]string{commoncontroller.NvidiaAnnotationHashKey: hash, commoncontroller.NvidiaAnnotationGenerationKey: "8"}
			metadata := metav1.ObjectMeta{Name: "graph-0-frontend", Namespace: "default", Generation: 1}
			var child client.Object = &grovev1alpha1.PodClique{ObjectMeta: metadata, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 2}, Status: grovev1alpha1.PodCliqueStatus{Replicas: 2, ReadyReplicas: 2, UpdatedReplicas: 2, ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1))}}
			if test.scalingGroup {
				child = &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metadata, Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 2}, Status: grovev1alpha1.PodCliqueScalingGroupStatus{Replicas: 2, AvailableReplicas: 2, UpdatedReplicas: 2, ScheduledReplicas: 2, ObservedGeneration: ptr.To(int64(1))}}
			}
			writes := 0
			funcs := groveScaleInterceptor(interceptor.Funcs{
				Update: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
					if desired, ok := object.(*unstructured.Unstructured); ok && desired.GetKind() == groveconstants.KindPodCliqueSet {
						current := &unstructured.Unstructured{}
						current.SetGroupVersionKind(desired.GroupVersionKind())
						require.NoError(t, delegated.Get(ctx, client.ObjectKeyFromObject(desired), current))
						if !apiequality.Semantic.DeepEqual(current.Object["spec"], desired.Object["spec"]) {
							desired.SetGeneration(current.GetGeneration() + 1)
						}
					}
					return delegated.Update(ctx, object, opts...)
				},
			}, func() { writes++ })
			providerClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithRESTMapper(groveScaleRESTMapper()).WithObjects(dgd, pcs, child).WithStatusSubresource(dgd, pcs).Build()
			kubeClient := interceptor.NewClient(providerClient, funcs)
			reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient, Config: config, RuntimeConfig: runtimeConfig, Recorder: events.NewFakeRecorder(10), DockerSecretRetriever: secrets}

			t.Log("Request an image rollout and capacity change in the same DGD edit")
			dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "runtime:new"
			dgd.Spec.Components[0].Replicas = ptr.To(test.replicas)
			result, err := reconciler.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.Zero(t, writes)
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			require.EqualValues(t, 9, pcs.Generation)
			require.EqualValues(t, 8, *pcs.Status.ObservedGeneration)
			require.Equal(t, "runtime:new", pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].Image)
			dgd.Status = result.Status

			t.Log("Fresh program instances keep the new spec pending despite old or missing progress")
			for range 2 {
				result, err = reconciler.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
				require.NoError(t, err)
				require.Zero(t, writes)
				require.Zero(t, result.Result, "the PCS watch resumes pending work")
				require.True(t, meta.IsStatusConditionTrue(result.Status.Conditions, "ScalingDeferred"))
				require.Contains(t, result.Status.Components, "frontend")
				dgd.Status = result.Status
			}

			t.Log("Acknowledged active progress still blocks both scaling directions")
			pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
			pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()}
			require.NoError(t, providerClient.Status().Update(ctx, pcs))
			result, err = reconciler.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.Zero(t, writes)
			require.True(t, meta.IsStatusConditionTrue(result.Status.Conditions, "ScalingDeferred"))
			dgd.Status = result.Status

			t.Log("Only completion of the acknowledged generation releases scaling")
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			pcs.Status.UpdateProgress.UpdateEndedAt = ptr.To(metav1.Now())
			require.NoError(t, providerClient.Status().Update(ctx, pcs))
			result, err = reconciler.newGroveProgram().Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.Equal(t, 1, writes)
			require.True(t, meta.IsStatusConditionFalse(result.Status.Conditions, "ScalingDeferred"))
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(child), child))
			switch live := child.(type) {
			case *grovev1alpha1.PodClique:
				require.Equal(t, test.replicas, live.Spec.Replicas)
			case *grovev1alpha1.PodCliqueScalingGroup:
				require.Equal(t, test.replicas, live.Spec.Replicas)
			}
		})
	}
}

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	t.Log("Build a ready frontend alongside an independently owned LPX component")
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
	lpxSource := newLPXHandoffSource(t, "node-local-v2-hybrid")
	dgd.Spec.Components = append(dgd.Spec.Components, lpxSource.Spec.Components[0])
	wantSpec := dgd.Spec.DeepCopy()
	podClique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "graph-0-frontend",
			Namespace:  "default",
			Generation: 1,
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 0},
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
	scaleUpdates := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithRESTMapper(groveScaleRESTMapper()).
		WithObjects(dgd, podClique).
		WithStatusSubresource(dgd, podClique).
		WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				object client.Object,
				options ...client.GetOption,
			) error {
				if _, ok := object.(*grovev1alpha1.PodClique); ok {
					podCliqueReads++
				}
				return reader.Get(ctx, key, object, options...)
			},
		}, func() { scaleUpdates++ })).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(string, string) ([]string, error) {
				return nil, nil
			},
		},
	}

	t.Log("Reconcile workloads and reuse the single child observation for readiness")
	result, err := reconciler.newGroveProgram().workloads.Reconcile(
		context.Background(),
		groveReconcileRequest{DGD: dgd, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController},
		nil,
		nil,
	)

	t.Log("Initial creation defers scaling while retaining the live readiness observation")
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStatePending, result.State)
	require.True(t, result.ScalingDeferred)
	require.Zero(t, scaleUpdates)
	require.Contains(t, result.ComponentStatus, "frontend")

	t.Log("Observe the PCS configuration without waiting for Grove's observed generation")
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(dgd), pcs))
	require.Nil(t, pcs.Status.ObservedGeneration)
	podCliqueReads = 0
	result, err = reconciler.newGroveProgram().workloads.Reconcile(t.Context(), groveReconcileRequest{DGD: dgd, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)

	t.Log("Verify scaling precedes one readiness observation")
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	assert.Equal(t, 1, scaleUpdates)
	assert.Equal(t, 1, podCliqueReads)
	require.Len(t, result.ComponentStatus, 1)
	assert.Equal(t, nvidiacomv1beta1.ComponentKindPodClique, result.ComponentStatus["frontend"].ComponentKind)
	assert.Equal(t, []string{"graph-0-frontend"}, result.ComponentStatus["frontend"].ComponentNames)
	assert.Equal(t, *wantSpec, dgd.Spec)
}

func TestGroveWorkloadsReconcilerUsesStableReadinessWithoutOrdinaryPodCliqueSet(t *testing.T) {
	t.Log("Use an LPX-only graph without an ordinary PodCliqueSet")
	const dgdName = "graph"
	source := newLPXHandoffSource(t, "node-local-v2-lpu-only")
	source.Name, source.Namespace, source.UID = dgdName, corev1.NamespaceDefault, "dgd-uid"
	pcsReads := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithRESTMapper(groveScaleRESTMapper()).
		WithObjects(source).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, object client.Object, options ...client.GetOption) error {
				if isGrovePodCliqueSetObject(object) {
					pcsReads++
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
		DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) {
			return nil, nil
		}},
	}

	t.Log("Report stable-resource readiness without looking up an ordinary PodCliqueSet")
	result, err := reconciler.newGroveProgram().workloads.Reconcile(t.Context(), groveReconcileRequest{DGD: source, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	require.Zero(t, pcsReads)
	pcsList := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, kubeClient.List(t.Context(), pcsList))
	require.Empty(t, pcsList.Items)
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
			dgd.GetComponentByName("prefill").PodTemplate.Spec.Containers[0].Env[0].Value = updatedWorkerVersion
			wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			require.NotEqual(t, currentHash, wantHash)

			var existingPCS *grovev1alpha1.PodCliqueSet
			if tt.existingPCS {
				existingPCS = &grovev1alpha1.PodCliqueSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:      dynamo.PCSNameForDGD(dgd, nil),
						Namespace: dgd.Namespace,
						OwnerReferences: []metav1.OwnerReference{
							*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
						},
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
						if isGrovePodCliqueSetObject(object) {
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
						if isGrovePodCliqueSetObject(object) {
							if tt.existingPCS {
								return apierrors.NewConflict(
									schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
									object.GetName(),
									errors.New("stale PodCliqueSet"),
								)
							}
						} else if _, ok := object.(*nvidiacomv1beta1.DynamoGraphDeployment); ok {
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
			)

			t.Log("Reconcile the full workload transition")
			observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
			_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: observedDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)

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
	lpxSource := newLPXHandoffSource(t, "node-local-v2-hybrid")
	dgd.Spec.Components = append(dgd.Spec.Components, lpxSource.Spec.Components[0])
	currentHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}
	dgd.GetComponentByName("prefill").PodTemplate.Spec.Containers[0].Env[0].Value = updatedWorkerVersion
	wantSpec := dgd.Spec.DeepCopy()
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	legacyPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.PCSNameForDGD(dgd, nil),
			Namespace: dgd.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
			},
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
				if isGrovePodCliqueSetObject(object) {
					pcsUpdateCalls++
				} else if _, ok := object.(*nvidiacomv1beta1.DynamoGraphDeployment); ok {
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
	)

	t.Log("Persist the PCS suffix without projecting the DGD hash from the write receipt")
	observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
	_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: observedDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
	require.NoError(t, err)

	t.Log("Verify the write receipt leaves the parent hash unchanged")
	storedPCS := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(legacyPCS), storedPCS))
	clique := podCliqueSetCliqueForComponent(storedPCS, "prefill")
	require.NotNil(t, clique)
	assert.Equal(t, wantHash, clique.Labels[consts.KubeLabelDynamoWorkerHash])
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
	assert.Equal(t, currentHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
	assert.Equal(t, *wantSpec, storedDGD.Spec)
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Zero(t, dgdUpdateCalls)

	t.Log("Observe the suffix on a later reconcile, then reject the parent projection")
	freshDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), freshDGD))
	_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: freshDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
	require.Error(t, err)
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Equal(t, 1, dgdUpdateCalls)

	t.Log("Retry projection from a fresh observation after the simulated controller restart")
	failDGDUpdate = false
	freshDGD = &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), freshDGD))
	_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: freshDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
	require.NoError(t, err)

	t.Log("Verify the retry commits the target hash without rewriting the PCS")
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
	assert.Equal(t, wantHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
	assert.Equal(t, *wantSpec, storedDGD.Spec)
	assert.Equal(t, 1, pcsUpdateCalls)
	assert.Equal(t, 2, dgdUpdateCalls)

	t.Log("Verify the completed transition is idempotent")
	idempotentDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), idempotentDGD))
	_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: idempotentDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
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
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
			},
		},
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
	_, _, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{
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
	_, _, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{desired: desired})

	t.Log("Verify the creation collision is returned to the caller")
	require.Error(t, err)
	assert.True(t, apierrors.IsAlreadyExists(err))
}

func TestGroveProviderOverridesUseObservedPCSReconciliation(t *testing.T) {
	t.Log("Build one Grove reconciler with an injectable ordinary update failure")
	ctx := context.Background()
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: types.UID("graph-uid")},
	}
	desired := &grovev1alpha1.PodCliqueSet{
		TypeMeta:   metav1.TypeMeta{APIVersion: provideroverride.GroveAPIVersion, Kind: provideroverride.TargetPodCliqueSet},
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{},
		},
	}
	rejectUpdate := false
	updateCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(ctx context.Context, writer client.WithWatch, object client.Object, opts ...client.UpdateOption) error {
				require.IsType(t, &unstructured.Unstructured{}, object, "opaque PCS updates must serialize only at the write boundary")
				updateCalls++
				if rejectUpdate {
					return errors.New("provider rejected update")
				}
				return writer.Update(ctx, object, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Create the PCS after composing an opaque root topology override")
	dgd.Spec.ProviderOverride = rootTopologyOverride(`{"topologyName":"gpu-topology","futureProviderField":{"enabled":true}}`)
	_, _, err := reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{desired: desired})
	require.NoError(t, err)
	assertLiveRootTopologyValue(t, kubeClient, "topologyName", "gpu-topology")

	t.Log("Reuse the typed render observation when the desired PCS is unchanged")
	observed := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.NoError(t, err)
	assert.Zero(t, updateCalls)

	t.Log("Reject an ordinary PCS update and verify the live resource remains unchanged")
	dgd.Spec.ProviderOverride = rootTopologyOverride(`{"topologyName":"changed"}`)
	rejectUpdate = true
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.ErrorContains(t, err, "provider rejected update")
	assert.Equal(t, 1, updateCalls)
	assertLiveRootTopologyValue(t, kubeClient, "topologyName", "gpu-topology")

	t.Log("Remove the override and let the same PCS update path prune its subtree")
	rejectUpdate = false
	dgd.Spec.ProviderOverride = nil
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.NoError(t, err)
	assert.Equal(t, 2, updateCalls)
	live := newUnstructuredGrovePodCliqueSet()
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), live))
	_, found, nestedErr := unstructured.NestedFieldNoCopy(live.Object, "spec", "template", "topologyConstraint")
	require.NoError(t, nestedErr)
	assert.False(t, found)
}

func TestPodCliqueSetObservesWorkerHash(t *testing.T) {
	dgd := createTestDGD("graph", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "v1"}},
		},
	})
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)

	unstampedPCS := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{
		Template: grovev1alpha1.PodCliqueSetTemplateSpec{
			Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
				Labels: map[string]string{consts.KubeLabelDynamoComponent: "prefill"},
			}},
		},
	}}
	stampedPCS := unstampedPCS.DeepCopy()
	stampedPCS.Spec.Template.Cliques[0].Labels[consts.KubeLabelDynamoWorkerHash] = wantHash

	tests := []struct {
		name               string
		pcs                *grovev1alpha1.PodCliqueSet
		acceptAllUnstamped bool
		want               bool
	}{
		{
			name:               "nil PCS is never observed",
			pcs:                nil,
			acceptAllUnstamped: true,
			want:               false,
		},
		{
			name:               "unstamped cliques accepted for legacy unsuffixed PCS",
			pcs:                unstampedPCS,
			acceptAllUnstamped: true,
			want:               true,
		},
		{
			name:               "unstamped cliques rejected for new PCS requiring canonical hash",
			pcs:                unstampedPCS,
			acceptAllUnstamped: false,
			want:               false,
		},
		{
			name:               "cliques bearing target hash accepted regardless of acceptAllUnstamped",
			pcs:                stampedPCS,
			acceptAllUnstamped: false,
			want:               true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := podCliqueSetObservesWorkerHash(groveReconcileRequest{DGD: dgd, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, tt.pcs, tt.acceptAllUnstamped)
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGroveWorkloadsReconciler_SkipsHashObservationWhenHashIsCurrent(t *testing.T) {
	t.Log("Build a DGD whose annotation already matches the desired hash")
	dgd := createTestDGD("graph", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "v1"}},
		},
	})
	currentHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: currentHash}

	t.Log("Seed a PCS carrying the current hash so the sync is a no-op")
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      dynamo.PCSNameForDGD(dgd, nil),
			Namespace: dgd.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
			},
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
			Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
				Labels: map[string]string{
					consts.KubeLabelDynamoComponent:  "prefill",
					consts.KubeLabelDynamoWorkerHash: currentHash,
				},
			}},
		}},
	}

	dgdUpdateCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, existingPCS).
		WithStatusSubresource(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(_ context.Context, _ client.WithWatch, object client.Object, _ ...client.UpdateOption) error {
				if _, ok := object.(*nvidiacomv1beta1.DynamoGraphDeployment); ok {
					dgdUpdateCalls++
				}
				return nil
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
	)

	t.Log("Reconcile: hash observation block must be skipped entirely when needsCommit is false")
	observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
	_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: observedDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
	require.NoError(t, err)

	assert.Zero(t, dgdUpdateCalls, "DGD must not be updated when the hash annotation is already current")
}

func TestGroveWorkloadsReconciler_DefersHashCommitUntilPCSWriteObserved(t *testing.T) {
	dgd := createTestDGD("graph", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Envs:          []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "v1"}},
		},
	})
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)

	tests := []struct {
		name        string
		existingPCS *grovev1alpha1.PodCliqueSet
	}{
		{
			name:        "new PCS — create defers commit, second reconcile commits",
			existingPCS: nil,
		},
		{
			name: "legacy unsuffixed PCS — bookkeeping write defers commit, second reconcile commits",
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dynamo.PCSNameForDGD(dgd, nil),
					Namespace: dgd.Namespace,
					OwnerReferences: []metav1.OwnerReference{
						*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
					},
				},
				Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
					Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
						Labels: map[string]string{consts.KubeLabelDynamoComponent: "prefill"},
					}},
				}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dgdUpdateCalls := 0
			builder := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(dgd).
				WithStatusSubresource(dgd).
				WithInterceptorFuncs(interceptor.Funcs{
					Update: func(
						ctx context.Context,
						writer client.WithWatch,
						object client.Object,
						opts ...client.UpdateOption,
					) error {
						if _, ok := object.(*nvidiacomv1beta1.DynamoGraphDeployment); ok {
							dgdUpdateCalls++
						}
						return writer.Update(ctx, object, opts...)
					},
				})
			if tt.existingPCS != nil {
				builder.WithObjects(tt.existingPCS)
			}
			kubeClient := builder.Build()
			workloads := newGroveWorkloadsReconciler(
				kubeClient,
				events.NewFakeRecorder(10),
				newDGDWorkerRolloutReconciler(kubeClient, nil),
				&configv1alpha1.OperatorConfiguration{},
				&commoncontroller.RuntimeConfig{},
				&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
			)

			t.Log("First reconcile writes the PCS; commit must be deferred")
			observedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), observedDGD))
			_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: observedDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
			require.NoError(t, err)
			assert.Zero(t, dgdUpdateCalls, "hash annotation must not be committed on the reconcile that writes the PCS")

			t.Log("Second reconcile is a no-op PCS sync; commit must proceed")
			freshDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), freshDGD))
			_, err = workloads.Reconcile(context.Background(), groveReconcileRequest{DGD: freshDGD, IsDelegated: (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController}, nil, nil)
			require.NoError(t, err)
			assert.Equal(t, 1, dgdUpdateCalls, "hash annotation must be committed once the PCS write is observed")

			storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD))
			assert.Equal(t, wantHash, storedDGD.Annotations[consts.AnnotationCurrentWorkerHashV2])
		})
	}
}

func isGrovePodCliqueSetObject(object client.Object) bool {
	if _, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
		return true
	}
	unstructuredObject, ok := object.(*unstructured.Unstructured)
	return ok && unstructuredObject.GetKind() == provideroverride.TargetPodCliqueSet
}

func rootTopologyOverride(topology string) *nvidiacomv1beta1.ProviderOverride {
	return &nvidiacomv1beta1.ProviderOverride{
		APIVersion: provideroverride.GroveAPIVersion,
		Target:     provideroverride.TargetPodCliqueSet,
		Value: apiextensionsv1.JSON{Raw: []byte(
			`{"spec":{"template":{"topologyConstraint":` + topology + `}}}`,
		)},
	}
}

func assertLiveRootTopologyValue(
	t *testing.T,
	kubeClient client.Client,
	field string,
	want interface{},
) {
	t.Helper()
	live := newUnstructuredGrovePodCliqueSet()
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	got, found, err := unstructured.NestedFieldNoCopy(
		live.Object,
		"spec",
		"template",
		"topologyConstraint",
		field,
	)
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, want, got)
}

func newUnstructuredGrovePodCliqueSet() *unstructured.Unstructured {
	object := &unstructured.Unstructured{}
	object.SetAPIVersion(provideroverride.GroveAPIVersion)
	object.SetKind(provideroverride.TargetPodCliqueSet)
	return object
}

func TestGroveProgram_ScalingDuringUpdatesPreservesStatus(t *testing.T) {
	for _, test := range []struct {
		name, strategy                                              string
		active, unobserved, missingObservation, configurationChange bool
		guard, mixedLPX, allowScaling, scaleDown                    bool
	}{
		{name: "active coherent update waits for completion", strategy: "Coherent", active: true, unobserved: true},
		{name: "coherent configuration with lagging observed generation defers scaling", strategy: "Coherent", unobserved: true},
		{name: "initial coherent configuration waits for Grove acknowledgement", strategy: "Coherent", missingObservation: true},
		{name: "default rolling recreate can scale during a rollout", active: true, unobserved: true, allowScaling: true},
		{name: "explicit rolling recreate can scale during a rollout", strategy: "RollingRecreate", active: true, unobserved: true, allowScaling: true},
		{name: "default rolling recreate can scale down during a rollout", active: true, unobserved: true, allowScaling: true, scaleDown: true},
		{name: "on delete with lagging observed generation can scale", strategy: "OnDelete", unobserved: true, allowScaling: true},
		{name: "a PCS write waits for cached configuration", strategy: "Coherent", configurationChange: true},
		{name: "webhook lags observed completion", strategy: "Coherent", guard: true},
		{name: "settled LPX child cannot wake rejected frontend scale", strategy: "Coherent", guard: true, mixedLPX: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe a PCS serving one frontend replica while the DGD requests a different capacity")
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid", Generation: 1, Annotations: map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "Coherent"}}, Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "frontend", ComponentType: consts.ComponentTypeFrontend, Replicas: ptr.To(int32(2))}}}}
			if test.strategy == "" {
				delete(dgd.Annotations, consts.KubeAnnotationGroveUpdateStrategy)
			} else {
				dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy] = test.strategy
			}
			if test.scaleDown {
				dgd.Spec.Components[0].Replicas = ptr.To(int32(0))
			}
			if test.mixedLPX {
				dgd.Spec.Components = append(dgd.Spec.Components, newLPXHandoffSource(t, "single_v2").Spec.Components[0])
			}
			config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}}
			runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true}}
			secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
			pcs, err := dynamo.GenerateGrovePodCliqueSet(t.Context(), dgd, (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController, config, runtimeConfig, nil, secrets, nil, nil, true, nil)
			require.NoError(t, err)
			pcs.Generation = 1
			pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment"))}
			pcs.Spec.Template.Cliques[0].Spec.Replicas = 1
			pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()}
			if !test.active {
				pcs.Status.UpdateProgress.UpdateEndedAt = ptr.To(metav1.Now())
			}
			pcs.Status.ObservedGeneration = ptr.To(int64(1))
			if test.unobserved {
				pcs.Generation = 2
			}
			if test.missingObservation {
				pcs.Status.ObservedGeneration = nil
				pcs.Status.UpdateProgress = nil
			}
			if test.configurationChange {
				pcs.Spec.UpdateStrategy.Type = grovev1alpha1.RollingRecreateStrategy
			}
			hash, err := commoncontroller.GetSpecHash(pcs)
			require.NoError(t, err)
			pcs.Annotations = map[string]string{commoncontroller.NvidiaAnnotationHashKey: hash, commoncontroller.NvidiaAnnotationGenerationKey: fmt.Sprint(pcs.Generation)}
			pclq := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-frontend", Namespace: "default", Generation: 1}, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}, Status: grovev1alpha1.PodCliqueStatus{Replicas: 1, ReadyReplicas: 1, UpdatedReplicas: 1, ScheduledReplicas: 1, ObservedGeneration: ptr.To(int64(1))}}
			writes := 0
			guardActive := test.guard
			funcs := groveScaleInterceptor(interceptor.Funcs{}, func() { writes++ })
			scaleUpdate := funcs.SubResourceUpdate
			funcs.SubResourceUpdate = func(ctx context.Context, kube client.Client, subresource string, object client.Object, options ...client.SubResourceUpdateOption) error {
				if subresource == "status" {
					return kube.SubResource(subresource).Update(ctx, object, options...)
				}
				if guardActive {
					return apierrors.NewForbidden(consts.PodCliqueGVR.GroupResource(), object.GetName(), errors.New("spec.replicas changes are not allowed while a coherent update is in progress on PodCliqueSet default/graph, complete the update before scaling"))
				}
				return scaleUpdate(ctx, kube, subresource, object, options...)
			}
			kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithRESTMapper(groveScaleRESTMapper()).WithObjects(dgd, pcs, pclq).WithStatusSubresource(dgd, &nvidiacomv1alpha1.LPXGraphDeployment{}).WithInterceptorFuncs(funcs).Build()

			t.Log("Keep an already-settled LPX child independent of frontend scaling")
			var lpxChild *nvidiacomv1alpha1.LPXGraphDeployment
			if test.mixedLPX {
				lpxChild, err = (&dgdLPXHandoff{client: kubeClient}).Reconcile(t.Context(), dgd)
				require.NoError(t, err)
				lpxChild.Status.ObservedGeneration = lpxChild.Generation
				lpxChild.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: lpxChild.Generation}}
				lpxChild.Status.Components = map[string]nvidiacomv1alpha1.LPXComponentStatus{"lpx": {ComponentReplicaStatus: nvidiacomv1beta1.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: ptr.To(int32(1))}}}
				require.NoError(t, kubeClient.Status().Update(t.Context(), lpxChild))
				dgd.Status.Components = map[string]nvidiacomv1beta1.ComponentReplicaStatus{"lpx": lpxChild.Status.Components["lpx"].ComponentReplicaStatus}
			}
			reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient, Config: config, RuntimeConfig: runtimeConfig, Recorder: events.NewFakeRecorder(10), DockerSecretRetriever: secrets}

			t.Log("Run the complete program and preserve component facts while applying the scaling policy")
			result, err := reconciler.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			if test.guard {
				require.Error(t, err)
				require.True(t, apierrors.IsForbidden(err))
				require.Equal(t, nvidiacomv1beta1.DGDStateFailed, result.Status.State)
			} else {
				require.NoError(t, err)
				if test.allowScaling {
					require.False(t, meta.IsStatusConditionTrue(result.Status.Conditions, "ScalingDeferred"))
				} else {
					require.Equal(t, nvidiacomv1beta1.DGDStatePending, result.Status.State)
					require.True(t, meta.IsStatusConditionTrue(result.Status.Conditions, "ScalingDeferred"))
				}
			}
			if test.allowScaling {
				require.Equal(t, 1, writes)
			} else {
				require.Zero(t, writes)
			}
			require.Contains(t, result.Status.Components, "frontend")
			require.NotNil(t, result.Status.Components["frontend"].GPUsPerReplica)
			require.Equal(t, []string{"graph-0-frontend"}, result.Status.Components["frontend"].ComponentNames)
			require.Zero(t, result.Result, "pending states use watches; errors use controller backoff")
			if test.mixedLPX {
				require.Equal(t, int32(1), result.Status.Components["lpx"].Replicas)
				observedChild := &nvidiacomv1alpha1.LPXGraphDeployment{}
				require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(lpxChild), observedChild))
				require.Equal(t, lpxChild.Spec.InputRevision, observedChild.Spec.InputRevision)
				require.Equal(t, lpxChild.ResourceVersion, observedChild.ResourceVersion)
			}

			t.Log("Persist program status before the next reconciliation")
			dgd.Status = result.Status

			t.Log("Retry failures or observe acknowledgement and completion of the current Coherent generation")
			guardActive = false
			if !test.guard && !test.allowScaling {
				require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
				pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
				if pcs.Status.UpdateProgress != nil {
					pcs.Status.UpdateProgress.UpdateEndedAt = ptr.To(metav1.Now())
				}
				require.NoError(t, kubeClient.Update(t.Context(), pcs))
			}
			result, err = reconciler.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.Equal(t, 1, writes)
			if !test.guard && !test.allowScaling {
				require.True(t, meta.IsStatusConditionFalse(result.Status.Conditions, "ScalingDeferred"))
			}
			require.Zero(t, result.Result)
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pclq), pclq))
			require.Equal(t, *dgd.Spec.Components[0].Replicas, pclq.Spec.Replicas)
		})
	}
}

func TestGroveProgram_CoherentAPIRejectionIsReported(t *testing.T) {
	for _, origin := range []string{"1.1.0", "1.6.0"} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Request Coherent explicitly against an API that rejects the strategy")
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "graph", Namespace: "default", UID: "dgd-uid",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: origin,
						consts.KubeAnnotationGroveUpdateStrategy:         "Coherent",
					},
				},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "frontend", ComponentType: consts.ComponentTypeFrontend,
					}},
				},
			}
			rejected := apierrors.NewInvalid(
				schema.GroupKind{Group: grovev1alpha1.SchemeGroupVersion.Group, Kind: "PodCliqueSet"},
				dgd.Name,
				field.ErrorList{field.NotSupported(field.NewPath("spec", "updateStrategy", "type"), "Coherent", []string{"RollingRecreate", "OnDelete"})},
			)
			rejectStrategy := true
			kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dgd).WithInterceptorFuncs(interceptor.Funcs{
				Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
					if isGrovePodCliqueSetObject(object) && rejectStrategy {
						return rejected
					}
					return delegated.Create(ctx, object, opts...)
				},
			}).Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:                kubeClient,
				Config:                &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}},
				RuntimeConfig:         &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
				Recorder:              events.NewFakeRecorder(10),
				DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
			}

			t.Log("Preserve the API rejection in the failure diagnostic and normal retry path")
			result, err := reconciler.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.ErrorIs(t, err, rejected)
			require.True(t, apierrors.IsInvalid(err))
			ready := meta.FindStatusCondition(result.Status.Conditions, "Ready")
			require.NotNil(t, ready)
			require.Equal(t, string(reasonFailedToReconcileResources), ready.Reason)
			require.Contains(t, ready.Message, "Coherent")
			pcs := &grovev1alpha1.PodCliqueSet{}
			require.True(t, apierrors.IsNotFound(kubeClient.Get(t.Context(), client.ObjectKeyFromObject(dgd), pcs)))

			t.Log("Retry the same explicit intent after the installed API accepts it")
			rejectStrategy = false
			_, err = reconciler.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(dgd), pcs))
			require.Equal(t, grovev1alpha1.CoherentStrategy, pcs.Spec.UpdateStrategy.Type)
		})
	}
}
