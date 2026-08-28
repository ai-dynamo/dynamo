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
	"maps"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestEnsureWorkloadProvider(t *testing.T) {
	listErr := errors.New("list failed")
	tests := []struct {
		name         string
		gate         features.Gates
		annotations  map[string]string
		workloads    []providerTestWorkload
		groveListErr error
		wantProvider workloadProvider
		wantErr      string
		wantErrIs    error
	}{
		{
			name: "materialized component remains authoritative when Grove is enabled",
			gate: features.Gates{Grove: true},
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
			},
			workloads:    []providerTestWorkload{{provider: workloadProviderGrove, owned: true}},
			wantProvider: workloadProviderComponent,
		},
		{
			name: "materialized DisaggregatedSet remains authoritative when Grove is enabled",
			gate: features.Gates{Grove: true},
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderDisaggregatedSet,
			},
			workloads:    []providerTestWorkload{{provider: workloadProviderGrove, owned: true}},
			wantProvider: workloadProviderDisaggregatedSet,
		},
		{
			name:         "no owned workloads select Grove from current intent",
			gate:         features.Gates{Grove: true},
			wantProvider: workloadProviderGrove,
		},
		{
			name: "explicit Grove opt-out locks in component",
			gate: features.Gates{Grove: true},
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "FALSE",
			},
			wantProvider: workloadProviderComponent,
		},
		{
			name: "explicit DisaggregatedSet intent selects DisaggregatedSet after Grove opt-out",
			gate: features.Gates{Grove: true},
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove:            consts.KubeLabelValueFalse,
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
			wantProvider: workloadProviderDisaggregatedSet,
		},
		{
			name: "explicit DisaggregatedSet intent is durable even when its gate is unavailable",
			annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
			wantProvider: workloadProviderDisaggregatedSet,
		},
		{
			name:         "owned DCD adopts component despite current Grove intent",
			gate:         features.Gates{Grove: true},
			workloads:    []providerTestWorkload{{provider: workloadProviderComponent, owned: true}},
			wantProvider: workloadProviderComponent,
		},
		{
			name: "owned PodCliqueSet adopts Grove despite current component intent",
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse,
			},
			workloads:    []providerTestWorkload{{provider: workloadProviderGrove, owned: true}},
			wantProvider: workloadProviderGrove,
		},
		{
			name: "owned DisaggregatedSet adopts DS despite current component intent",
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse,
			},
			workloads:    []providerTestWorkload{{provider: workloadProviderDisaggregatedSet, owned: true}},
			wantProvider: workloadProviderDisaggregatedSet,
		},
		{
			name: "DisaggregatedSet adopts alongside auxiliary DCDs",
			workloads: []providerTestWorkload{
				{provider: workloadProviderComponent, owned: true},
				{provider: workloadProviderDisaggregatedSet, owned: true},
			},
			wantProvider: workloadProviderDisaggregatedSet,
		},
		{
			name: "Grove adopts alongside auxiliary DCDs",
			workloads: []providerTestWorkload{
				{provider: workloadProviderComponent, owned: true},
				{provider: workloadProviderGrove, owned: true},
			},
			wantProvider: workloadProviderGrove,
		},
		{
			name: "DisaggregatedSet and Grove fail closed",
			workloads: []providerTestWorkload{
				{provider: workloadProviderGrove, owned: true},
				{provider: workloadProviderDisaggregatedSet, owned: true},
			},
			wantErr:   "owns workloads from multiple providers",
			wantErrIs: errConflictingWorkloadProviders,
		},
		{
			name:         "unannotated DGD with only DCDs adopts component",
			workloads:    []providerTestWorkload{{provider: workloadProviderComponent, owned: true}},
			wantProvider: workloadProviderComponent,
		},
		{
			name: "foreign workloads are ignored",
			gate: features.Gates{Grove: true},
			workloads: []providerTestWorkload{
				{provider: workloadProviderComponent},
				{provider: workloadProviderGrove},
				{provider: workloadProviderDisaggregatedSet},
			},
			wantProvider: workloadProviderGrove,
		},
		{
			name: "cluster without Grove API selects component from current intent",
			groveListErr: fmt.Errorf("discover Grove API: %w", &meta.NoKindMatchError{
				GroupKind:        grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet").GroupKind(),
				SearchedVersions: []string{grovev1alpha1.SchemeGroupVersion.Version},
			}),
			wantProvider: workloadProviderComponent,
		},
		{
			name:         "workload discovery failure does not freeze a provider",
			groveListErr: listErr,
			wantErr:      "list owned PodCliqueSets",
			wantErrIs:    listErr,
		},
		{
			name: "unsupported materialized provider fails",
			annotations: map[string]string{
				consts.KubeAnnotationWorkloadProvider: "unknown",
			},
			wantErr:   "unsupported workload provider",
			wantErrIs: errUnsupportedWorkloadProvider,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Store one DGD and the workload families visible during adoption")
			seed := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "graph",
					Namespace:   "default",
					UID:         types.UID("graph-uid"),
					Annotations: maps.Clone(tt.annotations),
				},
			}
			objects := []client.Object{seed}
			for i, workload := range tt.workloads {
				objects = append(objects, newProviderTestWorkload(t, seed, i, workload))
			}
			clientBuilder := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(objects...)
			if tt.groveListErr != nil {
				clientBuilder = clientBuilder.WithInterceptorFuncs(interceptor.Funcs{
					List: func(
						ctx context.Context,
						kubeClient client.WithWatch,
						list client.ObjectList,
						opts ...client.ListOption,
					) error {
						if _, ok := list.(*grovev1alpha1.PodCliqueSetList); ok {
							return tt.groveListErr
						}
						return kubeClient.List(ctx, list, opts...)
					},
				})
			}
			kubeClient := clientBuilder.Build()
			live := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(seed), live))
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        kubeClient,
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: tt.gate},
			}

			t.Log("Resolve and, when missing, persist the provider")
			provider, err := reconciler.ensureWorkloadProvider(t.Context(), live)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				assert.ErrorIs(t, err, tt.wantErrIs)

				t.Log("Verify a failed adoption did not freeze an arbitrary provider")
				stored := &nvidiacomv1beta1.DynamoGraphDeployment{}
				require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(seed), stored))
				if _, originallySelected := tt.annotations[consts.KubeAnnotationWorkloadProvider]; !originallySelected {
					assert.NotContains(t, stored.Annotations, consts.KubeAnnotationWorkloadProvider)
				}
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantProvider, provider)
			assert.Equal(t, string(tt.wantProvider), live.Annotations[consts.KubeAnnotationWorkloadProvider])

			t.Log("Verify the stored annotation matches the selected provider")
			stored := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(seed), stored))
			assert.Equal(t, string(tt.wantProvider), stored.Annotations[consts.KubeAnnotationWorkloadProvider])
		})
	}
}

type providerTestWorkload struct {
	provider workloadProvider
	owned    bool
}

func newProviderTestWorkload(
	t testing.TB,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	index int,
	workload providerTestWorkload,
) client.Object {
	t.Helper()

	var ownerReferences []metav1.OwnerReference
	if workload.owned {
		ownerReferences = []metav1.OwnerReference{
			*metav1.NewControllerRef(dgd, nvidiacomv1beta1.GroupVersion.WithKind("DynamoGraphDeployment")),
		}
	}
	objectMeta := metav1.ObjectMeta{
		Name:            fmt.Sprintf("%s-%d", workload.provider, index),
		Namespace:       dgd.Namespace,
		OwnerReferences: ownerReferences,
	}

	switch workload.provider {
	case workloadProviderComponent:
		return &nvidiacomv1beta1.DynamoComponentDeployment{ObjectMeta: objectMeta}
	case workloadProviderGrove:
		return &grovev1alpha1.PodCliqueSet{ObjectMeta: objectMeta}
	case workloadProviderDisaggregatedSet:
		ds := newDisaggregatedSetObject()
		ds.SetName(disaggregatedSetName(dgd))
		ds.SetNamespace(dgd.Namespace)
		ds.SetOwnerReferences(ownerReferences)
		return ds
	default:
		t.Fatalf("unsupported test workload provider %q", workload.provider)
		return nil
	}
}

func TestEnsureWorkloadProviderRestoresObjectAfterPatchFailure(t *testing.T) {
	t.Log("Inject a provider patch failure for an unannotated DGD")
	patchErr := errors.New("patch failed")
	seed := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(seed).
		WithInterceptorFuncs(interceptor.Funcs{
			Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
				return patchErr
			},
		}).
		Build()
	live := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(seed), live))
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
	}

	t.Log("Attempt to lock in the provider")
	_, err := reconciler.ensureWorkloadProvider(t.Context(), live)
	require.ErrorIs(t, err, patchErr)
	assert.NotErrorIs(t, err, reconcile.TerminalError(nil))

	t.Log("Verify the request object does not retain an unpersisted annotation")
	assert.Nil(t, live.Annotations)
}

func TestDynamoGraphDeploymentReconcileReportsUnsupportedWorkloadProvider(t *testing.T) {
	statusUpdateErr := errors.New("status update failed")
	tests := []struct {
		name            string
		statusUpdateErr error
	}{
		{
			name: "persists the failure before returning a terminal error",
		},
		{
			name:            "keeps a failed status update retryable",
			statusUpdateErr: statusUpdateErr,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Store a DGD with an unsupported immutable workload provider")
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "graph",
					Namespace:  "default",
					Generation: 4,
					Annotations: map[string]string{
						consts.KubeAnnotationWorkloadProvider: "unknown",
					},
				},
			}
			builder := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(dgd).
				WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphDeployment{})
			if tt.statusUpdateErr != nil {
				builder = builder.WithInterceptorFuncs(interceptor.Funcs{
					SubResourceUpdate: func(
						context.Context,
						client.Client,
						string,
						client.Object,
						...client.SubResourceUpdateOption,
					) error {
						return tt.statusUpdateErr
					},
				})
			}
			kubeClient := builder.Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        kubeClient,
				RuntimeConfig: &commoncontroller.RuntimeConfig{},
			}

			t.Log("Reconcile the invalid durable provider")
			_, err := reconciler.Reconcile(t.Context(), ctrl.Request{
				NamespacedName: client.ObjectKeyFromObject(dgd),
			})
			if tt.statusUpdateErr != nil {
				t.Log("Verify a failed diagnosis write remains retryable")
				require.ErrorIs(t, err, tt.statusUpdateErr)
				assert.NotErrorIs(t, err, reconcile.TerminalError(nil))
				return
			}
			require.ErrorIs(t, err, errUnsupportedWorkloadProvider)
			assert.ErrorIs(t, err, reconcile.TerminalError(nil))

			t.Log("Verify the immutable provider failure is visible in DGD status")
			stored := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(dgd), stored))
			ready := meta.FindStatusCondition(stored.Status.Conditions, "Ready")
			require.NotNil(t, ready)
			assert.Equal(t, metav1.ConditionFalse, ready.Status)
			assert.Equal(t, string(reasonUnsupportedWorkloadProvider), ready.Reason)
			assert.Contains(t, ready.Message, "unsupported workload provider")
		})
	}
}
