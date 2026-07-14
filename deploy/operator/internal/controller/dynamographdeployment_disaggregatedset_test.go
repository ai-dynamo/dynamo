/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
)

func TestSelectDisaggregatedSetComponents(t *testing.T) {
	t.Run("selects multinode worker roles", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "prefill",
						ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "frontend",
						ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					},
				},
			},
		}

		selection, reason := selectDisaggregatedSetComponents(dgd)
		require.Empty(t, reason)
		require.Equal(t, "prefill", selection.componentToRole["prefill"])
		require.Equal(t, "decode", selection.componentToRole["decode"])
		require.Len(t, selection.componentToRole, 2)
	})

	t.Run("rejects scaling adapter", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName:  "prefill",
						ComponentType:  nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:      &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
						Replicas:       ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
		}

		_, reason := selectDisaggregatedSetComponents(dgd)
		require.Contains(t, reason, "scalingAdapter")
	})

	t.Run("rejects more roles than the DS API supports", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
		for i := 0; i < maxDisaggregatedSetRoles+1; i++ {
			dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: fmt.Sprintf("worker-%d", i),
				ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
				Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
			})
		}

		_, reason := selectDisaggregatedSetComponents(dgd)
		require.Contains(t, reason, "at most 10 roles")
	})
}

func TestCheckDisaggregatedSetReadiness(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetGeneration(3)
	ds.Object["status"] = map[string]any{
		"observedGeneration": int64(2),
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
			map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(1)},
		},
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 2, "decode": 2},
	}

	ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "observed generation")
	require.Equal(t, int32(2), ptr.Deref(statuses["prefill"].ReadyReplicas, 0))

	ds.Object["status"].(map[string]any)["observedGeneration"] = int64(3)
	ready, reason, statuses = checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "decode")
	require.Equal(t, int32(1), ptr.Deref(statuses["decode"].ReadyReplicas, 0))

	ds.Object["status"].(map[string]any)["roleStatuses"] = []any{
		map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
		map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
	}
	ready, _, _ = checkDisaggregatedSetReadiness(ds, selection)
	require.True(t, ready)
}

func TestApplyDisaggregatedSetCheckpointStartupPoliciesCoordinatesSelectedRoles(t *testing.T) {
	dcds := map[string]*nvidiacomv1beta1.DynamoComponentDeployment{
		"prefill": {
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{Replicas: ptr.To(int32(2))},
			},
		},
		"decode": {
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{Replicas: ptr.To(int32(2))},
			},
		},
	}
	checkpointInfos := map[string]*checkpoint.CheckpointInfo{
		"prefill": {
			Enabled:       true,
			Ready:         false,
			StartupPolicy: nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
		},
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 2, "decode": 2},
	}

	gated, err := applyDisaggregatedSetCheckpointStartupPolicies(dcds, checkpointInfos, selection)
	require.NoError(t, err)
	require.True(t, gated)
	require.Equal(t, int32(0), ptr.Deref(dcds["prefill"].Spec.Replicas, -1))
	require.Equal(t, int32(0), ptr.Deref(dcds["decode"].Spec.Replicas, -1))
	require.Equal(t, map[string]int32{"prefill": 0, "decode": 0}, selection.desiredReplicas)
}

func TestDeleteStaleDisaggregatedSetComponentServices(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	labels := func(componentName string) map[string]string {
		return map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponent:           componentName,
		}
	}
	service := func(name, componentName string, owned bool) *corev1.Service {
		svc := &corev1.Service{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: dgd.Namespace, Labels: labels(componentName)}}
		if owned {
			svc.OwnerReferences = []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)}
		}
		return svc
	}
	current := service("demo-prefill-new", "prefill", true)
	stale := service("demo-prefill-old", "prefill", true)
	removedComponent := service("demo-removed-old", "removed", true)
	userManaged := service("demo-prefill-user", "prefill", false)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, current, stale, removedComponent, userManaged).Build(),
	}
	dcds := map[string]*nvidiacomv1beta1.DynamoComponentDeployment{
		"prefill": {ObjectMeta: metav1.ObjectMeta{Name: current.Name}},
	}

	require.NoError(t, reconciler.deleteStaleDisaggregatedSetComponentServices(t.Context(), dgd, dcds))
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(current), &corev1.Service{}))
	require.NoError(t, reconciler.Get(t.Context(), client.ObjectKeyFromObject(userManaged), &corev1.Service{}))
	require.True(t, apierrors.IsNotFound(reconciler.Get(t.Context(), client.ObjectKeyFromObject(stale), &corev1.Service{})))
	require.True(t, apierrors.IsNotFound(reconciler.Get(t.Context(), client.ObjectKeyFromObject(removedComponent), &corev1.Service{})))
}

func TestCheckDisaggregatedSetReadinessFallsBackToTargetRevisionChildLWS(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, leaderworkersetv1.AddToScheme(scheme))

	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetNamespace("default")
	ds.SetUID("ds-uid")
	typedDS := &disaggregatedsetv1.DisaggregatedSet{
		Spec: disaggregatedsetv1.DisaggregatedSetSpec{Roles: []disaggregatedsetv1.DisaggregatedRoleSpec{
			{Name: "prefill"},
			{Name: "decode"},
		}},
	}
	typedObject, err := runtime.DefaultUnstructuredConverter.ToUnstructured(typedDS)
	require.NoError(t, err)
	ds.Object["spec"] = typedObject["spec"]
	targetRevision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}
	owner := metav1.OwnerReference{
		APIVersion: disaggregatedSetGVK.GroupVersion().String(),
		Kind:       disaggregatedSetGVK.Kind,
		Name:       ds.GetName(),
		UID:        ds.GetUID(),
		Controller: ptr.To(true),
	}
	child := func(name, role, revision string, ready int32) *leaderworkersetv1.LeaderWorkerSet {
		return &leaderworkersetv1.LeaderWorkerSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:       name,
				Namespace:  ds.GetNamespace(),
				Generation: 1,
				Labels: map[string]string{
					disaggregatedsetv1.SetNameLabelKey:  ds.GetName(),
					disaggregatedsetv1.RoleLabelKey:     role,
					disaggregatedsetv1.RevisionLabelKey: revision,
				},
				OwnerReferences: []metav1.OwnerReference{owner},
			},
			Status: leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: 1,
				Replicas:           1,
				UpdatedReplicas:    ready,
				ReadyReplicas:      ready,
			},
		}
	}
	objects := []client.Object{
		child("demo-old-prefill", "prefill", "old", 1),
		child("demo-new-prefill", "prefill", targetRevision, 0),
		child("demo-decode", "decode", targetRevision, 1),
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build(),
	}

	ready, reason, statuses, err := reconciler.checkDisaggregatedSetReadiness(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, ready)
	require.Contains(t, reason, "demo-new-prefill")
	require.Equal(t, []string{"demo-new-prefill", "demo-old-prefill"}, statuses["prefill"].ComponentNames)
	require.Equal(t, int32(2), statuses["prefill"].Replicas)

	newPrefill := &leaderworkersetv1.LeaderWorkerSet{}
	require.NoError(t, reconciler.Get(t.Context(), types.NamespacedName{Name: "demo-new-prefill", Namespace: ds.GetNamespace()}, newPrefill))
	newPrefill.Status.UpdatedReplicas = 1
	newPrefill.Status.ReadyReplicas = 1
	require.NoError(t, reconciler.Update(t.Context(), newPrefill))

	ready, reason, _, err = reconciler.checkDisaggregatedSetReadiness(t.Context(), ds, selection)
	require.NoError(t, err)
	require.False(t, ready)
	require.Contains(t, reason, "old child")

	oldPrefill := &leaderworkersetv1.LeaderWorkerSet{}
	require.NoError(t, reconciler.Get(t.Context(), types.NamespacedName{Name: "demo-old-prefill", Namespace: ds.GetNamespace()}, oldPrefill))
	oldPrefill.Status.Replicas = 0
	oldPrefill.Status.UpdatedReplicas = 0
	oldPrefill.Status.ReadyReplicas = 0
	require.NoError(t, reconciler.Update(t.Context(), oldPrefill))

	ready, _, _, err = reconciler.checkDisaggregatedSetReadiness(t.Context(), ds, selection)
	require.NoError(t, err)
	require.True(t, ready)
}

func TestListOwnedSelectedDCDsSkipsUserManagedDCDs(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo",
			Namespace: "default",
			UID:       "demo-uid",
		},
	}
	owned := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-prefill",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           "prefill",
			},
			OwnerReferences: []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "prefill",
				ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
			},
		},
	}
	userManaged := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-decode",
			Namespace: "default",
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           "decode",
			},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
			},
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, owned, userManaged).Build(),
	}

	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{
			dynamo.GetDCDComponentName(owned):       "prefill",
			dynamo.GetDCDComponentName(userManaged): "decode",
		},
		desiredReplicas: map[string]int32{"prefill": 1, "decode": 1},
	}

	got, err := reconciler.listOwnedSelectedDCDs(t.Context(), dgd, selection)
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, owned.Name, got[0].Name)
}

func TestReconcileReplacementBeforeDisaggregatedSetCleanup(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})

	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(scheme).WithObjects(ds).Build(),
		RuntimeConfig: newTestRuntimeConfig(true),
	}
	key := types.NamespacedName{Name: ds.GetName(), Namespace: ds.GetNamespace()}

	result, err := reconciler.reconcileReplacementBeforeDisaggregatedSetCleanup(t.Context(), dgd, func() (ReconcileResult, error) {
		existing := newDisaggregatedSetObject()
		require.NoError(t, reconciler.Get(t.Context(), key, existing), "DisaggregatedSet must remain while its replacement is pending")
		return ReconcileResult{State: nvidiacomv1beta1.DGDStatePending}, nil
	})
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStatePending, result.State)
	require.NoError(t, reconciler.Get(t.Context(), key, newDisaggregatedSetObject()))

	result, err = reconciler.reconcileReplacementBeforeDisaggregatedSetCleanup(t.Context(), dgd, func() (ReconcileResult, error) {
		return ReconcileResult{State: nvidiacomv1beta1.DGDStateSuccessful}, nil
	})
	require.NoError(t, err)
	require.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	require.True(t, apierrors.IsNotFound(reconciler.Get(t.Context(), key, newDisaggregatedSetObject())))
}

type notFoundOnDeleteClient struct {
	client.Client
}

func (c notFoundOnDeleteClient) Delete(_ context.Context, obj client.Object, _ ...client.DeleteOption) error {
	return apierrors.NewNotFound(schema.GroupResource{Group: disaggregatedSetGVK.Group, Resource: "disaggregatedsets"}, obj.GetName())
}

func TestDeleteDisaggregatedSetIgnoresNotFoundRace(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})
	baseClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ds).Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: notFoundOnDeleteClient{Client: baseClient}}

	require.NoError(t, reconciler.deleteDisaggregatedSetIfExists(t.Context(), dgd))
}

func TestSyncDisaggregatedSetPreservesUnmanagedMetadata(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	current := newDisaggregatedSetObject()
	current.SetName("demo")
	current.SetNamespace("default")
	current.SetLabels(map[string]string{"example.com/keep": "label"})
	current.SetAnnotations(map[string]string{"example.com/keep": "annotation"})
	current.SetOwnerReferences([]metav1.OwnerReference{
		*dgdControllerOwnerReference(dgd),
		{APIVersion: "v1", Kind: "ConfigMap", Name: "keep", UID: "keep-uid"},
	})
	current.Object["spec"] = map[string]any{"roles": []any{}}
	desired := current.DeepCopy()
	desired.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name})
	desired.SetAnnotations(nil)
	desired.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})
	desired.Object["spec"] = map[string]any{"roles": []any{map[string]any{"name": "prefill"}, map[string]any{"name": "decode"}}}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(dgd, current).Build(),
	}

	modified, synced, err := reconciler.syncDisaggregatedSet(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, "label", synced.GetLabels()["example.com/keep"])
	require.Equal(t, "annotation", synced.GetAnnotations()["example.com/keep"])
	require.Len(t, synced.GetOwnerReferences(), 2)
}

func TestMapDisaggregatedSetChildLWSToDGD(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default", UID: "demo-uid"},
	}
	ds := newDisaggregatedSetObject()
	ds.SetName(disaggregatedSetName(dgd))
	ds.SetNamespace(dgd.Namespace)
	ds.SetUID("ds-uid")
	ds.SetOwnerReferences([]metav1.OwnerReference{*dgdControllerOwnerReference(dgd)})
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(ds).Build(),
	}
	child := &leaderworkersetv1.LeaderWorkerSet{ObjectMeta: metav1.ObjectMeta{
		Name:      "demo-revision-prefill",
		Namespace: dgd.Namespace,
		Labels:    map[string]string{disaggregatedsetv1.SetNameLabelKey: ds.GetName()},
	}}

	requests := reconciler.mapDisaggregatedSetChildLWSToDGD(t.Context(), child)
	require.Len(t, requests, 1)
	require.Equal(t, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, requests[0].NamespacedName)
}

func TestShouldUseDisaggregatedSet(t *testing.T) {
	twoEligibleDGD := func() *nvidiacomv1beta1.DynamoGraphDeployment {
		return &nvidiacomv1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "demo",
				Namespace: "default",
				Annotations: map[string]string{
					consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
				},
			},
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "prefill",
						ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					},
				},
			},
		}
	}

	t.Run("annotation missing falls back", func(t *testing.T) {
		dgd := twoEligibleDGD()
		delete(dgd.Annotations, consts.KubeAnnotationEnableDisaggregatedSet)
		r := &DynamoGraphDeploymentReconciler{
			RuntimeConfig: newTestRuntimeConfig(true),
		}
		use, reason := r.shouldUseDisaggregatedSet(dgd)
		require.False(t, use)
		require.Empty(t, reason)
	})

	t.Run("API unavailable falls back with reason", func(t *testing.T) {
		dgd := twoEligibleDGD()
		r := &DynamoGraphDeploymentReconciler{
			RuntimeConfig: newTestRuntimeConfig(false),
		}
		use, reason := r.shouldUseDisaggregatedSet(dgd)
		require.False(t, use)
		require.Contains(t, reason, "DisaggregatedSet API is not available")
	})

	t.Run("only one eligible role falls back with reason", func(t *testing.T) {
		dgd := twoEligibleDGD()
		dgd.Spec.Components = dgd.Spec.Components[:1]
		r := &DynamoGraphDeploymentReconciler{
			RuntimeConfig: newTestRuntimeConfig(true),
		}
		use, reason := r.shouldUseDisaggregatedSet(dgd)
		require.False(t, use)
		require.Contains(t, reason, "two eligible multinode worker roles")
	})

	t.Run("eligible DGD opts in", func(t *testing.T) {
		dgd := twoEligibleDGD()
		r := &DynamoGraphDeploymentReconciler{
			RuntimeConfig: newTestRuntimeConfig(true),
		}
		use, reason := r.shouldUseDisaggregatedSet(dgd)
		require.True(t, use)
		require.Empty(t, reason)
	})
}

func newTestRuntimeConfig(enabled bool) *commoncontroller.RuntimeConfig {
	return &commoncontroller.RuntimeConfig{DisaggregatedSetEnabled: enabled}
}
