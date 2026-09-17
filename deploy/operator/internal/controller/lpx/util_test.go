// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGetPodCliqueScalingGroup(t *testing.T) {
	for _, tc := range []struct {
		name        string
		templates   int
		missing     bool
		previousPCS bool
		deleting    bool
		replicas    int32
		wantGroup   bool
		wantError   string
	}{
		{name: "current group", templates: 1, replicas: 3, wantGroup: true},
		{name: "zero replicas is still an observed group", templates: 1, wantGroup: true},
		{name: "missing group", templates: 1, missing: true},
		{name: "previous PCS incarnation", templates: 1, previousPCS: true, wantError: `PodCliqueScalingGroup "pcs-0-engines" is not controlled by PodCliqueSet "pcs"`},
		{name: "deleting previous PCS incarnation", templates: 1, previousPCS: true, deleting: true, wantError: `PodCliqueScalingGroup "pcs-0-engines" is not controlled by PodCliqueSet "pcs"`},
		{name: "deleting group", templates: 1, deleting: true},
		{name: "missing template", wantError: "requires exactly one scaling-group template"},
		{name: "multiple templates", templates: 2, wantError: "requires exactly one scaling-group template"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe the one group under PCS ordinal zero and establish ownership once")
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: "default", UID: "pcs-uid"}}
			pcs.Spec.Template.PodCliqueScalingGroupConfigs = make([]grovev1alpha1.PodCliqueScalingGroupConfig, tc.templates)
			for index := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
				pcs.Spec.Template.PodCliqueScalingGroupConfigs[index].Name = "engines"
			}
			group := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pcs-0-engines", Namespace: pcs.Namespace,
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: tc.replicas},
			}
			if tc.previousPCS {
				group.OwnerReferences[0].UID = "previous-pcs"
			}
			if tc.deleting {
				group.Finalizers = []string{"example.com/cleanup"}
				now := metav1.Now()
				group.DeletionTimestamp = &now
			}
			var objects []client.Object
			if !tc.missing {
				objects = append(objects, group)
			}
			reader := newLPXTestClient(t, objects...)

			t.Log("Only a current non-deleting group supplies capacity, including an explicit zero")
			observed, err := getPodCliqueScalingGroup(t.Context(), reader, pcs)
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				require.Nil(t, observed)
				return
			}
			require.NoError(t, err)
			if tc.wantGroup {
				require.NotNil(t, observed)
				require.Equal(t, group.Name, observed.Name)
				require.Equal(t, tc.replicas, observed.Spec.Replicas)
				require.True(t, metav1.IsControlledBy(observed, pcs))
			} else {
				require.Nil(t, observed)
			}
		})
	}
}

func TestLPXCacheMissesPreservePublishedWorkload(t *testing.T) {
	for _, missing := range []string{"source", "group"} {
		t.Run(missing, func(t *testing.T) {
			t.Log("Hide one dependency from the cache while its objects still exist")
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, desired)
			createLPXTestObjects(t, t.Context(), r.Client, objects...)
			publishSelectedLPXForTest(t, t.Context(), r, child, desired)
			before := getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, desired.requests[0].Name)
			base := r.Client.(client.WithWatch)
			writes := 0
			r.Client = interceptor.NewClient(base, interceptor.Funcs{
				Get: func(ctx context.Context, delegated client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
					kind := ""
					switch object.(type) {
					case *v1beta1.DynamoGraphDeployment:
						kind = "source"
					case *grovev1alpha1.PodCliqueScalingGroup:
						kind = "group"
					}
					if kind == missing {
						return apierrors.NewNotFound(grovev1alpha1.SchemeGroupVersion.WithResource(kind).GroupResource(), key.Name)
					}
					return delegated.Get(ctx, key, object, opts...)
				},
				Create: func(context.Context, client.WithWatch, client.Object, ...client.CreateOption) error {
					writes++
					return errors.New("unexpected create during cache miss")
				},
				Delete: func(context.Context, client.WithWatch, client.Object, ...client.DeleteOption) error {
					writes++
					return errors.New("unexpected deletion during cache miss")
				},
			})

			t.Log("Wait without treating a missing cache observation as deletion authority")
			for range 2 {
				result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
				require.NoError(t, err)
				require.Zero(t, result, "dependency watches resume reconciliation")
			}
			require.Zero(t, writes)
			require.Equal(t, before, getTestPipelineRequest(t, t.Context(), base, before.Namespace, before.Name))
		})
	}
}

func TestLPXForeignRequestNameIsNeverAdopted(t *testing.T) {
	t.Log("Occupy the deterministic request name with another PCS's request")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	foreign := desired.requests[0].DeepCopy()
	foreignPCS := findLPXTestPodCliqueSet(t, objects).DeepCopy()
	foreignPCS.UID = "foreign-pcs"
	foreign.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(foreignPCS, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
	require.NoError(t, r.Create(t.Context(), foreign))
	before := getTestPipelineRequest(t, t.Context(), r.Client, foreign.Namespace, foreign.Name)

	t.Log("Finish PCS synchronization before testing the conflicting request name")
	result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.Zero(t, result)
	require.Equal(t, before, getTestPipelineRequest(t, t.Context(), r.Client, foreign.Namespace, foreign.Name))

	t.Log("An indexed miss followed by AlreadyExists waits without adoption or deletion")
	for range 2 {
		result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
		require.NoError(t, err)
		require.Zero(t, result, "LPR events resume reconciliation; name collisions are never adopted")
		require.Equal(t, before, getTestPipelineRequest(t, t.Context(), r.Client, foreign.Namespace, foreign.Name))
	}
}

func TestLPXDeselectionWaitsForParentDeletion(t *testing.T) {
	t.Log("Keep a published workload while its source removes the LPX component")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	before := getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, desired.requests[0].Name)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	dgd.Spec.Components = nil
	require.NoError(t, r.Update(t.Context(), dgd))

	t.Log("Wait for the parent's deletion and GC without hashing empty LPX inputs or deleting the PCS")
	for range 2 {
		result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
		require.NoError(t, err)
		require.Zero(t, result, "the source and child watches observe deselection")
	}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	require.True(t, pcs.DeletionTimestamp.IsZero())
	require.Equal(t, before, getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, before.Name))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Equal(t, v1alpha1.LPXReadyReasonPending, meta.FindStatusCondition(child.Status.Conditions, "Ready").Reason)
}

func TestLPXDeletionUsesSnapshotPreconditions(t *testing.T) {
	t.Log("Observe the PCS before a concurrent update")
	pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
		Name: "pcs", Namespace: "default", UID: "pcs-uid", ResourceVersion: "1",
	}}
	r := &graphReconciler{Client: newLPXTestClient(t, pcs)}
	changed := pcs.DeepCopy()
	changed.Labels = map[string]string{"example.com/concurrent": "update"}
	require.NoError(t, r.Update(t.Context(), changed))
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			options := (&client.DeleteOptions{}).ApplyOptions(opts)
			require.Equal(t, metav1.DeletePropagationForeground, *options.PropagationPolicy)
			require.Equal(t, pcs.UID, *options.Preconditions.UID)
			require.Equal(t, pcs.ResourceVersion, *options.Preconditions.ResourceVersion)
			return delegated.Delete(ctx, object, opts...)
		},
	})

	t.Log("Foreground deletion cannot remove a resource changed since observation")
	err := r.deletePodCliqueSet(t.Context(), pcs)
	require.True(t, apierrors.IsConflict(err), "expected a resource-version conflict: %v", err)
	stored := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), stored))
	require.Equal(t, changed, stored)
}
