// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGetPodCliqueScalingGroups(t *testing.T) {
	for _, tc := range []struct {
		name        string
		templates   int
		missing     bool
		previousPCS bool
		deleting    bool
		secondGroup bool
		listError   bool
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
		{name: "missing template", wantError: "requires at least one scaling-group template"},
		{name: "multiple templates with one observed group", templates: 2, wantGroup: true},
		{name: "multiple observed groups", templates: 2, secondGroup: true, wantGroup: true},
		{name: "list failure", templates: 2, listError: true, wantError: "group observation unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe the expected groups under PCS ordinal zero and establish ownership once")
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: "default", UID: "pcs-uid"}}
			pcs.Spec.Template.PodCliqueScalingGroupConfigs = make([]grovev1alpha1.PodCliqueScalingGroupConfig, tc.templates)
			for index := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
				pcs.Spec.Template.PodCliqueScalingGroupConfigs[index].Name = "engines"
				if index > 0 {
					pcs.Spec.Template.PodCliqueScalingGroupConfigs[index].Name = "pending"
				}
			}
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pcs-0-engines", Namespace: pcs.Namespace,
					Labels:          map[string]string{grovecommon.LabelPartOfKey: pcs.Name},
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
				},
				Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: tc.replicas},
			}
			if tc.previousPCS {
				pcsg.OwnerReferences[0].UID = "previous-pcs"
			}
			if tc.deleting {
				pcsg.Finalizers = []string{"example.com/cleanup"}
				now := metav1.Now()
				pcsg.DeletionTimestamp = &now
			}
			var objects []client.Object
			if !tc.missing {
				objects = append(objects, pcsg)
			}
			if tc.secondGroup {
				second := pcsg.DeepCopy()
				second.Name, second.Spec.Replicas = "pcs-0-pending", 4
				objects = append(objects, second)
			}

			t.Log("Ignore unrelated names, higher PCS ordinals, and matching names in other namespaces")
			for _, key := range []client.ObjectKey{
				{Namespace: pcs.Namespace, Name: "unrelated"},
				{Namespace: pcs.Namespace, Name: "pcs-1-engines"},
				{Namespace: "other", Name: pcsg.Name},
			} {
				objects = append(objects, &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{
					Name: key.Name, Namespace: key.Namespace,
					Labels: map[string]string{grovecommon.LabelPartOfKey: pcs.Name},
				}})
			}
			lists := 0
			listError := errors.New("group observation unavailable")
			reader := interceptor.NewClient(newLPXTestClient(t, objects...), interceptor.Funcs{
				Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					t.Fatal("group observation must not issue individual GETs")
					return nil
				},
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					lists++
					if tc.listError {
						return listError
					}
					return delegated.List(ctx, list, opts...)
				},
			})

			t.Log("Only a current non-deleting group supplies capacity, including an explicit zero")
			pcsgs, err := getPodCliqueScalingGroups(t.Context(), reader, pcs)
			if tc.templates == 0 {
				require.Zero(t, lists)
			} else {
				require.Equal(t, 1, lists)
			}
			observed := pcsgs[pcsg.Name]
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				require.Nil(t, observed)
				if tc.listError {
					require.ErrorIs(t, err, listError)
				}
				return
			}
			require.NoError(t, err)
			wantGroups := 0
			if tc.wantGroup {
				wantGroups++
				require.NotNil(t, observed)
				require.Equal(t, pcsg.Name, observed.Name)
				require.Equal(t, tc.replicas, observed.Spec.Replicas)
				require.True(t, metav1.IsControlledBy(observed, pcs))
			} else {
				require.Nil(t, observed)
			}
			if tc.secondGroup {
				wantGroups++
				require.NotNil(t, pcsgs["pcs-0-pending"])
				require.EqualValues(t, 4, pcsgs["pcs-0-pending"].Spec.Replicas)
			}
			require.Len(t, pcsgs, wantGroups)
		})
	}
}

func TestGetPodCliques(t *testing.T) {
	for _, tc := range []struct {
		name, wantError                       string
		missing, deleting, foreign, listError bool
	}{
		{name: "observed"},
		{name: "missing", missing: true},
		{name: "deleting", deleting: true},
		{name: "foreign", foreign: true, wantError: "is not controlled by"},
		{name: "foreign deleting", foreign: true, deleting: true, wantError: "is not controlled by"},
		{name: "list failure", listError: true, wantError: "clique list unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Observe owned cliques and ignore cliques outside this PCS and namespace")
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: "test"}}
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "pcs-0-workload", Namespace: pcs.Namespace, UID: "group"}}
			pclq := &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name: pcsg.Name + "-0-agent", Namespace: pcs.Namespace,
					Labels:          map[string]string{grovecommon.LabelPartOfKey: pcs.Name, grovecommon.LabelPodCliqueScalingGroup: pcsg.Name, grovecommon.LabelPodCliqueSetReplicaIndex: "0"},
					OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcsg, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
				},
				Spec: grovev1alpha1.PodCliqueSpec{RoleName: "agent", Replicas: 4},
			}
			if tc.foreign {
				pclq.OwnerReferences[0].UID = "foreign"
			}
			if tc.deleting {
				pclq.DeletionTimestamp, pclq.Finalizers = ptr.To(metav1.Now()), []string{"test"}
			}
			otherPCS, otherNamespace, otherPCSReplica := pclq.DeepCopy(), pclq.DeepCopy(), pclq.DeepCopy()
			otherPCS.Name, otherPCS.Labels[grovecommon.LabelPartOfKey] = "unrelated", "other-pcs"
			otherNamespace.Namespace = "other-namespace"
			otherPCSReplica.Name = "pcs-1-workload-0-agent"
			otherPCSReplica.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "1"
			otherPCSReplica.Labels[grovecommon.LabelPodCliqueScalingGroup] = "pcs-1-workload"
			objects := []client.Object{otherPCS, otherNamespace, otherPCSReplica}
			if !tc.missing {
				objects = append(objects, pclq)
			}
			lists := 0
			reader := interceptor.NewClient(newLPXTestClient(t, objects...), interceptor.Funcs{
				Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					t.Fatal("PodClique observation must use one scoped List")
					return nil
				},
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					lists++
					if tc.listError {
						return errors.New("clique list unavailable")
					}
					return delegated.List(ctx, list, opts...)
				},
			})

			t.Log("Validate once and omit missing or deleting cliques for watch-driven progress")
			pclqs, err := getPodCliques(t.Context(), reader, pcs, map[string]*grovev1alpha1.PodCliqueScalingGroup{pcsg.Name: pcsg})
			require.Equal(t, 1, lists)
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				require.Nil(t, pclqs)
				return
			}
			require.NoError(t, err)
			if tc.missing || tc.deleting {
				require.Empty(t, pclqs)
			} else {
				require.Len(t, pclqs, 1)
				require.Contains(t, pclqs, pclq.Name)
			}
		})
	}
}

func TestLPXMissingSourcePreservesPublishedWorkload(t *testing.T) {
	t.Log("Hide the source DGD from the cache while its published workload still exists")
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
			if _, ok := object.(*v1beta1.DynamoGraphDeployment); ok {
				return apierrors.NewNotFound(v1beta1.GroupVersion.WithResource("dynamographdeployments").GroupResource(), key.Name)
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

func TestLPXDeletingSourcePreservesPublishedWorkload(t *testing.T) {
	t.Log("Keep a published workload while the DGD is awaiting its own finalization")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, desired)
	createLPXTestObjects(t, t.Context(), r.Client, objects...)
	publishSelectedLPXForTest(t, t.Context(), r, child, desired)
	pcs := findLPXTestPodCliqueSet(t, objects)
	before := getTestPipelineRequest(t, t.Context(), r.Client, child.Namespace, desired.requests[0].Name)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	commoncontroller.AddFinalizer(dgd)
	require.NoError(t, r.Update(t.Context(), dgd))
	require.NoError(t, r.Delete(t.Context(), dgd))

	t.Log("Stop publication without directly deleting the PCS or its requests")
	for range 2 {
		result, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
		require.NoError(t, err)
		require.Zero(t, result, "the source and child watches observe deletion")
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
	cl := newLPXTestClient(t, pcs)
	changed := pcs.DeepCopy()
	changed.Labels = map[string]string{"example.com/concurrent": "update"}
	require.NoError(t, cl.Update(t.Context(), changed))
	cl = interceptor.NewClient(cl, interceptor.Funcs{
		Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
			options := (&client.DeleteOptions{}).ApplyOptions(opts)
			require.Equal(t, metav1.DeletePropagationForeground, *options.PropagationPolicy)
			require.Equal(t, pcs.UID, *options.Preconditions.UID)
			require.Equal(t, pcs.ResourceVersion, *options.Preconditions.ResourceVersion)
			return delegated.Delete(ctx, object, opts...)
		},
	})

	t.Log("Foreground deletion cannot remove a resource changed since observation")
	err := deletePodCliqueSet(t.Context(), cl, pcs)
	require.True(t, apierrors.IsConflict(err), "expected a resource-version conflict: %v", err)
	stored := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, cl.Get(t.Context(), client.ObjectKeyFromObject(pcs), stored))
	require.Equal(t, changed, stored)
}

func TestScalePodCliqueScalingGroup(t *testing.T) {
	conflict := apierrors.NewConflict(consts.PodCliqueScalingGroupGVR.GroupResource(), "group", errors.New("stale version"))
	for _, tc := range []struct {
		name        string
		replicas    int32
		err         error
		wantUpdates int
	}{
		{name: "scale out", replicas: 12, wantUpdates: 1},
		{name: "scale in", replicas: 2, wantUpdates: 1},
		{name: "scale to zero", replicas: 0, wantUpdates: 1},
		{name: "unchanged", replicas: 9},
		{name: "conflict", replicas: 12, err: conflict, wantUpdates: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Use an already-owned group observation without rendering a workload")
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "group", Namespace: "default", ResourceVersion: "7"},
				Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 9},
			}
			observed := pcsg.DeepCopy()
			updates := 0
			cl := interceptor.NewClient(newLPXTestClient(t), interceptor.Funcs{
				SubResourceUpdate: func(_ context.Context, _ client.Client, objectSubresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					require.Equal(t, "scale", objectSubresource)
					scale := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts).SubResourceBody.(*autoscalingv1.Scale)
					require.Equal(t, object.GetName(), scale.Name)
					require.Equal(t, "7", scale.ResourceVersion)
					require.Equal(t, tc.replicas, scale.Spec.Replicas)
					updates++
					if tc.err != nil {
						return tc.err
					}
					scale.SetResourceVersion("8")
					scale.SetGeneration(2)
					return nil
				},
			})

			t.Log("Report capacity writes while leaving the observation unchanged")
			changed, err := scalePodCliqueScalingGroup(t.Context(), cl, pcsg, tc.replicas)
			require.ErrorIs(t, err, tc.err)
			require.Equal(t, tc.wantUpdates, updates)
			require.Equal(t, tc.wantUpdates > 0 && tc.err == nil, changed)
			require.Equal(t, observed, pcsg)
		})
	}
}

func TestScalePodCliques(t *testing.T) {
	conflict := errors.New("scale conflict")
	for _, tc := range []struct {
		name        string
		replicas    int32
		failOnWrite int
		wantWrites  int
		wantApplied int
	}{
		{name: "converged", replicas: 1},
		{name: "scaled", replicas: 3, wantWrites: 2, wantApplied: 2},
		{name: "scale failure", replicas: 3, failOnWrite: 1, wantWrites: 1},
		{name: "partial scale failure", replicas: 3, failOnWrite: 2, wantWrites: 2, wantApplied: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Supply already-owned cliques under a two-replica group")
			pcsg := &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "test"},
				Spec:       grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 2},
			}
			pclqs := make(map[string]*grovev1alpha1.PodClique)
			want := make(map[string]*grovev1alpha1.PodClique)
			for _, name := range []string{"workload-0-worker", "workload-1-worker"} {
				pclq := &grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: pcsg.Namespace, ResourceVersion: "7", Generation: 1},
					Spec:       grovev1alpha1.PodCliqueSpec{Replicas: 1},
				}
				pclqs[name] = pclq
				want[name] = pclq.DeepCopy()
			}
			writes := 0
			cl := interceptor.NewClient(newLPXTestClient(t), interceptor.Funcs{
				SubResourceUpdate: func(_ context.Context, _ client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					require.Equal(t, "scale", subresource)
					scale := (&client.SubResourceUpdateOptions{}).ApplyOptions(opts).SubResourceBody.(*autoscalingv1.Scale)
					require.Equal(t, object.GetName(), scale.Name)
					require.Equal(t, "7", scale.ResourceVersion)
					require.Equal(t, int32(3), scale.Spec.Replicas)
					writes++
					if writes == tc.failOnWrite {
						return conflict
					}
					scale.SetResourceVersion("8")
					scale.SetGeneration(2)
					return nil
				},
			})

			t.Log("Report successful writes, including partial progress, without mutating observations")
			changed, err := scalePodCliques(t.Context(), cl, pcsg, pclqs, "worker", tc.replicas)
			if tc.failOnWrite > 0 {
				require.ErrorIs(t, err, conflict)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantWrites, writes)
			require.Equal(t, tc.wantApplied > 0, changed)
			require.Equal(t, want, pclqs)
		})
	}
}
