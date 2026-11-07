// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// newUnitsTestRegistry publishes matching legacy/units shapes and a larger units
// build. Each entry keeps its own content identity for real registry acquisition.
func newUnitsTestRegistry(t *testing.T) lpx.ModelRegistry {
	t.Helper()
	root := t.TempDir()
	for _, id := range []string{"legacy", "units", "units-wide"} {
		writeLPXCheckpointTestBuild(t, root, id, deploymentcapnpv1.CompilationMode_lpuOnly, checkpointTestModel, checkpointTestRevision)
		if id == "legacy" {
			continue
		}
		path := filepath.Join(root, id, "deployment.v1.capnp.bin")
		data, err := os.ReadFile(path)
		require.NoError(t, err)
		message, err := capnp.Unmarshal(data)
		require.NoError(t, err)
		deployment, err := deploymentcapnpv1.ReadRootDeployment(message)
		require.NoError(t, err)
		writable, _ := capnp.NewSingleSegmentMessage(nil)
		require.NoError(t, writable.SetRoot(deployment.ToPtr()))
		message = writable
		deployment, err = deploymentcapnpv1.ReadRootDeployment(message)
		require.NoError(t, err)
		config, err := deployment.Config()
		require.NoError(t, err)
		units, err := config.NewLpuUnits(1)
		require.NoError(t, err)
		reservations, err := units.At(0).NewReservations(2)
		require.NoError(t, err)
		partitions, err := deployment.Partitions()
		require.NoError(t, err)
		for i, id := range []uint32{7, 8} {
			ids, err := reservations.At(i).NewPartitionIds(1)
			require.NoError(t, err)
			ids.Set(0, id)
			hosts := uint32(2)
			if filepath.Base(filepath.Dir(path)) == "units-wide" {
				hosts = 4
				detail, err := partitions.At(i).Detail().Lpu()
				require.NoError(t, err)
				detail.SetNumChips(32)
			}
			reservations.At(i).Size().SetNodes(hosts)
		}
		connectors, err := units.At(0).NewConnectors(1)
		require.NoError(t, err)
		connectors.At(0).SetXt8888Gap()
		data, err = message.Marshal()
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(path, data, 0o600))
	}
	registry, err := lpx.NewModelRegistry((&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String(), nil)
	require.NoError(t, err)
	return registry
}

func removeUnitsTestConfigMounts(component *v1beta1.DynamoComponentDeploymentSharedSpec) {
	for i := range component.Roles {
		spec := &component.Roles[i].PodTemplate.Spec
		for _, containers := range [][]corev1.Container{spec.Containers, spec.InitContainers} {
			for j := range containers {
				containers[j].VolumeMounts = slices.DeleteFunc(containers[j].VolumeMounts, func(m corev1.VolumeMount) bool { return m.Name == "config" })
			}
		}
	}
}

func TestUnitsMixedPairPreservesPublishedWorkload(t *testing.T) {
	for _, changed := range []string{"draft", "lpx"} {
		t.Run(changed, func(t *testing.T) {
			t.Log("Publish a legacy pair through the existing controller test harness")
			registry := newUnitsTestRegistry(t)
			dgd := loadTestSpecDecodeDGD(t)
			for i := range dgd.Spec.Components {
				dgd.Spec.Components[i].LPX.BuildID = "legacy"
			}
			child := newLPXTestDeployment(t, dgd)
			r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, t.Context(), r.Client, objects...)
			publishSelectedLPXForTest(t, t.Context(), r, child, selected)
			pcs := observedLPXTestPodCliqueSet(t, t.Context(), r, child, selected)
			beforeRequests, err := r.getPipelineRequests(t.Context(), pcs)
			require.NoError(t, err)
			t.Log("Change one member to units and also change immutable draft composition")
			dgd.GetComponentByName(changed).LPX.BuildID = "units"
			dgd.GetComponentByName("draft").Replicas = new(int32(1))
			removeUnitsTestConfigMounts(dgd.GetComponentByName(changed))
			updateTestDGD(t, r, child, dgd)
			for range 2 {
				_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
				require.ErrorContains(t, err, "mixes builds with and without lpuUnits")
				got, err := getPodCliqueSet(t.Context(), r.Client, child)
				require.NoError(t, err)
				require.Equal(t, pcs, got)
				requests, err := r.getPipelineRequests(t.Context(), pcs)
				require.NoError(t, err)
				require.Equal(t, beforeRequests, requests)
			}
			t.Log("Report the rejected placement without publishing a partial workload")
			require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
			condition := meta.FindStatusCondition(child.Status.Conditions, placementResolvedCondition)
			require.NotNil(t, condition)
			require.Equal(t, metav1.ConditionFalse, condition.Status)
			require.Equal(t, "InvalidWorkload", condition.Reason)
		})
	}
}

func TestUnitsIndependentLegacyWorkload(t *testing.T) {
	t.Log("Resolve independent legacy and units workloads within the same graph")
	registry := newUnitsTestRegistry(t)
	dgd := loadTestDGD(t, lpx.PipelineSingle, "legacy")
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName = "units-engine"
	second.LPX.BuildID = "units"
	removeUnitsTestConfigMounts(second)
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	child := newLPXTestDeployment(t, dgd)
	r := newLPXTestReconciler(t, registry, child, dgd)
	_, err := r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	pcs, err := getPodCliqueSet(t.Context(), r.Client, child)
	require.NoError(t, err)
	require.NotNil(t, pcs)
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 2)
	t.Log("Publish configuration only for the independent legacy workload")
	maps := &corev1.ConfigMapList{}
	require.NoError(t, r.List(t.Context(), maps))
	require.Len(t, maps.Items, 1)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	condition := meta.FindStatusCondition(child.Status.Conditions, placementResolvedCondition)
	require.NotNil(t, condition)
	require.Equal(t, "MultipleSources", condition.Reason)
	require.Equal(t, "lpx=legacy; units-engine=units", condition.Message)
	t.Log("A second reconcile is a no-op while it waits for Grove observation")
	before := pcs.DeepCopy()
	_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)})
	require.NoError(t, err)
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	require.Equal(t, before, pcs)
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.NoError(t, r.List(t.Context(), requests))
	require.Empty(t, requests.Items)
}

func TestUnitsRenderLayoutDiff(t *testing.T) {
	t.Log("Compare real legacy and units requests and immutable PCS layout")
	registry := newUnitsTestRegistry(t)
	dgd := loadTestDGD(t, lpx.PipelineSingle, "legacy")
	child := newLPXTestDeployment(t, dgd)
	r, legacy := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
	before := renderLPXTestPodCliqueSet(t, t.Context(), r, child, dgd, legacy)
	for _, id := range []string{"units", "units-wide"} {
		next := dgd.DeepCopy()
		next.Spec.Components[0].LPX.BuildID = id
		removeUnitsTestConfigMounts(&next.Spec.Components[0])
		selected := resolveLPXTestWorkload(t, registry, t.Context(), child, next)
		after := renderLPXTestPodCliqueSet(t, t.Context(), r, child, next, selected)
		require.Equal(t, id == "units", podCliqueSetLayoutMatches(before, after), id)
		if id == "units" {
			require.Equal(t, legacy.requests[0].Spec, selected.requests[0].Spec)
		} else {
			require.NotEqual(t, legacy.requests[0].Spec, selected.requests[0].Spec)
		}
		require.NotEqual(t, legacy.requests[0].Annotations, selected.requests[0].Annotations)
	}
}

func TestUnitsReplacementAndRollback(t *testing.T) {
	t.Log("Publish the legacy geometry and retain its runtime configuration")
	ctx := t.Context()
	registry := newUnitsTestRegistry(t)
	dgd := loadTestDGD(t, lpx.PipelineSingle, "legacy")
	original := dgd.DeepCopy()
	child := newLPXTestDeployment(t, dgd)
	r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	objects := lpxMaterializedObjects(t, r, child, dgd, selected)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	publishSelectedLPXForTest(t, ctx, r, child, selected)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	_, err := r.Reconcile(ctx, request)
	require.NoError(t, err)
	pcs, err := getPodCliqueSet(ctx, r, child)
	require.NoError(t, err)
	maps := &corev1.ConfigMapList{}
	require.NoError(t, r.List(ctx, maps))
	require.Len(t, maps.Items, 1)
	originalMap := maps.Items[0].DeepCopy()

	t.Log("Observe foreground deletion and give each recreated PCS a new API identity")
	creates, deletes := 0, 0
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
		Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
			if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok {
				options := (&client.DeleteOptions{}).ApplyOptions(opts)
				require.Equal(t, metav1.DeletePropagationForeground, *options.PropagationPolicy)
				require.Equal(t, obj.GetUID(), *options.Preconditions.UID)
				deletes++
			}
			return c.Delete(ctx, obj, opts...)
		},
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok {
				creates++
				obj.SetUID(types.UID(fmt.Sprintf("units-pcs-%d", creates)))
			}
			return c.Create(ctx, obj, opts...)
		},
	})
	for _, buildID := range []string{"units-wide", "legacy"} {
		t.Logf("Move to %s with a known incompatible immutable layout", buildID)
		oldUID := pcs.UID
		oldRequests, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.NotEmpty(t, oldRequests)
		pcs.Finalizers = []string{"test/foreground-gc"}
		require.NoError(t, r.Update(ctx, pcs))
		dgd.Spec = *original.Spec.DeepCopy()
		dgd.Spec.Components[0].LPX.BuildID = buildID
		if buildID != "legacy" {
			removeUnitsTestConfigMounts(&dgd.Spec.Components[0])
		}
		updateTestDGD(t, r, child, dgd)
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
		require.False(t, pcs.DeletionTimestamp.IsZero())
		require.Equal(t, oldUID, pcs.UID)
		retained, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Equal(t, oldRequests, retained)
		if buildID == "units-wide" {
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(originalMap), &corev1.ConfigMap{}))
		}
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.Equal(t, creates, deletes-1, "a terminating PCS cannot be replaced before GC")

		t.Log("Emulate foreground garbage collection; the controller does not delete dependent state itself")
		for _, oldRequest := range oldRequests {
			require.True(t, metav1.IsControlledBy(oldRequest, pcs))
			require.NoError(t, r.Delete(ctx, oldRequest))
		}
		for _, obj := range objects {
			if _, ok := obj.(*grovev1alpha1.PodCliqueSet); !ok {
				require.NoError(t, client.IgnoreNotFound(r.Delete(ctx, obj)))
			}
		}
		pcs.Finalizers = nil
		require.NoError(t, r.Update(ctx, pcs))
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		pcs, err = getPodCliqueSet(ctx, r, child)
		require.NoError(t, err)
		require.NotNil(t, pcs)
		require.NotEqual(t, oldUID, pcs.UID)
		newRequests, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Empty(t, newRequests, "publication waits for the newly written PCS observation")

		t.Log("Materialize the new Grove hierarchy with the replacement PCS UID")
		selected = resolveLPXTestWorkload(t, registry, ctx, child, dgd)
		newUID := pcs.UID
		objects = materializeLPXTestPCS(t, child, pcs, selected.plan)
		pcs.UID = newUID
		for _, obj := range objects {
			refs := obj.GetOwnerReferences()
			for i := range refs {
				if refs[i].Kind == "PodCliqueSet" {
					refs[i].UID = newUID
				}
			}
			obj.SetOwnerReferences(refs)
			if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok {
				require.NoError(t, r.Update(ctx, obj))
			} else {
				require.NoError(t, r.Create(ctx, obj))
			}
		}
		for range 2 {
			_, err = r.Reconcile(ctx, request)
			require.NoError(t, err)
		}
		newRequests, err = r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Len(t, newRequests, len(selected.requests))
		for _, next := range newRequests {
			require.True(t, metav1.IsControlledBy(next, pcs))
			for _, old := range oldRequests {
				require.NotEqual(t, old.UID, next.UID)
			}
		}
		if buildID == "units-wide" {
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(originalMap), &corev1.ConfigMap{}), "old config remains until replacement readiness")
		}

		t.Log("Acknowledge scheduler and Grove readiness, then collect obsolete configuration")
		for _, obj := range objects {
			switch live := obj.(type) {
			case *grovev1alpha1.PodClique:
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(live), live))
				live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
				live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
				require.NoError(t, r.Update(ctx, live))
			case *grovev1alpha1.PodCliqueScalingGroup:
				require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(live), live))
				live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
				live.Status.AvailableReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
				require.NoError(t, r.Update(ctx, live))
			}
		}
		for _, next := range newRequests {
			next.Status = newTestPipelineRequest(child, pcs, next.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
			require.NoError(t, r.Update(ctx, next))
		}
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
		require.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(child.Status.Conditions, "Ready").Status)
		maps = &corev1.ConfigMapList{}
		require.NoError(t, r.List(ctx, maps))
		if buildID == "units-wide" {
			require.Empty(t, maps.Items)
		} else {
			require.Len(t, maps.Items, 1)
			require.Equal(t, originalMap.Data, maps.Items[0].Data)
		}

		t.Log("Reconcile the settled generation again without resource or status changes")
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
		beforePCS := pcs.DeepCopy()
		beforeChild := child.DeepCopy()
		beforeRequests, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
		require.Equal(t, beforePCS, pcs)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
		require.Equal(t, beforeChild, child)
		afterRequests, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Equal(t, beforeRequests, afterRequests)
	}
	require.Equal(t, 2, deletes)
	require.Equal(t, 2, creates)
}

// TestUnitsOperatorOnlyUpgrade uses the actual old reader's render of the same
// artifact and DGD. See testdata/units-old-operator.md for capture provenance.
func TestUnitsOperatorOnlyUpgrade(t *testing.T) {
	t.Log("Load the old operator output with its exact units-bearing artifact and unchanged DGD")
	data, err := os.ReadFile("testdata/units-old-operator.json")
	require.NoError(t, err)
	var old struct {
		Base       string
		Deployment []byte
		DGD        v1beta1.DynamoGraphDeployment
		PCS        grovev1alpha1.PodCliqueSet
		Requests   []lpxv1alpha1.LPUPipelineRequest
		Resources  []corev1.ConfigMap
	}
	require.NoError(t, json.Unmarshal(data, &old))
	require.Equal(t, "20a9321cf16eb11e95fd12194c3033c210f7b2be", old.Base)
	root := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(root, "units"), 0700))
	require.NoError(t, os.WriteFile(filepath.Join(root, "units", "deployment.v1.capnp.bin"), old.Deployment, 0600))
	registry, err := lpx.NewModelRegistry((&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String(), nil)
	require.NoError(t, err)
	ctx := t.Context()
	dgd := &old.DGD
	child := newLPXTestDeployment(t, dgd)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	beforeDGD := dgd.DeepCopy()
	beforeRevision := child.Spec.InputRevision
	require.NoError(t, r.stampClientIdentity(ctx, dgd, &old.PCS))
	objects := materializeLPXTestPCS(t, child, &old.PCS, desired.plan)
	createLPXTestObjects(t, ctx, r.Client, objects...)
	for i := range old.Resources {
		old.Resources[i].OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(child, child.GroupVersionKind())}
		require.NoError(t, r.Create(ctx, &old.Resources[i]))
	}
	for i := range old.Requests {
		old.Requests[i].OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(&old.PCS, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}
		require.NoError(t, r.Create(ctx, &old.Requests[i]))
	}
	pcs := &old.PCS
	oldUID := pcs.UID
	initialRequests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	expectedPCS := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)
	require.True(t, podCliqueSetLayoutMatches(pcs, expectedPCS))
	require.NotEqual(t, pcs.Spec, expectedPCS.Spec)
	request := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}

	t.Log("Activate units without editing the DGD or build and wait for Grove to observe the new template")
	for range 2 {
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
		require.Equal(t, oldUID, pcs.UID)
		require.True(t, pcs.DeletionTimestamp.IsZero())
		retained, err := r.getPipelineRequests(ctx, pcs)
		require.NoError(t, err)
		require.Equal(t, initialRequests, retained, "old Grove roles cannot retire immutable requests")
		for _, config := range old.Resources {
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(&config), &corev1.ConfigMap{}))
		}
	}
	require.Equal(t, expectedPCS.Spec.Template.PodCliqueScalingGroupConfigs, pcs.Spec.Template.PodCliqueScalingGroupConfigs)

	t.Log("Observe replacement role templates before retiring and replacing the immutable requests")
	fresh := materializeLPXTestPCS(t, child, expectedPCS, desired.plan)
	for _, obj := range fresh {
		clique, ok := obj.(*grovev1alpha1.PodClique)
		if !ok {
			continue
		}
		require.NoError(t, r.Delete(ctx, clique))
		clique.UID += "-units"
		require.NoError(t, r.Create(ctx, clique))
	}
	for range 3 {
		_, err = r.Reconcile(ctx, request)
		require.NoError(t, err)
	}
	nextRequests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	require.Len(t, nextRequests, len(desired.requests))
	for name, next := range nextRequests {
		require.NotEqual(t, initialRequests[name].UID, next.UID)
		require.True(t, metav1.IsControlledBy(next, pcs))
	}
	for _, config := range old.Resources {
		require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(&config), &corev1.ConfigMap{}), "configuration remains until readiness")
	}

	t.Log("Confirm the new workload is ready before removing the old ConfigMap")
	for _, obj := range fresh {
		switch live := obj.(type) {
		case *grovev1alpha1.PodClique:
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(live), live))
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.ReadyReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
			require.NoError(t, r.Update(ctx, live))
		case *grovev1alpha1.PodCliqueScalingGroup:
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(live), live))
			live.Status.Replicas, live.Status.UpdatedReplicas = live.Spec.Replicas, live.Spec.Replicas
			live.Status.AvailableReplicas, live.Status.ScheduledReplicas = live.Spec.Replicas, live.Spec.Replicas
			require.NoError(t, r.Update(ctx, live))
		}
	}
	for _, next := range nextRequests {
		next.Status = newTestPipelineRequest(child, pcs, next.Name, time.Now(), lpxv1alpha1.RequestPhaseBound).Status
		require.NoError(t, r.Update(ctx, next))
	}
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, metav1.ConditionTrue, meta.FindStatusCondition(child.Status.Conditions, "Ready").Status)
	maps := &corev1.ConfigMapList{}
	require.NoError(t, r.List(ctx, maps))
	require.Empty(t, maps.Items)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(dgd), dgd))
	require.Equal(t, beforeDGD.Spec, dgd.Spec)
	require.Equal(t, beforeDGD.Generation, dgd.Generation)
	require.Equal(t, beforeRevision, child.Spec.InputRevision)

	t.Log("The settled second reconcile changes no PCS, request or status")
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
	settledPCS, settledChild := pcs.DeepCopy(), child.DeepCopy()
	settledRequests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	_, err = r.Reconcile(ctx, request)
	require.NoError(t, err)
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
	require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(child), child))
	require.Equal(t, settledPCS, pcs)
	require.Equal(t, settledChild, child)
	afterRequests, err := r.getPipelineRequests(ctx, pcs)
	require.NoError(t, err)
	require.Equal(t, settledRequests, afterRequests)
}

// TestUnitsMigrationShapes compares the same artifact and DGD through the actual
// base reader and the units reader. The captures keep both metadata forms.
func TestUnitsMigrationShapes(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                   string
		agents                 int32
		partitions, connectors int
		metadataRemoved        bool
	}{
		{"hx-unpacked", 2, 2, 1, true},
		{"hx-packed", 1, 1, 0, true},
		{"xt-hybrid-local", 4, 2, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			t.Log("Compare old and new placement for unchanged artifact bytes and authored workload")
			data, err := os.ReadFile("testdata/units-old-" + tc.name + ".json")
			require.NoError(t, err)
			var old struct {
				Base       string
				Deployment []byte
				DGD        v1beta1.DynamoGraphDeployment
				PCS        grovev1alpha1.PodCliqueSet
				Requests   []lpxv1alpha1.LPUPipelineRequest
			}
			require.NoError(t, json.Unmarshal(data, &old))
			require.Equal(t, "20a9321cf16eb11e95fd12194c3033c210f7b2be", old.Base)
			root := t.TempDir()
			require.NoError(t, os.Mkdir(filepath.Join(root, "units"), 0700))
			require.NoError(t, os.WriteFile(filepath.Join(root, "units", "deployment.v1.capnp.bin"), old.Deployment, 0600))
			registry, err := lpx.NewModelRegistry((&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String(), nil)
			require.NoError(t, err)
			dgd := old.DGD.DeepCopy()
			child := newLPXTestDeployment(t, dgd)
			r, selected := newPreparedLPXTestReconciler(t, registry, t.Context(), child, dgd)
			after := renderLPXTestPodCliqueSet(t, t.Context(), r, child, dgd, selected)
			require.Equal(t, old.DGD.Spec, dgd.Spec)
			require.True(t, podCliqueSetLayoutMatches(&old.PCS, after), "this same-build migration rolls the existing PCS")
			require.NotEqual(t, old.PCS.Spec, after.Spec, "runtime templates change even when the layout is compatible")
			for _, pcs := range []*grovev1alpha1.PodCliqueSet{&old.PCS, after} {
				found := false
				for _, clique := range pcs.Spec.Template.Cliques {
					if clique.Name == "agt" {
						require.NotNil(t, clique.Spec.MinAvailable)
						require.Equal(t, tc.agents, *clique.Spec.MinAvailable)
						found = true
					}
				}
				require.True(t, found)
			}
			require.Len(t, old.Requests, 1)
			require.Len(t, selected.requests, 1)
			before, now := old.Requests[0].Spec, selected.requests[0].Spec
			require.Len(t, now.Partitions, tc.partitions)
			require.Len(t, now.PropSyncConnectors, tc.connectors)
			require.Equal(t, before.Partitions, now.Partitions)
			require.Equal(t, before.PropSyncConnectors, now.PropSyncConnectors)
			require.Equal(t, before.NodeLocal, now.NodeLocal)
			require.Equal(t, tc.metadataRemoved, !slices.Equal(before.AllocationMetadata.Raw, now.AllocationMetadata.Raw))
			// The only request-spec change in these fixtures is removal of legacy HX
			// allocation metadata. All scheduling fields keep the same values.
			before.AllocationMetadata = now.AllocationMetadata
			require.Equal(t, before, now)
			require.NotEqual(t, old.Requests[0].Annotations, selected.requests[0].Annotations)
		})
	}
}
