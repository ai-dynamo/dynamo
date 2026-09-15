/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

func TestLPXDeploymentPrimaryPredicateTracksOwnerIdentity(t *testing.T) {
	t.Log("Build one current private handoff and its primary-resource predicate")
	deployment := &nvidiacomv1alpha1.LPXGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "graph", Namespace: "workloads", Generation: 1,
			OwnerReferences: []metav1.OwnerReference{{Name: "graph", UID: "source-uid", Controller: ptr.To(true)}},
		},
	}
	filter := lpxDeploymentPrimaryPredicate()

	t.Log("Ignore status-only updates that do not change publication authority")
	statusOnly := deployment.DeepCopy()
	statusOnly.Status.ObservedGeneration = 1
	require.False(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: statusOnly}))

	t.Log("Wake when the controller owner identity changes without advancing generation")
	ownerChanged := deployment.DeepCopy()
	ownerChanged.OwnerReferences[0].UID = "replacement-source-uid"
	require.True(t, filter.Update(event.UpdateEvent{ObjectOld: deployment, ObjectNew: ownerChanged}))
}

func TestLPXSourceWatchMapsControllerOwnedMaterializations(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	source := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "workloads", UID: "source-uid"}}
	owned := &nvidiacomv1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name: "materialization", Namespace: source.Namespace,
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, nvidiacomv1beta1.DynamoGraphDeploymentGVK)},
	}}
	unrelated := owned.DeepCopy()
	unrelated.Name = "unrelated"
	unrelated.OwnerReferences[0].UID = "other-source-uid"
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(owned, unrelated).
		WithIndex(&nvidiacomv1alpha1.LPXGraphDeployment{}, lpxSourceOwnerIndex, lpxSourceOwnerReferences).Build()
	reconciler := &graphReconciler{Client: kube}

	require.Equal(t, []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(owned)}}, reconciler.mapLPXSourceToRequests(t.Context(), source))
}

func TestLPXTopologyBindingWatchMapsMatchingPoliciesWithinScope(t *testing.T) {
	t.Log("Build matching, unrelated, and out-of-scope graph policies")
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1alpha1.AddToScheme(scheme))
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	matching := topologyPolicyDGD("matching", "workloads", "fabric")
	unrelated := topologyPolicyDGD("unrelated", "workloads", "other-fabric")
	foreign := topologyPolicyDGD("foreign", "other", "fabric")
	ordinary := topologyPolicyDGD("ordinary", "workloads", "fabric")
	for _, source := range []*nvidiacomv1beta1.DynamoGraphDeployment{matching, unrelated, foreign} {
		source.Spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "lpx", ComponentType: nvidiacomv1beta1.ComponentTypeLPX}}
	}

	t.Log("Give each source a differently named materialization and retain a stale same-name child")
	objects := []client.Object{matching, unrelated, foreign, ordinary}
	for _, source := range []*nvidiacomv1beta1.DynamoGraphDeployment{matching, unrelated, foreign, ordinary} {
		source.UID = types.UID(source.Name + "-uid")
		objects = append(objects, &nvidiacomv1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
			Name: "materialization-" + source.Name, Namespace: source.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, nvidiacomv1beta1.DynamoGraphDeploymentGVK)},
		}})
	}
	stale := matching.DeepCopy()
	stale.UID = "previous-source-uid"
	objects = append(objects, &nvidiacomv1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Name: matching.Name, Namespace: matching.Namespace,
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(stale, nvidiacomv1beta1.DynamoGraphDeploymentGVK)},
	}})

	t.Log("Index dependency references and exact controller ownership without scanning")
	kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).
		WithIndex(&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxTopologyBindingRefIndex, lpxTopologyBindingReferences).
		WithIndex(&nvidiacomv1alpha1.LPXGraphDeployment{}, lpxSourceOwnerIndex, lpxSourceOwnerReferences).
		WithInterceptorFuncs(interceptor.Funcs{List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			options := (&client.ListOptions{}).ApplyOptions(opts)
			require.NotNil(t, options.FieldSelector, "topology events must use an index, not scan DGDs")
			return delegated.List(ctx, list, opts...)
		}}).Build()
	reconciler := &graphReconciler{
		Client:        kube,
		runtimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{LPX: true}},
		Config: &configv1alpha1.OperatorConfiguration{
			Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "workloads"},
			LPX:       configv1alpha1.LPXConfiguration{Enabled: true},
		},
	}

	t.Log("Accept namespaced graph events only in the restricted namespace")
	filter := reconciler.lpxControllerEventFilter()
	require.True(t, filter.Create(event.CreateEvent{Object: matching}))
	require.False(t, filter.Create(event.CreateEvent{Object: foreign}))

	t.Log("Admit cluster-scoped topology bindings for namespace-aware mapping")
	binding := &grovev1alpha1.ClusterTopologyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "fabric"},
	}
	require.True(t, filter.Create(event.CreateEvent{Object: binding}))
	require.False(t, filter.Create(event.CreateEvent{Object: &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "foreign-cluster-object"},
	}}))

	t.Log("Requeue only the allowed materialization owned by the current matching source UID")
	requests := reconciler.indexedLPXDependencyRequests(context.Background(), binding, lpxTopologyBindingRefIndex)
	require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "materialization-matching"}}}, requests)

	t.Log("Ignore bindings no graph consumes")
	require.Empty(t, reconciler.indexedLPXDependencyRequests(context.Background(), &grovev1alpha1.ClusterTopologyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "unused"},
	}, lpxTopologyBindingRefIndex))

	t.Log("Editing a source policy removes the old dependency and indexes the new one")
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(matching), matching))
	matching.Spec.Experimental.KvTransferPolicy.ClusterTopologyName = "new-fabric"
	require.NoError(t, kube.Update(t.Context(), matching))
	require.Empty(t, reconciler.indexedLPXDependencyRequests(t.Context(), &grovev1alpha1.ClusterTopologyBinding{ObjectMeta: metav1.ObjectMeta{Name: "fabric"}}, lpxTopologyBindingRefIndex))
	require.Len(t, reconciler.indexedLPXDependencyRequests(t.Context(), &grovev1alpha1.ClusterTopologyBinding{ObjectMeta: metav1.ObjectMeta{Name: "new-fabric"}}, lpxTopologyBindingRefIndex), 1)
}

func topologyPolicyDGD(name, namespace, bindingName string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Experimental: &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
				KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: bindingName},
			},
		},
	}
}

func TestLPXPodGangPredicateTracksOnlyMaterializationIdentity(t *testing.T) {
	t.Log("Accept scheduler-scoped publication witnesses and map their exact namespaced graph")
	predicate := lpxPodGangPredicate()
	gang := &groveschedulerv1alpha1.PodGang{
		ObjectMeta: metav1.ObjectMeta{
			Name: "agents", Namespace: "workloads", UID: "uid", Generation: 1,
			Annotations: map[string]string{dynamolpx.DeploymentNameAnnotation: "materialization"},
			Labels: map[string]string{
				grovecommon.LabelSchedulerName:            dynamolpx.SchedulerName,
				consts.KubeLabelDynamoGraphDeploymentName: "graph",
			},
		},
	}
	require.True(t, predicate.Create(event.CreateEvent{Object: gang}))
	require.True(t, predicate.Delete(event.DeleteEvent{Object: gang}))
	require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "materialization"}}},
		mapLPXChildToRequests(t.Context(), gang))

	t.Log("Ignore status-only scheduler updates")
	statusOnly := gang.DeepCopy()
	statusOnly.Status.Phase = groveschedulerv1alpha1.PodGangPhase("scheduled")
	require.False(t, predicate.Update(event.UpdateEvent{ObjectOld: gang, ObjectNew: statusOnly}))

	t.Log("Wake on PodGang spec and mirrored workload-annotation convergence")
	specChange := gang.DeepCopy()
	specChange.Generation++
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: gang, ObjectNew: specChange}))
	annotationChange := gang.DeepCopy()
	annotationChange.Annotations = map[string]string{dynamolpx.WorkloadDigestAnnotation: "sha256:updated"}
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: gang, ObjectNew: annotationChange}))

	t.Log("Accept the ordinary scheduler witness and reject unrelated PodGangs")
	ordinary := gang.DeepCopy()
	ordinary.Labels[grovecommon.LabelSchedulerName] = corev1.DefaultSchedulerName
	require.True(t, predicate.Create(event.CreateEvent{Object: ordinary}))
	unrelated := ordinary.DeepCopy()
	delete(unrelated.Annotations, dynamolpx.DeploymentNameAnnotation)
	require.False(t, predicate.Create(event.CreateEvent{Object: unrelated}))
	require.Empty(t, mapLPXChildToRequests(t.Context(), unrelated))
}

func TestLPXWorkloadEventPredicates(t *testing.T) {
	for _, test := range []struct {
		name      string
		object    client.Object
		predicate predicate.Predicate
		observe   func(client.Object)
	}{
		{"clique", &grovev1alpha1.PodClique{}, lpxPodCliqueEventPredicates(), func(o client.Object) { o.(*grovev1alpha1.PodClique).Status.ReadyReplicas++ }},
		{"scaling group", &grovev1alpha1.PodCliqueScalingGroup{}, lpxScalingGroupEventPredicates(), func(o client.Object) {
			o.(*grovev1alpha1.PodCliqueScalingGroup).Status.ObservedGeneration = ptr.To(int64(2))
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe creation, deletion and all metadata used to fence materialization")
			old := test.object
			old.SetName("grove-child")
			old.SetNamespace("workloads")
			old.SetAnnotations(map[string]string{
				dynamolpx.DeploymentNameAnnotation: "materialization",
				dynamolpx.WorkloadDigestAnnotation: "sha256:workload",
				lpxv1alpha1.PodRoleAnnotation:      lpxv1alpha1.PodRoleAgent,
			})
			require.True(t, test.predicate.Create(event.CreateEvent{Object: old}))
			require.True(t, test.predicate.Delete(event.DeleteEvent{Object: old}))
			require.False(t, test.predicate.Generic(event.GenericEvent{Object: old}))
			require.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "workloads", Name: "materialization"}}},
				mapLPXChildToRequests(t.Context(), old), "Grove children need no source DGD label for routing")

			t.Log("Ordinary readiness traffic does not wake the LPX controller")
			unrelated := old.DeepCopyObject().(client.Object)
			unrelated.SetAnnotations(nil)
			require.False(t, test.predicate.Create(event.CreateEvent{Object: unrelated}))
			observed := unrelated.DeepCopyObject().(client.Object)
			test.observe(observed)
			require.False(t, test.predicate.Update(event.UpdateEvent{ObjectOld: unrelated, ObjectNew: observed}))

			t.Log("All compiled roles and PodClique template-hash convergence remain observable")
			if clique, ok := old.(*grovev1alpha1.PodClique); ok {
				for _, role := range []string{lpxv1alpha1.PodRoleAgent, lpxv1alpha1.PodRoleConductor, lpxv1alpha1.PodRoleCyborgWorker} {
					current := clique.DeepCopy()
					current.Annotations[lpxv1alpha1.PodRoleAnnotation] = role
					require.True(t, test.predicate.Create(event.CreateEvent{Object: current}))
				}
				current := clique.DeepCopy()
				current.Status.CurrentPodTemplateHash = ptr.To("current-template")
				require.True(t, test.predicate.Update(event.UpdateEvent{ObjectOld: clique, ObjectNew: current}))
			}

			t.Log("Wake on role readiness, independently observed group generation and materialization fences")
			for _, change := range []struct {
				name   string
				mutate func(client.Object)
				want   bool
			}{
				{"unchanged", func(client.Object) {}, false},
				{"status noise", func(o client.Object) { o.SetResourceVersion("2") }, false},
				{"status observed", test.observe, true},
				{"stamp lost", func(o client.Object) { o.SetAnnotations(nil) }, true},
				{"workload changed", func(o client.Object) { o.GetAnnotations()[dynamolpx.WorkloadDigestAnnotation] = "sha256:updated" }, true},
				{"labels changed", func(o client.Object) { o.SetLabels(map[string]string{"changed": "true"}) }, true},
				{"owner changed", func(o client.Object) { o.SetOwnerReferences([]metav1.OwnerReference{{UID: "new-owner"}}) }, true},
				{"deleting", func(o client.Object) { now := metav1.Now(); o.SetDeletionTimestamp(&now) }, true},
				{"spec generation changed", func(o client.Object) { o.SetGeneration(o.GetGeneration() + 1) }, true},
			} {
				t.Run(change.name, func(t *testing.T) {
					updated := old.DeepCopyObject().(client.Object)
					change.mutate(updated)
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: old, ObjectNew: updated}))
					require.Equal(t, change.want, test.predicate.Update(event.UpdateEvent{ObjectOld: updated, ObjectNew: old}))
				})
			}
		})
	}
}

func TestLPXSourceWatchIgnoresOrdinaryChurn(t *testing.T) {
	t.Log("Ordinary capacity and DGD status edits must not rerun LPX compilation")
	source := newLPXTestSource(dynamolpx.PipelineSingle, "build-v2")
	source.Spec.Components = append(source.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "prefill", Replicas: ptr.To(int32(1))})
	changed := source.DeepCopy()
	changed.Generation++
	changed.Spec.Components[1].Replicas = ptr.To(int32(4))
	changed.Status.State = nvidiacomv1beta1.DGDStatePending
	predicate := lpxSourcePredicate()
	require.False(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))

	t.Log("Ignore ordinary creation and deletion, but keep LPX creation, deletion and deselection")
	ordinary := source.DeepCopy()
	ordinary.Spec.Components = ordinary.Spec.Components[1:]
	ordinary.Generation++
	require.False(t, predicate.Create(event.CreateEvent{Object: ordinary}))
	require.False(t, predicate.Delete(event.DeleteEvent{Object: ordinary}))
	require.True(t, predicate.Create(event.CreateEvent{Object: source}))
	require.True(t, predicate.Delete(event.DeleteEvent{Object: source}))
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: ordinary}))
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: ordinary, ObjectNew: source}))

	t.Log("LPX input changes and persisted restart selection wake the child")
	changed.Spec.Components[0].ComponentRole(nvidiacomv1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0].Image = "new-runtime"
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.Spec.Restart = &nvidiacomv1beta1.Restart{ID: "restart"}
	require.False(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed.Status.Restart = &nvidiacomv1beta1.RestartStatus{ObservedID: "restart", Phase: nvidiacomv1beta1.RestartPhaseRestarting, InProgress: []string{"lpx"}}
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.UID = "replacement"
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))

	t.Log("Inherited scheduler metadata still wakes the child without a new source generation")
	changed = source.DeepCopy()
	changed.Labels = map[string]string{"priorityClassName": "inference"}
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
	changed = source.DeepCopy()
	changed.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	require.True(t, predicate.Update(event.UpdateEvent{ObjectOld: source, ObjectNew: changed}))
}

func BenchmarkLPXSourceWatchStatusChurn(b *testing.B) {
	source := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "inference", Namespace: "workloads", Generation: 1},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "lpx", ComponentType: nvidiacomv1beta1.ComponentTypeLPX,
				LPX: &nvidiacomv1beta1.LPXConfig{BuildID: "model/build"},
				Roles: []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLPXConductor}, {
					Name: nvidiacomv1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "lpu-runtime:test"}}},
					},
				}},
			}},
		},
	}
	updated := source.DeepCopy()
	updated.Status.State = nvidiacomv1beta1.DGDStatePending
	update := event.UpdateEvent{ObjectOld: source, ObjectNew: updated}
	filter := lpxSourcePredicate()
	b.ReportAllocs()
	for b.Loop() {
		if filter.Update(update) {
			b.Fatal("status-only updates must not enqueue reconciliation")
		}
	}
}

func TestLPXDRAWatchMapsIndexedConsumers(t *testing.T) {
	t.Log("Index consumed regular and init-container references, deduplicating shared aliases")
	source := newLPXTestSource(dynamolpx.PipelineLPX, "build-v2")
	child := newLPXTestDeployment(t, source)
	child.Name = "materialization"
	pod := &dynamolpx.ServingComponent(source).ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
	pod.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "direct"}, {Name: "template"}}
	pod.Containers = append(pod.Containers, corev1.Container{Name: "sidecar", Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "direct"}}}})
	pod.InitContainers = []corev1.Container{
		{Name: "prepare", Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "init"}}}},
		{Name: "native-sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways), Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "init-template"}}}},
	}
	pod.ResourceClaims = []corev1.PodResourceClaim{
		{Name: "direct", ResourceClaimName: ptr.To("shared")},
		{Name: "template", ResourceClaimTemplateName: ptr.To("shared")},
		{Name: "init", ResourceClaimName: ptr.To("init-claim")},
		{Name: "init-template", ResourceClaimTemplateName: ptr.To("init-template")},
		{Name: "unused", ResourceClaimName: ptr.To("unused-claim"), ResourceClaimTemplateName: ptr.To("unused-template")},
	}
	require.ElementsMatch(t, []string{"shared", "init-claim"}, lpxDRAClaimReferences(false)(source))
	require.ElementsMatch(t, []string{"shared", "init-template"}, lpxDRAClaimReferences(true)(source))

	t.Log("Reference same-named claims and templates from two namespaces, excluding unused aliases")
	foreign := source.DeepCopy()
	foreign.Name, foreign.Namespace = "foreign", "other-namespace"
	foreign.UID = "foreign-source-uid"
	unused := source.DeepCopy()
	unused.Name = "unused"
	unused.UID = "unused-source-uid"
	unusedPod := &dynamolpx.ServingComponent(unused).ComponentRole(nvidiacomv1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
	unusedPod.Containers = unusedPod.Containers[:1]
	unusedPod.Containers[0].Resources.Claims, unusedPod.InitContainers = nil, nil
	claim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "shared", Namespace: source.Namespace},
		Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
			Name: "gpu", Exactly: &resourcev1.ExactDeviceRequest{DeviceClassName: "gpu"},
		}}}},
	}
	claimTemplate := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "shared", Namespace: source.Namespace},
		Spec: resourcev1.ResourceClaimTemplateSpec{Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
			Name: "gpu", FirstAvailable: []resourcev1.DeviceSubRequest{{Name: "first", DeviceClassName: "gpu"}, {Name: "second", DeviceClassName: "alternative-gpu"}},
		}}}}},
	}
	foreignTemplate := claimTemplate.DeepCopy()
	foreignTemplate.Namespace = foreign.Namespace

	t.Log("Route dependencies to current materializations, not same-name children from old source UIDs")
	foreignChild := newLPXTestDeployment(t, foreign)
	foreignChild.Name = "foreign-materialization"
	unusedChild := newLPXTestDeployment(t, unused)
	unusedChild.Name = "unused-materialization"
	staleChild := child.DeepCopy()
	staleChild.Name, staleChild.UID = source.Name, "stale-materialization-uid"
	staleChild.OwnerReferences[0].UID = "previous-source-uid"
	r := newLPXTestReconciler(t, nil, child, source, foreign, unused, foreignChild, unusedChild, staleChild, claim, claimTemplate, foreignTemplate)
	r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
		options := (&client.ListOptions{}).ApplyOptions(opts)
		require.NotNil(t, options.FieldSelector, "DRA events must use indexes at both hops")
		require.False(t, options.FieldSelector.Empty())
		return delegated.List(ctx, list, opts...)
	}})
	childRequest := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
	foreignRequest := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(foreignChild)}

	t.Log("Claim events remain namespace-local and unused aliases never enqueue a graph")
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), claim))
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), claimTemplate))
	require.Equal(t, []ctrl.Request{foreignRequest}, r.mapLPXDRADependencyToRequests(t.Context(), foreignTemplate))
	require.Empty(t, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.ResourceClaimTemplate{ObjectMeta: metav1.ObjectMeta{Namespace: source.Namespace, Name: "unrelated"}}))
	require.Empty(t, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "unrelated"}}))

	t.Log("Class events join exact and alternative references and enqueue each graph once")
	for _, class := range []string{"gpu", "alternative-gpu"} {
		require.ElementsMatch(t, []ctrl.Request{childRequest, foreignRequest}, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: class}}))
	}
	r.Config.Namespace.Restricted = "unwatched-namespace"
	require.Empty(t, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu"}}))
	r.Config.Namespace.Restricted = source.Namespace
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu"}}))

	t.Log("Editing only a direct claim updates its DeviceClass dependency")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(claim), claim))
	claim.Spec.Devices.Requests[0].Exactly.DeviceClassName = "replacement-claim-gpu"
	require.NoError(t, r.Update(t.Context(), claim))
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "replacement-claim-gpu"}}))
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu"}}))

	t.Log("A template-only edit removes its old class references without a DGD edit")
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(claimTemplate), claimTemplate))
	claimTemplate.Spec.Spec.Devices.Requests[0].FirstAvailable = []resourcev1.DeviceSubRequest{{Name: "replacement", DeviceClassName: "replacement-template-gpu"}}
	require.NoError(t, r.Update(t.Context(), claimTemplate))
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "replacement-template-gpu"}}))
	for _, class := range []string{"gpu", "alternative-gpu"} {
		require.Empty(t, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: class}}))
	}

	t.Log("Missing templates remain indexed, but no longer contribute a DeviceClass dependency")
	require.NoError(t, r.Delete(t.Context(), claimTemplate))
	require.Equal(t, []ctrl.Request{childRequest}, r.mapLPXDRADependencyToRequests(t.Context(), claimTemplate))
	require.Empty(t, r.mapLPXDRADependencyToRequests(t.Context(), &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "replacement-template-gpu"}}))

	t.Log("Dependency events never mutate the source or its revision; ordinary components are not indexed")
	stored := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(source), stored))
	require.Equal(t, source.Spec, stored.Spec)
	require.Equal(t, source.Generation, stored.Generation)
	source.Spec.Components[0].ComponentType = nvidiacomv1beta1.ComponentTypeDecode
	require.Empty(t, lpxDRAClaimReferences(false)(source))
	require.Empty(t, lpxDRAClaimReferences(true)(source))
}
