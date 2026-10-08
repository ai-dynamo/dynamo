/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"maps"
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	enginegroupcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/enginegroup"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveEngineGroupCreationCacheLag(t *testing.T) {
	t.Log("seed a colliding child that the informer has not observed yet")
	dgd := testSGLangEngineGroupDGD()
	pcs := &grove.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: dgd.Namespace, UID: "pcs-uid"}}
	group := &api.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Name: dynamo.EngineGroupNameForComponent(dgd.Name, "Worker", 0), Namespace: dgd.Namespace},
		Spec:       api.DynamoGraphDeploymentEngineGroupSpec{Replicas: 3},
	}
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(group).Build()
	hideChild := true
	lagged := interceptor.NewClient(kube, interceptor.Funcs{
		Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
			if _, isGroup := object.(*api.DynamoGraphDeploymentEngineGroup); isGroup && hideChild {
				return apierrors.NewNotFound(schema.GroupResource{Group: api.GroupVersion.Group, Resource: "dynamographdeploymentenginegroups"}, key.Name)
			}
			return c.Get(ctx, key, object, opts...)
		},
	})
	reconciler := &enginegroupcontroller.GroveChildrenReconciler{Client: lagged}

	t.Log("AlreadyExists remains pending without a readback, adoption or resetting its target")
	_, ready, err := reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)
	observed := &api.DynamoGraphDeploymentEngineGroup{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), observed))
	assert.Equal(t, int32(3), observed.Spec.Replicas)
	assert.Empty(t, observed.OwnerReferences)

	t.Log("once the informer catches up, refuse the conflicting child rather than adopting it")
	hideChild = false
	_, _, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.ErrorContains(t, err, "conflicting ownership")
}

func TestGroveEngineGroupsLifecycle(t *testing.T) {
	t.Log("declare one world with two initial allocations and create its Grove parent")
	dgd := testSGLangEngineGroupDGD()
	pcs := &grove.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: dgd.Namespace, UID: "pcs-uid"}}
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dgd, pcs).
		WithStatusSubresource(&api.DynamoGraphDeploymentEngineGroup{}).Build()
	reconciler := &enginegroupcontroller.GroveChildrenReconciler{Client: kube}

	t.Log("create the child once with a seed, declared policy and owner, leaving finalization to its controller")
	statuses, ready, err := reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)
	assert.Equal(t, int32(1), statuses["Worker"].Replicas)
	group := &api.DynamoGraphDeploymentEngineGroup{}
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.EngineGroupNameForComponent(dgd.Name, "Worker", 0)}
	require.NoError(t, kube.Get(t.Context(), key, group))
	assert.True(t, metav1.IsControlledBy(group, dgd))
	assert.Equal(t, "0", group.Labels[consts.KubeLabelDynamoEngineGroupWorldIndex])
	assert.Empty(t, group.Finalizers)
	assert.Equal(t, int32(2), group.Spec.Replicas)
	assert.Equal(t, int32(3), *group.Spec.Policy.MaxReplicas)
	assert.Equal(t, "http://graph-frontend.default.svc:8000/v1/completions", group.Annotations[consts.KubeAnnotationDynamoEngineGroupVerifyURL])

	t.Log("wait without adoption when Grove has not published the member clique")
	_, ready, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)

	t.Log("bind the one member clique after verifying its PCSG and PCS ownership chain")
	world := &grove.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker", Namespace: dgd.Namespace, UID: "world-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))}}}
	clique := &grove.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker-0-worker", Namespace: dgd.Namespace, UID: "clique-uid", Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name, grovecommon.LabelPodCliqueScalingGroupReplicaIndex: "0"}, OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}}}
	require.NoError(t, kube.Create(t.Context(), world))
	require.NoError(t, kube.Create(t.Context(), clique))
	_, ready, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)
	require.NoError(t, kube.Get(t.Context(), key, group))
	assert.Equal(t, string(clique.UID), group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])

	t.Log("simulate live scale and policy ownership moving to the child controller")
	group.Spec.Replicas = 3
	group.Spec.Policy.MinReplicas = ptr.To(int32(3))
	group.Generation = 2
	require.NoError(t, kube.Update(t.Context(), group))
	group.Status.Replicas = 3
	group.Status.Profile = &api.EngineGroupProfileStatus{GPUsPerReplica: 1}
	group.Status.Conditions = []metav1.Condition{
		{Type: "Available", Status: metav1.ConditionTrue, ObservedGeneration: 2},
		{Type: "TopologyKnown", Status: metav1.ConditionTrue, ObservedGeneration: 2},
	}
	require.NoError(t, kube.Status().Update(t.Context(), group))

	t.Log("a subsequent parent reconcile preserves both live fields and summarizes one verified world")
	statuses, ready, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.True(t, ready)
	assert.Equal(t, int32(1), *statuses["Worker"].ReadyReplicas)
	assert.Equal(t, int64(3), *statuses["Worker"].GPUsPerReplica)
	require.NoError(t, kube.Get(t.Context(), key, group))
	assert.Equal(t, int32(3), group.Spec.Replicas)
	assert.Equal(t, int32(3), *group.Spec.Policy.MinReplicas)

	t.Log("stale engine health cannot make the parent Ready for a new target generation")
	group.Generation = 3
	require.NoError(t, kube.Update(t.Context(), group))
	statuses, ready, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)
	assert.Equal(t, int32(0), *statuses["Worker"].ReadyReplicas)

	t.Log("leave a recreated clique pending for the child's restart handoff without overwriting its old binding")
	require.NoError(t, kube.Delete(t.Context(), clique))
	clique.ResourceVersion = ""
	clique.UID = types.UID("replacement-clique-uid")
	require.NoError(t, kube.Create(t.Context(), clique))
	_, ready, err = reconciler.Reconcile(t.Context(), dgd, pcs)
	require.NoError(t, err)
	assert.False(t, ready)
	require.NoError(t, kube.Get(t.Context(), key, group))
	assert.Equal(t, "clique-uid", group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	assert.Equal(t, int32(3), group.Spec.Replicas)
}

func TestGroveWorkloadsEngineGroupCapacityHandoff(t *testing.T) {
	t.Log("wire the actual Grove rendering, synchronization, scaling and child lifecycle path")
	dgd := testSGLangEngineGroupDGD()
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithRESTMapper(groveScaleRESTMapper()).
		WithObjects(dgd).WithStatusSubresource(dgd, &api.DynamoGraphDeploymentEngineGroup{}).
		WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client: kube, Config: &config.OperatorConfiguration{}, RuntimeConfig: &common.RuntimeConfig{}, Recorder: events.NewFakeRecorder(20),
		DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
	}
	workloads := reconciler.newGroveProgram().workloads

	t.Log("one DGD creates both its PCS and its child Engine Group without a manually authored resource")
	result, err := workloads.Reconcile(t.Context(), dgd, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, api.DGDStatePending, result.State)
	pcs := &grove.PodCliqueSet{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)}, pcs))
	assert.True(t, metav1.IsControlledBy(pcs, dgd))
	group := &api.DynamoGraphDeploymentEngineGroup{}
	groupKey := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.EngineGroupNameForComponent(dgd.Name, "Worker", 0)}
	require.NoError(t, kube.Get(t.Context(), groupKey, group))
	workerTemplate := podCliqueSetCliqueForComponent(pcs, "Worker")
	require.NotNil(t, workerTemplate)
	require.Equal(t, int32(2), workerTemplate.Spec.Replicas)

	t.Log("simulate Grove creating the independent world and its capacity clique, then scale that clique to three")
	world := &grove.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker", Namespace: dgd.Namespace, UID: "world-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))}}, Spec: grove.PodCliqueScalingGroupSpec{Replicas: 1}}
	clique := &grove.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker-0-worker", Namespace: dgd.Namespace, UID: "clique-uid", Labels: maps.Clone(workerTemplate.Labels), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}}, Spec: *workerTemplate.Spec.DeepCopy()}
	clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = "0"
	clique.Spec.Replicas = 3
	require.NoError(t, kube.Create(t.Context(), world))
	require.NoError(t, kube.Create(t.Context(), clique))
	group.Spec.Replicas = 3
	require.NoError(t, kube.Update(t.Context(), group))
	world.Spec.Replicas = 2
	require.NoError(t, kube.Update(t.Context(), world))

	t.Log("a parent reconcile binds the existing clique without resetting either live capacity target")
	_, err = workloads.Reconcile(t.Context(), dgd, nil, nil)
	require.NoError(t, err)
	require.NoError(t, kube.Get(t.Context(), groupKey, group))
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(clique), clique))
	assert.Equal(t, int32(3), group.Spec.Replicas)
	assert.Equal(t, int32(3), clique.Spec.Replicas)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(world), world))
	assert.Equal(t, int32(1), world.Spec.Replicas, "the DGD retains ownership of independent world count")
	assert.Equal(t, string(clique.UID), group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID])
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	assert.Equal(t, int32(2), podCliqueSetCliqueForComponent(pcs, "Worker").Spec.Replicas)

	t.Log("a frontend-only replica change does not mutate or recreate the live world")
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	_, err = workloads.Reconcile(t.Context(), dgd, nil, nil)
	require.NoError(t, err)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(clique), clique))
	assert.Equal(t, int32(3), clique.Spec.Replicas)

	t.Log("a changed engine image is blocked before it can rewrite a live world's template")
	dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Image = "runtime:2.0.0"
	_, err = workloads.Reconcile(t.Context(), dgd, nil, nil)
	require.ErrorContains(t, err, "coordinated world rollout")
}

func TestDGDEngineGroupDeletionIsDeferred(t *testing.T) {
	t.Log("give an owned world a controller finalizer while leaving another graph's world untouched")
	dgd := testSGLangEngineGroupDGD()
	group := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
		Name: "world", Namespace: dgd.Namespace, UID: "owned-world-uid", Finalizers: []string{"test-retirement"},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, api.GroupVersion.WithKind("DynamoGraphDeployment"))},
	}}
	unrelated := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: dgd.Namespace, UID: "unrelated-uid"}}
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(group, unrelated).Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kube}

	t.Log("request child deletion and keep the parent until engine retirement finishes")
	require.ErrorIs(t, reconciler.FinalizeResource(t.Context(), dgd), errEngineGroupRetirementPending)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), group))
	require.NotNil(t, group.DeletionTimestamp)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(unrelated), unrelated))
	assert.Nil(t, unrelated.DeletionTimestamp)

	t.Log("repeated parent reconciliation waits without rewriting the child's scale target")
	require.ErrorIs(t, reconciler.FinalizeResource(t.Context(), dgd), errEngineGroupRetirementPending)

	t.Log("when the child completes, the owned-world handoff no longer blocks the parent")
	group.Finalizers = nil
	require.NoError(t, kube.Update(t.Context(), group))
	pending, err := (&enginegroupcontroller.GroveChildrenReconciler{Client: kube}).RequestDeletion(t.Context(), dgd)
	require.NoError(t, err)
	assert.False(t, pending)
}

func testSGLangEngineGroupDGD() *api.DynamoGraphDeployment {
	// Declare the launch template locally rather than borrowing the runtime provider's fixtures.
	main := corev1.Container{
		Name:    consts.MainContainerName,
		Command: []string{"python3", "-m", dynamo.SGLangElasticEPBootstrapModule},
		Args: []string{
			"--model-path", "model", "--served-model-name", "model",
			"--tp", "2", "--dp", "2", "--nnodes", "2",
			"--enable-dp-attention", "--enable-dp-lm-head",
			"--moe-a2a-backend", "nixl", "--elastic-ep-backend", "mooncake",
			"--load-balance-method", "round_robin",
			"--elastic-ep-initial-size", "2", "--max-ep-size", "3",
			"--disable-cuda-graph", "--dist-init-addr", "rendezvous.test:24555",
		},
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
	}

	return &api.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid"},
		Spec: api.DynamoGraphDeploymentSpec{BackendFramework: "sglang", Components: []api.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "Frontend", ComponentType: api.ComponentTypeFrontend, Replicas: ptr.To(int32(1)), PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:1.1.0"}}}}},
			{ComponentName: "Worker", ComponentType: api.ComponentTypeWorker, Replicas: ptr.To(int32(1)), EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2, Policy: &api.ComponentEngineGroupPolicy{MinSize: ptr.To(int32(2)), MaxSize: ptr.To(int32(3))}}, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{main}}}},
		}},
	}
}
