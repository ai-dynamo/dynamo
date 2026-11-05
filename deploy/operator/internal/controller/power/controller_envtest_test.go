//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"testing"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	waitTimeout  = 10 * time.Second
	waitInterval = 100 * time.Millisecond
)

// startPowerController runs only this controller, so no DGD reconcile can contribute to convergence.
func startPowerController(t *testing.T, env *operatorenv.TestEnv, restricted string) {
	t.Helper()
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = restricted
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
	env.StartManager(func(mgr ctrl.Manager) error {
		if err := mgr.GetFieldIndexer().IndexField(context.Background(), &corev1.Pod{}, dynamo.ComponentPodIndex, dynamo.ComponentPodIndexValues); err != nil {
			return err
		}
		return Setup(mgr, config, runtimeConfig)
	})
}

// testPod is a worker Pod of the DGD's worker component created from a retained 300 W template.
func testPod(dgd *v1beta1.DynamoGraphDeployment, name string, owner client.Object, gvk schema.GroupVersionKind) *corev1.Pod {
	pod := &corev1.Pod{Name: name, Namespace: dgd.Namespace, Labels: map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoSelector: "worker",
	}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}}}
	if owner != nil {
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, gvk)}
	}
	return pod
}

// podCapIs reads a Pod's cap. Polling helpers report a failed read as false instead of failing
// the test from the polling goroutine, which can outlive the test's context.
func podCapIs(c client.Client, pod *corev1.Pod, want string) bool {
	current := &corev1.Pod{}
	return c.Get(context.Background(), client.ObjectKeyFromObject(pod), current) == nil && current.Annotations[consts.KubeAnnotationGPUPowerLimit] == want
}

// conditionReason reads the DGD's power condition reason; present is false when it is absent.
func conditionReason(c client.Client, dgd *v1beta1.DynamoGraphDeployment) (reason string, present bool, err error) {
	current := &v1beta1.DynamoGraphDeployment{}
	if err := c.Get(context.Background(), client.ObjectKeyFromObject(dgd), current); err != nil {
		return "", false, err
	}
	condition := meta.FindStatusCondition(current.Status.Conditions, ConditionType)
	if condition == nil {
		return "", false, nil
	}
	return condition.Reason, true, nil
}

func TestControllerRepairsReplacementAndAdoptedPods(t *testing.T) {
	t.Log("Start only the power controller and store a component graph edited to 350 W")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	startPowerController(t, env, env.Namespace())
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	require.NoError(t, c.Create(t.Context(), dgd))
	rs := &appsv1.ReplicaSet{Name: "worker-rs", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}, Spec: appsv1.ReplicaSetSpec{
		Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"worker": "power"}},
		Template: corev1.PodTemplateSpec{Labels: map[string]string{"worker": "power"}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}}},
	}}
	require.NoError(t, c.Create(t.Context(), rs))

	t.Log("A worker Pod created from the retained 300 W template is repaired by its create event")
	retained := testPod(dgd, "retained", rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
	require.NoError(t, c.Create(t.Context(), retained))
	require.Eventually(t, func() bool { return podCapIs(c, retained, "350") }, waitTimeout, waitInterval)
	require.Eventually(t, func() bool {
		reason, _, err := conditionReason(c, dgd)
		return err == nil && reason == "PodAnnotationsMatch"
	}, waitTimeout, waitInterval)

	t.Log("A later replacement Pod is repaired without any DGD change")
	replacement := testPod(dgd, "replacement", rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
	require.NoError(t, c.Create(t.Context(), replacement))
	require.Eventually(t, func() bool { return podCapIs(c, replacement, "350") }, waitTimeout, waitInterval)

	t.Log("An orphan worker Pod is skipped until adoption changes only its owner references")
	orphan := testPod(dgd, "orphan", nil, schema.GroupVersionKind{})
	require.NoError(t, c.Create(t.Context(), orphan))
	require.Never(t, func() bool { return !podCapIs(c, orphan, "300") }, 2*time.Second, waitInterval)
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(orphan), orphan))
	orphan.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}
	require.NoError(t, c.Update(t.Context(), orphan))
	require.Eventually(t, func() bool { return podCapIs(c, orphan, "350") }, waitTimeout, waitInterval)
}

func TestControllerWaitsForGroveUpdate(t *testing.T) {
	t.Log("Start only the power controller and store a Grove graph edited to 350 W")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	startPowerController(t, env, env.Namespace())
	dgd := testDGD(env.Namespace(), "grove")
	dgd.UID, dgd.Generation = "", 0
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	require.NoError(t, c.Create(t.Context(), dgd))

	t.Log("Grove has not acknowledged the DGD's PodCliqueSet; its worker Pod still carries 300 W")
	pcs := &grovev1alpha1.PodCliqueSet{Name: dynamo.PCSNameForDGD(dgd, (*v1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController), Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}}
	require.NoError(t, c.Create(t.Context(), pcs))
	clique := &grovev1alpha1.PodClique{Name: "worker", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
	require.NoError(t, c.Create(t.Context(), clique))
	pod := testPod(dgd, "worker", clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))
	require.NoError(t, c.Create(t.Context(), pod))
	require.Eventually(t, func() bool {
		reason, _, err := conditionReason(c, dgd)
		return err == nil && reason == "WaitingForWorkerUpdate"
	}, waitTimeout, waitInterval)
	require.Never(t, func() bool { return !podCapIs(c, pod, "300") }, 2*time.Second, waitInterval)

	t.Log("Grove acknowledges and finishes the update; the PodCliqueSet event releases the repair")
	// The test CRD has no status subresource, so this update itself advances the generation by one.
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	pcs.Status.ObservedGeneration = new(pcs.Generation + 1)
	require.NoError(t, c.Update(t.Context(), pcs))
	require.Eventually(t, func() bool { return podCapIs(c, pod, "350") }, waitTimeout, waitInterval)

	t.Log("A new unacknowledged generation waits again, and deleting that PodCliqueSet ends the wait")
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	pcs.Spec.Replicas = 2
	require.NoError(t, c.Update(t.Context(), pcs))
	require.Eventually(t, func() bool {
		reason, _, err := conditionReason(c, dgd)
		return err == nil && reason == "WaitingForWorkerUpdate"
	}, waitTimeout, waitInterval)
	require.NoError(t, c.Delete(t.Context(), pcs))
	require.Eventually(t, func() bool {
		reason, present, err := conditionReason(c, dgd)
		return err == nil && present && reason != "WaitingForWorkerUpdate"
	}, waitTimeout, waitInterval)
}

func TestControllerIgnoresFilteredNamespace(t *testing.T) {
	t.Log("Restrict the operator to another namespace and store a capped graph needing repair")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	startPowerController(t, env, "another-operator-namespace")
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	require.NoError(t, c.Create(t.Context(), dgd))
	rs := &appsv1.ReplicaSet{Name: "worker-rs", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}, Spec: appsv1.ReplicaSetSpec{
		Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"worker": "power"}},
		Template: corev1.PodTemplateSpec{Labels: map[string]string{"worker": "power"}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}}},
	}}
	require.NoError(t, c.Create(t.Context(), rs))
	pod := testPod(dgd, "worker", rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
	require.NoError(t, c.Create(t.Context(), pod))

	t.Log("No Pod is patched and the DGD gets no condition")
	require.Never(t, func() bool {
		_, present, err := conditionReason(c, dgd)
		return err == nil && (present || !podCapIs(c, pod, "300"))
	}, 3*time.Second, waitInterval)
}

func TestApplyConditionCoexistsWithDGDStatusUpdate(t *testing.T) {
	t.Log("Store a DGD and apply the power condition with the controller's field manager")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	require.NoError(t, c.Create(t.Context(), dgd))
	applied := metav1.Condition{Type: ConditionType, Status: metav1.ConditionTrue, Reason: "PodAnnotationsMatch", Message: "All owned live worker Pod annotations match", ObservedGeneration: dgd.Generation}
	require.NoError(t, applyCondition(t.Context(), c, dgd, &applied))

	t.Log("An ordinary DGD status update, as the DGD controller persists, keeps the power condition")
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	require.NotNil(t, meta.FindStatusCondition(dgd.Status.Conditions, ConditionType))
	meta.SetStatusCondition(&dgd.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Ready"})
	dgd.Status.State = v1beta1.DGDStateSuccessful
	require.NoError(t, c.Status().Update(t.Context(), dgd))
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	require.Equal(t, "PodAnnotationsMatch", meta.FindStatusCondition(dgd.Status.Conditions, ConditionType).Reason)

	t.Log("Removing the condition with an empty apply leaves the DGD controller's status intact")
	require.NoError(t, applyCondition(t.Context(), c, dgd, nil))
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
	require.Nil(t, meta.FindStatusCondition(dgd.Status.Conditions, ConditionType))
	require.Equal(t, "Ready", meta.FindStatusCondition(dgd.Status.Conditions, "Ready").Reason)
	require.Equal(t, v1beta1.DGDStateSuccessful, dgd.Status.State)
}

func TestReplicaSetOwnershipUsesMetadataCache(t *testing.T) {
	t.Log("Seed an owner with an engine template that ownership observation does not need")
	env := sharedEnv.ForTest(t)
	dgd := testDGD(env.Namespace(), "component")
	rs := &appsv1.ReplicaSet{Name: "power-owner", Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}, Spec: appsv1.ReplicaSetSpec{
		Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"worker": "power"}},
		Template: corev1.PodTemplateSpec{Labels: map[string]string{"worker": "power"}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "engine", Image: "example/engine:test"}}}},
	}}
	require.NoError(t, env.Client().Create(t.Context(), rs))
	pod := &corev1.Pod{Namespace: env.Namespace(), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}}

	t.Log("Register only a metadata informer and forbid lazy creation of typed informers")
	cached, err := cache.New(env.RESTConfig(), cache.Options{Scheme: env.Client().Scheme(), ReaderFailOnMissingInformer: true, DefaultNamespaces: map[string]cache.Config{env.Namespace(): {}}})
	require.NoError(t, err)
	metadata := &metav1.PartialObjectMetadata{}
	metadata.SetGroupVersionKind(appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))
	_, err = cached.GetInformer(t.Context(), metadata)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- cached.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		require.NoError(t, <-done)
	})
	syncCtx, syncCancel := context.WithTimeout(ctx, waitTimeout)
	defer syncCancel()
	require.True(t, cached.WaitForCacheSync(syncCtx))

	t.Log("Ownership resolves through metadata while a typed ReplicaSet read is explicitly uncached")
	owned, pending, err := podOwned(ctx, cached, dgd, pod)
	require.NoError(t, err)
	require.True(t, owned)
	require.False(t, pending)
	err = cached.Get(ctx, client.ObjectKeyFromObject(rs), &appsv1.ReplicaSet{})
	require.ErrorContains(t, err, "not cached")
	pod.OwnerReferences[0].UID = "stale-owner-uid"
	owned, pending, err = podOwned(ctx, cached, dgd, pod)
	require.NoError(t, err)
	require.False(t, owned)
	require.False(t, pending)
}
