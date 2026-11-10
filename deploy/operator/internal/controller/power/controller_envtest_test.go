//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"errors"
	"testing"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
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
		if err := mgr.GetFieldIndexer().IndexField(context.Background(), &corev1.Pod{}, podcontroller.ComponentIndex, podcontroller.ComponentIndexValues); err != nil {
			return err
		}
		return Setup(mgr, config, runtimeConfig)
	})
}

// testPod is a labeled worker Pod of the DGD's worker component created from a retained 300 W template.
func testPod(dgd *v1beta1.DynamoGraphDeployment, name string) *corev1.Pod {
	return &corev1.Pod{Name: name, Namespace: dgd.Namespace, Labels: map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker",
	}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}, Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}}}
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

func TestControllerRepairsLabeledWorkerPods(t *testing.T) {
	t.Log("Start only the power controller and store a component graph edited to 350 W")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	startPowerController(t, env, env.Namespace())
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	require.NoError(t, c.Create(t.Context(), dgd))

	t.Log("A labeled worker Pod created from the retained 300 W template is repaired by its create event")
	retained := testPod(dgd, "retained")
	require.NoError(t, c.Create(t.Context(), retained))
	require.Eventually(t, func() bool { return podCapIs(c, retained, "350") }, waitTimeout, waitInterval)
	require.Eventually(t, func() bool {
		reason, _, err := conditionReason(c, dgd)
		return err == nil && reason == "PodAnnotationsMatch"
	}, waitTimeout, waitInterval)

	t.Log("A replacement Pod owned through Grove's beta API, whose owner is not stored, is repaired without any DGD change")
	replacement := testPod(dgd, "replacement")
	replacement.OwnerReferences = []metav1.OwnerReference{{APIVersion: "grove.io/v1beta1", Kind: "PodClique", Name: "absent-clique", UID: "absent-clique-uid", Controller: new(true)}}
	require.NoError(t, c.Create(t.Context(), replacement))
	require.Eventually(t, func() bool { return podCapIs(c, replacement, "350") }, waitTimeout, waitInterval)

	t.Log("A Pod labeled for another component keeps its annotation until its labels move it into the capped worker")
	moved := testPod(dgd, "moved")
	moved.Labels[consts.KubeLabelDynamoComponent] = "uncapped"
	require.NoError(t, c.Create(t.Context(), moved))
	require.Never(t, func() bool { return !podCapIs(c, moved, "300") }, 2*time.Second, waitInterval)
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(moved), moved))
	moved.Labels[consts.KubeLabelDynamoComponent] = "worker"
	require.NoError(t, c.Update(t.Context(), moved))
	require.Eventually(t, func() bool { return podCapIs(c, moved, "350") }, waitTimeout, waitInterval)
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
	pcs := &grovev1alpha1.PodCliqueSet{Name: "legacy-pcs", Namespace: env.Namespace(), Labels: map[string]string{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}}
	require.NoError(t, c.Create(t.Context(), pcs))
	pod := testPod(dgd, "worker")
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
	pod := testPod(dgd, "worker")
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

func TestReconcileReturnsAPIErrors(t *testing.T) {
	for _, scenario := range []string{"list", "conflict", "rejection", "status-too"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Create a graph and a worker needing repair")
			env := sharedEnv.ForTest(t)
			c := env.Client()
			dgd := testDGD(env.Namespace(), "component")
			dgd.UID, dgd.Generation = "", 0
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			require.NoError(t, c.Create(t.Context(), dgd))
			watchClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: c.Scheme()})
			require.NoError(t, err)
			podReader := testClient(t).WithObjects(testWorkerPod(dgd, "component")).Build()
			propagationErr := errors.New("API denied")
			if scenario == "conflict" {
				propagationErr = apierrors.NewConflict(schema.GroupResource{Resource: "pods"}, "worker", propagationErr)
			}
			statusErr := errors.New("status denied")
			wrapped := interceptor.NewClient(watchClient, interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if scenario == "list" {
						return propagationErr
					}
					return podReader.List(ctx, list, opts...)
				},
				Patch: func(context.Context, client.WithWatch, client.Object, client.Patch, ...client.PatchOption) error {
					return propagationErr
				},
				SubResourceApply: func(ctx context.Context, c client.Client, sub string, obj runtime.ApplyConfiguration, opts ...client.SubResourceApplyOption) error {
					if scenario == "status-too" {
						return statusErr
					}
					return c.SubResource(sub).Apply(ctx, obj, opts...)
				},
			})

			t.Log("The reconcile reports errors for rate-limited retry after attempting the condition")
			result, err := (&reconciler{client: wrapped}).Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
			require.ErrorIs(t, err, propagationErr)
			require.Zero(t, result)
			if scenario == "status-too" {
				require.ErrorIs(t, err, statusErr)
			} else {
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
				condition := meta.FindStatusCondition(dgd.Status.Conditions, ConditionType)
				require.NotNil(t, condition)
				wantReason := "PodPatchFailed"
				if scenario == "list" {
					wantReason = "PodObservationFailed"
				}
				require.Equal(t, wantReason, condition.Reason)
			}
		})
	}
}

func TestReconcileRestoresConditionWithStaleCache(t *testing.T) {
	t.Log("Cache a converged graph, then overwrite the stored condition")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	require.NoError(t, c.Create(t.Context(), dgd))
	podReaderInitial := testClient(t).Build()
	initialClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: c.Scheme()})
	require.NoError(t, err)
	r := &reconciler{client: interceptor.NewClient(initialClient, interceptor.Funcs{List: func(ctx context.Context, _ client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
		return podReaderInitial.List(ctx, list, opts...)
	}})}
	req := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)}
	_, err = r.Reconcile(t.Context(), req)
	require.NoError(t, err)
	require.NoError(t, c.Get(t.Context(), req.NamespacedName, dgd))
	cached := dgd.DeepCopy()
	want := *meta.FindStatusCondition(cached.Status.Conditions, ConditionType)
	meta.SetStatusCondition(&dgd.Status.Conditions, metav1.Condition{Type: ConditionType, Status: metav1.ConditionFalse, Reason: "Stale"})
	meta.SetStatusCondition(&dgd.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Ready"})
	require.NoError(t, c.Status().Update(t.Context(), dgd))
	watchClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: c.Scheme()})
	require.NoError(t, err)
	podReader := testClient(t).Build()
	r.client = interceptor.NewClient(watchClient, interceptor.Funcs{List: func(ctx context.Context, _ client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
		return podReader.List(ctx, list, opts...)
	}, Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if graph, ok := obj.(*v1beta1.DynamoGraphDeployment); ok {
			*graph = *cached.DeepCopy()
			return nil
		}
		return c.Get(ctx, key, obj, opts...)
	}})

	t.Log("Repeated reconciles restore persisted status without a manager or self-condition watch")
	for i := range 3 {
		t.Logf("pass %d", i)
		_, err := r.Reconcile(t.Context(), req)
		require.NoError(t, err)
		require.NoError(t, c.Get(t.Context(), req.NamespacedName, dgd))
		require.Equal(t, want, *meta.FindStatusCondition(dgd.Status.Conditions, ConditionType))
		require.Equal(t, "Ready", meta.FindStatusCondition(dgd.Status.Conditions, "Ready").Reason)
	}
}

func TestReconcileRemovesConditionWithStaleCache(t *testing.T) {
	t.Log("Store an uncapped graph and keep a cached view that misses the power condition written next")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	dgd := testDGD(env.Namespace(), "component")
	dgd.UID, dgd.Generation = "", 0
	delete(dgd.Spec.Components[0].PodTemplate.Annotations, consts.KubeAnnotationGPUPowerLimit)
	require.NoError(t, c.Create(t.Context(), dgd))
	cached := dgd.DeepCopy()
	key := client.ObjectKeyFromObject(dgd)
	applied := metav1.Condition{Type: ConditionType, Status: metav1.ConditionTrue, Reason: "PodAnnotationsMatch", Message: "All live worker Pod annotations match", ObservedGeneration: dgd.Generation}
	require.NoError(t, applyCondition(t.Context(), c, dgd, &applied))
	require.NoError(t, c.Get(t.Context(), key, dgd))
	meta.SetStatusCondition(&dgd.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, Reason: "Ready"})
	require.NoError(t, c.Status().Update(t.Context(), dgd))
	watchClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: c.Scheme()})
	require.NoError(t, err)
	r := &reconciler{client: interceptor.NewClient(watchClient, interceptor.Funcs{Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
		if graph, ok := obj.(*v1beta1.DynamoGraphDeployment); ok {
			*graph = *cached.DeepCopy()
			return nil
		}
		return c.Get(ctx, key, obj, opts...)
	}})}

	t.Log("A reconcile against the stale cache still removes the persisted condition and keeps other status")
	_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)
	require.NoError(t, c.Get(t.Context(), key, dgd))
	require.Nil(t, meta.FindStatusCondition(dgd.Status.Conditions, ConditionType))
	require.Equal(t, "Ready", meta.FindStatusCondition(dgd.Status.Conditions, "Ready").Reason)

	t.Log("Repeating the removal on a DGD with status writes nothing")
	resourceVersion := dgd.ResourceVersion
	_, err = r.Reconcile(t.Context(), ctrl.Request{NamespacedName: key})
	require.NoError(t, err)
	require.NoError(t, c.Get(t.Context(), key, dgd))
	require.Equal(t, resourceVersion, dgd.ResourceVersion)
}
