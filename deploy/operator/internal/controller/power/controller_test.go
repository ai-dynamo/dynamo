// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"errors"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// testDGD describes one capped single-node worker at 300 W on the given workload provider.
func testDGD(namespace, provider string) *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		Name: "power", Namespace: namespace, UID: "graph-uid", Generation: 2,
		Annotations: map[string]string{consts.KubeAnnotationWorkloadProvider: provider},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker", ComponentType: consts.ComponentTypeWorker, Replicas: new(int32(1)),
				PodTemplate: &corev1.PodTemplateSpec{
					Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"},
					Spec:        corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}},
				},
			}},
		},
	}
}

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	require.NoError(t, grovev1alpha1.AddToScheme(scheme))
	return scheme
}

// testClient registers the production per-component Pod index that propagation queries.
func testClient(t *testing.T) *fake.ClientBuilder {
	t.Helper()
	return fake.NewClientBuilder().WithScheme(testScheme(t)).WithIndex(&corev1.Pod{}, dynamo.ComponentPodIndex, dynamo.ComponentPodIndexValues)
}

// testOwners constructs a fresh complete owner chain and one worker Pod at 300 W.
func testOwners(dgd *v1beta1.DynamoGraphDeployment, provider string) ([]client.Object, *corev1.Pod) {
	pod := &corev1.Pod{Name: "worker-pod", Namespace: dgd.Namespace, UID: "pod-uid", Labels: map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoSelector: "worker",
	}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300", "other": "retained"}}
	if provider == "grove" {
		pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: dgd.Namespace, UID: "pcs-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}}
		group := &grovev1alpha1.PodCliqueScalingGroup{Name: "group", Namespace: dgd.Namespace, UID: "group-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
		clique := &grovev1alpha1.PodClique{Name: "clique", Namespace: dgd.Namespace, UID: "clique-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}}
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))}
		return []client.Object{pcs, group, clique}, pod
	}
	dcd := &v1beta1.DynamoComponentDeployment{Name: "dcd", Namespace: dgd.Namespace, UID: "dcd-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}}
	deployment := &appsv1.Deployment{Name: "deployment", Namespace: dgd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}}
	rs := &appsv1.ReplicaSet{Name: "rs", Namespace: dgd.Namespace, UID: "rs-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(deployment, appsv1.SchemeGroupVersion.WithKind("Deployment"))}}
	pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}
	return []client.Object{dcd, deployment, rs}, pod
}

func TestPropagate(t *testing.T) {
	for _, provider := range []string{"component", "grove"} {
		for _, scenario := range []string{"repair", "matching", "empty", "terminal", "foreign", "orphan", "missing-parent", "parent-arrives", "direct-grove", "list-error", "owner-error", "patch-error", "conflict", "deleted"} {
			t.Run(provider+"/"+scenario, func(t *testing.T) {
				t.Log("Observe owned workers after a cap edit to 350 W")
				dgd := testDGD("default", provider)
				dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
				owners, pod := testOwners(dgd, provider)
				var lateParent client.Object
				switch scenario {
				case "parent-arrives":
					lateParent = owners[len(owners)-1].DeepCopyObject().(client.Object)
					owners = owners[:len(owners)-1]
				case "direct-grove":
					if provider == "grove" {
						owners[2].SetOwnerReferences([]metav1.OwnerReference{*metav1.NewControllerRef(owners[0], grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))})
						owners = []client.Object{owners[0], owners[2]}
					}
				case "matching":
					pod.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
				case "terminal":
					pod.Status.Phase = corev1.PodSucceeded
				case "foreign":
					pod.OwnerReferences[0].UID = "stale-owner"
				case "orphan":
					pod.OwnerReferences = nil
				case "missing-parent":
					owners = owners[:len(owners)-1]
				}
				objects := append(owners, dgd)
				if scenario != "empty" {
					objects = append(objects, pod)
				}
				c := testClient(t).WithObjects(objects...).WithInterceptorFuncs(interceptor.Funcs{
					Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
						if _, ok := obj.(*grovev1alpha1.PodCliqueSet); !ok && scenario == "owner-error" {
							return errors.New("owner read denied")
						}
						return c.Get(ctx, key, obj, opts...)
					},
					List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
						if scenario == "list-error" {
							return errors.New("list denied")
						}
						return c.List(ctx, list, opts...)
					},
					Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
						switch scenario {
						case "patch-error":
							return errors.New("patch denied")
						case "conflict":
							return apierrors.NewConflict(schema.GroupResource{Resource: "pods"}, obj.GetName(), errors.New("concurrent edit"))
						case "deleted":
							return apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, obj.GetName())
						}
						return c.Patch(ctx, obj, patch, opts...)
					},
				}).Build()
				r := &Reconciler{client: c, groveEnabled: true}
				condition, requeueAfter := r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))

				t.Log("Report the worst Pod outcome and retry only what no watch event will deliver")
				require.Equal(t, dgd.Generation, condition.ObservedGeneration)
				switch scenario {
				case "list-error", "owner-error":
					require.Equal(t, metav1.ConditionUnknown, condition.Status)
					require.Equal(t, "PodObservationFailed", condition.Reason)
					require.Equal(t, failureRetryAfter, requeueAfter)
				case "patch-error":
					require.Equal(t, metav1.ConditionFalse, condition.Status)
					require.Equal(t, "PodPatchFailed", condition.Reason)
					require.Equal(t, failureRetryAfter, requeueAfter)
				case "conflict":
					require.Equal(t, metav1.ConditionFalse, condition.Status)
					require.Equal(t, "PodPatchFailed", condition.Reason)
					require.Equal(t, retryAfter, requeueAfter)
				case "missing-parent", "parent-arrives":
					require.Equal(t, metav1.ConditionUnknown, condition.Status)
					require.Equal(t, retryAfter, requeueAfter)
					if lateParent != nil {
						t.Log("The cache later observes the parent and a retry repairs the Pod")
						require.NoError(t, c.Create(t.Context(), lateParent))
						condition, _ = r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))
						require.Equal(t, "PodAnnotationMismatch", condition.Reason)
					}
				case "deleted", "direct-grove":
					require.Equal(t, "PodAnnotationMismatch", condition.Reason)
					require.Zero(t, requeueAfter)
				case "repair":
					require.Equal(t, "PodAnnotationMismatch", condition.Reason)
					require.Zero(t, requeueAfter)
					require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
					require.Equal(t, "350", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
					require.Equal(t, "retained", pod.Annotations["other"])
					t.Log("A later observation establishes convergence")
					condition, requeueAfter = r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))
					require.Equal(t, metav1.ConditionTrue, condition.Status)
					require.Zero(t, requeueAfter)
				default:
					require.Equal(t, metav1.ConditionTrue, condition.Status)
					require.Zero(t, requeueAfter)
				}
			})
		}
	}
}

func TestPropagateMixedPodOutcomes(t *testing.T) {
	for _, scenario := range []string{"conflict-and-denied", "denied-only"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Observe several owned workers whose repairs end differently in one pass")
			dgd := testDGD("default", "component")
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			owners, conflicting := testOwners(dgd, "component")
			denied := conflicting.DeepCopy()
			denied.Name, denied.UID = "denied-pod", "denied-uid"
			matching := conflicting.DeepCopy()
			matching.Name, matching.UID = "matching-pod", "matching-uid"
			matching.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			pods := []client.Object{denied, matching}
			if scenario == "conflict-and-denied" {
				pods = append(pods, conflicting)
			}
			c := testClient(t).WithObjects(append(append(owners, dgd), pods...)...).WithInterceptorFuncs(interceptor.Funcs{
				Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
					switch obj.GetName() {
					case conflicting.Name:
						return apierrors.NewConflict(schema.GroupResource{Resource: "pods"}, obj.GetName(), errors.New("concurrent update"))
					case denied.Name:
						return errors.New("patch denied")
					}
					return c.Patch(ctx, obj, patch, opts...)
				},
			}).Build()
			r := &Reconciler{client: c}
			condition, requeueAfter := r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))

			t.Log("A failed repair outranks a matching Pod and the shortest retry wins")
			require.Equal(t, metav1.ConditionFalse, condition.Status)
			require.Equal(t, "PodPatchFailed", condition.Reason)
			want := failureRetryAfter
			if scenario == "conflict-and-denied" {
				want = retryAfter
			}
			require.Equal(t, want, requeueAfter)
		})
	}
}

func TestPropagateMultipleCaps(t *testing.T) {
	t.Log("Different worker components keep their own desired caps in one observation")
	dgd := testDGD("default", "component")
	other := *dgd.Spec.Components[0].DeepCopy()
	other.ComponentName = "second"
	other.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "325"
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	dgd.Spec.Components = append(dgd.Spec.Components, other)
	owners, pod := testOwners(dgd, "component")
	second := pod.DeepCopy()
	second.Name, second.UID = "second-pod", "second-uid"
	second.Labels[consts.KubeLabelDynamoComponent] = "second"
	c := testClient(t).WithObjects(append(owners, dgd, pod, second)...).Build()
	r := &Reconciler{client: c}
	r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(second), second))
	require.Equal(t, "350", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
	require.Equal(t, "325", second.Annotations[consts.KubeAnnotationGPUPowerLimit])
}

func TestPropagateGroveBarrier(t *testing.T) {
	for _, state := range []string{"missing-observed-generation", "stale-observed-generation", "update-in-progress", "finished", "foreign", "absent", "read-error"} {
		t.Run(state, func(t *testing.T) {
			t.Log("Observe the DGD's own PodCliqueSet in the given update state")
			dgd := testDGD("default", "grove")
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			owners, pod := testOwners(dgd, "grove")
			pcs := owners[0].(*grovev1alpha1.PodCliqueSet)
			pcs.Name, pcs.Generation = dynamo.PCSNameForDGD(dgd, (*v1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController), 3
			switch state {
			case "stale-observed-generation":
				pcs.Status.ObservedGeneration = new(int64(2))
			case "update-in-progress":
				pcs.Status.ObservedGeneration = new(int64(3))
				pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()}
			case "finished", "foreign":
				pcs.Status.ObservedGeneration = new(int64(3))
			}
			if state == "foreign" {
				pcs.Status.ObservedGeneration = nil
				pcs.OwnerReferences[0].UID = "other-graph"
			}
			objects := append(owners, dgd, pod)
			if state == "absent" {
				objects = objects[1:]
			}
			podReads := 0
			c := testClient(t).WithObjects(objects...).WithStatusSubresource(&grovev1alpha1.PodCliqueSet{}).WithInterceptorFuncs(interceptor.Funcs{
				Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok && state == "read-error" {
						return errors.New("PodCliqueSet read denied")
					}
					return c.Get(ctx, key, obj, opts...)
				},
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					podReads++
					return c.List(ctx, list, opts...)
				},
			}).Build()
			r := &Reconciler{client: c, groveEnabled: true}
			condition, requeueAfter := r.propagate(t.Context(), dgd, dynamo.DesiredWorkerPowerCaps(dgd))

			t.Log("An unfinished update defers all Pod reads; the PodCliqueSet watch, not a timer, wakes the next pass")
			switch state {
			case "missing-observed-generation", "stale-observed-generation", "update-in-progress":
				require.Zero(t, podReads)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, "WaitingForWorkerUpdate", condition.Reason)
				require.Zero(t, requeueAfter)
			case "read-error":
				require.Zero(t, podReads)
				require.Equal(t, "PodObservationFailed", condition.Reason)
				require.Equal(t, failureRetryAfter, requeueAfter)
			default:
				require.Equal(t, 1, podReads)
				require.NotEqual(t, "WaitingForWorkerUpdate", condition.Reason)
			}
		})
	}
}

func TestReconcileSkipsUnobservableGraphs(t *testing.T) {
	for _, scenario := range []string{"no-caps", "grove-disabled"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Store a worker Pod at 300 W under a graph this controller must not observe")
			provider := "component"
			if scenario == "grove-disabled" {
				provider = "grove"
			}
			dgd := testDGD("default", provider)
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			if scenario == "no-caps" {
				delete(dgd.Spec.Components[0].PodTemplate.Annotations, consts.KubeAnnotationGPUPowerLimit)
			}
			owners, pod := testOwners(dgd, provider)
			podReads, podPatches := 0, 0
			c := testClient(t).WithObjects(append(owners, dgd, pod)...).WithStatusSubresource(dgd).WithInterceptorFuncs(interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if _, ok := list.(*corev1.PodList); ok {
						podReads++
					}
					return c.List(ctx, list, opts...)
				},
				Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
					if _, ok := obj.(*corev1.Pod); ok {
						podPatches++
					}
					return c.Patch(ctx, obj, patch, opts...)
				},
			}).Build()

			t.Log("Reconcile with Grove disabled, as after an operator restart without the Grove integration")
			result, err := (&Reconciler{client: c, groveEnabled: false}).Reconcile(t.Context(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dgd)})
			require.NoError(t, err)
			require.Zero(t, result.RequeueAfter)

			t.Log("No Pod is read or patched and no condition is written")
			require.Zero(t, podReads)
			require.Zero(t, podPatches)
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
			require.Equal(t, "300", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
			require.Nil(t, meta.FindStatusCondition(dgd.Status.Conditions, ConditionType))
		})
	}
}
