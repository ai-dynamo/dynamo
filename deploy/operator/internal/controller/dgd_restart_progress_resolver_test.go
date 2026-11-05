/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestComponentRestartAppliedRequiresEveryTemplate(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		sharedID              string
		leaderID, workerID    string
		multinode, noTemplate bool
		requestedID           string
		wantApplied           bool
	}{
		{name: "requested restart on the shared template", sharedID: "requested", requestedID: "requested", wantApplied: true},
		{name: "ready old revision", sharedID: "previous", requestedID: "requested"},
		{name: "requested restart on every role template", sharedID: "requested", leaderID: "requested", workerID: "requested", multinode: true, requestedID: "requested", wantApplied: true},
		{name: "only one role template updated", sharedID: "requested", leaderID: "requested", workerID: "previous", multinode: true, requestedID: "requested"},
		{name: "annotation missing", requestedID: "requested"},
		{name: "no pod template", noTemplate: true, requestedID: "requested"},
		{name: "empty requested restart", requestedID: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build the observed DCD templates")
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				Name: "graph-frontend", Namespace: "default",
			}
			if !tc.noTemplate {
				dcd.Spec.PodTemplate = restartPodTemplate(tc.sharedID)
			}
			if tc.multinode {
				dcd.Spec.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: "leader", PodTemplate: restartPodTemplate(tc.leaderID)},
					{Name: "worker", PodTemplate: restartPodTemplate(tc.workerID)},
				}
			}
			original := dcd.DeepCopy()

			t.Log("Require every template to observe the requested restart without mutating the DCD")
			require.Equal(t, tc.wantApplied, componentRestartApplied(dcd, tc.requestedID))
			require.Equal(t, original, dcd)
		})
	}
}

func TestGroveRestartAppliedRequiresEverySelectedClique(t *testing.T) {
	for _, tc := range []struct {
		name        string
		cliques     []*grovev1alpha1.PodCliqueTemplateSpec
		requestedID string
		wantApplied bool
	}{
		{
			name: "every clique of the component has the requested restart",
			cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
				restartClique("decode-ldr", "decode", "requested"),
				restartClique("decode-wkr", "decode", "requested"),
				nil,
			},
			requestedID: "requested",
			wantApplied: true,
		},
		{
			name: "ready old revision",
			cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
				restartClique("decode-ldr", "decode", "previous"),
				restartClique("decode-wkr", "decode", "previous"),
			},
			requestedID: "requested",
		},
		{
			name: "leader updated but worker on the old revision",
			cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
				restartClique("decode-ldr", "decode", "requested"),
				restartClique("decode-wkr", "decode", "previous"),
			},
			requestedID: "requested",
		},
		{
			name:        "no clique for the component",
			cliques:     []*grovev1alpha1.PodCliqueTemplateSpec{restartClique("frontend", "frontend", "requested")},
			requestedID: "requested",
		},
		{
			name: "only another component has the requested restart",
			cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
				restartClique("frontend", "frontend", "requested"),
				restartClique("decode", "decode", ""),
			},
			requestedID: "requested",
		},
		{
			name:        "empty requested restart",
			cliques:     []*grovev1alpha1.PodCliqueTemplateSpec{restartClique("decode", "decode", "")},
			requestedID: "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build the observed PodCliqueSet cliques")
			pcs := &grovev1alpha1.PodCliqueSet{
				Name: "graph", Namespace: "default",
				Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
					Cliques: tc.cliques,
				}},
			}
			original := pcs.DeepCopy()

			t.Log("Require every selected clique to observe the requested restart without mutating the PCS")
			require.Equal(t, tc.wantApplied, groveRestartApplied(pcs, "decode", tc.requestedID))
			require.Equal(t, original, pcs)
		})
	}
}

// restartPodTemplate describes a pod template observed at restartID; an empty
// restartID omits the annotation.
func restartPodTemplate(restartID string) *corev1.PodTemplateSpec {
	template := &corev1.PodTemplateSpec{}
	if restartID != "" {
		template.Annotations = map[string]string{consts.RestartAnnotation: restartID}
	}
	return template
}

// restartClique describes a rendered clique of componentName observed at
// restartID; an empty restartID omits the annotation.
func restartClique(name, componentName, restartID string) *grovev1alpha1.PodCliqueTemplateSpec {
	template := &grovev1alpha1.PodCliqueTemplateSpec{
		Name:   name,
		Labels: map[string]string{consts.KubeLabelDynamoComponent: componentName},
	}
	if restartID != "" {
		template.Annotations = map[string]string{consts.RestartAnnotation: restartID}
	}
	return template
}

// The Grove program lists Pods through a manager-cache field index that the
// raw envtest client lacks, so this drives the program through an indexed fake
// client that emulates PodCliqueSet generations. The DGD has no status
// subresource here, so Update persists the status the outer controller would.
func TestGroveRestartWaitsForRequestedRevisionAfterFailedWrite(t *testing.T) {
	for _, strategy := range []nvidiacomv1beta1.RestartStrategyType{
		nvidiacomv1beta1.RestartStrategyTypeSequential,
		nvidiacomv1beta1.RestartStrategyTypeParallel,
	} {
		t.Run(string(strategy), func(t *testing.T) {
			ctx := t.Context()
			order := []string{"frontend", "decode"}
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				Name:        "grove-restart-retry-" + strings.ToLower(string(strategy)),
				Namespace:   "default",
				Annotations: map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "frontend", ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
							Replicas: new(int32(1)), RuntimeVersionOverride: "1.0.0", PodTemplate: envtestDSTestPodTemplate(),
						},
						{
							ComponentName: "decode", ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
							Replicas: new(int32(1)), RuntimeVersionOverride: "1.0.0", PodTemplate: envtestDSTestPodTemplate(),
						},
					},
				},
			}
			failWrite := false
			pcsGVK := grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet")
			injectedConflict := apierrors.NewConflict(grovev1alpha1.Resource("podcliquesets"), dgd.Name, fmt.Errorf("injected restart write conflict"))
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(dgd).
				WithIndex(&corev1.Pod{}, podcontroller.ComponentIndex, podcontroller.ComponentIndexValues).
				WithRESTMapper(groveScaleRESTMapper()).
				WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{
					Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
						if gvk, _ := apiutil.GVKForObject(obj, c.Scheme()); gvk == pcsGVK && failWrite {
							return injectedConflict
						}
						return c.Create(ctx, obj, opts...)
					},
					Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
						// The program writes the PodCliqueSet as unstructured; emulate
						// the API server by starting a new generation on a spec change.
						desired, isProgramWrite := obj.(*unstructured.Unstructured)
						if !isProgramWrite || desired.GroupVersionKind() != pcsGVK {
							return c.Update(ctx, obj, opts...)
						}
						if failWrite {
							return injectedConflict
						}
						stored := &unstructured.Unstructured{}
						stored.SetGroupVersionKind(pcsGVK)
						if err := c.Get(ctx, client.ObjectKeyFromObject(desired), stored); err != nil {
							return err
						}
						desired.SetGeneration(stored.GetGeneration())
						if !equality.Semantic.DeepEqual(stored.Object["spec"], desired.Object["spec"]) {
							desired.SetGeneration(stored.GetGeneration() + 1)
						}
						return c.Update(ctx, desired, opts...)
					},
				}, nil)).
				Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   kubeClient,
				Recorder: events.NewFakeRecorder(100),
				Config: &configv1alpha1.OperatorConfiguration{
					Namespace: configv1alpha1.NamespaceConfiguration{Restricted: dgd.Namespace},
				},
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
			}
			program := reconciler.newGroveProgram()
			pcsName := dynamo.PCSNameForDGD(dgd, (*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController)

			t.Log("Observe the ready baseline before requesting a new restart")
			_, err := program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			markRestartGroveComponentsReady(t, reconciler.Client, dgd.Namespace, pcsName, order)
			dgd.Spec.Restart = &nvidiacomv1beta1.Restart{
				ID:       "restart-retry",
				Strategy: &nvidiacomv1beta1.RestartStrategy{Type: strategy},
			}
			if strategy == nvidiacomv1beta1.RestartStrategyTypeSequential {
				dgd.Spec.Restart.Strategy.Order = order
			}
			require.NoError(t, reconciler.Client.Update(ctx, dgd))

			t.Log("Fail the PodCliqueSet write and persist the returned restart status")
			failWrite = true
			result, err := program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.Error(t, err)
			require.Equal(t, nvidiacomv1beta1.RestartPhaseRestarting, result.Status.Restart.Phase)
			dgd.Status = result.Status
			require.NoError(t, reconciler.Client.Update(ctx, dgd))
			require.Equal(t, "restart-retry", dgd.Status.Restart.ObservedID)

			t.Log("Retry against the ready old revision without advancing or completing")
			pending := []string{"frontend"}
			if strategy == nvidiacomv1beta1.RestartStrategyTypeParallel {
				pending = order
			}
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.Error(t, err)
			require.Equal(t, nvidiacomv1beta1.RestartPhaseRestarting, result.Status.Restart.Phase)
			require.ElementsMatch(t, pending, result.Status.Restart.InProgress)
			observed := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, reconciler.Client.Get(ctx, client.ObjectKey{Name: pcsName, Namespace: dgd.Namespace}, observed))
			for _, clique := range observed.Spec.Template.Cliques {
				require.Empty(t, clique.Annotations[consts.RestartAnnotation])
			}

			t.Log("Accept the write but wait until the requested revision becomes ready")
			failWrite = false
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.ElementsMatch(t, pending, result.Status.Restart.InProgress)
			dgd.Status = result.Status
			require.NoError(t, reconciler.Client.Update(ctx, dgd))
			require.Equal(t, "restart-retry", dgd.Status.Restart.ObservedID)
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.ElementsMatch(t, pending, result.Status.Restart.InProgress)

			t.Log("Advance or complete only after the requested revision is ready")
			markRestartGroveComponentsReady(t, reconciler.Client, dgd.Namespace, pcsName, order)
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			if strategy == nvidiacomv1beta1.RestartStrategyTypeParallel {
				require.Equal(t, nvidiacomv1beta1.RestartPhaseCompleted, result.Status.Restart.Phase)
			} else {
				require.Equal(t, nvidiacomv1beta1.RestartPhaseRestarting, result.Status.Restart.Phase)
				require.Equal(t, []string{"decode"}, result.Status.Restart.InProgress)
			}
		})
	}
}

// markRestartGroveComponentsReady reports the PodCliqueSet as observed and
// replaces each component's PodClique with a ready one, standing in for the
// Grove controller.
func markRestartGroveComponentsReady(t *testing.T, c client.Client, namespace, pcsName string, components []string) {
	t.Helper()
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, c.Get(t.Context(), client.ObjectKey{Name: pcsName, Namespace: namespace}, pcs))
	pcs.Status.ObservedGeneration = new(pcs.Generation)
	require.NoError(t, c.Update(t.Context(), pcs))

	// Replace each PodClique so its ready status belongs to its current generation.
	for _, componentName := range components {
		clique := &grovev1alpha1.PodClique{
			Name: dynamo.GroveComponentResourceName(pcsName, componentName), Namespace: namespace, Generation: 1,
			Spec:   grovev1alpha1.PodCliqueSpec{Replicas: 1},
			Status: grovev1alpha1.PodCliqueStatus{ObservedGeneration: new(int64(1)), Replicas: 1, UpdatedReplicas: 1, ReadyReplicas: 1},
		}
		require.NoError(t, client.IgnoreNotFound(c.Delete(t.Context(), clique)))
		require.NoError(t, c.Create(t.Context(), clique))
	}
}
