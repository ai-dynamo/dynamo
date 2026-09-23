// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/yaml"
)

// newLPXHandoffSource loads a fresh authored workload without compiler or child-controller fixtures.
func newLPXHandoffSource(t *testing.T, fixture string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("../dynamo/lpx/testdata/from_dgd_yaml", fixture+".input.yaml"))
	require.NoError(t, err)
	source := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}}
	require.NoError(t, yaml.Unmarshal(data, source))
	source.GetComponentByName("lpu").ComponentName = "lpx"
	source.Generation = 3
	return source
}

// newLPXHandoffFixture creates only the parent-owned handoff through its production path.
func newLPXHandoffFixture(t *testing.T, fixture string) (*v1alpha1.LPXGraphDeployment, *v1beta1.DynamoGraphDeployment, client.Client) {
	t.Helper()
	source := newLPXHandoffSource(t, fixture)
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithRESTMapper(groveScaleRESTMapper()).
		WithStatusSubresource(&v1beta1.DynamoGraphDeployment{}, &v1alpha1.LPXGraphDeployment{}).
		WithObjects(source).Build()
	child, err := (&dgdLPXHandoff{client: kube}).Reconcile(t.Context(), source)
	require.NoError(t, err)
	child.UID, child.Generation = "child-uid", 3
	require.NoError(t, kube.Update(t.Context(), child))
	return child, source, kube
}

func TestProjectWithoutExternallyManagedComponentsDoesNotMutateSource(t *testing.T) {
	for _, test := range []struct {
		name      string
		longNames bool
		lpxOnly   bool
	}{
		{name: "mixed"},
		{name: "mixed long names", longNames: true},
		{name: "LPX only", lpxOnly: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Project the exact DGD-owned subset without changing the source")
			source := newLPXHandoffSource(t, "node-local-v2-hybrid")
			lpxComponent := &source.Spec.Components[0]
			if test.longNames {
				source.Name = strings.Repeat("graph-", 10) + "a"
				lpxComponent.ComponentName = strings.Repeat("l", 30)
			}
			lpxName := lpxComponent.ComponentName
			if !test.lpxOnly {
				source.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{ClusterTopologyName: "test-topology", PackDomain: "rack"}
				source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "ordinary-frontend", ComponentType: v1beta1.ComponentTypeFrontend,
					Replicas: ptr.To(int32(1)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "frontend"}},
					}},
				})
			}
			before := source.DeepCopy()
			ordinary := projectWithoutExternallyManagedComponents(source)
			require.Nil(t, ordinary.GetComponentByName(lpxName))
			require.Len(t, ordinary.Spec.Components, len(source.Spec.Components)-1)

			t.Log("Seed only the projected PCS identity, including the truncated-name case")
			pcsName := dynamo.PCSNameForDGD(ordinary.Name, ordinary.Spec.Components)
			if test.longNames {
				require.NotEqual(t, source.Name, pcsName)
				require.NotEqual(t, dynamo.PCSNameForDGD(source.Name, source.Spec.Components), pcsName)
			}
			pcs := &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: pcsName, Namespace: source.Namespace, Generation: 1},
				Status: grovev1alpha1.PodCliqueSetStatus{ObservedGeneration: ptr.To(int64(1)), Conditions: []metav1.Condition{{
					Type: groveconstants.ConditionTopologyLevelsUnavailable, Status: metav1.ConditionTrue,
					Reason: groveconstants.ConditionReasonClusterTopologyNotFound, Message: "missing ordinary topology",
				}}},
			}
			clique := &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{Name: dynamo.GroveComponentResourceName(ordinary, "ordinary-frontend"), Namespace: source.Namespace, Generation: 1},
				Spec:       grovev1alpha1.PodCliqueSpec{Replicas: 1},
				Status:     grovev1alpha1.PodCliqueStatus{ObservedGeneration: ptr.To(int64(1)), Replicas: 1, UpdatedReplicas: 1, ReadyReplicas: 1},
			}
			kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(pcs, clique).Build()

			t.Log("Render and observe restarts against that same ordinary PCS")
			renderer := newGroveWorkloadRenderer(kube, &configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}, nil)
			rendered, err := renderer.Render(t.Context(), source, ordinary, nil, nil, false)
			require.NoError(t, err)
			require.NotNil(t, rendered.existing)
			require.Equal(t, pcsName, rendered.existing.Name)
			if test.lpxOnly {
				require.Nil(t, rendered.desired)
			} else {
				require.Equal(t, pcsName, rendered.desired.Name)
			}
			remaining := resolveCompositeGroveRestartProgress(t.Context(), source, ordinary,
				[]string{lpxName, "ordinary-frontend"}, newGroveRestartProgressResolver(kube), newLPXRestartProgressResolver(kube))
			require.Equal(t, []string{lpxName}, remaining)

			t.Log("Observe the graph-level topology through the same ordinary PCS")
			result := newWorkloadProgramResult(source)
			newDGDGroveTopologyConditionReconciler(kube).Reconcile(t.Context(), ordinary, &result)
			condition := meta.FindStatusCondition(result.Status.Conditions, v1beta1.ConditionTypeTopologyLevelsAvailable)
			if test.lpxOnly {
				require.Nil(t, condition)
				require.Empty(t, result.Events)
			} else {
				require.NotNil(t, condition)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, v1beta1.ConditionReasonTopologyDefinitionNotFound, condition.Reason)
				require.Equal(t, "missing ordinary topology", condition.Message)
			}

			t.Log("Keep the source unchanged after observations and projected component-slice edits")
			require.Equal(t, before, source)
			if len(ordinary.Spec.Components) > 0 {
				ordinary.Spec.Components[0].ComponentName = "ordinary-copy"
			}
			require.Equal(t, before, source)
		})
	}
}

func TestLPXHandoffCreatesOnlyAnOwnedReference(t *testing.T) {
	t.Log("Create the generated handoff from a real source DGD")
	source := newLPXHandoffSource(t, "node-local-v2-lpu-only")
	kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(source).Build()
	handoff := &dgdLPXHandoff{client: kube}
	child, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.NotEmpty(t, child.Spec.InputRevision)
	require.Equal(t, metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK), metav1.GetControllerOf(child))
	revision, err := dynamo.LPXInputRevision(source, child.Annotations[dynamo.LPXRestartAnnotation])
	require.NoError(t, err)
	require.Equal(t, revision, child.Spec.InputRevision)
	require.NotContains(t, child.Annotations, "lpx.nvidia.com/podcliqueset-name")
	pending := mergeLPXChildStatus(source, child, ReconcileResult{State: v1beta1.DGDStatePending})
	require.Equal(t, v1beta1.ComponentKindPodCliqueScalingGroup, pending.ComponentStatus["lpx"].ComponentKind)

	t.Log("Reject foreign references before returning a child for status projection")
	for _, invalidIdentity := range []string{"foreign-owner", "missing-owner"} {
		foreign := child.DeepCopy()
		switch invalidIdentity {
		case "foreign-owner":
			foreign.OwnerReferences[0].UID = "another-owner"
		case "missing-owner":
			foreign.OwnerReferences = nil
		}
		foreignClient := fake.NewClientBuilder().WithScheme(kube.Scheme()).WithObjects(foreign).Build()
		_, err := (&dgdLPXHandoff{client: foreignClient}).Reconcile(t.Context(), source)
		require.ErrorContains(t, err, "adoption is not supported", invalidIdentity)
	}

	t.Log("Refuse adoption after the source name is reused with a different UID")
	source.UID = "replacement-source"
	_, err = handoff.Reconcile(t.Context(), source)
	require.ErrorContains(t, err, "adoption is not supported")

	t.Log("Do not block finalization on or delete the child owned by the prior source")
	require.NoError(t, handoff.Finalize(t.Context(), source))
	stored := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
	require.True(t, stored.DeletionTimestamp.IsZero())
}

func TestLPXChildStatusRequiresObservedResultsAndCompleteEngine(t *testing.T) {
	t.Log("Project exactly one complete LPX engine alongside an ordinary component")
	child, source, _ := newLPXHandoffFixture(t, "node-local-v2-hybrid")
	child.Status.ObservedGeneration = child.Generation
	child.Status.Components = map[string]v1alpha1.LPXComponentStatus{"lpx": {
		ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{
			Replicas: 3, AvailableReplicas: ptr.To(int32(3)),
			GPUsPerEngine: ptr.To(int64(8)), GPUsPerReplica: ptr.To(int64(8)),
		},
	}}
	source.Status.Placement = &v1beta1.PlacementStatus{Score: ptr.To(0.92), State: v1beta1.PlacementScoreStateReported}
	meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: child.Generation, Reason: "Ready"})
	result := ReconcileResult{State: v1beta1.DGDStateSuccessful, ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{"prefill": {Replicas: 4, AvailableReplicas: ptr.To(int32(4))}}}
	result = mergeLPXChildStatus(source, child, result)
	require.Equal(t, v1beta1.DGDStateSuccessful, result.State)
	require.Len(t, result.ComponentStatus, 2)
	require.Equal(t, child.Status.Components["lpx"].ComponentReplicaStatus, result.ComponentStatus["lpx"])

	t.Log("Reject missing or stale children and stale Ready conditions")
	for _, failure := range []string{"missing-child", "generation", "condition"} {
		stale := child.DeepCopy()
		switch failure {
		case "missing-child":
			stale = nil
		case "generation":
			stale.Status.ObservedGeneration--
		case "condition":
			stale.Status.Conditions[0].ObservedGeneration--
		}
		pending := mergeLPXChildStatus(source, stale, ReconcileResult{State: v1beta1.DGDStateSuccessful})
		require.Equal(t, v1beta1.DGDStatePending, pending.State, failure)
	}

	t.Log("A partially ready child retains logical replica counts and GPU capacity in public status")
	partial := child.Status.Components["lpx"]
	partial.AvailableReplicas = ptr.To(int32(2))
	child.Status.Components["lpx"] = partial
	result.Reason = "CheckpointReady"
	meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionFalse, ObservedGeneration: child.Generation, Reason: v1alpha1.LPXReadyReasonPending, Message: "Waiting for removed LPX requests"})
	result = mergeLPXChildStatus(source, child, result)
	require.Equal(t, v1beta1.DGDStatePending, result.State)
	require.Equal(t, Reason(v1alpha1.LPXReadyReasonPending), result.Reason)
	require.Equal(t, partial.ComponentReplicaStatus, result.ComponentStatus["lpx"])
	source.Status.Components = result.ComponentStatus
	request := &v1beta1.DynamoGraphDeploymentRequest{}
	require.True(t, updateDeploymentInfo(request, source))
	require.Equal(t, ptr.To(int32(7)), request.Status.DeploymentInfo.Replicas)
	require.Equal(t, ptr.To(int32(6)), request.Status.DeploymentInfo.AvailableReplicas)

	t.Log("Retirement remains pending in final DGD status after ordinary checkpoint readiness")
	projected := newWorkloadProgramResult(source)
	projected.applyReconcileResult(source.Generation, result)
	require.Equal(t, v1beta1.DGDStatePending, projected.Status.State)
	require.Equal(t, source.Status.Placement, projected.Status.Placement)
	ready := meta.FindStatusCondition(projected.Status.Conditions, "Ready")
	require.NotNil(t, ready)
	require.Equal(t, metav1.ConditionFalse, ready.Status)
	require.Equal(t, child.Status.Conditions[0].Reason, ready.Reason)
	require.Equal(t, child.Status.Conditions[0].Message, ready.Message)
	require.Equal(t, source.Generation, ready.ObservedGeneration)
}

func TestLPXFailureProjectionRequiresCurrentCondition(t *testing.T) {
	for _, test := range []struct {
		name   string
		failed bool
	}{
		{"unobserved-result", true},
		{"observed-result", true},
		{"old-condition", false},
		{"cleared-condition", false},
		{"deleting", false},
		{"ordinary-failure", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Only a current child failure may bypass the complete-observation gate")
			child, source, _ := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
			child.Status.Components = map[string]v1alpha1.LPXComponentStatus{"lpx": {ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 1}}}
			child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionFalse, ObservedGeneration: child.Generation, Reason: v1alpha1.LPXReadyReasonFailed, Message: "Check the namespace quota"}}
			result := ReconcileResult{State: v1beta1.DGDStateSuccessful}
			status := v1beta1.DynamoGraphDeploymentStatus{}
			switch test.name {
			case "observed-result":
				child.Status.ObservedGeneration = child.Generation
				result = ReconcileResult{State: v1beta1.DGDStatePending, Reason: "RolloutInProgress"}
				status.RollingUpdate = &v1beta1.RollingUpdateStatus{Phase: v1beta1.RollingUpdatePhaseInProgress}
			case "old-condition":
				child.Status.Conditions[0].ObservedGeneration--
			case "cleared-condition":
				child.Status.Conditions[0].Reason = v1alpha1.LPXReadyReasonPending
			case "deleting":
				child.DeletionTimestamp = ptr.To(metav1.Now())
			case "ordinary-failure":
				result = ReconcileResult{State: v1beta1.DGDStateFailed, Reason: "OrdinaryFailure", Message: "Existing ordinary failure"}
			}
			result = mergeLPXChildStatus(source, child, result)
			require.Equal(t, test.failed, result.State == v1beta1.DGDStateFailed)
			if test.name == "ordinary-failure" {
				require.Equal(t, Reason("OrdinaryFailure"), result.Reason)
				require.Equal(t, Message("Existing ordinary failure"), result.Message)
			} else if test.failed {
				require.Equal(t, Reason(child.Status.Conditions[0].Reason), result.Reason)
				require.Equal(t, Message(child.Status.Conditions[0].Message), result.Message)
			} else {
				require.Equal(t, Reason("LPXChildPending"), result.Reason)
			}

			t.Log("A failure condition does not make stale result payloads current")
			if test.name == "observed-result" {
				require.Equal(t, int32(1), result.ComponentStatus["lpx"].Replicas)

				t.Log("A retained ordinary rollout cannot hide the child failure in final DGD status")
				projected := workloadProgramResult{Status: status}
				projected.applyReconcileResult(source.Generation, result)
				require.Equal(t, v1beta1.DGDStateFailed, projected.Status.State)
				ready := meta.FindStatusCondition(projected.Status.Conditions, "Ready")
				require.NotNil(t, ready)
				require.Equal(t, metav1.ConditionFalse, ready.Status)
				require.Equal(t, child.Status.Conditions[0].Reason, ready.Reason)
				require.Equal(t, child.Status.Conditions[0].Message, ready.Message)
				require.Equal(t, source.Generation, ready.ObservedGeneration)
			} else {
				require.Zero(t, result.ComponentStatus["lpx"].Replicas)
			}
		})
	}
}

func TestLPXRestartUsesPersistedSelectionAndCurrentChildStatus(t *testing.T) {
	for _, strategy := range []v1beta1.RestartStrategyType{v1beta1.RestartStrategyTypeParallel} {
		t.Run(string(strategy), func(t *testing.T) {
			t.Log("Request an LPX-only restart without requiring an ordinary PCS")
			_, source, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
			source.Spec.Restart = &v1beta1.Restart{ID: "restart-1", Strategy: &v1beta1.RestartStrategy{Type: strategy}}
			handoff := &dgdLPXHandoff{client: kube}
			unchanged, err := handoff.Reconcile(t.Context(), source)
			require.NoError(t, err)
			require.Empty(t, unchanged.Annotations[dynamo.LPXRestartAnnotation])
			program := (&DynamoGraphDeploymentReconciler{
				Client: kube, RuntimeConfig: &commoncontroller.RuntimeConfig{},
			}).newGroveProgram()
			ordinaryDGD := projectWithoutExternallyManagedComponents(source)
			progress := func(ctx context.Context, source *v1beta1.DynamoGraphDeployment, inProgress []string) []string {
				return resolveCompositeGroveRestartProgress(
					ctx, source, ordinaryDGD, inProgress, program.restartProgress, program.lpxRestartProgress,
				)
			}
			restart := newDGDRestartReconciler().Resolve(t.Context(), source, &source.Status, progress)
			require.Equal(t, []string{"lpx"}, restart.Status.InProgress)

			t.Log("Deliver the token only after the DGD has persisted its component selection")
			source.Status.Restart = restart.Status
			updated, err := handoff.Reconcile(t.Context(), source)
			require.NoError(t, err)
			require.Equal(t, "restart-1", updated.Annotations[dynamo.LPXRestartAnnotation])
			require.Equal(t, []string{"lpx"}, progress(t.Context(), source, []string{"lpx"}))
			updated.Status.ObservedGeneration = updated.Generation
			updated.Status.Components = map[string]v1alpha1.LPXComponentStatus{"lpx": {
				ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 1, AvailableReplicas: ptr.To(int32(1))},
				Conditions:             []metav1.Condition{{Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: updated.Generation}},
			}}
			meta.SetStatusCondition(&updated.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: updated.Generation, Reason: "Ready"})
			require.NoError(t, kube.Status().Update(t.Context(), updated))

			t.Log("Complete the restart only from the current child's full-engine readiness")
			require.Empty(t, progress(t.Context(), source, []string{"lpx"}))
			completed := newDGDRestartReconciler().Resolve(t.Context(), source, &source.Status, progress)
			require.Equal(t, v1beta1.RestartPhaseCompleted, completed.Status.Phase)
			updated.Status.ObservedGeneration--
			require.NoError(t, kube.Status().Update(t.Context(), updated))
			require.Equal(t, []string{"lpx"}, progress(t.Context(), source, []string{"lpx"}))

			t.Log("A second request cannot borrow the completed first restart's selection")
			source.Status.Restart = completed.Status
			source.Spec.Restart.ID = "restart-2"
			source.Generation++
			revision, err := dynamo.LPXInputRevision(source, updated.Annotations[dynamo.LPXRestartAnnotation])
			require.NoError(t, err)
			require.Equal(t, revision, updated.Spec.InputRevision)
			unchanged, err = handoff.Reconcile(t.Context(), source)
			require.NoError(t, err)
			require.Equal(t, updated, unchanged)

			t.Log("Deliver the second restart only after its own selection is persisted")
			source.Status.Restart = newDGDRestartReconciler().Resolve(t.Context(), source, &source.Status, progress).Status
			next, err := handoff.Reconcile(t.Context(), source)
			require.NoError(t, err)
			require.Equal(t, "restart-2", next.Annotations[dynamo.LPXRestartAnnotation])
			require.NotEqual(t, updated.Spec.InputRevision, next.Spec.InputRevision)
		})
	}
}

func TestParallelRestartStartsAllLPXAndOrdinaryWorkloadsBeforeReadiness(t *testing.T) {
	t.Log("Request one parallel restart of two independent LPX workloads and a frontend")
	child, source, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
	require.NoError(t, rbacv1.AddToScheme(kube.Scheme()))
	second := source.Spec.Components[0].DeepCopy()
	second.ComponentName = "second"
	source.Spec.Components = append(source.Spec.Components, *second, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend"}}}},
	})
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	source.Spec.Restart = &v1beta1.Restart{ID: "parallel-restart", Strategy: &v1beta1.RestartStrategy{Type: v1beta1.RestartStrategyTypeParallel}}
	require.NoError(t, kube.Update(t.Context(), source))
	config := &configv1alpha1.OperatorConfiguration{}
	config.Namespace.Restricted = source.Namespace
	program := (&DynamoGraphDeploymentReconciler{
		Client: kube, Config: config, Recorder: events.NewFakeRecorder(100),
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true}},
	}).newGroveProgram()

	t.Log("Persist selection of all three components before publishing their restart")
	result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, []string{"frontend", "lpx", "second"}, result.Status.Restart.InProgress)
	source.Status = result.Status
	require.NoError(t, kube.Status().Update(t.Context(), source))

	t.Log("Deliver the same token to LPX and the ordinary Pod template while every workload is pending")
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.DGDStatePending, result.Status.State)
	require.Equal(t, []string{"frontend", "lpx", "second"}, result.Status.Restart.InProgress)
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Equal(t, source.Spec.Restart.ID, child.Annotations[dynamo.LPXRestartAnnotation])
	require.Empty(t, child.Status.Components)
	ordinary := projectWithoutExternallyManagedComponents(source)
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{
		Namespace: source.Namespace, Name: dynamo.PCSNameForDGD(ordinary.Name, ordinary.Spec.Components),
	}, pcs))
	require.Len(t, pcs.Spec.Template.Cliques, 1)
	require.Equal(t, source.Spec.Restart.ID, pcs.Spec.Template.Cliques[0].Annotations[consts.RestartAnnotation])
	require.Nil(t, pcs.Status.ObservedGeneration)
	source.Status = result.Status
	require.NoError(t, kube.Status().Update(t.Context(), source))

	t.Log("One ready LPX workload cannot finish the restart while the second is pending")
	child.Status.ObservedGeneration = child.Generation
	child.Status.Components = map[string]v1alpha1.LPXComponentStatus{"lpx": {
		ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 1, AvailableReplicas: ptr.To(int32(1))},
		Conditions:             []metav1.Condition{{Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: child.Generation}},
	}}
	require.NoError(t, kube.Status().Update(t.Context(), child))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.RestartPhaseRestarting, result.Status.Restart.Phase)
	require.Contains(t, result.Status.Restart.InProgress, "second")

	t.Log("Both ready LPX workloads still wait for the ordinary component")
	child.Status.Components["second"] = child.Status.Components["lpx"]
	meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{
		Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: child.Generation, Reason: "Ready",
	})
	require.NoError(t, kube.Status().Update(t.Context(), child))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.RestartPhaseRestarting, result.Status.Restart.Phase)
	require.Equal(t, []string{"frontend"}, result.Status.Restart.InProgress)
	source.Status = result.Status
	require.NoError(t, kube.Status().Update(t.Context(), source))

	t.Log("Complete only after Grove observes the frontend's ready replacement")
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
	require.NoError(t, kube.Update(t.Context(), pcs))
	require.NoError(t, kube.Create(t.Context(), &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: dynamo.GroveComponentResourceName(ordinary, "frontend"), Namespace: source.Namespace, Generation: 1},
		Spec:       grovev1alpha1.PodCliqueSpec{Replicas: 1},
		Status:     grovev1alpha1.PodCliqueStatus{ObservedGeneration: ptr.To(int64(1)), Replicas: 1, UpdatedReplicas: 1, ReadyReplicas: 1},
	}))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.RestartPhaseCompleted, result.Status.Restart.Phase)
	require.Empty(t, result.Status.Restart.InProgress)
}

func TestLPXHandoffOrdinaryScalingPreservesReadiness(t *testing.T) {
	t.Log("Seed the shared child's current readiness")
	child, source, kube := newLPXHandoffFixture(t, "node-local-v2-specdecode")
	source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1)),
	}, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "frontend:0"}}}},
	})
	child.Status.ObservedGeneration = child.Generation
	child.Status.Components = map[string]v1alpha1.LPXComponentStatus{
		"lpx": {
			ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 1, AvailableReplicas: ptr.To(int32(1))},
		},
		"draft": {
			ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 2, AvailableReplicas: ptr.To(int32(2))},
		},
	}
	meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: child.Generation, Reason: "Ready"})
	require.NoError(t, kube.Status().Update(t.Context(), child))
	beforeChild := child.DeepCopy()
	handoff := &dgdLPXHandoff{client: kube}

	t.Log("Reorder the pair and make five frontend-only edits without changing the child")
	source.Spec.Components[0], source.Spec.Components[1] = source.Spec.Components[1], source.Spec.Components[0]
	for edit := range 5 {
		source.GetComponentByName("frontend").PodTemplate.Spec.Containers[0].Image = "frontend:" + strconv.Itoa(edit+1)
		source.Generation++
		beforeSource := source.DeepCopy()
		current, err := handoff.Reconcile(t.Context(), source)
		require.NoError(t, err)
		require.Equal(t, beforeChild, current)
		ordinary := ReconcileResult{State: v1beta1.DGDStateSuccessful, Reason: "Ready"}
		projected := newWorkloadProgramResult(source)
		ordinary = mergeLPXChildStatus(source, current, ordinary)
		projected.applyReconcileResult(source.Generation, ordinary)
		require.Equal(t, v1beta1.DGDStateSuccessful, projected.Status.State)
		require.True(t, meta.IsStatusConditionTrue(projected.Status.Conditions, "Ready"))
		require.Equal(t, source.Generation, projected.Status.ObservedGeneration)
		require.Equal(t, beforeSource, source)
		source.Status = projected.Status
	}
	require.Equal(t, int64(8), source.Generation)

	t.Log("Scale prefill without changing the child or its readiness")
	source.GetComponentByName("prefill").Replicas = ptr.To(int32(4))
	source.Generation++
	current, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.Equal(t, beforeChild, current)

	t.Log("LPX capacity updates the handoff without modifying its child's durable attempt")
	source.GetComponentByName("draft").Replicas = ptr.To(int32(3))
	source.Generation++
	updated, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.NotEqual(t, beforeChild.Spec.InputRevision, updated.Spec.InputRevision)
	require.Equal(t, strconv.FormatInt(source.Generation, 10), updated.Annotations[lpx.DGDGenerationAnnotation])
	require.Equal(t, beforeChild.Status, updated.Status)
	// Fake clients do not advance generations; the isolated API test covers that transition.
	updated.Generation++
	pending := mergeLPXChildStatus(source, updated, ReconcileResult{State: v1beta1.DGDStateSuccessful})
	require.Equal(t, v1beta1.DGDStatePending, pending.State)
}

func TestSpecDecodeRestartRollsTheSharedChildOnce(t *testing.T) {
	for _, scenario := range []struct {
		name     string
		strategy v1beta1.RestartStrategy
	}{
		{"parallel", v1beta1.RestartStrategy{Type: v1beta1.RestartStrategyTypeParallel}},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Persist a requested restart of the authored draft and target")
			_, source, kube := newLPXHandoffFixture(t, "node-local-v2-specdecode")
			source.Spec.Restart = &v1beta1.Restart{ID: "pair-restart", Strategy: &scenario.strategy}
			var childReads, pcsReads int
			var failChildRead bool
			observed := interceptor.NewClient(kube.(client.WithWatch), interceptor.Funcs{
				Get: func(ctx context.Context, reader client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
					if _, pcs := obj.(*grovev1alpha1.PodCliqueSet); pcs {
						pcsReads++
					}
					if _, child := obj.(*v1alpha1.LPXGraphDeployment); child {
						childReads++
						if failChildRead {
							return apierrors.NewServiceUnavailable("child observation unavailable")
						}
					}
					return reader.Get(ctx, key, obj, opts...)
				},
			})
			program := (&DynamoGraphDeploymentReconciler{
				Client: observed, RuntimeConfig: &commoncontroller.RuntimeConfig{},
			}).newGroveProgram()
			ordinaryDGD := projectWithoutExternallyManagedComponents(source)
			progress := func(ctx context.Context, source *v1beta1.DynamoGraphDeployment, inProgress []string) []string {
				return resolveCompositeGroveRestartProgress(
					ctx, source, ordinaryDGD, inProgress, program.restartProgress, program.lpxRestartProgress,
				)
			}
			restarter := newDGDRestartReconciler()
			source.Status.Restart = restarter.Resolve(t.Context(), source, &source.Status, progress).Status
			require.NotEmpty(t, source.Status.Restart.InProgress)

			t.Log("Both selected members deliver one token to the same child")
			handoff := &dgdLPXHandoff{client: kube}
			child, err := handoff.Reconcile(t.Context(), source)
			require.NoError(t, err)
			require.Equal(t, "pair-restart", child.Annotations[dynamo.LPXRestartAnnotation])
			require.NotEmpty(t, progress(t.Context(), source, source.Status.Restart.InProgress))
			child.Status.ObservedGeneration = child.Generation
			child.Status.Components = map[string]v1alpha1.LPXComponentStatus{
				"lpx": {
					ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 1},
					Conditions:             []metav1.Condition{{Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: child.Generation}},
				},
				"draft": {
					ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 2},
					Conditions:             []metav1.Condition{{Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: child.Generation}},
				},
			}
			meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: child.Generation, Reason: "Ready"})
			require.NoError(t, kube.Status().Update(t.Context(), child))
			before := child.DeepCopy()

			t.Log("One failed child observation keeps both ready members pending")
			failChildRead, childReads = true, 0
			require.Equal(t, []string{"draft", "lpx"}, progress(t.Context(), source, []string{"draft", "lpx"}))
			require.Equal(t, 1, childReads)
			failChildRead = false

			t.Log("Finish each selected member without changing the shared child revision again")
			for range 2 {
				childReads = 0
				require.Empty(t, progress(t.Context(), source, source.Status.Restart.InProgress))
				require.Equal(t, min(len(source.Status.Restart.InProgress), 1), childReads)
				source.Status.Restart = restarter.Resolve(t.Context(), source, &source.Status, progress).Status
				current, err := handoff.Reconcile(t.Context(), source)
				require.NoError(t, err)
				require.Equal(t, before, current)
			}
			require.Equal(t, v1beta1.RestartPhaseCompleted, source.Status.Restart.Phase)

			t.Log("A missing draft status cannot borrow the target's readiness")
			delete(child.Status.Components, "draft")
			require.NoError(t, kube.Status().Update(t.Context(), child))
			require.Equal(t, []string{"draft"}, progress(t.Context(), source, []string{"draft", "lpx"}))
			require.Zero(t, pcsReads)

			t.Log("A missing ordinary PCS keeps only live ordinary restart members pending")
			source.Spec.Components = append(source.Spec.Components,
				v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1))},
				v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			)
			ordinaryDGD = projectWithoutExternallyManagedComponents(source)
			child.Status.Components["draft"] = v1alpha1.LPXComponentStatus{
				ComponentReplicaStatus: v1beta1.ComponentReplicaStatus{Replicas: 2},
				Conditions:             []metav1.Condition{{Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionTrue, ObservedGeneration: child.Generation}},
			}
			require.NoError(t, kube.Status().Update(t.Context(), child))
			requested := []string{"draft", "frontend", "lpx", "prefill", "removed"}
			beforeSource := source.DeepCopy()
			childReads, pcsReads = 0, 0
			require.Equal(t, []string{"frontend", "prefill"}, progress(t.Context(), source, requested))
			require.Equal(t, 1, childReads)
			require.Equal(t, 1, pcsReads)

			t.Log("An unobserved ordinary PCS cannot hold back either ready LPX member")
			pcs := &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: dynamo.PCSNameForDGD(ordinaryDGD.Name, ordinaryDGD.Spec.Components), Namespace: source.Namespace, Generation: 2},
				Status:     grovev1alpha1.PodCliqueSetStatus{ObservedGeneration: ptr.To(int64(1))},
			}
			require.NoError(t, kube.Create(t.Context(), pcs))
			childReads, pcsReads = 0, 0
			require.Equal(t, []string{"frontend", "prefill"}, progress(t.Context(), source, requested))
			require.Equal(t, 1, childReads)
			require.Equal(t, 1, pcsReads)

			t.Log("A failed child read cannot hold back a ready ordinary member or reorder pending members")
			pcs.Status.ObservedGeneration = ptr.To(pcs.Generation)
			require.NoError(t, kube.Update(t.Context(), pcs))
			prefill := &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{Name: dynamo.GroveComponentResourceName(ordinaryDGD, "prefill"), Namespace: source.Namespace, Generation: 1},
				Spec:       grovev1alpha1.PodCliqueSpec{Replicas: 1},
				Status:     grovev1alpha1.PodCliqueStatus{ObservedGeneration: ptr.To(int64(1)), Replicas: 1, UpdatedReplicas: 1, ReadyReplicas: 1},
			}
			require.NoError(t, kube.Create(t.Context(), prefill))
			failChildRead, childReads, pcsReads = true, 0, 0
			require.Equal(t, []string{"draft", "frontend", "lpx"}, progress(t.Context(), source, requested))
			require.Equal(t, 1, childReads)
			require.Equal(t, 1, pcsReads)
			require.Equal(t, beforeSource, source)
			require.Equal(t, []string{"draft", "frontend", "lpx", "prefill", "removed"}, requested)
		})
	}
}

func TestLPXRemovalSurvivesOrdinaryReconcileFailure(t *testing.T) {
	for _, test := range []struct {
		name      string
		retainLPX bool
		childGone bool
	}{
		{name: "remove"},
		{name: "retain", retainLPX: true},
		{name: "pending child already gone", childGone: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Edit a mixed DGD while ordinary shared-resource reconciliation will fail")
			child, source, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
			if test.childGone {
				require.NoError(t, kube.Delete(t.Context(), child))
			} else {
				child.Finalizers = []string{"test.example/child-cleanup"}
				require.NoError(t, kube.Update(t.Context(), child))
			}
			if !test.retainLPX {
				source.Spec.Components = nil
			} else {
				source.Spec.Components[0].Replicas = ptr.To(int32(2))
			}
			source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill,
			})
			source.Generation++
			require.NoError(t, kube.Update(t.Context(), source))
			source.Status.Placement = &v1beta1.PlacementStatus{Score: ptr.To(0.92), State: v1beta1.PlacementScoreStateReported}
			source.Status.Components = map[string]v1beta1.ComponentReplicaStatus{
				"lpx": {Replicas: 1}, "prefill": {Replicas: 2},
			}
			if test.childGone {
				source.Status.Components["lpx"] = v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindPodCliqueScalingGroup}
			}
			previousStatus := source.Status.DeepCopy()
			program := (&DynamoGraphDeploymentReconciler{
				Client: kube, Config: &configv1alpha1.OperatorConfiguration{}, Recorder: events.NewFakeRecorder(10),
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true}},
			}).newGroveProgram()

			t.Log("Preserve the ordinary error while deleting only a deselected child")
			result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
			require.ErrorContains(t, err, "RBAC manager not initialized")
			require.Equal(t, v1beta1.DGDStateFailed, result.Status.State)
			stored := &v1alpha1.LPXGraphDeployment{}
			if test.childGone {
				require.True(t, apierrors.IsNotFound(kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored)))
			} else {
				require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
				require.Equal(t, !test.retainLPX, !stored.DeletionTimestamp.IsZero())
				require.Equal(t, child.Spec, stored.Spec)
			}
			require.Equal(t, previousStatus, &source.Status)
			require.Equal(t, source.Status.Components["prefill"], result.Status.Components["prefill"])
			if !test.retainLPX {
				require.NotContains(t, result.Status.Components, "lpx")
			} else {
				require.Equal(t, source.Status.Components, result.Status.Components)
			}
			require.Equal(t, source.Status.Placement, result.Status.Placement)
		})
	}
}

func TestLPXChildFinalizationAndDeselectionWaitForCleanup(t *testing.T) {
	t.Log("Deselect LPX while the child reports unfinished cleanup through its finalizer")
	child, source, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
	child.Finalizers = []string{"test.example/child-cleanup"}
	require.NoError(t, kube.Update(t.Context(), child))
	handoff := &dgdLPXHandoff{client: kube}
	source.Spec.Components = nil
	deleting, err := handoff.Reconcile(t.Context(), source)
	require.NoError(t, err)
	require.NotNil(t, deleting)
	require.ErrorContains(t, handoff.Finalize(t.Context(), source), "finish cleanup")
	stored := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
	require.False(t, stored.DeletionTimestamp.IsZero())

	t.Log("The outer DGD finalizer must wait for LPX before inspecting checkpoint cleanup")
	snapshotReads := 0
	observed := interceptor.NewClient(kube.(client.WithWatch), interceptor.Funcs{
		List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, snapshot := list.(*snapshotv1alpha1.SnapshotJobList); snapshot {
				snapshotReads++
			}
			return delegated.List(ctx, list, opts...)
		},
	})
	parent := &DynamoGraphDeploymentReconciler{Client: observed}
	require.ErrorContains(t, parent.FinalizeResource(t.Context(), source), "waiting for LPXGraphDeployment")
	require.Zero(t, snapshotReads)

	t.Log("Finish parent finalization only after the child releases its cleanup finalizer")
	stored.Finalizers = nil
	require.NoError(t, kube.Update(t.Context(), stored))
	require.NoError(t, handoff.Finalize(t.Context(), source))
	require.True(t, apierrors.IsNotFound(kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored)))
}

func TestLPXPendingDownloadDoesNotBlockOrdinaryWorkloads(t *testing.T) {
	t.Log("Leave the child pending while the DGD desires an ordinary prefill workload")
	_, source, kube := newLPXHandoffFixture(t, "node-local-v2-lpu-only")
	require.NoError(t, rbacv1.AddToScheme(kube.Scheme()))
	source.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(configv1alpha1.DiscoveryBackendKubernetes)
	source.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
	source.Spec.Annotations = map[string]string{"example.com/model-discovery": "enabled"}
	source.Spec.Labels = map[string]string{"example.com/policy": "enabled"}
	source.Spec.BackendFramework = string(dynamo.BackendFrameworkVLLM)
	source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "vllm-runtime"}}}},
		ProviderOverride: &v1beta1.ProviderOverride{
			APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Target: "PodCliqueTemplateSpec",
			Value: apiextensionsv1.JSON{Raw: []byte(`{"topologyConstraint":{"topologyName":"fabric","pack":{"required":"rack"}}}`)},
		},
	})
	require.NoError(t, kube.Update(t.Context(), source))
	child, err := (&dgdLPXHandoff{client: kube}).Reconcile(t.Context(), source)
	require.NoError(t, err)
	child.Status.ObservedGeneration = child.Generation
	child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionFalse, ObservedGeneration: child.Generation, Reason: v1alpha1.LPXReadyReasonPending, Message: "Waiting for model downloads"}}
	require.NoError(t, kube.Status().Update(t.Context(), child))
	config := &configv1alpha1.OperatorConfiguration{}
	config.Namespace.Restricted = source.Namespace
	parent := &DynamoGraphDeploymentReconciler{
		Client: kube, Config: config, Recorder: events.NewFakeRecorder(100),
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true, Checkpoint: true}},
	}
	program := parent.newGroveProgram()

	t.Log("Publish ordinary resources independently of the child's pending download")
	result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.DGDStatePending, result.Status.State)
	pcs := &grovev1alpha1.PodCliqueSet{}
	ordinaryDGD := projectWithoutExternallyManagedComponents(source)
	key := client.ObjectKey{Namespace: source.Namespace, Name: dynamo.PCSNameForDGD(ordinaryDGD.Name, ordinaryDGD.Spec.Components)}
	require.NoError(t, kube.Get(t.Context(), key, pcs))
	require.True(t, metav1.IsControlledBy(pcs, source))
	require.Len(t, pcs.Spec.Template.Cliques, 1)
	require.Equal(t, "prefill", pcs.Spec.Template.Cliques[0].Name)
	require.Equal(t, &grovev1alpha1.TopologyConstraint{
		TopologyName: "fabric", Pack: &grovev1alpha1.TopologyPackConstraint{RequiredDomain: "rack"},
	}, pcs.Spec.Template.Cliques[0].TopologyConstraint)

	t.Log("Model discovery remains DGD-owned while the LPX child is downloading")
	service := &corev1.Service{}
	serviceKey := client.ObjectKey{Namespace: source.Namespace, Name: dynamo.GenerateServiceName("test/model")}
	require.NoError(t, kube.Get(t.Context(), serviceKey, service))
	require.Equal(t, corev1.ClusterIPNone, service.Spec.ClusterIP)
	require.False(t, service.Spec.PublishNotReadyAddresses)
	require.Equal(t, map[string]string{consts.KubeLabelDynamoBaseModelHash: dynamo.HashModelName("test/model")}, service.Spec.Selector)
	require.Equal(t, dynamo.HashModelName("test/model"), service.Labels[consts.KubeLabelDynamoBaseModelHash])
	require.Equal(t, "enabled", service.Annotations["example.com/model-discovery"])
	require.True(t, metav1.IsControlledBy(service, source))

	t.Log("Observe the ordinary PCS from a fresh parent reconcile before committing its worker hash")
	freshSource := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(source), freshSource))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: freshSource})
	require.NoError(t, err)
	require.Equal(t, v1beta1.DGDStatePending, result.Status.State)
	source = freshSource

	t.Log("A pending ordinary checkpoint retains its startup and scaling gates alongside the pending child")
	prefill := source.GetComponentByName("prefill")
	prefill.Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
		Enabled: true, CheckpointRef: ptr.To(friendlyCheckpointName), StartupPolicy: v1beta1.CheckpointStartupPolicyWaitForCheckpoint,
	}}
	prefill.ScalingAdapter = &v1beta1.ScalingAdapter{}
	require.NoError(t, kube.Update(t.Context(), source))
	compatibilityHash, err := program.sharedResources.checkpoints.snapshotCompatibilityHashForComponent(
		source,
		"prefill",
		prefill,
	)
	require.NoError(t, err)
	require.NotEmpty(t, compatibilityHash)
	snapshot := dgdTestPodSnapshot(friendlyCheckpointName, compatibilityHash, false)
	snapshot.Namespace = source.Namespace
	require.NoError(t, kube.Create(t.Context(), snapshot))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.DGDStatePending, result.Status.State)
	require.Equal(t, string(reasonWaitingForCheckpoint), meta.FindStatusCondition(result.Status.Conditions, "Ready").Reason)
	require.Equal(t, v1beta1.ComponentCheckpointStatus{CheckpointName: friendlyCheckpointName}, result.Status.Checkpoints["prefill"])
	require.NoError(t, kube.Get(t.Context(), key, pcs))
	require.Zero(t, pcs.Spec.Template.Cliques[0].Spec.Replicas)
	adapters := &v1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	require.NoError(t, kube.List(t.Context(), adapters))
	require.Empty(t, adapters.Items)

	t.Log("An actionable child failure remains visible while the ordinary checkpoint is still pending")
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	meta.SetStatusCondition(&child.Status.Conditions, metav1.Condition{
		Type: "Ready", Status: metav1.ConditionFalse, ObservedGeneration: child.Generation,
		Reason: v1alpha1.LPXReadyReasonFailed, Message: "publish LPX workload: storage volume mount is missing",
	})
	require.NoError(t, kube.Status().Update(t.Context(), child))
	result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: source})
	require.NoError(t, err)
	require.Equal(t, v1beta1.DGDStateFailed, result.Status.State)
	failed := meta.FindStatusCondition(result.Status.Conditions, "Ready")
	require.Equal(t, v1alpha1.LPXReadyReasonFailed, failed.Reason)
	require.Contains(t, failed.Message, "storage volume mount is missing")
	require.Equal(t, v1beta1.ComponentCheckpointStatus{CheckpointName: friendlyCheckpointName}, result.Status.Checkpoints["prefill"])

	t.Log("Removing the last ordinary component deletes its PCS without disturbing the child")
	source.Spec.Components = source.Spec.Components[:1]
	for range 2 {
		_, err = program.workloads.Reconcile(t.Context(), source, projectWithoutExternallyManagedComponents(source), nil, nil)
		require.NoError(t, err)
		require.True(t, apierrors.IsNotFound(kube.Get(t.Context(), key, pcs)))
	}
	stored := &v1alpha1.LPXGraphDeployment{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(child), stored))
	require.Equal(t, child, stored)
}
