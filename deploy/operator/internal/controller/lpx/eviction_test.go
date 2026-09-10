/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"sort"
	"strings"
	"testing"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	lpuEvictionTestNamespace = "test-ns"
	lpuEvictionTestDGD       = "test-dgd"
	lpuEvictionTestComponent = "lpx-worker"
	lpuEvictionTestPCSG      = "lpx-worker"
	lpuEvictionTestModel     = "test-model"
)

func newLPUEvictionReconciler(objs ...client.Object) (*lpuEvictionReconciler, client.Client) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)

	cb := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&corev1.Pod{}).
		WithObjects(newLPUEvictionConfigMap())
	for _, o := range objs {
		cb = cb.WithObjects(o)
	}
	c := cb.Build()

	return &lpuEvictionReconciler{Client: c, Recorder: events.NewFakeRecorder(16)}, c
}

func newLPUEvictionConfigMap() *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: lpuEvictionTestDGD + "-lpu", Namespace: lpuEvictionTestNamespace},
		Data: map[string]string{
			"nodes_per_partition":    "2\n1",
			"partition_node_offsets": "0\n2",
		},
	}
}

func newLPUEvictionPod(name string, replicaIdx string, podIndex string, deleting bool, conditionReason string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: lpuEvictionTestNamespace,
			UID:       types.UID(name + "-uid"),
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName:    lpuEvictionTestDGD,
				commonconsts.KubeLabelDynamoComponent:              lpuEvictionTestComponent,
				commonconsts.KubeLabelDynamoComponentType:          commonconsts.ComponentTypeLPX,
				grovecommon.LabelPodCliqueScalingGroup:             lpuEvictionTestPCSG,
				grovecommon.LabelPodCliqueScalingGroupReplicaIndex: replicaIdx,
			},
			Annotations: map[string]string{
				commonconsts.AnnotationExtraResourcesHash: dynamolpx.LPUConfigMapHash(newLPUEvictionConfigMap()),
				lpxv1alpha1.PodRoleAnnotation:             lpxv1alpha1.PodRoleAgent,
				dynamolpx.WorkloadModeAnnotation:          string(lpxv1alpha1.WorkloadModeV2LPUOnly),
			},
		},
		Spec:   corev1.PodSpec{SchedulerName: dynamolpx.SchedulerName},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	if podIndex != "" {
		pod.Labels[grovecommon.LabelPodCliquePodIndex] = podIndex
	}
	pod.Annotations[lpxv1alpha1.PodModelAnnotation] = lpuEvictionTestModel
	if conditionReason != "" {
		pod.Status.Conditions = []corev1.PodCondition{{
			Type:   corev1.DisruptionTarget,
			Status: corev1.ConditionTrue,
			Reason: conditionReason,
		}}
	}
	if deleting {
		now := metav1.Now()
		pod.DeletionTimestamp = &now
		pod.Finalizers = []string{"test-finalizer"}
	}
	return pod
}

func newLPUOnlyConductor() *corev1.Pod {
	pod := newLPUEvictionPod("conductor-0", "0", "", false, "")
	pod.Annotations[lpxv1alpha1.PodRoleAnnotation] = lpxv1alpha1.PodRoleConductor
	return pod
}

func reconcileLPUEviction(t *testing.T, r *lpuEvictionReconciler, name string) ctrl.Result {
	t.Helper()

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: name, Namespace: lpuEvictionTestNamespace},
	})
	require.NoError(t, err)
	return result
}

func remainingLPUEvictionPods(t *testing.T, c client.Client) []string {
	t.Helper()

	var remaining corev1.PodList
	require.NoError(t, c.List(context.Background(), &remaining, client.InNamespace(lpuEvictionTestNamespace)))

	names := make([]string, 0, len(remaining.Items))
	for _, pod := range remaining.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		names = append(names, pod.Name)
	}
	sort.Strings(names)
	return names
}

func lpuEvictionPodNames(pods []corev1.Pod) []string {
	names := make([]string, 0, len(pods))
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	sort.Strings(names)
	return names
}

func TestLPUEviction_PodsForTrigger(t *testing.T) {
	t.Run("lpu-gpu rejects a runtime table newer than the trigger", func(t *testing.T) {
		trigger := newLPUEvictionPod("agent-0", "0", "0", false, "")
		trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
		trigger.Annotations[commonconsts.AnnotationExtraResourcesHash] = "stale"
		sibling := newLPUEvictionPod("agent-1", "0", "1", false, "")
		r, _ := newLPUEvictionReconciler(trigger, sibling)

		pods, err := r.podsForTrigger(context.Background(), trigger)

		require.ErrorContains(t, err, "does not match trigger pod")
		assert.Nil(t, pods)
	})

	t.Run("lpu-gpu excludes Pods from a concurrent rollout", func(t *testing.T) {
		trigger := newLPUEvictionPod("agent-0", "0", "0", false, "")
		trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
		samePartition := newLPUEvictionPod("agent-1", "0", "1", false, "")
		samePartition.Annotations[commonconsts.AnnotationExtraResourcesHash] = "other"
		r, _ := newLPUEvictionReconciler(trigger, samePartition)

		pods, err := r.podsForTrigger(context.Background(), trigger)

		require.NoError(t, err)
		assert.Equal(t, []string{"agent-0"}, lpuEvictionPodNames(pods))
	})

	t.Run("lpu-gpu pod without Grove index returns error", func(t *testing.T) {
		trigger := newLPUEvictionPod("agent-0", "0", "", false, "")
		trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
		sibling := newLPUEvictionPod("agent-1", "0", "1", false, "")
		r, _ := newLPUEvictionReconciler(trigger, sibling)

		pods, err := r.podsForTrigger(context.Background(), trigger)

		require.Error(t, err)
		assert.Nil(t, pods)
	})

	t.Run("lpu-gpu pod without model filters only by partition", func(t *testing.T) {
		trigger := newLPUEvictionPod("agent-0", "0", "0", false, "")
		trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
		delete(trigger.Annotations, lpxv1alpha1.PodModelAnnotation)
		sibling := newLPUEvictionPod("agent-1", "0", "0", false, "")
		r, _ := newLPUEvictionReconciler(trigger, sibling)

		pods, err := r.podsForTrigger(context.Background(), trigger)

		require.NoError(t, err)
		require.NotNil(t, pods)
		assert.Equal(t, []string{"agent-0", "agent-1"}, lpuEvictionPodNames(pods))
	})
}

func TestLPUEviction_RuntimePartitionTableUsesBoundedGeneratedName(t *testing.T) {
	t.Log("Create the generated runtime table for a maximum-length DGD name")
	dgdName := strings.Repeat("a", validation.DNS1123SubdomainMaxLength)
	config := newLPUEvictionConfigMap()
	config.Name = dynamolpx.LPUConfigMapName(dgdName)

	t.Log("Override the Agent config volume without replacing its generated partition table")
	trigger := newLPUEvictionPod("agent-0", "0", "0", false, "")
	trigger.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName] = dgdName
	trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
	trigger.Annotations[commonconsts.AnnotationExtraResourcesHash] = dynamolpx.LPUConfigMapHash(config)
	trigger.Spec.Volumes = []corev1.Volume{{
		Name: "config", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	}}
	r, _ := newLPUEvictionReconciler(trigger, config)

	t.Log("Read the bounded generated ConfigMap despite the non-ConfigMap volume override")
	partitions, err := r.runtimePartitionByPodIndex(t.Context(), trigger)
	require.NoError(t, err)
	require.Equal(t, map[string]int{"0": 0, "1": 0, "2": 1}, partitions)
}

func TestLPUEviction_LPUOnlyDeletesSamePCSGReplica(t *testing.T) {
	tests := []struct {
		name                string
		reason              string
		deletedConcurrently bool
	}{
		{name: "taint manager", reason: podDisruptionReasonTaintManagerDeletion},
		{name: "eviction api", reason: podDisruptionReasonEvictionAPI},
		{name: "deleted concurrently", reason: podDisruptionReasonTaintManagerDeletion, deletedConcurrently: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Observe a disrupted Agent alongside its conductor and another replica")
			trigger := newLPUEvictionPod("agent-0", "0", "0", true, tt.reason)
			conductor := newLPUOnlyConductor()
			agent := newLPUEvictionPod("agent-1", "0", "1", false, "")
			otherReplica := newLPUEvictionPod("agent-other-replica", "1", "0", false, "")

			r, c := newLPUEvictionReconciler(
				trigger,
				conductor,
				agent,
				otherReplica,
			)
			if tt.deletedConcurrently {
				r.Client = interceptor.NewClient(c.(client.WithWatch), interceptor.Funcs{
					Delete: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.DeleteOption) error {
						// Another actor removes the sibling before this reconciler's delete reaches the server.
						require.NoError(t, delegated.Delete(ctx, object, opts...))
						return delegated.Delete(ctx, object, opts...)
					},
				})
			}

			t.Log("Delete only the same replica's Agent and report only our successful deletion")
			result := reconcileLPUEviction(t, r, "agent-0")
			assert.Equal(t, ctrl.Result{}, result)
			assert.Equal(t, []string{"agent-other-replica", "conductor-0"}, remainingLPUEvictionPods(t, c))
			recorded := r.Recorder.(*events.FakeRecorder).Events
			if tt.deletedConcurrently {
				require.Empty(t, recorded)
			} else {
				require.Len(t, recorded, 1)
				require.Contains(t, <-recorded, "deleted 1 LPU pods")
			}

			t.Log("Reconcile the retained deleting trigger without another mutation or warning")
			assert.Equal(t, ctrl.Result{}, reconcileLPUEviction(t, r, "agent-0"))
			require.Empty(t, recorded)
		})
	}
}

func TestLPUEviction_LPUGPUDeletesSamePartitionOnly(t *testing.T) {
	trigger := newLPUEvictionPod("stage-0-partition-3", "0", "0", true, podDisruptionReasonTaintManagerDeletion)
	samePartition := newLPUEvictionPod("stage-1-partition-3", "0", "1", false, "")
	otherPartition := newLPUEvictionPod("stage-0-partition-4", "0", "2", false, "")
	otherModel := newLPUEvictionPod("stage-0-draft-partition-3", "0", "0", false, "")
	otherModel.Annotations[lpxv1alpha1.PodModelAnnotation] = lpuEvictionTestModel + "-other"
	noPartition := newLPUOnlyConductor()
	otherReplica := newLPUEvictionPod("replica-1-partition-3", "1", "0", false, "")
	trigger.Annotations[dynamolpx.WorkloadModeAnnotation] = string(lpxv1alpha1.WorkloadModeV2StrictHybrid)
	require.NotContains(t, trigger.Annotations, lpxv1alpha1.PodPartitionIDAnnotation)

	r, c := newLPUEvictionReconciler(
		trigger,
		samePartition,
		otherPartition,
		otherModel,
		noPartition,
		otherReplica,
	)

	result := reconcileLPUEviction(t, r, "stage-0-partition-3")
	assert.Equal(t, ctrl.Result{}, result)
	assert.Equal(t, []string{"conductor-0", "replica-1-partition-3", "stage-0-draft-partition-3", "stage-0-partition-4"}, remainingLPUEvictionPods(t, c))
}

func TestLPUEviction_DisruptionConditionWithoutDeletionTimestampDoesNotCascade(t *testing.T) {
	trigger := newLPUEvictionPod("agent-0", "0", "0", false, podDisruptionReasonTaintManagerDeletion)
	sibling := newLPUEvictionPod("agent-1", "0", "1", false, "")

	r, c := newLPUEvictionReconciler(trigger, sibling)

	result := reconcileLPUEviction(t, r, "agent-0")
	assert.Equal(t, ctrl.Result{}, result)
	assert.Equal(t, []string{"agent-0", "agent-1"}, remainingLPUEvictionPods(t, c))

	var updated corev1.Pod
	require.NoError(t, c.Get(context.Background(), client.ObjectKeyFromObject(trigger), &updated))
	assert.Empty(t, updated.Finalizers)
}

func TestLPUEviction_PodPredicateEnqueuesOnlyDeletingPods(t *testing.T) {
	predicate := (&lpuEvictionReconciler{}).evictionPredicate()
	ordinary := newLPUEvictionPod("agent-0", "0", "0", false, "")
	deleting := newLPUEvictionPod("agent-0", "0", "0", true, podDisruptionReasonTaintManagerDeletion)

	assert.False(t, predicate.Create(event.CreateEvent{Object: ordinary}))
	assert.False(t, predicate.Update(event.UpdateEvent{ObjectOld: ordinary, ObjectNew: ordinary.DeepCopy()}))
	assert.True(t, predicate.Update(event.UpdateEvent{ObjectOld: ordinary, ObjectNew: deleting}))
}

func TestLPUEviction_Noops(t *testing.T) {
	tests := []struct {
		name            string
		triggerName     string
		conditionReason string
		phase           corev1.PodPhase
		statusReason    string
		missingMode     bool
		nonLPU          bool
		conductor       bool
	}{
		{name: "other disruption reason", conditionReason: "OtherDisruptionReason"},
		{name: "failed evicted without disruption condition", phase: corev1.PodFailed, statusReason: "Evicted"},
		{name: "missing workload mode", conditionReason: podDisruptionReasonTaintManagerDeletion, missingMode: true},
		{name: "non LPX pod", triggerName: "worker-0", conditionReason: podDisruptionReasonTaintManagerDeletion, nonLPU: true},
		{name: "conductor", triggerName: "conductor-0", conditionReason: podDisruptionReasonTaintManagerDeletion, conductor: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Observe a deleting Pod outside the Agent cascade contract")
			triggerName := tt.triggerName
			if triggerName == "" {
				triggerName = "agent-0"
			}
			trigger := newLPUEvictionPod(triggerName, "0", "0", true, tt.conditionReason)
			trigger.Status.Phase = tt.phase
			trigger.Status.Reason = tt.statusReason
			if tt.nonLPU {
				trigger.Labels[commonconsts.KubeLabelDynamoComponentType] = commonconsts.ComponentTypeWorker
			}
			if tt.conductor {
				trigger.Annotations[lpxv1alpha1.PodRoleAnnotation] = lpxv1alpha1.PodRoleConductor
			}
			if tt.missingMode {
				delete(trigger.Annotations, dynamolpx.WorkloadModeAnnotation)
			}
			sibling := newLPUEvictionPod("agent-1", "0", "1", false, "")
			r, c := newLPUEvictionReconciler(trigger, sibling)

			t.Log("Reconcile without deleting the surviving Agent")
			result := reconcileLPUEviction(t, r, trigger.Name)
			assert.Equal(t, ctrl.Result{}, result)
			assert.Equal(t, []string{"agent-1"}, remainingLPUEvictionPods(t, c))
		})
	}
}

func TestLPUEviction_MissingGroveLabelsReturnsError(t *testing.T) {
	trigger := newLPUEvictionPod("agent-0", "0", "0", true, podDisruptionReasonTaintManagerDeletion)
	delete(trigger.Labels, grovecommon.LabelPodCliqueScalingGroup)
	sibling := newLPUEvictionPod("agent-1", "0", "1", false, "")

	r, c := newLPUEvictionReconciler(trigger, sibling)

	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: trigger.Name, Namespace: lpuEvictionTestNamespace},
	})
	require.Error(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	assert.Equal(t, []string{"agent-1"}, remainingLPUEvictionPods(t, c))
}
