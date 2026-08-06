/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/event"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

func TestSelectDisaggregatedSetComponents(t *testing.T) {
	t.Run("selects multinode worker roles", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "prefill",
						ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
					{
						ComponentName: "frontend",
						ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					},
				},
			},
		}

		selection, reason := selectDisaggregatedSetComponents(dgd)
		require.Empty(t, reason)
		require.Equal(t, "prefill", selection.componentToRole["prefill"])
		require.Equal(t, "decode", selection.componentToRole["decode"])
		require.Len(t, selection.componentToRole, 2)
	})

	t.Run("rejects scaling adapter", func(t *testing.T) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
			Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
				Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName:  "prefill",
						ComponentType:  nvidiacomv1beta1.ComponentTypePrefill,
						Multinode:      &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
						Replicas:       ptr.To(int32(2)),
					},
					{
						ComponentName: "decode",
						ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
						Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
		}

		_, reason := selectDisaggregatedSetComponents(dgd)
		require.Contains(t, reason, "scalingAdapter")
	})
}

func TestDisaggregatedSetChildNamesFitDNSLabelLimit(t *testing.T) {
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: strings.Repeat("d", 63)},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: strings.Repeat("p", 63),
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
				{
					ComponentName: strings.Repeat("q", 63),
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
			},
		},
	}

	selection, reason := selectDisaggregatedSetComponents(dgd)
	require.Empty(t, reason)
	require.Len(t, selection.componentToRole, 2)
	setName := disaggregatedSetName(dgd)
	require.LessOrEqual(t, len(setName), maxDisaggregatedSetNameLength)
	for _, roleName := range selection.componentToRole {
		require.LessOrEqual(t, len(roleName), maxDisaggregatedSetRoleNameLength)
		childName := disaggregatedsetutils.GenerateName(setName, roleName, strings.Repeat("a", disaggregatedSetRevisionLength))
		require.LessOrEqual(t, len(childName), 63)
	}
}

func TestCheckDisaggregatedSetReadiness(t *testing.T) {
	ds := newDisaggregatedSetObject()
	ds.SetName("demo")
	ds.SetGeneration(3)
	ds.Object["status"] = map[string]any{
		"observedGeneration": int64(2),
		"roleStatuses": []any{
			map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
			map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(1)},
		},
	}
	selection := disaggregatedSetSelection{
		componentToRole: map[string]string{"prefill": "prefill", "decode": "decode"},
		desiredReplicas: map[string]int32{"prefill": 2, "decode": 2},
	}

	t.Log("stale observedGeneration keeps the DisaggregatedSet unready")
	ready, reason, statuses := checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "observed generation")
	require.Equal(t, int32(2), ptr.Deref(statuses["prefill"].ReadyReplicas, 0))

	t.Log("lagging decode role readiness keeps the DisaggregatedSet unready")
	ds.Object["status"].(map[string]any)["observedGeneration"] = int64(3)
	ready, reason, statuses = checkDisaggregatedSetReadiness(ds, selection)
	require.False(t, ready)
	require.Contains(t, reason, "decode")
	require.Equal(t, int32(1), ptr.Deref(statuses["decode"].ReadyReplicas, 0))

	t.Log("all roles at desired ready replicas report ready")
	ds.Object["status"].(map[string]any)["roleStatuses"] = []any{
		map[string]any{"name": "prefill", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
		map[string]any{"name": "decode", "replicas": int64(2), "updatedReplicas": int64(2), "readyReplicas": int64(2)},
	}
	ready, _, _ = checkDisaggregatedSetReadiness(ds, selection)
	require.True(t, ready)
}

func TestDisaggregatedSetPredicatesObserveRoutingMetadata(t *testing.T) {
	baseDS := newDisaggregatedSetObject()
	baseDS.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "demo"})
	relabeledDS := baseDS.DeepCopy()
	relabeledDS.SetLabels(map[string]string{consts.KubeLabelDynamoGraphDeploymentName: "other"})
	reownedDS := baseDS.DeepCopy()
	reownedDS.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       dynamoGraphDeploymentKind,
		Name:       "demo",
		UID:        "demo-uid",
		Controller: ptr.To(true),
	}})

	require.False(t, disaggregatedSetStatusChanged(baseDS, baseDS.DeepCopy()))
	require.True(t, disaggregatedSetStatusChanged(baseDS, relabeledDS))
	require.True(t, disaggregatedSetStatusChanged(baseDS, reownedDS))

	baseLWS := &leaderworkersetv1.LeaderWorkerSet{ObjectMeta: metav1.ObjectMeta{
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: "demo",
		},
	}}
	relabeledLWS := baseLWS.DeepCopy()
	relabeledLWS.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "other",
	})

	require.False(t, leaderWorkerSetStatusChanged(baseLWS, baseLWS.DeepCopy()))
	require.True(t, leaderWorkerSetStatusChanged(baseLWS, relabeledLWS))
}

func TestWorkloadRoutingAnnotationsChanged(t *testing.T) {
	t.Log("no change in routing annotations does not trigger update predicate")
	oldDGD := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				consts.KubeAnnotationEnableGrove:            consts.KubeLabelValueFalse,
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueFalse,
			},
		},
	}
	newDGD := oldDGD.DeepCopy()
	require.False(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))

	t.Log("enabling DisaggregatedSet triggers update predicate")
	newDGD = oldDGD.DeepCopy()
	newDGD.Annotations[consts.KubeAnnotationEnableDisaggregatedSet] = consts.KubeLabelValueTrue
	require.True(t, workloadRoutingAnnotationsChanged(updateEvent(oldDGD, newDGD)))
}

func updateEvent(oldObj, newObj *nvidiacomv1beta1.DynamoGraphDeployment) event.UpdateEvent {
	return event.UpdateEvent{ObjectOld: oldObj, ObjectNew: newObj}
}
