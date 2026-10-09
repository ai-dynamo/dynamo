/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/event"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/workermetadata"
)

// admissionTestFrontend describes one frontend pod and what it publishes.
type admissionTestFrontend struct {
	ready bool
	// capable frontends publish admission records.
	capable bool
	// servesReplacement frontends list the replacement worker's model card.
	servesReplacement bool
}

// admissionTestPod builds a pod of one DGD component owned by the DCD named selector.
func admissionTestPod(name, component, componentType, workerHash, selector string, ready bool) *corev1.Pod {
	status := corev1.ConditionFalse
	if ready {
		status = corev1.ConditionTrue
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       types.UID(name + "-uid"),
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoComponent:           component,
				consts.KubeLabelDynamoComponentType:       componentType,
				consts.KubeLabelDynamoWorkerHash:          workerHash,
				consts.KubeLabelDynamoSelector:            selector,
			},
		},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: status}},
		},
	}
}

// admissionTestMetadata builds the DynamoWorkerMetadata resource that pod publishes.
func admissionTestMetadata(pod *corev1.Pod, data map[string]any) *unstructured.Unstructured {
	metadata := workermetadata.New()
	metadata.SetName(pod.Name)
	metadata.SetNamespace(pod.Namespace)
	metadata.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion: "v1", Kind: "Pod", Name: pod.Name, UID: pod.UID,
	}})
	metadata.Object["spec"] = map[string]any{"data": data}
	return metadata
}

// admissionTestRolloutDCD builds a worker DCD of one generation with one available replica.
func admissionTestRolloutDCD(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, workerHash string, created metav1.Time) *nvidiacomv1beta1.DynamoComponentDeployment {
	return createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:              dynamo.GetDCDResourceName(dgd, "prefill", workerHash),
			Namespace:         "default",
			CreationTimestamp: created,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          workerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypePrefill,
				ServiceName:   "prefill",
				Replicas:      ptr.To(int32(1)),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				Replicas:          1,
				AvailableReplicas: ptr.To(int32(1)),
			},
		},
	})
}

func TestBuildRollingUpdateContext_WaitsForFrontendsToServeReplacement(t *testing.T) {
	const (
		oldCard = "test-dgd-old/prefill/generate/1"
		newCard = "test-dgd-new/prefill/generate/2"
	)
	ready := admissionTestFrontend{ready: true, capable: true, servesReplacement: true}
	lagging := admissionTestFrontend{ready: true, capable: true}
	legacy := admissionTestFrontend{ready: true}
	tests := []struct {
		name      string
		frontends []admissionTestFrontend
		// oldCards and newCards report whether each generation published its model card.
		oldCards      bool
		newCards      bool
		wantOldTarget int32
	}{
		{name: "every ready frontend serves the replacement", frontends: []admissionTestFrontend{ready, ready}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "one frontend does not serve the replacement yet", frontends: []admissionTestFrontend{ready, lagging}, oldCards: true, newCards: true, wantOldTarget: 1},
		{name: "no frontend serves the replacement yet", frontends: []admissionTestFrontend{lagging, lagging}, oldCards: true, newCards: true, wantOldTarget: 1},
		{name: "frontends without admission records roll on pod availability", frontends: []admissionTestFrontend{legacy, legacy}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "a frontend without admission records blocks while another publishes them", frontends: []admissionTestFrontend{ready, legacy}, oldCards: true, newCards: true, wantOldTarget: 1},
		{name: "unready frontends are not waited for", frontends: []admissionTestFrontend{ready, {capable: true}}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "without a ready frontend the rollout uses pod availability", frontends: []admissionTestFrontend{{capable: true}}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "a replacement that has not published its card is not served", frontends: []admissionTestFrontend{ready}, oldCards: true, newCards: false, wantOldTarget: 1},
		{name: "a component that publishes no cards rolls on pod availability", frontends: []admissionTestFrontend{lagging}, oldCards: false, newCards: false, wantOldTarget: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD whose prefill component rolls with maxSurge=1 and maxUnavailable=0")
			dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(len(tt.frontends))),
				},
				"prefill": {
					ComponentType: consts.ComponentTypePrefill,
					Replicas:      ptr.To(int32(1)),
					Annotations: map[string]string{
						KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
						KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
					},
				},
			})
			dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash}
			newHash := betaDGDWorkersSpecHash(t, dgd)

			t.Log("Both generations have one Ready replica")
			oldDCD := admissionTestRolloutDCD(t, dgd, testOldWorkerHash, metav1.Now())
			newDCD := admissionTestRolloutDCD(t, dgd, newHash, metav1.Now())
			oldPod := admissionTestPod("prefill-old", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, oldDCD.Name, true)
			newPod := admissionTestPod("prefill-new", "prefill", consts.ComponentTypePrefill, newHash, newDCD.Name, true)
			objs := []runtime.Object{oldDCD, newDCD, oldPod, newPod}

			t.Log("Each worker publishes its discovery records")
			oldCards := map[string]any{}
			if tt.oldCards {
				oldCards[oldCard] = map[string]any{"type": "Model", "namespace": "test-dgd-old"}
			}
			newCards := map[string]any{}
			if tt.newCards {
				newCards[newCard] = map[string]any{"type": "Model", "namespace": "test-dgd-new"}
			}
			objs = append(objs,
				admissionTestMetadata(oldPod, map[string]any{"model_cards": oldCards}),
				admissionTestMetadata(newPod, map[string]any{"model_cards": newCards}),
			)

			t.Log("Each frontend publishes the workers it serves")
			for i, frontend := range tt.frontends {
				pod := admissionTestPod(fmt.Sprintf("frontend-%d", i), "frontend", consts.ComponentTypeFrontend, "", "", frontend.ready)
				objs = append(objs, pod)
				if !frontend.capable {
					continue
				}
				members := []any{oldCard}
				if frontend.servesReplacement {
					members = append(members, newCard)
				}
				objs = append(objs, admissionTestMetadata(pod, map[string]any{"event_sources": map[string]any{
					"capability": map[string]any{
						"topic":    workermetadata.FrontendAdmissionTopic,
						"scope":    map[string]any{"kind": "namespace", "name": "test-dgd"},
						"metadata": map[string]any{"protocol": workermetadata.FrontendAdmissionProtocol, "capability": true},
					},
					"members": map[string]any{
						"topic":    workermetadata.FrontendAdmissionTopic,
						"scope":    map[string]any{"kind": "namespace", "name": "test-dgd-new"},
						"metadata": map[string]any{"protocol": workermetadata.FrontendAdmissionProtocol, "members": members},
					},
				}}))
			}

			t.Log("Plan the rollout step")
			r := createTestReconcilerWithStatus(dgd, withObjects(objs...))
			result, err := r.buildRollingUpdateContext(context.Background(), dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.wantOldTarget, result.OldWorkerReplicaTargetsByComponent["prefill"])
			assert.Equal(t, int32(1), result.NewWorkerReplicaTargetsByComponent["prefill"])
		})
	}
}

func TestBuildRollingUpdateContext_RollForwardKeepsTheServedGeneration(t *testing.T) {
	t.Log("Build a DGD whose prefill component rolls with maxSurge=1 and maxUnavailable=0")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"frontend": {
			ComponentType: consts.ComponentTypeFrontend,
			Replicas:      ptr.To(int32(1)),
		},
		"prefill": {
			ComponentType: consts.ComponentTypePrefill,
			Replicas:      ptr.To(int32(1)),
			Annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
			},
		},
	})
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash}

	t.Log("The original generation serves; a failed newer generation left a Ready prefill no frontend serves")
	const failedHash = "failedh2"
	servedDCD := admissionTestRolloutDCD(t, dgd, testOldWorkerHash, metav1.NewTime(metav1.Now().Add(-time.Hour)))
	failedDCD := admissionTestRolloutDCD(t, dgd, failedHash, metav1.Now())
	servedPod := admissionTestPod("prefill-served", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, servedDCD.Name, true)
	failedPod := admissionTestPod("prefill-failed", "prefill", consts.ComponentTypePrefill, failedHash, failedDCD.Name, true)
	frontend := admissionTestPod("frontend-0", "frontend", consts.ComponentTypeFrontend, "", "", true)
	objs := []runtime.Object{
		servedDCD, failedDCD, servedPod, failedPod, frontend,
		admissionTestMetadata(servedPod, map[string]any{"model_cards": map[string]any{
			"test-dgd-served/prefill/generate/1": map[string]any{"type": "Model", "namespace": "test-dgd-served"},
		}}),
		admissionTestMetadata(failedPod, map[string]any{"model_cards": map[string]any{
			"test-dgd-failed/prefill/generate/2": map[string]any{"type": "Model", "namespace": "test-dgd-failed"},
		}}),
		admissionTestMetadata(frontend, map[string]any{"event_sources": map[string]any{
			"capability": map[string]any{
				"topic":    workermetadata.FrontendAdmissionTopic,
				"scope":    map[string]any{"kind": "namespace", "name": "test-dgd"},
				"metadata": map[string]any{"protocol": workermetadata.FrontendAdmissionProtocol, "capability": true},
			},
			"members": map[string]any{
				"topic":    workermetadata.FrontendAdmissionTopic,
				"scope":    map[string]any{"kind": "namespace", "name": "test-dgd-served"},
				"metadata": map[string]any{"protocol": workermetadata.FrontendAdmissionProtocol, "members": []any{"test-dgd-served/prefill/generate/1"}},
			},
		}}),
	}

	t.Log("Plan the first step toward a new generation")
	r := createTestReconcilerWithStatus(dgd, withObjects(objs...))
	result, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)

	t.Log("Surge room comes from the unserved replica, not the serving one")
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByDCD[servedDCD.Name])
	assert.Equal(t, int32(0), result.OldWorkerReplicaTargetsByDCD[failedDCD.Name])
}

func TestDGDFrontendPodEventPredicate(t *testing.T) {
	t.Log("Build ready and unready frontends and a worker")
	unready := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", false)
	ready := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", true)
	worker := admissionTestPod("worker", "prefill", consts.ComponentTypePrefill, "hash", "worker-dcd", true)
	pred := dgdFrontendPodEventPredicate()

	t.Log("Membership and readiness changes of frontends requeue; other events do not")
	assert.True(t, pred.Create(event.CreateEvent{Object: ready}))
	assert.True(t, pred.Delete(event.DeleteEvent{Object: ready}))
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: unready, ObjectNew: ready}))
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: ready, ObjectNew: unready}))
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: ready, ObjectNew: ready.DeepCopy()}))
	assert.False(t, pred.Create(event.CreateEvent{Object: worker}))
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: worker, ObjectNew: worker.DeepCopy()}))
	assert.False(t, pred.Generic(event.GenericEvent{Object: ready}))

	t.Log("A frontend maps to its DGD")
	requests := mapDGDFrontendPodToRequests(context.Background(), ready)
	require.Len(t, requests, 1)
	assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "test-dgd"}, requests[0].NamespacedName)
	assert.Empty(t, mapDGDFrontendPodToRequests(context.Background(), worker))
}

func TestWorkerMetadataChangedPredicate(t *testing.T) {
	t.Log("Build a projected resource and a copy whose records changed")
	pod := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", true)
	original := admissionTestMetadata(pod, map[string]any{"event_sources": map[string]any{}})
	unchanged := original.DeepCopy()
	unchanged.SetResourceVersion("2")
	changed := admissionTestMetadata(pod, map[string]any{"event_sources": map[string]any{
		"capability": map[string]any{"topic": workermetadata.FrontendAdmissionTopic},
	}})
	pred := workerMetadataChangedPredicate()

	t.Log("Only changes to the projected records requeue")
	assert.True(t, pred.Create(event.CreateEvent{Object: original}))
	assert.True(t, pred.Delete(event.DeleteEvent{Object: original}))
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: original, ObjectNew: changed}))
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: original, ObjectNew: unchanged}))
	assert.False(t, pred.Generic(event.GenericEvent{Object: original}))
}

func TestMapWorkerMetadataToDGDRequests(t *testing.T) {
	t.Log("Build a resource owned by a DGD pod and one owned by an unknown pod")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{})
	pod := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", true)
	published := admissionTestMetadata(pod, map[string]any{})
	orphan := admissionTestMetadata(admissionTestPod("gone", "frontend", consts.ComponentTypeFrontend, "", "", true), map[string]any{})
	r := createTestDGDReconcilerWithStatus(dgd, withObjects(pod))

	t.Log("The resource maps to the DGD of the pod that published it")
	requests := r.mapWorkerMetadataToDGDRequests(context.Background(), published)
	require.Len(t, requests, 1)
	assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "test-dgd"}, requests[0].NamespacedName)
	assert.Empty(t, r.mapWorkerMetadataToDGDRequests(context.Background(), orphan))
}
