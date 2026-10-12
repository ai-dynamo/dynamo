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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
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
	// template is the pod-template hash; empty means admissionTestTemplate.
	template          string
	publishes         bool
	unsynced          bool
	servesReplacement bool
	// fromEarlierPod marks a record published by an earlier pod of the same name.
	fromEarlierPod bool
	terminating    bool
}

const admissionTestTemplate = "frontend-1"

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

// admissionTestFrontendPod builds a frontend pod created from pod template template.
func admissionTestFrontendPod(name, template string, ready bool) *corev1.Pod {
	pod := admissionTestPod(name, "frontend", consts.ComponentTypeFrontend, "", "", ready)
	pod.Labels[appsv1.DefaultDeploymentUniqueLabelKey] = template
	return pod
}

// admissionTestMetadata builds the DynamoWorkerMetadata resource that pod publishes.
func admissionTestMetadata(pod *corev1.Pod, data map[string]any) *unstructured.Unstructured {
	return admissionTestMetadataOwnedBy(pod, pod.UID, data)
}

// admissionTestMetadataOwnedBy builds a DynamoWorkerMetadata resource that a
// pod with pod's name and uid published.
func admissionTestMetadataOwnedBy(pod *corev1.Pod, uid types.UID, data map[string]any) *unstructured.Unstructured {
	metadata := workermetadata.New()
	metadata.SetName(pod.Name)
	metadata.SetNamespace(pod.Namespace)
	metadata.SetOwnerReferences([]metav1.OwnerReference{{
		APIVersion: "v1", Kind: "Pod", Name: pod.Name, UID: uid,
	}})
	metadata.Object["spec"] = map[string]any{"data": data}
	return metadata
}

// admissionTestRecord is the discovery data of a frontend that publishes the
// workers with base model-card keys members.
func admissionTestRecord(synced bool, members ...string) map[string]any {
	byEndpoint := map[string]any{}
	for _, member := range members {
		endpoint, hexID := member[:strings.LastIndex(member, "/")], member[strings.LastIndex(member, "/")+1:]
		ids, _ := byEndpoint[endpoint].([]any)
		byEndpoint[endpoint] = append(ids, hexID)
	}
	return map[string]any{"event_sources": map[string]any{
		"admission": map[string]any{
			"topic":    workermetadata.FrontendAdmissionTopic,
			"scope":    map[string]any{"kind": "namespace", "name": "test-dgd"},
			"metadata": map[string]any{"protocol": workermetadata.FrontendAdmissionProtocol, "synced": synced, "members": byEndpoint},
		},
	}}
}

// admissionTestCards is the discovery data of a worker that registers cards.
func admissionTestCards(cards ...string) map[string]any {
	registered := map[string]any{}
	for _, card := range cards {
		registered[card] = map[string]any{"type": "Model"}
	}
	return map[string]any{"model_cards": registered}
}

// admissionTestRolloutDCD builds a worker DCD of one generation with available Ready replicas.
func admissionTestRolloutDCD(
	t testing.TB,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component, componentType, workerHash string,
	replicas, available int32,
	created metav1.Time,
) *nvidiacomv1beta1.DynamoComponentDeployment {
	return createTestDCD(t, dgd, &nvidiacomv1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:              dynamo.GetDCDResourceName(dgd, component, workerHash),
			Namespace:         "default",
			CreationTimestamp: created,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoWorkerHash:          workerHash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: componentType,
				ServiceName:   component,
				Replicas:      ptr.To(replicas),
			},
		},
		Status: nvidiacomv1alpha1.DynamoComponentDeploymentStatus{
			Service: &nvidiacomv1alpha1.ServiceReplicaStatus{
				Replicas:          replicas,
				AvailableReplicas: ptr.To(available),
			},
		},
	})
}

// admissionTestRollingDGD builds a DGD whose worker components roll with
// maxSurge=1 and maxUnavailable, and whose frontend component declares frontends.
func admissionTestRollingDGD(frontends int32, maxUnavailable string, workers map[string]int32) *nvidiacomv1beta1.DynamoGraphDeployment {
	services := map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Replicas: ptr.To(frontends)},
	}
	for component, replicas := range workers {
		services[component] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
			ComponentType: component,
			Replicas:      ptr.To(replicas),
			Annotations: map[string]string{
				KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
				KubeAnnotationDeploymentRollingUpdateMaxUnavailable: maxUnavailable,
			},
		}
	}
	dgd := createTestDGD("test-dgd", services)
	dgd.Annotations = map[string]string{consts.AnnotationCurrentWorkerHashV2: testOldWorkerHash}
	return dgd
}

func TestBuildRollingUpdateContext_WaitsForFrontendsToServeReplacement(t *testing.T) {
	const (
		oldCard = "test-dgd-old/prefill/generate/1"
		newCard = "test-dgd-new/prefill/generate/2"
	)
	serving := admissionTestFrontend{ready: true, publishes: true, servesReplacement: true}
	lagging := admissionTestFrontend{ready: true, publishes: true}
	unsynced := admissionTestFrontend{ready: true, publishes: true, unsynced: true, servesReplacement: true}
	silent := admissionTestFrontend{ready: true}
	tests := []struct {
		name      string
		frontends []admissionTestFrontend
		// recorded is the DGD's record of frontend templates that publish admissions.
		recorded string
		oldCards bool
		newCards bool
		// newCardsFromEarlierPod marks the replacement's card as published by an
		// earlier pod of the same name.
		newCardsFromEarlierPod bool
		wantOldTarget          int32
		wantRecorded           string
		// wantCondition is the reason of the ReplacementsServed condition; empty means none.
		wantCondition string
	}{
		{name: "every ready frontend serves the replacement", frontends: []admissionTestFrontend{serving, serving}, oldCards: true, newCards: true, wantOldTarget: 0, wantRecorded: admissionTestTemplate, wantCondition: reasonReplacementsServed},
		{name: "one frontend does not serve the replacement yet", frontends: []admissionTestFrontend{serving, lagging}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonWaitingForFrontends},
		{name: "no frontend serves the replacement yet", frontends: []admissionTestFrontend{lagging, lagging}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonWaitingForFrontends},
		{name: "an unsynced frontend serves nothing it lists", frontends: []admissionTestFrontend{serving, unsynced}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendRecordsMissing},
		{name: "frontends that never published records roll on pod availability", frontends: []admissionTestFrontend{silent, silent}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "a frontend without records blocks while another publishes them", frontends: []admissionTestFrontend{serving, silent}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendRecordsMissing},
		{name: "a frontend of a recorded template that lost its records blocks", frontends: []admissionTestFrontend{silent}, recorded: admissionTestTemplate, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendRecordsMissing},
		{name: "after a downgrade, frontends of an unrecorded template roll on pod availability", frontends: []admissionTestFrontend{{ready: true, template: "frontend-0"}}, recorded: admissionTestTemplate, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "unready frontends are not waited for", frontends: []admissionTestFrontend{serving, {publishes: true}}, oldCards: true, newCards: true, wantOldTarget: 0, wantRecorded: admissionTestTemplate, wantCondition: reasonReplacementsServed},
		{name: "without a ready frontend the rollout uses pod availability", frontends: []admissionTestFrontend{{publishes: true}}, oldCards: true, newCards: true, wantOldTarget: 0},
		{name: "without a ready frontend a recorded gate holds", frontends: []admissionTestFrontend{{publishes: true}}, recorded: admissionTestTemplate, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendRecordsMissing},
		{name: "a replacement that has not published its card is not served", frontends: []admissionTestFrontend{serving}, oldCards: true, newCards: false, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonWaitingForFrontends},
		{name: "a component that publishes no cards rolls on pod availability", frontends: []admissionTestFrontend{lagging}, oldCards: false, newCards: false, wantOldTarget: 0, wantRecorded: admissionTestTemplate},
		{name: "a record of an earlier frontend pod with the same name does not count", frontends: []admissionTestFrontend{serving, {ready: true, publishes: true, servesReplacement: true, fromEarlierPod: true}}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendRecordsMissing},
		{name: "a card of an earlier worker pod with the same name does not count", frontends: []admissionTestFrontend{serving}, oldCards: true, newCards: true, newCardsFromEarlierPod: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonWaitingForFrontends},
		{name: "frontends rolling out from a template without records keep the old replica", frontends: []admissionTestFrontend{{ready: true, template: "frontend-0"}, {}}, oldCards: true, newCards: true, wantOldTarget: 1, wantCondition: reasonFrontendsRolling},
		{name: "frontends rolling out between publishing templates keep the old replica", frontends: []admissionTestFrontend{serving, {template: "frontend-2"}}, oldCards: true, newCards: true, wantOldTarget: 1, wantRecorded: admissionTestTemplate, wantCondition: reasonFrontendsRolling},
		{name: "a terminating frontend of an earlier template does not hold the rollout", frontends: []admissionTestFrontend{serving, {template: "frontend-0", terminating: true}}, oldCards: true, newCards: true, wantOldTarget: 0, wantRecorded: admissionTestTemplate, wantCondition: reasonReplacementsServed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD whose prefill component rolls with maxSurge=1 and maxUnavailable=0")
			dgd := admissionTestRollingDGD(int32(len(tt.frontends)), "0", map[string]int32{consts.ComponentTypePrefill: 1})
			if tt.recorded != "" {
				dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates] = tt.recorded
			}
			newHash := betaDGDWorkersSpecHash(t, dgd)

			t.Log("Both generations have one Ready replica")
			oldDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 1, 1, metav1.Now())
			newDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, newHash, 1, 1, metav1.Now())
			oldPod := admissionTestPod("prefill-old", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, oldDCD.Name, true)
			newPod := admissionTestPod("prefill-new", "prefill", consts.ComponentTypePrefill, newHash, newDCD.Name, true)
			objs := []runtime.Object{oldDCD, newDCD, oldPod, newPod}

			t.Log("Each worker publishes its discovery records")
			var oldCards, newCards []string
			if tt.oldCards {
				oldCards = append(oldCards, oldCard)
			}
			if tt.newCards {
				newCards = append(newCards, newCard)
			}
			newPodUID := newPod.UID
			if tt.newCardsFromEarlierPod {
				newPodUID = "earlier-" + newPod.UID
			}
			objs = append(objs,
				admissionTestMetadata(oldPod, admissionTestCards(oldCards...)),
				admissionTestMetadataOwnedBy(newPod, newPodUID, admissionTestCards(newCards...)),
			)

			t.Log("Each frontend publishes the workers it serves")
			for i, frontend := range tt.frontends {
				template := frontend.template
				if template == "" {
					template = admissionTestTemplate
				}
				pod := admissionTestFrontendPod(fmt.Sprintf("frontend-%d", i), template, frontend.ready)
				if frontend.terminating {
					pod.DeletionTimestamp = ptr.To(metav1.Now())
					pod.Finalizers = []string{"test.nvidia.com/hold"}
				}
				objs = append(objs, pod)
				if !frontend.publishes {
					continue
				}
				members := []string{oldCard}
				if frontend.servesReplacement {
					members = append(members, newCard)
				}
				uid := pod.UID
				if frontend.fromEarlierPod {
					uid = "earlier-" + pod.UID
				}
				objs = append(objs, admissionTestMetadataOwnedBy(pod, uid, admissionTestRecord(!frontend.unsynced, members...)))
			}

			t.Log("Plan the rollout step")
			r := createTestReconcilerWithStatus(dgd, withObjects(objs...))
			plan, err := r.planRollingUpdate(context.Background(), dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.wantOldTarget, plan.context.OldWorkerReplicaTargetsByComponent["prefill"])
			assert.Equal(t, int32(1), plan.context.NewWorkerReplicaTargetsByComponent["prefill"])

			t.Log("The ReplacementsServed condition reports what the rollout waits for")
			if tt.wantCondition == "" {
				assert.Nil(t, plan.replacementsServed)
			} else {
				require.NotNil(t, plan.replacementsServed)
				assert.Equal(t, tt.wantCondition, plan.replacementsServed.Reason)
				assert.Equal(t, tt.wantCondition == reasonReplacementsServed,
					plan.replacementsServed.Status == metav1.ConditionTrue)
			}

			t.Log("The DGD records the frontend templates that publish admissions")
			stored := &nvidiacomv1beta1.DynamoGraphDeployment{}
			require.NoError(t, r.Get(context.Background(), client.ObjectKeyFromObject(dgd), stored))
			assert.Equal(t, tt.wantRecorded, stored.Annotations[consts.AnnotationFrontendAdmissionTemplates])
		})
	}
}

func TestBuildRollingUpdateContext_KeepsWaitingWhenFrontendRecordsDisappear(t *testing.T) {
	const (
		oldCard = "test-dgd-old/prefill/generate/1"
		newCard = "test-dgd-new/prefill/generate/2"
	)
	t.Log("Build a prefill rollout with one ready frontend that does not serve the replacement yet")
	dgd := admissionTestRollingDGD(1, "0", map[string]int32{consts.ComponentTypePrefill: 1})
	newHash := betaDGDWorkersSpecHash(t, dgd)
	oldDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 1, 1, metav1.Now())
	newDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, newHash, 1, 1, metav1.Now())
	oldPod := admissionTestPod("prefill-old", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, oldDCD.Name, true)
	newPod := admissionTestPod("prefill-new", "prefill", consts.ComponentTypePrefill, newHash, newDCD.Name, true)
	frontend := admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)
	r := createTestReconcilerWithStatus(dgd, withObjects(
		oldDCD, newDCD, oldPod, newPod, frontend,
		admissionTestMetadata(oldPod, admissionTestCards(oldCard)),
		admissionTestMetadata(newPod, admissionTestCards(newCard)),
		admissionTestMetadata(frontend, admissionTestRecord(true, oldCard)),
	))

	t.Log("The first step keeps the old replica and records the frontend's template")
	result, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByComponent["prefill"])
	assert.Equal(t, admissionTestTemplate, dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates])

	t.Log("The frontend's records disappear while it stays ready")
	metadata := workermetadata.New()
	metadata.SetNamespace(dgd.Namespace)
	metadata.SetName(frontend.Name)
	require.NoError(t, r.Delete(context.Background(), metadata))

	t.Log("The next step still keeps the old replica")
	result, err = r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByComponent["prefill"])
}

func TestBuildRollingUpdateContext_PendingFrontendsKeepAvailableOldWorkers(t *testing.T) {
	t.Log("Build a decode rollout that tolerates one unavailable replica, with a recorded frontend template")
	dgd := admissionTestRollingDGD(1, "1", map[string]int32{consts.ComponentTypeDecode: 2})
	dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates] = admissionTestTemplate
	newHash := betaDGDWorkersSpecHash(t, dgd)
	oldDCD := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, testOldWorkerHash, 2, 2, metav1.Now())
	newDCD := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, newHash, 1, 0, metav1.Now())
	objs := []runtime.Object{oldDCD, newDCD}
	for i := range 2 {
		pod := admissionTestPod(fmt.Sprintf("decode-old-%d", i), "decode", consts.ComponentTypeDecode, testOldWorkerHash, oldDCD.Name, true)
		objs = append(objs, pod, admissionTestMetadata(pod, admissionTestCards(fmt.Sprintf("test-dgd-old/decode/generate/%d", i))))
	}

	t.Log("The only frontend restarted and has not published records yet")
	objs = append(objs, admissionTestFrontendPod("frontend-0", admissionTestTemplate, true))
	r := createTestReconcilerWithStatus(dgd, withObjects(objs...))

	t.Log("maxUnavailable would allow retiring an old replica, but which ones serve is unknown")
	result, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(2), result.OldWorkerReplicaTargetsByComponent["decode"])
}

func TestBuildRollingUpdateContext_KeepsAnOldRoleWhileItsGenerationStillServes(t *testing.T) {
	const (
		oldPrefillCard = "test-dgd-old/prefill/generate/1"
		newPrefillCard = "test-dgd-new/prefill/generate/2"
	)
	tests := []struct {
		name string
		// newDecodes is how many replacement decode replicas every frontend serves.
		newDecodes           int32
		wantOldPrefillTarget int32
		wantOldDecodeTarget  int32
	}{
		{name: "the old prefill stays while old decodes are still needed", newDecodes: 1, wantOldPrefillTarget: 1, wantOldDecodeTarget: 1},
		{name: "the old generation retires once replacements cover every role", newDecodes: 2, wantOldPrefillTarget: 0, wantOldDecodeTarget: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD with one prefill and two decode replicas rolling with maxSurge=1, maxUnavailable=0")
			dgd := admissionTestRollingDGD(1, "0", map[string]int32{
				consts.ComponentTypePrefill: 1,
				consts.ComponentTypeDecode:  2,
			})
			newHash := betaDGDWorkersSpecHash(t, dgd)
			now := metav1.Now()

			t.Log("The old generation is complete; the new one has a prefill and the given decodes")
			oldPrefill := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 1, 1, now)
			newPrefill := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, newHash, 1, 1, now)
			oldDecode := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, testOldWorkerHash, 3-tt.newDecodes, 3-tt.newDecodes, now)
			newDecode := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, newHash, tt.newDecodes, tt.newDecodes, now)
			objs := []runtime.Object{oldPrefill, newPrefill, oldDecode, newDecode}
			members := []string{oldPrefillCard, newPrefillCard}
			addWorker := func(name, component, componentType, hash, dcd, card string) {
				pod := admissionTestPod(name, component, componentType, hash, dcd, true)
				objs = append(objs, pod, admissionTestMetadata(pod, admissionTestCards(card)))
				members = append(members, card)
			}
			addWorker("prefill-old", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, oldPrefill.Name, oldPrefillCard)
			addWorker("prefill-new", "prefill", consts.ComponentTypePrefill, newHash, newPrefill.Name, newPrefillCard)
			for i := range 3 - tt.newDecodes {
				addWorker(fmt.Sprintf("decode-old-%d", i), "decode", consts.ComponentTypeDecode, testOldWorkerHash, oldDecode.Name, fmt.Sprintf("test-dgd-old/decode/generate/1%d", i))
			}
			for i := range tt.newDecodes {
				addWorker(fmt.Sprintf("decode-new-%d", i), "decode", consts.ComponentTypeDecode, newHash, newDecode.Name, fmt.Sprintf("test-dgd-new/decode/generate/2%d", i))
			}

			t.Log("The frontend serves both complete generations")
			frontend := admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)
			objs = append(objs, frontend, admissionTestMetadata(frontend, admissionTestRecord(true, members...)))
			r := createTestReconcilerWithStatus(dgd, withObjects(objs...))

			t.Log("Plan the rollout step")
			result, err := r.buildRollingUpdateContext(context.Background(), dgd)
			require.NoError(t, err)
			assert.Equal(t, tt.wantOldPrefillTarget, result.OldWorkerReplicaTargetsByComponent["prefill"])
			assert.Equal(t, tt.wantOldPrefillTarget, result.OldWorkerReplicaTargetsByDCD[oldPrefill.Name])
			assert.Equal(t, tt.wantOldDecodeTarget, result.OldWorkerReplicaTargetsByComponent["decode"])
		})
	}
}

func TestBuildRollingUpdateContext_RollForwardKeepsTheServedGeneration(t *testing.T) {
	t.Log("Build a DGD whose prefill component rolls with maxSurge=1 and maxUnavailable=0")
	dgd := admissionTestRollingDGD(1, "0", map[string]int32{consts.ComponentTypePrefill: 1})

	t.Log("The original generation serves; a failed newer generation left a Ready prefill no frontend serves")
	const failedHash = "failedh2"
	servedDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 1, 1, metav1.NewTime(metav1.Now().Add(-time.Hour)))
	failedDCD := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, failedHash, 1, 1, metav1.Now())
	servedPod := admissionTestPod("prefill-served", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, servedDCD.Name, true)
	failedPod := admissionTestPod("prefill-failed", "prefill", consts.ComponentTypePrefill, failedHash, failedDCD.Name, true)
	frontend := admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)
	objs := []runtime.Object{
		servedDCD, failedDCD, servedPod, failedPod, frontend,
		admissionTestMetadata(servedPod, admissionTestCards("test-dgd-served/prefill/generate/1")),
		admissionTestMetadata(failedPod, admissionTestCards("test-dgd-failed/prefill/generate/2")),
		admissionTestMetadata(frontend, admissionTestRecord(true, "test-dgd-served/prefill/generate/1")),
	}

	t.Log("Plan the first step toward a new generation")
	r := createTestReconcilerWithStatus(dgd, withObjects(objs...))
	result, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)

	t.Log("Surge room comes from the unserved replica, not the serving one")
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByDCD[servedDCD.Name])
	assert.Equal(t, int32(0), result.OldWorkerReplicaTargetsByDCD[failedDCD.Name])
}

func TestServingOldGeneration(t *testing.T) {
	older := metav1.NewTime(metav1.Now().Add(-time.Hour))
	newer := metav1.Now()
	dcd := func(name, hash string, created metav1.Time) *nvidiacomv1beta1.DynamoComponentDeployment {
		return &nvidiacomv1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: created,
			Labels:            map[string]string{consts.KubeLabelDynamoWorkerHash: hash},
		}}
	}
	step := func(gated bool, servedA, servedB int32) workerRolloutStep {
		return workerRolloutStep{
			gated:     gated,
			oldDCDs:   []*nvidiacomv1beta1.DynamoComponentDeployment{dcd("role-a", "hasha", older), dcd("role-b", "hashb", newer)},
			servedOld: map[string]int32{"role-a": servedA, "role-b": servedB},
		}
	}
	tests := []struct {
		name      string
		steps     []workerRolloutStep
		want      string
		wantFound bool
	}{
		{name: "the most served generation keeps serving", steps: []workerRolloutStep{step(true, 2, 1), step(true, 1, 1)}, want: "hasha", wantFound: true},
		{name: "the newest generation keeps serving on a tie", steps: []workerRolloutStep{step(true, 1, 1), step(true, 1, 1)}, want: "hashb", wantFound: true},
		{name: "components that do not wait for frontends do not count", steps: []workerRolloutStep{step(true, 1, 1), step(false, 5, 0)}, want: "hashb", wantFound: true},
		{name: "no generation is kept when frontends serve no old replica", steps: []workerRolloutStep{step(true, 0, 0)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Pick the old generation that keeps serving")
			got, found := servingOldGeneration(tt.steps)
			assert.Equal(t, tt.wantFound, found)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestBuildRollingUpdateContext_RollForwardFromTwoServedGenerationsFreesSurge(t *testing.T) {
	const (
		partialHash  = "partialh2"
		cardAPrefill = "test-dgd-a/prefill/generate/1"
		cardADecode  = "test-dgd-a/decode/generate/2"
		cardBPrefill = "test-dgd-b/prefill/generate/3"
		cardBDecode  = "test-dgd-b/decode/generate/4"
	)
	t.Log("Build a DGD with one prefill and two decode replicas rolling with maxSurge=1, maxUnavailable=0")
	dgd := admissionTestRollingDGD(1, "0", map[string]int32{
		consts.ComponentTypePrefill: 1,
		consts.ComponentTypeDecode:  2,
	})
	createdA := metav1.NewTime(metav1.Now().Add(-time.Hour))
	createdB := metav1.NewTime(metav1.Now().Add(-30 * time.Minute))

	t.Log("Generations A and B each serve one prefill and one decode; B's second decode failed")
	prefillA := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 1, 1, createdA)
	decodeA := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, testOldWorkerHash, 1, 1, createdA)
	prefillB := admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, partialHash, 1, 1, createdB)
	decodeB := admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, partialHash, 2, 1, createdB)
	podPrefillA := admissionTestPod("prefill-a", "prefill", consts.ComponentTypePrefill, testOldWorkerHash, prefillA.Name, true)
	podDecodeA := admissionTestPod("decode-a", "decode", consts.ComponentTypeDecode, testOldWorkerHash, decodeA.Name, true)
	podPrefillB := admissionTestPod("prefill-b", "prefill", consts.ComponentTypePrefill, partialHash, prefillB.Name, true)
	podDecodeB := admissionTestPod("decode-b", "decode", consts.ComponentTypeDecode, partialHash, decodeB.Name, true)
	frontend := admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)
	r := createTestReconcilerWithStatus(dgd, withObjects(
		prefillA, decodeA, prefillB, decodeB, podPrefillA, podDecodeA, podPrefillB, podDecodeB, frontend,
		admissionTestMetadata(podPrefillA, admissionTestCards(cardAPrefill)),
		admissionTestMetadata(podDecodeA, admissionTestCards(cardADecode)),
		admissionTestMetadata(podPrefillB, admissionTestCards(cardBPrefill)),
		admissionTestMetadata(podDecodeB, admissionTestCards(cardBDecode)),
		admissionTestMetadata(frontend, admissionTestRecord(true, cardAPrefill, cardADecode, cardBPrefill, cardBDecode)),
	))

	t.Log("The first step toward generation C retires A's prefill and keeps B serving")
	result, err := r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(0), result.OldWorkerReplicaTargetsByDCD[prefillA.Name])
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByDCD[prefillB.Name])
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByDCD[decodeB.Name])

	t.Log("Once A's prefill is gone the frontend serves only B, and C's prefill gets the surge room")
	prefillA = admissionTestRolloutDCD(t, dgd, "prefill", consts.ComponentTypePrefill, testOldWorkerHash, 0, 0, createdA)
	decodeB = admissionTestRolloutDCD(t, dgd, "decode", consts.ComponentTypeDecode, partialHash, 1, 1, createdB)
	r = createTestReconcilerWithStatus(dgd, withObjects(
		prefillA, decodeA, prefillB, decodeB, podDecodeA, podPrefillB, podDecodeB, frontend,
		admissionTestMetadata(podDecodeA, admissionTestCards(cardADecode)),
		admissionTestMetadata(podPrefillB, admissionTestCards(cardBPrefill)),
		admissionTestMetadata(podDecodeB, admissionTestCards(cardBDecode)),
		admissionTestMetadata(frontend, admissionTestRecord(true, cardBPrefill, cardBDecode)),
	))
	result, err = r.buildRollingUpdateContext(context.Background(), dgd)
	require.NoError(t, err)
	assert.Equal(t, int32(1), result.NewWorkerReplicaTargetsByComponent["prefill"])
	assert.Equal(t, int32(1), result.OldWorkerReplicaTargetsByDCD[prefillB.Name])
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
}

func TestAdmissionEventsRequeueOnlyDGDsThatAwaitFrontends(t *testing.T) {
	tests := []struct {
		name string
		// workerHash is the DGD's recorded worker hash; empty means the desired one.
		workerHash  string
		rolling     bool
		wantRequeue bool
	}{
		{name: "a rolling DGD", rolling: true, wantRequeue: true},
		{name: "a DGD whose workers are about to roll", workerHash: testOldWorkerHash, wantRequeue: true},
		{name: "an idle DGD", wantRequeue: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD with a frontend and a worker, and the frontend's discovery records")
			dgd := admissionTestRollingDGD(1, "0", map[string]int32{consts.ComponentTypeDecode: 1})
			dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = betaDGDWorkersSpecHash(t, dgd)
			if tt.workerHash != "" {
				dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = tt.workerHash
			}
			if tt.rolling {
				dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress}
			}
			frontend := admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)
			records := admissionTestMetadata(frontend, admissionTestRecord(true))
			r := createTestDGDReconcilerWithStatus(dgd, withObjects(frontend))

			t.Log("Frontend pod and record events requeue the DGD only while it awaits frontends")
			var want []ctrl.Request
			if tt.wantRequeue {
				want = []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "default", Name: "test-dgd"}}}
			}
			assert.Equal(t, want, r.mapDGDFrontendPodToRequests(context.Background(), frontend))
			assert.Equal(t, want, r.mapWorkerMetadataToDGDRequests(context.Background(), records))
		})
	}

	t.Log("Pods of other components and of unknown DGDs requeue nothing")
	r := createTestDGDReconcilerWithStatus(createTestDGD("other-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{}))
	worker := admissionTestPod("worker", "prefill", consts.ComponentTypePrefill, "hash", "worker-dcd", true)
	assert.Empty(t, r.mapDGDFrontendPodToRequests(context.Background(), worker))
	assert.Empty(t, r.mapDGDFrontendPodToRequests(context.Background(),
		admissionTestFrontendPod("frontend-0", admissionTestTemplate, true)))
}

func TestWorkerMetadataChangedPredicate(t *testing.T) {
	t.Log("Build a projected resource and a copy whose records changed")
	pod := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", true)
	original := admissionTestMetadata(pod, map[string]any{"event_sources": map[string]any{}})
	unchanged := original.DeepCopy()
	unchanged.SetResourceVersion("2")
	changed := admissionTestMetadata(pod, admissionTestRecord(true))
	pred := workerMetadataChangedPredicate()

	t.Log("Only changes to the projected records requeue")
	assert.True(t, pred.Create(event.CreateEvent{Object: original}))
	assert.True(t, pred.Delete(event.DeleteEvent{Object: original}))
	assert.True(t, pred.Update(event.UpdateEvent{ObjectOld: original, ObjectNew: changed}))
	assert.False(t, pred.Update(event.UpdateEvent{ObjectOld: original, ObjectNew: unchanged}))
	assert.False(t, pred.Generic(event.GenericEvent{Object: original}))
}

func TestMapWorkerMetadataToDGDRequests(t *testing.T) {
	t.Log("Build a resource owned by a pod of a rolling DGD, one owned by an unknown pod, and one owned by an earlier pod of the same name")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{})
	dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress}
	pod := admissionTestPod("frontend", "frontend", consts.ComponentTypeFrontend, "", "", true)
	published := admissionTestMetadata(pod, map[string]any{})
	orphan := admissionTestMetadata(admissionTestPod("gone", "frontend", consts.ComponentTypeFrontend, "", "", true), map[string]any{})
	earlier := admissionTestMetadataOwnedBy(pod, "earlier-uid", map[string]any{})
	r := createTestDGDReconcilerWithStatus(dgd, withObjects(pod))

	t.Log("Only the resource of the current pod maps to its DGD")
	requests := r.mapWorkerMetadataToDGDRequests(context.Background(), published)
	require.Len(t, requests, 1)
	assert.Equal(t, types.NamespacedName{Namespace: "default", Name: "test-dgd"}, requests[0].NamespacedName)
	assert.Empty(t, r.mapWorkerMetadataToDGDRequests(context.Background(), orphan))
	assert.Empty(t, r.mapWorkerMetadataToDGDRequests(context.Background(), earlier))
}

func TestReplacementsServedCondition(t *testing.T) {
	gatedStep := func(name string, served, available int32) workerRolloutStep {
		return workerRolloutStep{componentName: name, gated: true, newServed: served, newAvailable: available}
	}
	ready := workermetadata.FrontendAdmission{Capable: true, Synced: true}
	tests := []struct {
		name        string
		admission   frontendAdmissionView
		steps       []workerRolloutStep
		wantStatus  metav1.ConditionStatus
		wantReason  string
		wantMessage string
	}{
		{name: "no component waits for frontends", admission: frontendAdmissionView{}, steps: []workerRolloutStep{{componentName: "decode"}}},
		{
			name:        "every available replacement is served",
			admission:   frontendAdmissionView{gated: true, admissions: []workermetadata.FrontendAdmission{ready}},
			steps:       []workerRolloutStep{gatedStep("decode", 2, 2)},
			wantStatus:  metav1.ConditionTrue,
			wantReason:  reasonReplacementsServed,
			wantMessage: "Every ready frontend serves the available replacement workers",
		},
		{
			name:        "replacements wait for frontends",
			admission:   frontendAdmissionView{gated: true, admissions: []workermetadata.FrontendAdmission{ready}},
			steps:       []workerRolloutStep{gatedStep("decode", 1, 2), gatedStep("prefill", 0, 1), gatedStep("encode", 1, 1)},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonWaitingForFrontends,
			wantMessage: "Available replacement workers that every ready frontend serves: decode 1/2, prefill 0/1",
		},
		{
			name:        "a ready frontend has no current records",
			admission:   frontendAdmissionView{gated: true, pending: true, admissions: []workermetadata.FrontendAdmission{ready, {}}, unsynced: []string{"frontend-1"}},
			steps:       []workerRolloutStep{gatedStep("decode", 0, 2)},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonFrontendRecordsMissing,
			wantMessage: "Frontends without current serving records: frontend-1; keeping every available previous worker",
		},
		{
			name:        "no frontend is ready",
			admission:   frontendAdmissionView{gated: true, pending: true},
			steps:       []workerRolloutStep{gatedStep("decode", 0, 2)},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonFrontendRecordsMissing,
			wantMessage: "No frontend is ready; keeping every available previous worker",
		},
		{
			name:        "frontends are rolling out",
			admission:   frontendAdmissionView{gated: true, pending: true, frontendsRolling: true, admissions: []workermetadata.FrontendAdmission{{}}, unsynced: []string{"frontend-0"}},
			steps:       []workerRolloutStep{gatedStep("decode", 0, 2)},
			wantStatus:  metav1.ConditionFalse,
			wantReason:  reasonFrontendsRolling,
			wantMessage: "Frontends are rolling out; keeping every available previous worker until they finish",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the condition for the rollout step")
			condition := replacementsServedCondition(7, tt.admission, tt.steps)
			if tt.wantReason == "" {
				assert.Nil(t, condition)
				return
			}
			require.NotNil(t, condition)
			assert.Equal(t, metav1.Condition{
				Type:               conditionTypeReplacementsServed,
				Status:             tt.wantStatus,
				ObservedGeneration: 7,
				Reason:             tt.wantReason,
				Message:            tt.wantMessage,
			}, *condition)
		})
	}
}

func TestApplyReconcileResultSetsAndClearsReplacementsServed(t *testing.T) {
	t.Log("A rollout step that waits for frontends sets the condition")
	result := workloadProgramResult{}
	waiting := metav1.Condition{
		Type:    conditionTypeReplacementsServed,
		Status:  metav1.ConditionFalse,
		Reason:  reasonWaitingForFrontends,
		Message: "Available replacement workers that every ready frontend serves: decode 1/2",
	}
	result.applyReconcileResult(3, ReconcileResult{ReplacementsServed: &waiting})
	got := apimeta.FindStatusCondition(result.Status.Conditions, conditionTypeReplacementsServed)
	require.NotNil(t, got)
	assert.Equal(t, waiting.Message, got.Message)

	t.Log("A step that no longer waits for frontends removes it")
	result.applyReconcileResult(3, ReconcileResult{})
	assert.Nil(t, apimeta.FindStatusCondition(result.Status.Conditions, conditionTypeReplacementsServed))
}

func BenchmarkPlanRollingUpdate(b *testing.B) {
	for _, workersPerRole := range []int32{10, 100, 1000} {
		b.Run(fmt.Sprintf("workers_per_role=%d", workersPerRole), func(b *testing.B) {
			b.Log("Build a prefill and decode rollout where both generations run every replica")
			dgd := admissionTestRollingDGD(2, "0", map[string]int32{
				consts.ComponentTypePrefill: workersPerRole,
				consts.ComponentTypeDecode:  workersPerRole,
			})
			newHash := betaDGDWorkersSpecHash(b, dgd)
			now := metav1.Now()
			var objs []runtime.Object
			var members []string
			for _, role := range []string{consts.ComponentTypePrefill, consts.ComponentTypeDecode} {
				for _, hash := range []string{testOldWorkerHash, newHash} {
					dcd := admissionTestRolloutDCD(b, dgd, role, role, hash, workersPerRole, workersPerRole, now)
					objs = append(objs, dcd)
					for i := range workersPerRole {
						name := fmt.Sprintf("%s-%s-%d", role, hash, i)
						pod := admissionTestPod(name, role, role, hash, dcd.Name, true)
						card := fmt.Sprintf("test-dgd-%s/%s/generate/%x", hash, role, i)
						objs = append(objs, pod, admissionTestMetadata(pod, admissionTestCards(card)))
						members = append(members, card)
					}
				}
			}

			b.Log("Two frontends serve every worker")
			for i := range 2 {
				frontend := admissionTestFrontendPod(fmt.Sprintf("frontend-%d", i), admissionTestTemplate, true)
				objs = append(objs, frontend, admissionTestMetadata(frontend, admissionTestRecord(true, members...)))
			}
			r := createTestReconcilerWithStatus(dgd, withObjects(objs...))

			b.Log("Plan rollout steps")
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := r.planRollingUpdate(ctx, dgd); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
