//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"sync"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lwscontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/lws"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestFailureRecordsPersistUsesServerUIDAndResourceVersion(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := t.Context()
	now := time.Date(2026, 11, 9, 15, 0, 0, 0, time.UTC)

	t.Log("Create a DCD and read the server-assigned UID/resourceVersion")
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		Name:      "failure-records-cas",
		Namespace: env.Namespace(),
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:      new(int32(1)),
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, dcd))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), dcd))
	require.NotEmpty(t, dcd.UID)
	require.NotEmpty(t, dcd.ResourceVersion)
	preWriteRV := dcd.ResourceVersion

	t.Log("Seed an unrelated status condition that metadata Patch must not touch")
	dcd.Status.Conditions = []metav1.Condition{{
		Type:               "ForeignReady",
		Status:             metav1.ConditionTrue,
		Reason:             "Seeded",
		Message:            "keep-me",
		LastTransitionTime: metav1.NewTime(now),
	}}
	require.NoError(t, env.Client().Status().Update(ctx, dcd))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), dcd))
	preWriteRV = dcd.ResourceVersion

	batch := failureRecordBatch(dcd, 0, "leader-uid", "pod-uid", "decode-0", now)

	t.Log("Persist the first CurrentFailure against the live object")
	result, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), dcd, batch, now)
	require.NoError(t, err)
	require.True(t, result.Written)

	stored := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), stored))
	require.Contains(t, stored.Annotations, lwscontroller.FailureRecordsAnnotationKey)
	assert.NotEqual(t, preWriteRV, stored.ResourceVersion)
	require.NotNil(t, findCondition(stored.Status.Conditions, "ForeignReady"))
	assert.Equal(t, "keep-me", findCondition(stored.Status.Conditions, "ForeignReady").Message)

	t.Log("A genuinely new identity against a stale RV must Conflict without mutation")
	stale := stored.DeepCopy()
	stale.ResourceVersion = preWriteRV
	staleBatch := failureRecordBatch(dcd, 1, "leader-uid-stale", "pod-uid-stale", "decode-1", now)
	beforeConflict := stored.DeepCopy()
	conflict, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), stale, staleBatch, now.Add(time.Second))
	require.NoError(t, err)
	assert.True(t, conflict.Conflict)
	assert.False(t, conflict.Written)
	assert.False(t, conflict.Noop)
	assert.Equal(t, lwscontroller.FailureRecordConflictRequeue, conflict.RequeueAfter)

	afterConflict := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), afterConflict))
	assert.Equal(t, beforeConflict.Annotations[lwscontroller.FailureRecordsAnnotationKey],
		afterConflict.Annotations[lwscontroller.FailureRecordsAnnotationKey])
	assert.Equal(t, beforeConflict.Spec, afterConflict.Spec)
	assert.Equal(t, beforeConflict.Status.Conditions, afterConflict.Status.Conditions)

	t.Log("Foreign annotations and status survive a successful owned-key write")
	fresh := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), fresh))
	if fresh.Annotations == nil {
		fresh.Annotations = map[string]string{}
	}
	fresh.Annotations["example.com/foreign"] = "keep-me"
	require.NoError(t, env.Client().Update(ctx, fresh))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), fresh))

	againBatch := failureRecordBatch(dcd, 1, "leader-uid-2", "pod-uid-2", "decode-1", now)
	again, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), fresh, againBatch, now.Add(2*time.Second))
	require.NoError(t, err)
	require.True(t, again.Written)

	final := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), final))
	assert.Equal(t, "keep-me", final.Annotations["example.com/foreign"])
	assert.Contains(t, final.Annotations, lwscontroller.FailureRecordsAnnotationKey)
	require.NotNil(t, findCondition(final.Status.Conditions, "ForeignReady"))
}

func TestFailureRecordsRejectSameNameReplacementUID(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := t.Context()
	now := time.Date(2026, 11, 9, 15, 0, 0, 0, time.UTC)

	t.Log("Create a DCD, persist a ledger, then recreate under the same name")
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		Name:      "failure-records-recreate",
		Namespace: env.Namespace(),
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:      new(int32(1)),
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, dcd))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), dcd))
	oldUID := dcd.UID
	oldRV := dcd.ResourceVersion
	batch := failureRecordBatch(dcd, 0, "leader-uid", "pod-uid", "decode-0", now)
	_, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), dcd, batch, now)
	require.NoError(t, err)

	require.NoError(t, env.Client().Delete(ctx, dcd))
	require.Eventually(t, func() bool {
		err := env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), &nvidiacomv1beta1.DynamoComponentDeployment{})
		return apierrors.IsNotFound(err)
	}, 30*time.Second, 100*time.Millisecond)

	replacement := &nvidiacomv1beta1.DynamoComponentDeployment{
		Name:      dcd.Name,
		Namespace: dcd.Namespace,
		Spec:      dcd.Spec,
	}
	require.NoError(t, env.Client().Create(ctx, replacement))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(replacement), replacement))
	require.NotEqual(t, oldUID, replacement.UID)
	liveAnnotation := replacement.Annotations[lwscontroller.FailureRecordsAnnotationKey]
	liveSpec := replacement.Spec.DeepCopy()
	liveStatus := replacement.Status.DeepCopy()

	t.Log("Old UID + old RV against the replacement must not mutate it")
	staleReq := replacement.DeepCopy()
	staleReq.UID = oldUID
	staleReq.ResourceVersion = oldRV
	staleBatch := failureRecordBatch(replacement, 2, "old-uid-leader", "old-uid-pod", "decode-old", now)
	staleBatch[0].DCDUID = oldUID
	out, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), staleReq, staleBatch, now.Add(time.Second))
	require.NoError(t, err)
	assert.False(t, out.Written)
	assert.True(t, out.Conflict)
	assert.False(t, out.Unavailable)
	assert.Equal(t, lwscontroller.FailureRecordConflictRequeue, out.RequeueAfter)

	afterStale := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(replacement), afterStale))
	assert.Equal(t, liveAnnotation, afterStale.Annotations[lwscontroller.FailureRecordsAnnotationKey])
	assert.Equal(t, *liveSpec, afterStale.Spec)
	assert.Equal(t, *liveStatus, afterStale.Status)

	t.Log("Wrong UID + fresh RV is Invalid/unavailable, not Conflict, and leaves the object unchanged")
	wrongUID := replacement.DeepCopy()
	wrongUID.UID = "definitely-not-the-live-uid"
	wrongBatch := failureRecordBatch(replacement, 3, "wrong-uid-leader", "wrong-uid-pod", "decode-wrong", now)
	wrongBatch[0].DCDUID = wrongUID.UID
	wrongOut, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), wrongUID, wrongBatch, now.Add(2*time.Second))
	require.NoError(t, err)
	assert.False(t, wrongOut.Written)
	assert.False(t, wrongOut.Conflict)
	assert.True(t, wrongOut.Unavailable)
	assert.Equal(t, time.Duration(0), wrongOut.RequeueAfter)

	afterWrong := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(replacement), afterWrong))
	assert.Equal(t, afterStale.Annotations[lwscontroller.FailureRecordsAnnotationKey],
		afterWrong.Annotations[lwscontroller.FailureRecordsAnnotationKey])
	assert.Equal(t, afterStale.Spec, afterWrong.Spec)

	t.Log("An inherited old-UID ledger on a replacement object stays unavailable")
	inherited := afterWrong.DeepCopy()
	if inherited.Annotations == nil {
		inherited.Annotations = map[string]string{}
	}
	inherited.Annotations[lwscontroller.FailureRecordsAnnotationKey] =
		`{"version":1,"dcdUID":"` + string(oldUID) + `","entries":[]}`
	require.NoError(t, env.Client().Update(ctx, inherited))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(replacement), inherited))
	legacyBatch := failureRecordBatch(inherited, 4, "legacy-leader", "legacy-pod", "decode-legacy", now)
	legacyOut, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), inherited, legacyBatch, now.Add(3*time.Second))
	require.NoError(t, err)
	assert.True(t, legacyOut.Unavailable)
	assert.False(t, legacyOut.Written)
}

func TestFailureRecordsSimultaneousWritersPreserveUnion(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	now := time.Date(2026, 11, 9, 15, 0, 0, 0, time.UTC)

	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		Name:      "failure-records-race",
		Namespace: env.Namespace(),
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:      new(int32(1)),
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, dcd))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), dcd))

	batchA := failureRecordBatch(dcd, 0, "race-leader-a", "race-pod-a", "decode-a", now)
	batchB := failureRecordBatch(dcd, 1, "race-leader-b", "race-pod-b", "decode-b", now)

	start := make(chan struct{})
	done := make(chan struct{})
	var wg sync.WaitGroup
	results := make([]lwscontroller.PersistResult, 2)
	errs := make([]error, 2)
	run := func(idx int, obj *nvidiacomv1beta1.DynamoComponentDeployment, batch []lwscontroller.Observation) {
		defer wg.Done()
		select {
		case <-start:
		case <-ctx.Done():
			errs[idx] = ctx.Err()
			return
		}
		results[idx], errs[idx] = lwscontroller.PersistFailureRecords(ctx, env.Client(), obj, batch, now)
	}

	wg.Add(2)
	go run(0, dcd.DeepCopy(), batchA)
	go run(1, dcd.DeepCopy(), batchB)
	close(start)
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("simultaneous writers timed out: %v", ctx.Err())
	}
	require.NoError(t, errs[0])
	require.NoError(t, errs[1])

	var winnerIdx, loserIdx int
	switch {
	case results[0].Written && results[1].Conflict:
		winnerIdx, loserIdx = 0, 1
	case results[1].Written && results[0].Conflict:
		winnerIdx, loserIdx = 1, 0
	default:
		t.Fatalf("expected one Written and one Conflict, got %#v %#v", results[0], results[1])
	}
	assert.Equal(t, lwscontroller.FailureRecordConflictRequeue, results[loserIdx].RequeueAfter)
	_ = winnerIdx

	loserBatch := batchB
	if loserIdx == 0 {
		loserBatch = batchA
	}
	fresh := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), fresh))
	second, err := lwscontroller.PersistFailureRecords(ctx, env.Client(), fresh, loserBatch, now.Add(time.Second))
	require.NoError(t, err)
	require.True(t, second.Written)

	final := &nvidiacomv1beta1.DynamoComponentDeployment{}
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(dcd), final))
	raw := final.Annotations[lwscontroller.FailureRecordsAnnotationKey]
	require.NotEmpty(t, raw)
	assert.Contains(t, raw, "race-leader-a")
	assert.Contains(t, raw, "race-leader-b")
}

func failureRecordBatch(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	group int,
	leaderUID, podUID, podName string,
	now time.Time,
) []lwscontroller.Observation {
	exit := int32(23)
	finished := metav1.NewTime(now.Add(-30 * time.Second))
	readyLTT := metav1.NewTime(now.Add(-45 * time.Second))
	g := group
	return []lwscontroller.Observation{{
		Verdict:                 lwscontroller.VerdictCurrentFailure,
		Reason:                  lwscontroller.ReasonCurrentMainNonzeroExit,
		Namespace:               dcd.Namespace,
		DCDName:                 dcd.Name,
		DCDUID:                  dcd.UID,
		DCDGeneration:           dcd.Generation,
		LWSName:                 dcd.Name,
		LWSUID:                  "lws-uid",
		LWSGeneration:           1,
		GroupIndex:              &g,
		Revision:                "rev-1",
		LeaderUID:               types.UID(leaderUID),
		PodName:                 podName,
		PodUID:                  types.UID(podUID),
		PodResourceVersion:      "11",
		Container:               "main",
		ExitCode:                &exit,
		FinishedAt:              &finished,
		Ready:                   corev1.ConditionFalse,
		ReadyLastTransitionTime: &readyLTT,
		ReadOrigin:              lwscontroller.ReadOriginManagerCache,
	}}
}

func findCondition(conditions []metav1.Condition, condType string) *metav1.Condition {
	for i := range conditions {
		if conditions[i].Type == condType {
			return &conditions[i]
		}
	}
	return nil
}
