// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"gotest.tools/v3/golden"
	"sigs.k8s.io/yaml"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestPipelineRequestDeadlineFailureSurvivesReadyChanges(t *testing.T) {
	t.Log("A scheduling cycle begins after Ready last became false")
	dgd := loadTestDGD(t, lpx.PipelineSingle, "build-v2")
	child := newLPXTestDeployment(t, dgd)
	setReadyCondition(child, v1beta1.DGDStatePending, "Waiting for scheduling")
	transition := metav1.NewTime(time.Now().Add(-time.Hour).UTC().Truncate(time.Second))
	child.Status.Conditions[0].LastTransitionTime = transition
	r := newLPXTestReconciler(t, nil, child, dgd)
	request := newTestPipelineRequest(child, newTestPodCliqueSet(child), "expired", time.Now().Add(-time.Minute), lpxv1alpha1.RequestPhasePending)
	previous := child.Status.DeepCopy()

	t.Log("Persist the failure independently of Ready's unchanged boolean transition")
	setSchedulingFailedCondition(child, true)
	require.NoError(t, r.updateStatus(t.Context(), child, previous))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Len(t, child.Status.Conditions, 2)
	require.True(t, transition.Equal(&child.Status.Conditions[0].LastTransitionTime))
	require.Equal(t, v1alpha1.LPXReadyReasonFailed, child.Status.Conditions[0].Reason)
	schedulingFailed := meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition)
	require.NotNil(t, schedulingFailed)
	require.True(t, schedulingFailed.LastTransitionTime.After(transition.Time))
	require.Equal(t, metav1.ConditionTrue, schedulingFailed.Status)
	require.Equal(t, child.Generation, schedulingFailed.ObservedGeneration)
	require.Equal(t, pipelineRequestDeadlineExceededReason, schedulingFailed.Reason)
	require.Nil(t, meta.FindStatusCondition(child.Status.Conditions, "Failed"))
	require.True(t, schedulingFailureCoversPipelineRequests(child, []*lpxv1alpha1.LPUPipelineRequest{request}))
	failure := schedulingFailed.DeepCopy()

	t.Log("Other pending, failed, and ready observations cannot authorize a scheduling retry")
	for _, state := range []v1beta1.DGDState{v1beta1.DGDStatePending, v1beta1.DGDStateFailed, v1beta1.DGDStateSuccessful} {
		previous = child.Status.DeepCopy()
		setReadyCondition(child, state, "An unrelated observation")
		require.NoError(t, r.updateStatus(t.Context(), child, previous))
		require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
		require.Len(t, child.Status.Conditions, 2)
		require.Equal(t, failure, meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition))
		require.True(t, isSchedulingFailedConditionCurrent(child))
		require.True(t, schedulingFailureCoversPipelineRequests(child, []*lpxv1alpha1.LPUPipelineRequest{request}))
	}

	t.Log("A later reconcile restores the deadline diagnostic without renewing cleanup authorization")
	previous = child.Status.DeepCopy()
	setSchedulingFailedCondition(child, false)
	require.NoError(t, r.updateStatus(t.Context(), child, previous))
	require.NoError(t, r.Get(t.Context(), client.ObjectKeyFromObject(child), child))
	require.Len(t, child.Status.Conditions, 2)
	require.Equal(t, v1alpha1.LPXReadyReasonFailed, child.Status.Conditions[0].Reason)
	require.Equal(t, failure, meta.FindStatusCondition(child.Status.Conditions, schedulingFailedCondition))
}

func TestLPXReadyConditionClassifiesState(t *testing.T) {
	for _, test := range []struct {
		state  v1beta1.DGDState
		status metav1.ConditionStatus
		reason string
	}{
		{v1beta1.DGDStatePending, metav1.ConditionFalse, v1alpha1.LPXReadyReasonPending},
		{v1beta1.DGDStateFailed, metav1.ConditionFalse, v1alpha1.LPXReadyReasonFailed},
		{v1beta1.DGDStateSuccessful, metav1.ConditionTrue, v1alpha1.LPXReadyReasonReady},
	} {
		t.Run(string(test.state), func(t *testing.T) {
			t.Log("One Ready condition carries the condition and the complete diagnostic")
			child := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 7}}
			setReadyCondition(child, test.state, "current observation")
			require.Len(t, child.Status.Conditions, 1)
			ready := child.Status.Conditions[0]
			require.Equal(t, "Ready", ready.Type)
			require.Equal(t, test.status, ready.Status)
			require.Equal(t, test.reason, ready.Reason)
			require.Equal(t, "current observation", ready.Message)
			require.Equal(t, child.Generation, ready.ObservedGeneration)
		})
	}
}

func TestLPXStatusSkipsUnchangedObservations(t *testing.T) {
	for _, state := range []v1beta1.DGDState{v1beta1.DGDStatePending, v1beta1.DGDStateSuccessful, v1beta1.DGDStateFailed} {
		t.Run(string(state), func(t *testing.T) {
			t.Log("Start from a persisted status with stable condition transition times")
			child := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", Generation: 1, ResourceVersion: "1"}}
			setReadyCondition(child, state, "current observation")
			transition := metav1.NewTime(time.Now().Add(-time.Hour).UTC().Truncate(time.Second))
			for index := range child.Status.Conditions {
				child.Status.Conditions[index].LastTransitionTime = transition
			}
			child.Status.Components = map[string]v1alpha1.LPXComponentStatus{
				"engine": {Conditions: []metav1.Condition{child.Status.Conditions[0]}},
			}
			r := &graphReconciler{Client: newLPXTestClient(t, child)}
			previous := child.Status.DeepCopy()
			writes := 0
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				SubResourceUpdate: func(ctx context.Context, delegated client.Client, subresource string, object client.Object, opts ...client.SubResourceUpdateOption) error {
					require.Equal(t, "status", subresource)
					writes++
					return delegated.SubResource(subresource).Update(ctx, object, opts...)
				},
			})

			t.Log("Message-only observations preserve transition time and an unchanged final status skips the write")
			different := v1beta1.DGDStateSuccessful
			if state == different {
				different = v1beta1.DGDStatePending
			}
			setReadyCondition(child, state, "not externally visible")
			setReadyCondition(child, state, "current observation")
			component := child.Status.Components["engine"]
			component.Conditions = []metav1.Condition{readyCondition(child.Generation, different, "intermediate observation")}
			meta.SetStatusCondition(&component.Conditions, readyCondition(child.Generation, state, "current observation"))
			child.Status.Components["engine"] = component
			require.Zero(t, writes)
			require.NoError(t, r.updateStatus(t.Context(), child, previous))
			require.Zero(t, writes)
			require.Equal(t, previous, &child.Status)

			t.Log("A graph transition is persisted without overwriting independently observed component readiness")
			setReadyCondition(child, different, "new observation")
			require.Zero(t, writes)
			require.NoError(t, r.updateStatus(t.Context(), child, previous))
			require.Equal(t, 1, writes)
			require.True(t, meta.FindStatusCondition(child.Status.Conditions, "Ready").LastTransitionTime.After(transition.Time))
			require.True(t, apiequality.Semantic.DeepEqual(previous.Components["engine"].Conditions, child.Status.Components["engine"].Conditions))

			t.Log("Observing the same component state for a new generation preserves its transition time")
			previous = child.Status.DeepCopy()
			child.Generation++
			require.NoError(t, r.Update(t.Context(), child))
			component = child.Status.Components["engine"]
			component.Conditions = []metav1.Condition{readyCondition(child.Generation, state, "current observation")}
			child.Status.Components["engine"] = component
			require.NoError(t, r.updateStatus(t.Context(), child, previous))
			require.Equal(t, 2, writes)
			ready := meta.FindStatusCondition(child.Status.Components["engine"].Conditions, v1alpha1.LPXReadyCondition)
			require.Equal(t, child.Generation, ready.ObservedGeneration)
			require.True(t, transition.Equal(&ready.LastTransitionTime))
		})
	}
}

// Update with: go test ./internal/controller/lpx -run TestPipelineRequestReadyConditionGolden -args -update
func TestPipelineRequestReadyConditionGolden(t *testing.T) {
	t.Log("Load scheduler receipts and preserve exact condition diagnostics in the golden")
	payload, err := os.ReadFile("testdata/pipeline-request-statuses.yaml")
	require.NoError(t, err)
	var statuses []lpxv1alpha1.LPUPipelineRequestStatus
	require.NoError(t, yaml.UnmarshalStrict(payload, &statuses))
	conditions := make(map[string]metav1.Condition, len(statuses))
	for _, status := range statuses {
		t.Run(string(status.Phase), func(t *testing.T) {
			t.Log("Project the current receipt without altering it or changing Ready's transition time")
			deployment := &v1alpha1.LPXGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Generation: 7},
				Status: v1alpha1.LPXGraphDeploymentStatus{Conditions: []metav1.Condition{{
					Type: v1alpha1.LPXReadyCondition, Status: metav1.ConditionFalse,
					LastTransitionTime: metav1.NewTime(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)),
				}}},
			}
			request := &lpxv1alpha1.LPUPipelineRequest{ObjectMeta: metav1.ObjectMeta{Generation: 7}, Status: status.DeepCopy()}
			before := request.DeepCopy()
			requests := map[string]*lpxv1alpha1.LPUPipelineRequest{"observed": request}
			require.False(t, setPipelineRequestReadyCondition(&deployment.Status.Conditions, deployment.Generation, requests))
			require.Len(t, deployment.Status.Conditions, 1)
			conditions[string(status.Phase)] = *meta.FindStatusCondition(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)

			t.Log("A failure outranks missing receipts and repeated map observations stay stable")
			requests["first-open"] = &lpxv1alpha1.LPUPipelineRequest{}
			require.False(t, setPipelineRequestReadyCondition(&deployment.Status.Conditions, deployment.Generation, requests))
			wantReason := v1alpha1.LPXReadyReasonPending
			if status.Phase == lpxv1alpha1.RequestPhaseUnsupported || status.Phase == "future" {
				wantReason = v1alpha1.LPXReadyReasonFailed
			}
			condition := *meta.FindStatusCondition(deployment.Status.Conditions, v1alpha1.LPXReadyCondition)
			require.Equal(t, wantReason, condition.Reason)
			for range 10 {
				require.False(t, setPipelineRequestReadyCondition(&deployment.Status.Conditions, deployment.Generation, requests))
				require.Equal(t, condition, *meta.FindStatusCondition(deployment.Status.Conditions, v1alpha1.LPXReadyCondition))
			}
			require.Equal(t, before, request)
		})
	}
	actual, err := json.MarshalIndent(conditions, "", "  ")
	require.NoError(t, err)
	golden.Assert(t, string(actual)+"\n", "pipeline-request-ready.golden.json")
}

func TestPipelineRequestReadyConditionRequiresCurrentReceipts(t *testing.T) {
	for _, tc := range []struct {
		name     string
		phase    lpxv1alpha1.RequestPhase
		observed *int64
		runtime  v1beta1.DGDState
		allBound bool
	}{
		{name: "Bound preserves runtime readiness", phase: lpxv1alpha1.RequestPhaseBound, observed: ptr.To(int64(7)), runtime: v1beta1.DGDStateSuccessful, allBound: true},
		{name: "Bound still needs runtime readiness", phase: lpxv1alpha1.RequestPhaseBound, observed: ptr.To(int64(7)), runtime: v1beta1.DGDStatePending, allBound: true},
		{name: "stale Bound", phase: lpxv1alpha1.RequestPhaseBound, observed: ptr.To(int64(6)), runtime: v1beta1.DGDStateSuccessful},
		{name: "unobserved Bound", phase: lpxv1alpha1.RequestPhaseBound, runtime: v1beta1.DGDStateSuccessful},
		{name: "stale failure", phase: lpxv1alpha1.RequestPhaseUnsupported, observed: ptr.To(int64(6)), runtime: v1beta1.DGDStatePending},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Combine scheduler freshness with the previously observed runtime condition")
			child := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 7}}
			setReadyCondition(child, tc.runtime, "Runtime observation")
			before := child.Status.DeepCopy()
			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{Generation: 7},
				Status: &lpxv1alpha1.LPUPipelineRequestStatus{
					Phase: tc.phase, ObservedGeneration: tc.observed,
					Diagnostics: []lpxv1alpha1.StatusDiagnostic{{Detail: "Old diagnostic"}},
				},
			}
			require.Equal(t, tc.allBound, setPipelineRequestReadyCondition(&child.Status.Conditions, child.Generation, map[string]*lpxv1alpha1.LPUPipelineRequest{"request": request}))
			if tc.allBound {
				require.Equal(t, before, &child.Status)
			} else {
				ready := meta.FindStatusCondition(child.Status.Conditions, v1alpha1.LPXReadyCondition)
				require.Equal(t, v1alpha1.LPXReadyReasonPending, ready.Reason)
				require.NotContains(t, ready.Message, "Old diagnostic")
			}
		})
	}
}

func TestSchedulingFailureCoversPipelineRequests(t *testing.T) {
	failedAt := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name            string
		conditionStatus metav1.ConditionStatus
		reason          string
		generation      int64
		noCondition     bool
		noFailureTime   bool
		start           *metav1.Time
		created         *metav1.Time
		noStatus        bool
		wantCurrent     bool
		wantCovered     bool
	}{
		{name: "no condition", noCondition: true},
		{name: "false condition", conditionStatus: metav1.ConditionFalse, reason: pipelineRequestDeadlineExceededReason, generation: 2},
		{name: "unrelated failure", conditionStatus: metav1.ConditionTrue, reason: "OtherFailure", generation: 2},
		{name: "previous generation covers older cycle", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 1, start: ptr.To(metav1.NewTime(failedAt.Add(-time.Minute))), wantCovered: true},
		{name: "previous generation does not cover newer cycle", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 1, start: ptr.To(metav1.NewTime(failedAt.Add(time.Second)))},
		{name: "missing failure timestamp", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, noFailureTime: true, wantCurrent: true},
		{name: "older cycle", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, start: ptr.To(metav1.NewTime(failedAt.Add(-time.Minute))), wantCurrent: true, wantCovered: true},
		{name: "same timestamp", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, start: ptr.To(metav1.NewTime(failedAt)), wantCurrent: true},
		{name: "newer cycle", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, start: ptr.To(metav1.NewTime(failedAt.Add(time.Second))), wantCurrent: true},
		{name: "no scheduler receipt uses creation", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, noStatus: true, wantCurrent: true, wantCovered: true},
		{name: "no scheduling start uses creation", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, wantCurrent: true, wantCovered: true},
		{name: "zero scheduling start uses creation", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, start: &metav1.Time{}, wantCurrent: true, wantCovered: true},
		{name: "creation after failure is not covered", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, created: ptr.To(metav1.NewTime(failedAt.Add(time.Second))), noStatus: true, wantCurrent: true},
		{name: "unpublished request is not covered", conditionStatus: metav1.ConditionTrue, reason: pipelineRequestDeadlineExceededReason, generation: 2, created: &metav1.Time{}, noStatus: true, wantCurrent: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("A durable failure authorizes cleanup of older cycles independently of retry generation")
			deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 2}}
			if !tc.noCondition {
				condition := metav1.Condition{
					Type: schedulingFailedCondition, Status: tc.conditionStatus, Reason: tc.reason,
					ObservedGeneration: tc.generation, LastTransitionTime: metav1.NewTime(failedAt),
				}
				if tc.noFailureTime {
					condition.LastTransitionTime = metav1.Time{}
				}
				deployment.Status.Conditions = []metav1.Condition{condition}
			}
			request := &lpxv1alpha1.LPUPipelineRequest{
				ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.NewTime(failedAt.Add(-time.Hour))},
				Status:     &lpxv1alpha1.LPUPipelineRequestStatus{SchedulingStartedAt: tc.start},
			}
			if tc.created != nil {
				request.CreationTimestamp = *tc.created
			}
			if tc.noStatus {
				request.Status = nil
			}
			require.Equal(t, tc.wantCurrent, isSchedulingFailedConditionCurrent(deployment))
			require.Equal(t, tc.wantCovered, schedulingFailureCoversPipelineRequests(deployment, []*lpxv1alpha1.LPUPipelineRequest{request}))
		})
	}
}

func TestSetSchedulingFailedCondition(t *testing.T) {
	for _, tc := range []struct {
		name               string
		previousGeneration int64
		renew              bool
		wantGeneration     int64
	}{
		{name: "first failure", renew: true, wantGeneration: 2},
		{name: "current failure unchanged", previousGeneration: 2, wantGeneration: 2},
		{name: "previous failure unchanged", previousGeneration: 1, wantGeneration: 1},
		{name: "new expired cycle", previousGeneration: 2, renew: true, wantGeneration: 2},
		{name: "new generation expires", previousGeneration: 1, renew: true, wantGeneration: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Keep the recorded generation and timestamp unless a newly expired cycle requires renewal")
			deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 2}}
			before := metav1.NewTime(time.Now().Add(-time.Hour))
			if tc.previousGeneration != 0 {
				deployment.Status.Conditions = []metav1.Condition{{
					Type: schedulingFailedCondition, Status: metav1.ConditionTrue, Reason: pipelineRequestDeadlineExceededReason,
					ObservedGeneration: tc.previousGeneration, LastTransitionTime: before,
				}}
			}
			setSchedulingFailedCondition(deployment, tc.renew)
			condition := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
			require.Equal(t, tc.wantGeneration, condition.ObservedGeneration)
			require.Equal(t, metav1.ConditionTrue, condition.Status)
			if tc.renew {
				require.True(t, condition.LastTransitionTime.After(before.Time))
			} else {
				require.Equal(t, before, condition.LastTransitionTime)
			}
		})
	}
}

func TestClearPreviousSchedulingFailure(t *testing.T) {
	for _, tc := range []struct {
		name               string
		recordedGeneration int64
		wantCleared        bool
	}{
		{name: "no failure"},
		{name: "previous generation", recordedGeneration: 1, wantCleared: true},
		{name: "same generation", recordedGeneration: 2},
		{name: "newer observed generation", recordedGeneration: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Only a later desired generation authorizes retrying expired pipeline requests")
			deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Generation: 2}}
			if tc.recordedGeneration > 0 {
				deployment.Status.Conditions = []metav1.Condition{{
					Type: schedulingFailedCondition, Status: metav1.ConditionTrue,
					Reason: pipelineRequestDeadlineExceededReason, ObservedGeneration: tc.recordedGeneration,
				}}
			}
			before := deployment.Status.DeepCopy()
			acknowledgeSchedulingRetry(deployment)
			if !tc.wantCleared {
				require.Equal(t, *before, deployment.Status)
				return
			}
			condition := meta.FindStatusCondition(deployment.Status.Conditions, schedulingFailedCondition)
			require.Equal(t, metav1.ConditionFalse, condition.Status)
			require.Equal(t, deployment.Generation, condition.ObservedGeneration)
			require.Equal(t, "LPXSchedulingRetryAuthorized", condition.Reason)
		})
	}
}

func TestPipelineRequestDiagnosticMessageLimit(t *testing.T) {
	for _, tc := range []struct{ name, detail string }{
		{name: "ASCII", detail: strings.Repeat("a", 4096)},
		{name: "multibyte", detail: strings.Repeat("界", 4096/len("界"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Truncate oversized scheduler diagnostics to the condition limit with an ASCII suffix")
			diagnostics := make([]lpxv1alpha1.StatusDiagnostic, 32)
			for i := range diagnostics {
				diagnostics[i] = lpxv1alpha1.StatusDiagnostic{Code: "code", Subject: "subject", Detail: tc.detail}
			}
			message := pipelineRequestDiagnosticMessage("LPX scheduler diagnostic summary", diagnostics)
			require.LessOrEqual(t, len(message), maxConditionMessageSize)
			require.True(t, utf8.ValidString(message))
			require.True(t, strings.HasSuffix(message, "..."))
		})
	}
}
