/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestLargeCommittedInventoryIsBounded(t *testing.T) {
	dgd, dgpb, qualification := statusTestObjects(1)
	committed := int32(math.MaxInt32)
	dgd.Spec.Components[0].Replicas = &committed
	dgpb.Spec.BudgetWatts = math.MaxInt64
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": committed}

	t.Log("Aggregate a representable MaxInt32 committed target without per-slot allocation")
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		qualification,
		time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build large committed inventory: %v", err)
	}
	wantWatts := int64(math.MaxInt32) * 350
	if len(status.Components) != 1 || status.Components[0].InGatePhysicalGPUs != committed ||
		status.Ledger.InGateReservedWatts != wantWatts || status.Ledger.TotalChargedWatts != wantWatts {
		t.Fatalf("large committed inventory = %+v, want %d in-gate GPUs and %d watts", status, committed, wantWatts)
	}
}

func TestDeletionNeverUndercounts(t *testing.T) {
	t.Log("Charge a terminating assigned Pod conservatively until deletion is observed")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	now := metav1.NewTime(time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC))
	terminating := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", &now)
	inventory := dgdPowerBudgetInventory{Pods: []corev1.Pod{terminating}}
	withTerminating, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		inventory,
		powerReportHistory{PodUIDs: []string{"pod-uid"}},
		qualification,
		now.Time,
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("buildDGPBInventoryStatus() terminating error = %v", err)
	}
	if withTerminating.Ledger.TotalChargedWatts != 700 ||
		len(withTerminating.Components) != 1 ||
		withTerminating.Components[0].TerminatingReplicas != 1 ||
		withTerminating.Components[0].UnknownPhysicalGPUs != 1 {
		t.Fatalf("terminating inventory = %#v, want one terminating 700W unknown charge", withTerminating)
	}

	t.Log("Replace the disappeared Pod with the durable committed in-gate reservation")
	afterDeletion, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		qualification,
		now.Time,
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("buildDGPBInventoryStatus() deleted error = %v", err)
	}
	if afterDeletion.Ledger.TotalChargedWatts != 350 ||
		afterDeletion.Components[0].InGatePhysicalGPUs != 1 {
		t.Fatalf("post-deletion inventory = %#v, want committed 350W in-gate reservation", afterDeletion)
	}
}

func TestRolloutExtrasAreDisjointAndCloseFence(t *testing.T) {
	t.Log("Build one committed Pod plus one terminating rollout-extra Pod")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	deleting := metav1.NewTime(time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC))

	t.Log("Publish the extra watts in their disjoint class and close the phase fence")
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{Pods: []corev1.Pod{
			powerStatusTestPod(dgd, "worker-00-current", "current", nil),
			powerStatusTestPod(dgd, "worker-00-old", "old", &deleting),
		}},
		powerReportHistory{All: true},
		qualification,
		time.Date(2026, 8, 15, 12, 0, 1, 0, time.UTC),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("buildDGPBInventoryStatus() rollout error = %v", err)
	}
	if status.Ledger.UnknownWatts != 700 || status.Ledger.RolloutExtraWatts != 700 ||
		status.Ledger.TotalChargedWatts != 1400 {
		t.Fatalf("rollout ledger = %#v, want 700W committed unknown plus disjoint 700W extra", status.Ledger)
	}
	if status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying ||
		!powerbudget.ReplicaFenceClosed(status.Phase) {
		t.Fatalf("rollout phase = %q, want closed Applying", status.Phase)
	}
}

func TestRolloutReservationSurvivesCacheLagUntilObservedCompletion(t *testing.T) {
	t.Log("Persist a pre-child-write rollout reservation before DCD inventory catches up")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	dgpb.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		InGateReservedWatts: 350,
		RolloutExtraWatts:   350,
		TotalChargedWatts:   700,
	}

	t.Log("A DGPB-triggered pass with stale child caches must not erase the reservation")
	pending, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		qualification,
		time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build cache-lag inventory: %v", err)
	}
	if pending.Ledger.RolloutExtraWatts != 350 || pending.Ledger.TotalChargedWatts != 700 ||
		pending.RolloutInProgress {
		t.Fatalf("pending reservation = %#v, want retained unseen 350W extra", pending)
	}

	t.Log("Once watched rollout state is visible, record that the reservation became active")
	dgpb.Status = pending
	active, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{RolloutInProgress: true},
		powerReportHistory{},
		qualification,
		time.Date(2026, 8, 15, 12, 0, 1, 0, time.UTC),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build active rollout inventory: %v", err)
	}
	if !active.RolloutInProgress || active.Ledger.RolloutExtraWatts != 350 {
		t.Fatalf("active reservation = %#v, want active retained 350W extra", active)
	}

	t.Log("Release only after active rollout state and all watched extra obligations disappear")
	dgpb.Status = active
	completed, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		qualification,
		time.Date(2026, 8, 15, 12, 0, 2, 0, time.UTC),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build completed rollout inventory: %v", err)
	}
	if completed.RolloutInProgress || completed.Ledger.RolloutExtraWatts != 0 {
		t.Fatalf("completed reservation = %#v, want rollout extra released", completed)
	}
}

func TestReportedPodHistoryDoesNotConsumeNewPodInGate(t *testing.T) {
	t.Log("Build one established reported Pod and one new never-reported Pod")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 2}
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(350)
	report, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       string(dgd.UID),
		Component:    "worker-00",
		PodUID:       "established-uid",
		Node:         "node-a",
		AllocationID: expectedPowerAllocationID("established-uid", nvidiacomv1beta1.MainContainerName, []string{"GPU-1"}),
		GPUs: []powerbudget.AgentGPUReport{{
			UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
			ConstraintMinWatts: 300, ConstraintMaxWatts: 700,
			PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
			WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: now,
		}},
	})
	if err != nil {
		t.Fatalf("encode established report: %v", err)
	}
	established := powerStatusTestPod(dgd, "worker-00-established", "established-uid", nil)
	established.Annotations = map[string]string{powerbudget.AgentReportAnnotation: string(report)}
	newPod := powerStatusTestPod(dgd, "worker-00-new", "new-uid", nil)

	t.Log("Charge only the established allocation as reported and preserve B_c for the new Pod")
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{Pods: []corev1.Pod{established, newPod}},
		powerReportHistory{PodUIDs: []string{"established-uid"}},
		qualification,
		now,
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build mixed report-history status: %v", err)
	}
	if status.Ledger.EnforcedWatts != 350 || status.Ledger.InGateReservedWatts != 350 ||
		status.Ledger.UnknownWatts != 0 || status.Ledger.TotalChargedWatts != 700 {
		t.Fatalf("mixed history ledger = %#v, want 350W enforced plus 350W inGate", status.Ledger)
	}
}

func TestFreshEvidenceExpiresToUnknown(t *testing.T) {
	t.Log("Build a fresh exact-readback report for one committed Pod")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(350)
	report, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       string(dgd.UID),
		Component:    "worker-00",
		PodUID:       "pod-uid",
		Node:         "node-a",
		AllocationID: expectedPowerAllocationID("pod-uid", nvidiacomv1beta1.MainContainerName, []string{"GPU-1"}),
		GPUs: []powerbudget.AgentGPUReport{{
			UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
			ConstraintMinWatts: 300, ConstraintMaxWatts: 700,
			PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
			WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: now,
		}},
	})
	if err != nil {
		t.Fatalf("encode report: %v", err)
	}
	pod := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", nil)
	pod.Annotations = map[string]string{powerbudget.AgentReportAnnotation: string(report)}

	t.Log("Use exact enforced watts while evidence is fresh")
	fresh, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}},
		powerReportHistory{PodUIDs: []string{"pod-uid"}},
		qualification,
		now,
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build fresh status: %v", err)
	}
	if fresh.Ledger.EnforcedWatts != 350 || fresh.Ledger.UnknownWatts != 0 {
		t.Fatalf("fresh ledger = %#v, want exact 350W", fresh.Ledger)
	}

	t.Log("Reclassify the same evidence at U_c and close the fence after expiry")
	expired, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}},
		powerReportHistory{PodUIDs: []string{"pod-uid"}},
		qualification,
		now.Add(time.Minute+time.Nanosecond),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build expired status: %v", err)
	}
	if expired.Ledger.UnknownWatts != 700 || expired.Ledger.EnforcedWatts != 0 ||
		expired.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale {
		t.Fatalf("expired status = %#v, want 700W unknown and Stale", expired)
	}
}

func TestLiveRangeDriftIsUnqualifiedAndConservativelyCharged(t *testing.T) {
	for _, test := range []struct {
		name        string
		liveMin     int64
		liveMax     int64
		wantUnknown int64
	}{
		{name: "minimum drift", liveMin: 299, liveMax: 700, wantUnknown: 700},
		{name: "lower maximum drift", liveMin: 300, liveMax: 699, wantUnknown: 700},
		{name: "higher maximum drift", liveMin: 300, liveMax: 800, wantUnknown: 800},
	} {
		t.Run(test.name, func(t *testing.T) {
			dgd, dgpb, qualification := statusTestObjects(1)
			dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
			now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
			enforced := int64(350)
			report, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
				Version:      powerbudget.AgentReportDocumentVersion,
				DGDUID:       string(dgd.UID),
				Component:    "worker-00",
				PodUID:       "pod-uid",
				Node:         "node-a",
				AllocationID: expectedPowerAllocationID("pod-uid", nvidiacomv1beta1.MainContainerName, []string{"GPU-1"}),
				GPUs: []powerbudget.AgentGPUReport{{
					UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
					ConstraintMinWatts: test.liveMin, ConstraintMaxWatts: test.liveMax,
					PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
					WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
					EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: now,
				}},
			})
			if err != nil {
				t.Fatalf("encode drifted report: %v", err)
			}
			pod := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", nil)
			pod.Annotations = map[string]string{powerbudget.AgentReportAnnotation: string(report)}

			status, err := buildDGPBInventoryStatus(
				dgd,
				dgpb,
				dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}},
				powerReportHistory{PodUIDs: []string{"pod-uid"}},
				qualification,
				now,
				time.Minute,
				false,
			)
			if err != nil {
				t.Fatalf("build drifted inventory: %v", err)
			}
			if status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified ||
				status.Ledger.EnforcedWatts != 0 ||
				status.Ledger.UnknownWatts != test.wantUnknown {
				t.Fatalf("drifted status = %#v, want Unqualified and %dW unknown", status, test.wantUnknown)
			}
		})
	}
}

func TestHealthyApplyingReopensReplicaFence(t *testing.T) {
	t.Log("Start from a durable committed target whose workload is still Applying")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(350)
	report, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       string(dgd.UID),
		Component:    "worker-00",
		PodUID:       "pod-uid",
		Node:         "node-a",
		AllocationID: expectedPowerAllocationID("pod-uid", nvidiacomv1beta1.MainContainerName, []string{"GPU-1"}),
		GPUs: []powerbudget.AgentGPUReport{{
			UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
			ConstraintMinWatts: 300, ConstraintMaxWatts: 700,
			PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
			WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: now,
		}},
	})
	if err != nil {
		t.Fatalf("encode healthy report: %v", err)
	}
	pod := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", nil)
	pod.Annotations = map[string]string{powerbudget.AgentReportAnnotation: string(report)}

	t.Log("Fresh exact evidence for every committed slot reopens the phase-derived fence")
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}},
		powerReportHistory{PodUIDs: []string{"pod-uid"}},
		qualification,
		now,
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build healthy Applying status: %v", err)
	}
	if status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle ||
		powerbudget.ReplicaFenceClosed(status.Phase) {
		t.Fatalf("healthy phase = %q, want open Idle", status.Phase)
	}

	t.Log("The reopened fence permits a subsequent in-budget DGDSA increase")
	dgpb.Status = status
	decision, err := admitTransactionalReplicaVector(
		dgd,
		dgpb,
		powerbudget.ReplicaVector{"worker-00": 1},
		powerbudget.ReplicaVector{"worker-00": 2},
	)
	if err != nil || !decision.Accepted {
		t.Fatalf("post-convergence admission = (%+v, %v), want accepted", decision, err)
	}
}

func TestStatusBounded(t *testing.T) {
	t.Log("Encode aggregate status at the supported component limit")
	dgd, dgpb, qualification := statusTestObjects(powerbudget.MaxPowerManagedComponents)
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		qualification,
		time.Now(),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("buildDGPBInventoryStatus() at limit error = %v", err)
	}
	if len(status.Components) != powerbudget.MaxPowerManagedComponents {
		t.Fatalf("status components = %d, want %d", len(status.Components), powerbudget.MaxPowerManagedComponents)
	}
	if _, err := powerbudget.EncodeStatusSnapshot(status); err != nil {
		t.Fatalf("EncodeStatusSnapshot() at limit error = %v", err)
	}

	t.Log("Reject one more component rather than emitting unbounded status")
	tooLarge, tooLargeDGPB, tooLargeQualification := statusTestObjects(powerbudget.MaxPowerManagedComponents + 1)
	if _, err := buildDGPBInventoryStatus(
		tooLarge,
		tooLargeDGPB,
		dgdPowerBudgetInventory{},
		powerReportHistory{},
		tooLargeQualification,
		time.Now(),
		time.Minute,
		false,
	); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("buildDGPBInventoryStatus() over limit error = %v, want bounded rejection", err)
	}
}

func TestFailClosed(t *testing.T) {
	t.Log("Evaluate each unsupported topology or qualification mode against the phase fence")
	tests := []struct {
		name            string
		mutate          func(*nvidiacomv1beta1.DynamoGraphDeployment)
		qualificationOK bool
		grovePathway    bool
		wantPhase       nvidiacomv1beta1.DynamoGraphPowerBudgetPhase
	}{
		{name: "Grove", grovePathway: true, qualificationOK: true, wantPhase: nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale},
		{
			name: "multinode",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			},
			qualificationOK: true,
			wantPhase:       nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale,
		},
		{
			name: "checkpoint enabled",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			},
			qualificationOK: true,
			wantPhase:       nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale,
		},
		{name: "unknown SKU", wantPhase: nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build the watched inventory and require a closed fail-safe phase")
			dgd, dgpb, qualification := statusTestObjects(1)
			dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
			if tc.mutate != nil {
				tc.mutate(dgd)
			}
			if !tc.qualificationOK {
				qualification = powerbudget.QualificationIndex{}
			}
			status, err := buildDGPBInventoryStatus(
				dgd,
				dgpb,
				dgdPowerBudgetInventory{},
				powerReportHistory{},
				qualification,
				time.Now(),
				time.Minute,
				tc.grovePathway,
			)
			if err != nil {
				t.Fatalf("buildDGPBInventoryStatus() error = %v", err)
			}
			if status.Phase != tc.wantPhase || !powerbudget.ReplicaFenceClosed(status.Phase) {
				t.Fatalf("phase = %q, want closed %q", status.Phase, tc.wantPhase)
			}
		})
	}
}

func powerStatusTestPod(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	name string,
	uid types.UID,
	deletionTimestamp *metav1.Time,
) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         dgd.Namespace,
			Name:              name,
			UID:               uid,
			DeletionTimestamp: deletionTimestamp,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           "worker-00",
				consts.KubeLabelDynamoSelector:            "worker-00",
				consts.KubeLabelDynamoComponentType:       string(nvidiacomv1beta1.ComponentTypeWorker),
			},
		},
		Spec: corev1.PodSpec{NodeName: "node-a"},
	}
}

func statusTestObjects(componentCount int) (
	*nvidiacomv1beta1.DynamoGraphDeployment,
	*nvidiacomv1beta1.DynamoGraphPowerBudget,
	powerbudget.QualificationIndex,
) {
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "graph",
			UID:       types.UID("dgd-uid"),
		},
	}
	for i := 0; i < componentCount; i++ {
		name := fmt.Sprintf("worker-%02d", i)
		dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName:  name,
			ComponentType:  nvidiacomv1beta1.ComponentTypeWorker,
			Replicas:       ptr.To(int32(1)),
			ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
			PodTemplate: &corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
					consts.KubeAnnotationGPUPowerLimit: "350",
				}},
				Spec: corev1.PodSpec{
					NodeSelector: map[string]string{qualifiedGPUProductLabel: "test-sku"},
					Containers: []corev1.Container{{
						Name: nvidiacomv1beta1.MainContainerName,
						Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
							corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
						}},
					}},
				},
			},
		})
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name, Generation: 1},
		Spec: nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
			BudgetWatts: 100_000,
			Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
		},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID:                  string(dgd.UID),
			Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
			CommittedReplicaTargets: map[string]int32{},
		},
	}
	qualification := powerbudget.QualificationIndex{
		"test-sku": {MinWatts: 300, MaxWatts: 700},
	}
	return dgd, dgpb, qualification
}
