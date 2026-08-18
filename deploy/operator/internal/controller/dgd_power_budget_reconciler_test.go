/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func TestInventoryEpoch(t *testing.T) {
	t.Log("Persist the first semantic inventory observation and advance the epoch")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "graph", Generation: 1},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID:         "dgd-uid",
			InventoryEpoch: 7,
			Phase:          nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()

	desired := dgpb.Status
	desired.ObservedGeneration = dgpb.Generation
	desired.InventoryEpoch = 8
	updated, err := persistDGPBInventoryStatus(context.Background(), kubeClient, dgpb, desired)
	if err != nil || !updated {
		t.Fatalf("persistDGPBInventoryStatus() initial = (%v, %v), want (true, nil)", updated, err)
	}
	if dgpb.Status.InventoryEpoch != 8 {
		t.Fatalf("inventoryEpoch = %d, want 8", dgpb.Status.InventoryEpoch)
	}

	t.Log("Build timestamp-varied reports for an otherwise identical semantic observation")
	labels := map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "graph",
		consts.KubeLabelDynamoComponent:           "worker",
		consts.KubeLabelDynamoSelector:            "worker",
		consts.KubeLabelDynamoComponentType:       string(nvidiacomv1beta1.ComponentTypeWorker),
	}
	oldPod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Labels:      labels,
		Annotations: map[string]string{powerbudget.AgentReportAnnotation: inventoryEpochReport(t, time.Unix(100, 0).UTC())},
	}}
	newPod := oldPod.DeepCopy()
	newPod.Annotations[powerbudget.AgentReportAnnotation] = inventoryEpochReport(t, time.Unix(200, 0).UTC())
	if !dgdWorkerPodEventPredicate().Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}) {
		t.Fatal("timestamp-only report refresh was not observed")
	}

	t.Log("Observe a timestamp-only refresh without a status write or epoch increment")
	updated, err = persistDGPBInventoryStatus(context.Background(), kubeClient, dgpb, dgpb.Status)
	if err != nil || updated {
		t.Fatalf("persistDGPBInventoryStatus() timestamp refresh = (%v, %v), want (false, nil)", updated, err)
	}
	if dgpb.Status.InventoryEpoch != 8 {
		t.Fatalf("timestamp-only inventoryEpoch = %d, want 8", dgpb.Status.InventoryEpoch)
	}

	t.Log("Advance the epoch for a semantic rollout change")
	desired = dgpb.Status
	desired.RolloutInProgress = true
	desired.InventoryEpoch = 9
	updated, err = persistDGPBInventoryStatus(context.Background(), kubeClient, dgpb, desired)
	if err != nil || !updated {
		t.Fatalf("persistDGPBInventoryStatus() semantic change = (%v, %v), want (true, nil)", updated, err)
	}
	if dgpb.Status.InventoryEpoch != 9 {
		t.Fatalf("semantic-change inventoryEpoch = %d, want 9", dgpb.Status.InventoryEpoch)
	}
}

func TestInventoryStatusConflictRequiresReevaluation(t *testing.T) {
	t.Log("Keep a stale inventory writer from overwriting a newer committed reservation")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "graph"},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			InventoryEpoch:          7,
			CommittedReplicaTargets: map[string]int32{"worker": 2},
			Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
				EnforcedWatts: 600, TotalChargedWatts: 600,
			},
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	stale := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stale); err != nil {
		t.Fatalf("read stale inventory snapshot: %v", err)
	}

	newer := stale.DeepCopy()
	newer.Status.CommittedReplicaTargets = map[string]int32{"worker": 3}
	newer.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 600, InGateReservedWatts: 300, TotalChargedWatts: 900,
	}
	if err := kubeClient.Status().Update(context.Background(), newer); err != nil {
		t.Fatalf("publish newer admission reservation: %v", err)
	}

	desired := stale.Status
	desired.InventoryEpoch++
	updated, err := persistDGPBInventoryStatus(context.Background(), kubeClient, stale, desired)
	if !apierrors.IsConflict(err) || updated {
		t.Fatalf("stale inventory write = (%v, %v), want (false, conflict)", updated, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read power budget after conflict: %v", err)
	}
	if stored.Status.CommittedReplicaTargets["worker"] != 3 ||
		stored.Status.Ledger.InGateReservedWatts != 300 ||
		stored.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf("stale inventory overwrote committed reservation: %+v", stored.Status)
	}
}

func TestSemanticInventoryFingerprint(t *testing.T) {
	t.Log("Fingerprint one bounded watched inventory")
	dgd, dgpb, _ := statusTestObjects(1)
	desired := dgpb.Status

	pod := corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Namespace: dgd.Namespace,
		Name:      "worker-00-pod",
		UID:       types.UID("pod-a"),
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponent:           "worker-00",
			consts.KubeLabelDynamoSelector:            "worker-00",
			consts.KubeLabelDynamoComponentType:       string(nvidiacomv1beta1.ComponentTypeWorker),
		},
	}, Spec: corev1.PodSpec{NodeName: "node-a"}}
	base := dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}}
	baseFingerprint, err := calculatePowerInventoryFingerprint(dgd, base, desired, powerReportHistory{})
	if err != nil {
		t.Fatalf("calculate base fingerprint: %v", err)
	}

	t.Log("Change the fingerprint for a same-charge Pod UID replacement")
	replacement := base
	replacement.Pods = []corev1.Pod{*pod.DeepCopy()}
	replacement.Pods[0].UID = types.UID("pod-b")
	replacementFingerprint, err := calculatePowerInventoryFingerprint(dgd, replacement, desired, powerReportHistory{})
	if err != nil {
		t.Fatalf("calculate replacement fingerprint: %v", err)
	}
	if replacementFingerprint == baseFingerprint {
		t.Fatal("same-charge Pod UID replacement did not change inventory fingerprint")
	}

	t.Log("Change the fingerprint when a DGDSA requested replica changes")
	adapterInventory := base
	adapterInventory.ScalingAdapters = []nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{{
		ObjectMeta: metav1.ObjectMeta{Name: "graph-worker-00", UID: types.UID("adapter-a"), Generation: 1},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: 1,
			DGDRef: nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
				Name: dgd.Name, ServiceName: "worker-00",
			},
		},
	}}
	requestedOne, err := calculatePowerInventoryFingerprint(dgd, adapterInventory, desired, powerReportHistory{})
	if err != nil {
		t.Fatalf("calculate requested-one fingerprint: %v", err)
	}
	adapterInventory.ScalingAdapters[0].Spec.Replicas = 2
	requestedTwo, err := calculatePowerInventoryFingerprint(dgd, adapterInventory, desired, powerReportHistory{})
	if err != nil {
		t.Fatalf("calculate requested-two fingerprint: %v", err)
	}
	if requestedOne == requestedTwo {
		t.Fatal("DGDSA requested-replica change did not change inventory fingerprint")
	}

	t.Log("Normalize report observation timestamps out of the semantic fingerprint")
	withReport := base
	withReport.Pods = []corev1.Pod{*pod.DeepCopy()}
	withReport.Pods[0].Annotations = map[string]string{
		powerbudget.AgentReportAnnotation: semanticFingerprintReport(t, dgd, &pod, time.Unix(100, 0).UTC()),
	}
	firstTimestamp, err := calculatePowerInventoryFingerprint(
		dgd, withReport, desired, powerReportHistory{PodUIDs: []string{"pod-a"}},
	)
	if err != nil {
		t.Fatalf("calculate first timestamp fingerprint: %v", err)
	}
	withReport.Pods[0].Annotations[powerbudget.AgentReportAnnotation] = semanticFingerprintReport(
		t,
		dgd,
		&pod,
		time.Unix(200, 0).UTC(),
	)
	secondTimestamp, err := calculatePowerInventoryFingerprint(
		dgd, withReport, desired, powerReportHistory{PodUIDs: []string{"pod-a"}},
	)
	if err != nil {
		t.Fatalf("calculate second timestamp fingerprint: %v", err)
	}
	if firstTimestamp != secondTimestamp {
		t.Fatal("timestamp-only report refresh changed inventory fingerprint")
	}
}

func TestPowerReportExpiryRequeue(t *testing.T) {
	t.Log("Schedule reconciliation at the earliest decoded report expiry")
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	enforced := int64(350)
	report, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version: powerbudget.AgentReportDocumentVersion, DGDUID: "dgd", Component: "worker",
		PodUID: "pod", Node: "node", AllocationID: "allocation",
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
	pods := []corev1.Pod{{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
		powerbudget.AgentReportAnnotation: string(report),
	}}}}
	if got := nextPowerReportRequeue(pods, now, time.Minute); got != time.Minute {
		t.Fatalf("freshness requeue = %s, want 1m", got)
	}
	if got := nextPowerReportRequeue(pods, now.Add(time.Minute), time.Minute); got != time.Nanosecond {
		t.Fatalf("boundary freshness requeue = %s, want 1ns", got)
	}
	if got := nextPowerReportRequeue(pods, now.Add(time.Minute+time.Nanosecond), time.Minute); got != 0 {
		t.Fatalf("expired freshness requeue = %s, want 0", got)
	}
}

func TestPowerInventoryPredicates(t *testing.T) {
	t.Log("Observe Pod termination metadata that changes the published inventory")
	labels := map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: "graph",
		consts.KubeLabelDynamoComponent:           "worker",
		consts.KubeLabelDynamoSelector:            "worker",
		consts.KubeLabelDynamoComponentType:       string(nvidiacomv1beta1.ComponentTypeWorker),
	}
	oldPod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: labels}}
	newPod := oldPod.DeepCopy()
	deleting := metav1.NewTime(time.Now())
	newPod.DeletionTimestamp = &deleting
	if !dgdWorkerPodEventPredicate().Update(event.UpdateEvent{ObjectOld: oldPod, ObjectNew: newPod}) {
		t.Fatal("Pod deletion-timestamp transition was not observed")
	}
	t.Log("Observe DCD creation as a new inventory dependency")
	if !dgdDCDInventoryEventPredicate().Create(event.CreateEvent{Object: &nvidiacomv1beta1.DynamoComponentDeployment{}}) {
		t.Fatal("DCD creation was not observed")
	}

	t.Log("Observe deployment-wide rollout status transitions on the primary DGD")
	oldDGD := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
	}}}
	newDGD := oldDGD.DeepCopy()
	newDGD.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress}
	if !dgdPowerInventoryStatusPredicate().Update(event.UpdateEvent{ObjectOld: oldDGD, ObjectNew: newDGD}) {
		t.Fatal("DGD rollout-status transition was not observed")
	}
}

func inventoryEpochReport(t *testing.T, observedAt time.Time) string {
	t.Helper()
	enforced := int64(350)
	encoded, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       "dgd-uid",
		Component:    "worker",
		PodUID:       "pod-uid",
		Node:         "node-a",
		AllocationID: "allocation-a",
		GPUs: []powerbudget.AgentGPUReport{{
			UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
			ConstraintMinWatts: 300, ConstraintMaxWatts: 700,
			PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
			WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: observedAt,
		}},
	})
	if err != nil {
		t.Fatalf("encode inventory-epoch report: %v", err)
	}
	return string(encoded)
}

func semanticFingerprintReport(
	t *testing.T,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	pod *corev1.Pod,
	observedAt time.Time,
) string {
	t.Helper()
	enforced := int64(350)
	encoded, err := powerbudget.EncodeAgentReport(powerbudget.AgentReport{
		Version:      powerbudget.AgentReportDocumentVersion,
		DGDUID:       string(dgd.UID),
		Component:    "worker-00",
		PodUID:       string(pod.UID),
		Node:         pod.Spec.NodeName,
		AllocationID: expectedPowerAllocationID(string(pod.UID), nvidiacomv1beta1.MainContainerName, []string{"GPU-1"}),
		GPUs: []powerbudget.AgentGPUReport{{
			UUID: "GPU-1", RequestedWatts: 350, TargetWatts: 350,
			ConstraintMinWatts: 300, ConstraintMaxWatts: 700,
			PolicyOutcome: powerbudget.AgentPolicyOutcomeAnnotated,
			WriteOutcome:  powerbudget.AgentWriteOutcomeSucceeded, ReadbackOutcome: powerbudget.AgentReadbackOutcomeSucceeded,
			EnforcedCapWatts: &enforced, Actuator: powerbudget.AgentActuatorNVML, ObservedAt: observedAt,
		}},
	})
	if err != nil {
		t.Fatalf("encode semantic-fingerprint report: %v", err)
	}
	return string(encoded)
}

type noReadAfterStatusPatchClient struct {
	client.Client
	patched bool
}

func (c *noReadAfterStatusPatchClient) Get(
	ctx context.Context,
	key client.ObjectKey,
	object client.Object,
	opts ...client.GetOption,
) error {
	if c.patched {
		return fmt.Errorf("read after status patch")
	}
	return c.Client.Get(ctx, key, object, opts...)
}

func (c *noReadAfterStatusPatchClient) Status() client.SubResourceWriter {
	return &trackStatusPatchWriter{SubResourceWriter: c.Client.Status(), patched: &c.patched}
}

type trackStatusPatchWriter struct {
	client.SubResourceWriter
	patched *bool
}

type failAfterInventoryStatePatchClient struct {
	client.Client
	failNext bool
}

func (c *failAfterInventoryStatePatchClient) Patch(
	ctx context.Context,
	object client.Object,
	patch client.Patch,
	opts ...client.PatchOption,
) error {
	if err := c.Client.Patch(ctx, object, patch, opts...); err != nil {
		return err
	}
	if c.failNext {
		if _, isDGPB := object.(*nvidiacomv1beta1.DynamoGraphPowerBudget); isDGPB &&
			object.GetAnnotations()[powerInventoryStateAnnotation] != "" {
			c.failNext = false
			return fmt.Errorf("injected response loss after inventory-state patch")
		}
	}
	return nil
}

func (w *trackStatusPatchWriter) Patch(
	ctx context.Context,
	object client.Object,
	patch client.Patch,
	opts ...client.SubResourcePatchOption,
) error {
	if err := w.SubResourceWriter.Patch(ctx, object, patch, opts...); err != nil {
		return err
	}
	*w.patched = true
	return nil
}

func TestCacheLagConverges(t *testing.T) {
	t.Log("Patch aggregate inventory status without an immediate read-after-write")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core scheme: %v", err)
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{
		Namespace: "test",
		Name:      "graph",
		UID:       types.UID("dgd-uid"),
		Annotations: map[string]string{
			nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
		},
	}}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name, Generation: 1},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID: string(dgd.UID),
			Phase:  nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		},
	}
	baseClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	trackedClient := &noReadAfterStatusPatchClient{Client: baseClient}
	reconciler := &DynamoGraphDeploymentReconciler{Client: trackedClient}

	updated, _, err := reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || !updated {
		t.Fatalf("reconcileDGDPowerBudgetInventory() first = (%v, %v), want (true, nil)", updated, err)
	}
	if !trackedClient.patched {
		t.Fatal("inventory reconciliation did not patch status")
	}
	t.Log("Persist the matching operator-private convergence marker without a controller readback")
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := baseClient.Get(
		context.Background(),
		types.NamespacedName{Namespace: dgpb.Namespace, Name: dgpb.Name},
		stored,
	); err != nil {
		t.Fatalf("read persisted DGPB in test: %v", err)
	}
	state, valid := loadPowerInventoryState(stored)
	if !valid || state.TargetEpoch != stored.Status.InventoryEpoch || state.TargetEpoch != 1 {
		t.Fatalf("persisted inventory state = (%+v, valid=%v), status epoch=%d", state, valid, stored.Status.InventoryEpoch)
	}

	t.Log("Converge on a later cached observation without another write")
	trackedClient.patched = false
	updated, _, err = reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || updated {
		t.Fatalf("reconcileDGDPowerBudgetInventory() converged = (%v, %v), want (false, nil)", updated, err)
	}
}

func TestReportHistorySurvivesAmbiguousMarkerPatchFailure(t *testing.T) {
	t.Log("Initialize one never-reported assigned Pod and its B_c inventory marker")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core scheme: %v", err)
	}
	dgd, dgpb, qualification := statusTestObjects(1)
	dgd.Annotations = map[string]string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
	}
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 1}
	pod := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", nil)
	baseClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb, &pod).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:               baseClient,
		PowerQualification:   qualification,
		PowerReportFreshness: time.Minute,
		PowerNow:             func() time.Time { return now },
	}
	updated, _, err := reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || !updated {
		t.Fatalf("initialize inventory = (%v, %v), want (true, nil)", updated, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgpb.Namespace, Name: dgpb.Name}
	if err := baseClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read initialized DGPB: %v", err)
	}
	initialEpoch := stored.Status.InventoryEpoch
	if stored.Status.Ledger.InGateReservedWatts != 350 {
		t.Fatalf("initial ledger = %#v, want 350W inGate", stored.Status.Ledger)
	}

	t.Log("Persist report history first, then lose the successful patch response")
	currentPod := &corev1.Pod{}
	podKey := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	if err := baseClient.Get(context.Background(), podKey, currentPod); err != nil {
		t.Fatalf("read Pod before report: %v", err)
	}
	currentPod.Annotations = map[string]string{
		powerbudget.AgentReportAnnotation: semanticFingerprintReport(t, dgd, currentPod, now),
	}
	if err := baseClient.Update(context.Background(), currentPod); err != nil {
		t.Fatalf("publish report: %v", err)
	}
	failingClient := &failAfterInventoryStatePatchClient{Client: baseClient, failNext: true}
	reconciler.Client = failingClient
	if _, _, err := reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd); err == nil {
		t.Fatal("history marker response loss was not surfaced")
	}
	if failingClient.failNext {
		t.Fatal("history reconciliation did not attempt the marker patch")
	}
	if err := baseClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read DGPB after ambiguous marker patch: %v", err)
	}
	state, valid := loadPowerInventoryState(stored)
	if !valid || !powerReportHistoryEqual(state, powerReportHistory{PodUIDs: []string{"pod-uid"}}) {
		t.Fatalf("durable history after response loss = (%+v, valid=%v)", state, valid)
	}
	if stored.Status.InventoryEpoch != initialEpoch {
		t.Fatalf("status advanced before history durability: got %d, want %d", stored.Status.InventoryEpoch, initialEpoch)
	}

	t.Log("Remove the report and retry from durable history without returning to B_c")
	if err := baseClient.Get(context.Background(), podKey, currentPod); err != nil {
		t.Fatalf("read Pod before report removal: %v", err)
	}
	delete(currentPod.Annotations, powerbudget.AgentReportAnnotation)
	if err := baseClient.Update(context.Background(), currentPod); err != nil {
		t.Fatalf("remove report: %v", err)
	}
	reconciler.Client = baseClient
	updated, _, err = reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || !updated {
		t.Fatalf("retry missing report inventory = (%v, %v), want (true, nil)", updated, err)
	}
	if err := baseClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read retried DGPB: %v", err)
	}
	if stored.Status.Ledger.UnknownWatts != 700 || stored.Status.Ledger.InGateReservedWatts != 0 {
		t.Fatalf("retried ledger = %#v, want 700W unknown and no inGate regression", stored.Status.Ledger)
	}
	if stored.Status.InventoryEpoch <= initialEpoch {
		t.Fatalf("retried epoch = %d, want > %d", stored.Status.InventoryEpoch, initialEpoch)
	}
}

func TestReportedPodHistoryIsBoundedAndDurable(t *testing.T) {
	t.Log("Keep a newly assigned Pod in the B_c class until any report exists")
	dgd, dgpb, _ := statusTestObjects(1)
	pod := powerStatusTestPod(dgd, "worker-00-pod", "pod-uid", nil)
	inventory := dgdPowerBudgetInventory{Pods: []corev1.Pod{pod}}
	state, valid := loadPowerInventoryState(dgpb)
	if valid {
		t.Fatal("new DGPB unexpectedly had an inventory state marker")
	}
	if got := mergeReportedPowerPods(dgpb, inventory, state, valid); got.All || len(got.PodUIDs) != 0 {
		t.Fatalf("new assigned Pod history = %+v, want empty", got)
	}

	t.Log("Record even malformed nonempty evidence so removal can never return to B_c")
	inventory.Pods[0].Annotations = map[string]string{powerbudget.AgentReportAnnotation: "{"}
	reported := mergeReportedPowerPods(dgpb, inventory, state, valid)
	if reported.All || len(reported.PodUIDs) != 1 || reported.PodUIDs[0] != "pod-uid" {
		t.Fatalf("reported Pod history = %+v, want [pod-uid]", reported)
	}
	fingerprint := strings.Repeat("a", 64)
	dgpb.Annotations = map[string]string{powerInventoryStateAnnotation: fmt.Sprintf(
		`{"v":1,"e":1,"f":%q,"p":["pod-uid"]}`,
		fingerprint,
	)}
	state, valid = loadPowerInventoryState(dgpb)
	if !valid {
		t.Fatal("persisted component history marker was rejected")
	}
	inventory.Pods[0].Annotations = nil
	reported = mergeReportedPowerPods(dgpb, inventory, state, valid)
	if reported.All || len(reported.PodUIDs) != 1 || reported.PodUIDs[0] != "pod-uid" {
		t.Fatalf("history after report removal = %+v, want [pod-uid]", reported)
	}

	t.Log("Fail closed for every current component if established metadata is missing or corrupt")
	dgpb.Status.InventoryEpoch = 2
	dgpb.Annotations[powerInventoryStateAnnotation] = "corrupt"
	state, valid = loadPowerInventoryState(dgpb)
	reported = mergeReportedPowerPods(dgpb, inventory, state, valid)
	if valid || !reported.All || len(reported.PodUIDs) != 0 {
		t.Fatalf("corrupt-state history = (%+v, valid=%v), want fail-closed saturation", reported, valid)
	}

	t.Log("Saturate to one bounded fail-closed bit instead of growing history without limit")
	fullState := powerInventoryState{ReportedPodUIDs: make([]string, maxReportedPodUIDs)}
	for i := range fullState.ReportedPodUIDs {
		fullState.ReportedPodUIDs[i] = fmt.Sprintf("pod-%04d", i)
	}
	inventory.Pods[0].UID = "pod-overflow"
	inventory.Pods[0].Annotations = map[string]string{powerbudget.AgentReportAnnotation: "{"}
	reported = mergeReportedPowerPods(dgpb, inventory, fullState, true)
	if !reported.All || len(reported.PodUIDs) != 0 {
		t.Fatalf("saturated history = %+v, want bounded all-reported mode", reported)
	}
}

func TestCreateTimeVectorBeforeWorkloads(t *testing.T) {
	t.Log("Create one DGDSA seed for each nonzero DGD component before workloads")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "graph",
			UID:       types.UID("dgd-uid"),
			Annotations: map[string]string{
				nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
				nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation:   "2400",
				nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation: "1",
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:  "prefill",
					ComponentType:  nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:       ptr.To(int32(2)),
					ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
				},
				{
					ComponentName:  "decode",
					ComponentType:  nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:       ptr.To(int32(3)),
					ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
				},
			},
		},
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID: string(dgd.UID),
			Phase:  nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing,
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
		).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}

	wait, err := reconciler.reconcileTransactionalPowerBootstrap(context.Background(), dgd)
	if err != nil {
		t.Fatalf("reconcileTransactionalPowerBootstrap() seed error = %v", err)
	}
	if !wait {
		t.Fatal("reconcileTransactionalPowerBootstrap() seed wait = false, want true")
	}
	for component, want := range map[string]int32{"prefill": 2, "decode": 3} {
		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
		key := types.NamespacedName{Namespace: dgd.Namespace, Name: generateAdapterName(dgd.Name, component)}
		if err := kubeClient.Get(context.Background(), key, adapter); err != nil {
			t.Fatalf("get %s seed: %v", component, err)
		}
		if adapter.Spec.Replicas != want {
			t.Fatalf("%s seed replicas = %d, want %d", component, adapter.Spec.Replicas, want)
		}
	}

	t.Log("Preserve the create-only seed without rereading later DGD replica mutations")
	dgd.Spec.Components[0].Replicas = ptr.To(int32(8))
	created, err := newDGDScalingAdaptersReconciler(reconciler.Client, reconciler.Recorder).
		ReconcileCreateTimeSeeds(context.Background(), dgd)
	if err != nil || created {
		t.Fatalf("ReconcileCreateTimeSeeds() repeat = (%v, %v), want (false, nil)", created, err)
	}
	adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
	prefillKey := types.NamespacedName{Namespace: dgd.Namespace, Name: generateAdapterName(dgd.Name, "prefill")}
	if err := kubeClient.Get(context.Background(), prefillKey, adapter); err != nil {
		t.Fatalf("get preserved prefill seed: %v", err)
	}
	if adapter.Spec.Replicas != 2 {
		t.Fatalf("prefill seed replicas = %d after DGD mutation, want create-only value 2", adapter.Spec.Replicas)
	}
}

func TestSeedOnce(t *testing.T) {
	for _, tc := range []struct {
		name              string
		budgetWatts       int64
		wantCommitted     int32
		wantDGDReplica    int32
		wantApplyingPhase bool
	}{
		{name: "safe seed is reserved", budgetWatts: 700, wantCommitted: 2, wantDGDReplica: 9, wantApplyingPhase: true},
		{name: "unsafe seed stays requested", budgetWatts: 599, wantCommitted: 0, wantDGDReplica: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Create a transactional worker, empty durable vector, and no adapter")
			scheme := runtime.NewScheme()
			if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
				t.Fatalf("add v1alpha1 scheme: %v", err)
			}
			if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
				t.Fatalf("add v1beta1 scheme: %v", err)
			}
			dgd := transactionalReplicaTestDGD(2)
			dgpb := transactionalReplicaTestDGPB(dgd, tc.budgetWatts, 0, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing)
			kubeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(dgd, dgpb).
				WithStatusSubresource(
					&nvidiacomv1beta1.DynamoGraphPowerBudget{},
					&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
				).
				Build()
			reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}

			t.Log("Create the adapter exactly once from the initial DGD replica intent")
			wait, err := reconciler.reconcileTransactionalPowerBootstrap(context.Background(), dgd)
			if err != nil || !wait {
				t.Fatalf("create seed = (%v, %v), want (true, nil)", wait, err)
			}
			adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
			adapterKey := types.NamespacedName{Namespace: dgd.Namespace, Name: generateAdapterName(dgd.Name, "worker")}
			if err := kubeClient.Get(context.Background(), adapterKey, adapter); err != nil {
				t.Fatalf("read created seed: %v", err)
			}
			if adapter.Spec.Replicas != 2 {
				t.Fatalf("seed replicas = %d, want 2", adapter.Spec.Replicas)
			}

			t.Log("Change the DGD after seeding and persist the zero bootstrap vector")
			dgd.Spec.Components[0].Replicas = ptr.To(int32(9))
			if err := kubeClient.Update(context.Background(), dgd); err != nil {
				t.Fatalf("mutate DGD after seed: %v", err)
			}
			wait, err = reconciler.reconcileTransactionalPowerBootstrap(context.Background(), dgd)
			if err != nil || !wait {
				t.Fatalf("persist zero vector = (%v, %v), want (true, nil)", wait, err)
			}
			pendingSeed := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
			if err := kubeClient.Get(context.Background(), adapterKey, pendingSeed); err != nil {
				t.Fatalf("read zero-vector pending seed: %v", err)
			}
			if pendingSeed.Status.RequestedReplicas != 2 || pendingSeed.Status.CommittedReplicas != 0 ||
				pendingSeed.Status.PendingReason != nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterPendingReason("UnenforcedBaseline") {
				t.Fatalf("zero-vector pending status = %+v, want requested=2 committed=0 reason=UnenforcedBaseline", pendingSeed.Status)
			}

			t.Log("Evaluate only the durable adapter request against the zero committed vector")
			wait, err = reconciler.reconcileTransactionalPowerBootstrap(context.Background(), dgd)
			if err != nil {
				t.Fatalf("evaluate seed: %v", err)
			}
			storedAdapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
			if err := kubeClient.Get(context.Background(), adapterKey, storedAdapter); err != nil {
				t.Fatalf("read preserved seed: %v", err)
			}
			if storedAdapter.Spec.Replicas != 2 {
				t.Fatalf("seed was overwritten from the mutated DGD: got %d, want 2", storedAdapter.Spec.Replicas)
			}
			wantPendingReason := nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterPendingReason("")
			if tc.wantCommitted == 0 {
				wantPendingReason = nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterPendingReason("BudgetExceeded")
			}
			if storedAdapter.Status.RequestedReplicas != 2 ||
				storedAdapter.Status.CommittedReplicas != tc.wantCommitted ||
				storedAdapter.Status.ActualReplicas != 0 ||
				storedAdapter.Status.Replicas != 0 ||
				storedAdapter.Status.PendingReason != wantPendingReason {
				t.Fatalf(
					"seed status = %+v, want requested=2 committed=%d actual=0 pending=%q",
					storedAdapter.Status,
					tc.wantCommitted,
					wantPendingReason,
				)
			}
			storedDGPB := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
			if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
				t.Fatalf("read committed vector: %v", err)
			}
			if got := storedDGPB.Status.CommittedReplicaTargets["worker"]; got != tc.wantCommitted {
				t.Fatalf("committed seed = %d, want %d", got, tc.wantCommitted)
			}
			storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
			if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
				t.Fatalf("read mirrored DGD: %v", err)
			}
			if got := ptr.Deref(storedDGD.GetComponentByName("worker").Replicas, int32(1)); got != tc.wantDGDReplica {
				t.Fatalf("DGD replicas after decision = %d, want %d", got, tc.wantDGDReplica)
			}
			if tc.wantApplyingPhase && storedDGPB.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying {
				t.Fatalf("safe seed phase = %q, want Applying", storedDGPB.Status.Phase)
			}
			if !wait {
				t.Log("Rejected request converged immediately because the DGD was already at the durable zero target")
			}
		})
	}
}

func TestZeroSeedPublishesBelowMinimum(t *testing.T) {
	t.Log("Create an all-zero transactional seed against a positive minimum endpoint")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(0)
	dgpb := transactionalReplicaTestDGPB(dgd, 1000, 0, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing)
	dgpb.Status.CommittedReplicaTargets = nil
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, dgpb).
		WithStatusSubresource(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
		).
		Build()
	reconciler := newDGDScalingAdaptersReconciler(kubeClient, nil)

	t.Log("Cross the create-only adapter and zero-vector durability boundaries")
	for step := 1; step <= 2; step++ {
		wait, err := reconciler.ReconcileTransactionalReplicas(context.Background(), dgd)
		if err != nil || !wait {
			t.Fatalf("zero seed boundary %d = (%v, %v), want (true, nil)", step, wait, err)
		}
	}

	t.Log("Evaluate the numerically equal zero seed and publish the exact floor rejection")
	wait, err := reconciler.ReconcileTransactionalReplicas(context.Background(), dgd)
	if err != nil || wait {
		t.Fatalf("zero seed admission = (%v, %v), want (false, nil)", wait, err)
	}
	adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
	key := types.NamespacedName{Namespace: dgd.Namespace, Name: generateAdapterName(dgd.Name, "worker")}
	if err := kubeClient.Get(context.Background(), key, adapter); err != nil {
		t.Fatalf("read zero seed adapter: %v", err)
	}
	if adapter.Spec.Replicas != 0 || adapter.Status.RequestedReplicas != 0 ||
		adapter.Status.CommittedReplicas != 0 ||
		adapter.Status.PendingReason != nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterPendingReason("BelowMinimum") {
		t.Fatalf("zero seed status = %+v, want requested=0 committed=0 pending=BelowMinimum", adapter.Status)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read zero seed power budget: %v", err)
	}
	if stored.Status.CommittedReplicaTargets["worker"] != 0 ||
		stored.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing {
		t.Fatalf("zero seed changed durable commitment: %+v", stored.Status)
	}
}

func TestReserveBeforeMirror(t *testing.T) {
	t.Log("Create a healthy committed baseline with a newer DGDSA request")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgpb := transactionalReplicaTestDGPB(dgd, 1000, 2, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle)
	adapter := transactionalReplicaTestAdapter(dgd, "worker", 3)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, dgpb, adapter).
		WithStatusSubresource(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
		).
		Build()
	reconciler := newDGDScalingAdaptersReconciler(kubeClient, nil)

	t.Log("Persist the accepted vector and stop before modifying the DGD")
	wait, err := reconciler.ReconcileTransactionalReplicas(context.Background(), dgd)
	if err != nil || !wait {
		t.Fatalf("reserve request = (%v, %v), want (true, nil)", wait, err)
	}
	storedDGPB := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
		t.Fatalf("read reservation: %v", err)
	}
	if got := storedDGPB.Status.CommittedReplicaTargets["worker"]; got != 3 {
		t.Fatalf("committed replicas = %d, want 3", got)
	}
	if storedDGPB.Status.Ledger.EnforcedWatts != 600 ||
		storedDGPB.Status.Ledger.InGateReservedWatts != 300 ||
		storedDGPB.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf(
			"committed snapshot ledger = %+v, want existing 600W plus durable 300W reservation",
			storedDGPB.Status.Ledger,
		)
	}
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
		t.Fatalf("read DGD before mirror: %v", err)
	}
	if got := ptr.Deref(storedDGD.GetComponentByName("worker").Replicas, int32(1)); got != 2 {
		t.Fatalf("DGD changed before reservation boundary: got %d, want 2", got)
	}

	t.Log("Mirror the already durable vector on the next reconciliation")
	wait, err = reconciler.ReconcileTransactionalReplicas(context.Background(), storedDGD)
	if err != nil || !wait {
		t.Fatalf("mirror request = (%v, %v), want (true, nil)", wait, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
		t.Fatalf("read mirrored DGD: %v", err)
	}
	if got := ptr.Deref(storedDGD.GetComponentByName("worker").Replicas, int32(1)); got != 3 {
		t.Fatalf("mirrored DGD replicas = %d, want 3", got)
	}
}

func TestCommittedReservationConflictRequiresReevaluation(t *testing.T) {
	t.Log("Read a DGPB snapshot that becomes stale before the committed reservation write")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgpb := transactionalReplicaTestDGPB(dgd, 1000, 2, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	stale := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stale); err != nil {
		t.Fatalf("read stale candidate: %v", err)
	}
	newer := stale.DeepCopy()
	newer.Status.InventoryEpoch++
	if err := kubeClient.Status().Update(context.Background(), newer); err != nil {
		t.Fatalf("publish concurrent status: %v", err)
	}

	t.Log("Optimistic status patching rejects the stale decision instead of overwriting it")
	reconciler := newDGDScalingAdaptersReconciler(kubeClient, nil)
	err := reconciler.persistCommittedReplicaVector(
		context.Background(),
		stale,
		powerbudget.ReplicaVector{"worker": 3},
		nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			EnforcedWatts:       600,
			InGateReservedWatts: 300,
			TotalChargedWatts:   900,
		},
		true,
	)
	if !apierrors.IsConflict(err) {
		t.Fatalf("stale reservation error = %v, want conflict", err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read status after conflict: %v", err)
	}
	if stored.Status.InventoryEpoch != newer.Status.InventoryEpoch ||
		stored.Status.CommittedReplicaTargets["worker"] != 2 {
		t.Fatalf("conflict overwrote newer status: %+v", stored.Status)
	}
}

func TestRestartConverges(t *testing.T) {
	t.Log("Start after a reservation commit but before its DGD mirror")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgpb := transactionalReplicaTestDGPB(dgd, 1000, 3, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying)
	adapter := transactionalReplicaTestAdapter(dgd, "worker", 3)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, dgpb, adapter).
		WithStatusSubresource(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
		).
		Build()

	t.Log("A fresh reconciler mirrors the durable vector without reseeding")
	restarted := newDGDScalingAdaptersReconciler(kubeClient, nil)
	wait, err := restarted.ReconcileTransactionalReplicas(context.Background(), dgd)
	if err != nil || !wait {
		t.Fatalf("restart mirror = (%v, %v), want (true, nil)", wait, err)
	}
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
		t.Fatalf("read restarted DGD: %v", err)
	}
	if got := ptr.Deref(storedDGD.GetComponentByName("worker").Replicas, int32(1)); got != 3 {
		t.Fatalf("restart mirrored replicas = %d, want 3", got)
	}

	t.Log("A later retry is an idempotent no-op")
	wait, err = newDGDScalingAdaptersReconciler(kubeClient, nil).
		ReconcileTransactionalReplicas(context.Background(), storedDGD)
	if err != nil || wait {
		t.Fatalf("converged retry = (%v, %v), want (false, nil)", wait, err)
	}
	storedAdapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(adapter), storedAdapter); err != nil {
		t.Fatalf("read adapter after restart: %v", err)
	}
	if storedAdapter.Spec.Replicas != 3 {
		t.Fatalf("restart reseeded adapter to %d, want 3", storedAdapter.Spec.Replicas)
	}
}

func TestLatestVector(t *testing.T) {
	t.Log("Create two healthy committed workers with one newer component request")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 scheme: %v", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgd.Spec.Components[0].ComponentName = "prefill"
	dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:  "decode",
		ComponentType:  nvidiacomv1beta1.ComponentTypeWorker,
		Replicas:       ptr.To(int32(2)),
		ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
	})
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name},
		Spec: nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
			BudgetWatts: 2400,
			Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
		},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID:                  string(dgd.UID),
			Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
			CommittedReplicaTargets: map[string]int32{"prefill": 2, "decode": 2},
			Components: []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
				transactionalReplicaStatusComponent("prefill", 2),
				transactionalReplicaStatusComponent("decode", 2),
			},
			Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
				EnforcedWatts: 1200, TotalChargedWatts: 1200,
			},
		},
	}
	prefillAdapter := transactionalReplicaTestAdapter(dgd, "prefill", 3)
	decodeAdapter := transactionalReplicaTestAdapter(dgd, "decode", 2)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, dgpb, prefillAdapter, decodeAdapter).
		WithStatusSubresource(
			&nvidiacomv1beta1.DynamoGraphPowerBudget{},
			&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{},
		).
		Build()
	reconciler := newDGDScalingAdaptersReconciler(kubeClient, nil)

	t.Log("Commit the first observed complete vector atomically")
	wait, err := reconciler.ReconcileTransactionalReplicas(context.Background(), dgd)
	if err != nil || !wait {
		t.Fatalf("commit first vector = (%v, %v), want (true, nil)", wait, err)
	}
	storedDGPB := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
		t.Fatalf("read first vector: %v", err)
	}
	if got := storedDGPB.Status.CommittedReplicaTargets; got["prefill"] != 3 || got["decode"] != 2 {
		t.Fatalf("first committed vector = %v, want prefill=3 decode=2", got)
	}

	t.Log("Accept a newer request while first completing the prior durable DGD mirror")
	storedDecode := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(decodeAdapter), storedDecode); err != nil {
		t.Fatalf("read decode adapter: %v", err)
	}
	storedDecode.Spec.Replicas = 3
	if err := kubeClient.Update(context.Background(), storedDecode); err != nil {
		t.Fatalf("publish latest decode request: %v", err)
	}
	wait, err = reconciler.ReconcileTransactionalReplicas(context.Background(), dgd)
	if err != nil || !wait {
		t.Fatalf("mirror prior vector = (%v, %v), want (true, nil)", wait, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
		t.Fatalf("read vector after mirror: %v", err)
	}
	if got := storedDGPB.Status.CommittedReplicaTargets; got["prefill"] != 3 || got["decode"] != 2 {
		t.Fatalf("new request partially changed prior vector: %v", got)
	}

	t.Log("After enforcement settles, converge to the latest complete request vector")
	storedDGPB.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	storedDGPB.Status.Components = []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
		transactionalReplicaStatusComponent("prefill", 3),
		transactionalReplicaStatusComponent("decode", 2),
	}
	storedDGPB.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 1500, TotalChargedWatts: 1500,
	}
	if err := kubeClient.Status().Update(context.Background(), storedDGPB); err != nil {
		t.Fatalf("settle first vector inventory: %v", err)
	}
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgd), storedDGD); err != nil {
		t.Fatalf("read mirrored DGD: %v", err)
	}
	wait, err = reconciler.ReconcileTransactionalReplicas(context.Background(), storedDGD)
	if err != nil || !wait {
		t.Fatalf("commit latest vector = (%v, %v), want (true, nil)", wait, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), storedDGPB); err != nil {
		t.Fatalf("read latest vector: %v", err)
	}
	if got := storedDGPB.Status.CommittedReplicaTargets; got["prefill"] != 3 || got["decode"] != 3 {
		t.Fatalf("latest committed vector = %v, want prefill=3 decode=3", got)
	}
}

func TestReplacementReservation(t *testing.T) {
	t.Log("Reserve the write-order peak while the old target is still live")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgpb := transactionalReplicaTestDGPB(dgd, 1000, 2, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle)
	bindTransactionalReplicaTestDGPB(dgpb, dgd)
	dgpb.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 600, TotalChargedWatts: 600,
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	rollingUpdateCtx := dynamo.RollingUpdateContext{
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
		NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
	}

	t.Log("The new DCD is written before old targets, so old=2 plus new=1 needs one extra")
	reserved, err := newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			rollingUpdateCtx,
			map[string]int32{"worker": 2},
			map[string]int32{"worker": 0},
		)
	if err != nil || !reserved {
		t.Fatalf("replacement reservation = (%v, %v), want (true, nil)", reserved, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read replacement reservation: %v", err)
	}
	if stored.Status.Ledger.RolloutExtraWatts != 300 || stored.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf("replacement peak ledger = %+v, want 300W extra and 900W total", stored.Status.Ledger)
	}

	t.Log("A restart before the child write reuses the durable unseen reservation")
	reserved, err = newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			rollingUpdateCtx,
			map[string]int32{"worker": 2},
			map[string]int32{"worker": 0},
		)
	if err != nil || !reserved {
		t.Fatalf("restart reservation reuse = (%v, %v), want (true, nil)", reserved, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read reused replacement reservation: %v", err)
	}
	if stored.Status.Ledger.RolloutExtraWatts != 300 || stored.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf("restart double-charged replacement ledger: %+v", stored.Status.Ledger)
	}

	t.Log("Do not reuse the floor after the first extra target is visible through a stale false marker")
	stored.Spec.BudgetWatts = 1200
	if err := kubeClient.Update(context.Background(), stored); err != nil {
		t.Fatalf("raise test budget for the second target: %v", err)
	}
	visibleSecondTarget := dynamo.RollingUpdateContext{
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
		NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
	}
	reserved, err = newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			visibleSecondTarget,
			map[string]int32{"worker": 2},
			map[string]int32{"worker": 1},
		)
	if err != nil || !reserved {
		t.Fatalf("visible second-target reservation = (%v, %v), want (true, nil)", reserved, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read visible second-target reservation: %v", err)
	}
	if stored.Status.Ledger.RolloutExtraWatts != 600 || stored.Status.Ledger.TotalChargedWatts != 1200 {
		t.Fatalf("visible extra reused consumed floor: %+v", stored.Status.Ledger)
	}

	t.Log("Reuse a retained peak floor between rollout waves when no current extra remains")
	stored.Spec.BudgetWatts = 900
	if err := kubeClient.Update(context.Background(), stored); err != nil {
		t.Fatalf("restore exact-budget test spec: %v", err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("refresh exact-budget test object: %v", err)
	}
	stored.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 600, RolloutExtraWatts: 300, TotalChargedWatts: 900,
	}
	stored.Status.RolloutInProgress = true
	if err := kubeClient.Status().Update(context.Background(), stored); err != nil {
		t.Fatalf("restore retained peak floor: %v", err)
	}
	secondWave := dynamo.RollingUpdateContext{
		OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
		NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
	}
	reserved, err = newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			secondWave,
			map[string]int32{"worker": 1},
			map[string]int32{"worker": 1},
		)
	if err != nil || !reserved {
		t.Fatalf("second-wave floor reuse = (%v, %v), want (true, nil)", reserved, err)
	}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read second-wave floor reuse: %v", err)
	}
	if stored.Status.Ledger.RolloutExtraWatts != 300 || stored.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf("second wave double-charged retained floor: %+v", stored.Status.Ledger)
	}

	t.Log("After old capacity has vacated, replacement consumes committed slots without another reservation")
	stored.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
	if err := kubeClient.Status().Update(context.Background(), stored); err != nil {
		t.Fatalf("close fence after old capacity vacates: %v", err)
	}
	reserved, err = newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			rollingUpdateCtx,
			map[string]int32{"worker": 1},
			map[string]int32{"worker": 0},
		)
	if err != nil || !reserved {
		t.Fatalf("replacement slot reuse = (%v, %v), want (true, nil)", reserved, err)
	}
}

func TestTerminatingCapacityReservation(t *testing.T) {
	t.Log("Reserve a replacement while prior committed Pods are still terminating")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := transactionalReplicaTestDGD(2)
	dgpb := transactionalReplicaTestDGPB(dgd, 900, 2, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying)
	bindTransactionalReplicaTestDGPB(dgpb, dgd)
	dgpb.Status.Components[0].TerminatingReplicas = 2
	dgpb.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 600, TotalChargedWatts: 600,
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()

	reserved, err := newDGDWorkerRolloutReconciler(kubeClient, nil).
		reserveTransactionalRolloutExtras(
			context.Background(),
			dgd,
			dynamo.RollingUpdateContext{
				OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 0},
				NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
			},
			map[string]int32{"worker": 0},
			map[string]int32{"worker": 0},
		)
	if err != nil || !reserved {
		t.Fatalf("terminating replacement reservation = (%v, %v), want (true, nil)", reserved, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read terminating replacement reservation: %v", err)
	}
	if stored.Status.Ledger.RolloutExtraWatts != 300 || stored.Status.Ledger.TotalChargedWatts != 900 {
		t.Fatalf("terminating replacement ledger = %+v, want 300W extra and 900W total", stored.Status.Ledger)
	}
}

func TestRolloutExtras(t *testing.T) {
	for _, tc := range []struct {
		name         string
		budgetWatts  int64
		wantReserved bool
	}{
		{name: "surge fits", budgetWatts: 900, wantReserved: true},
		{name: "surge exceeds budget", budgetWatts: 899, wantReserved: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Evaluate one proposed surge replica against the current aggregate ledger")
			scheme := runtime.NewScheme()
			if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
				t.Fatalf("add v1beta1 scheme: %v", err)
			}
			dgd := transactionalReplicaTestDGD(2)
			dgpb := transactionalReplicaTestDGPB(dgd, tc.budgetWatts, 2, nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying)
			bindTransactionalReplicaTestDGPB(dgpb, dgd)
			kubeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(dgpb).
				WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
				Build()
			rollingUpdateCtx := dynamo.RollingUpdateContext{
				OldWorkerReplicaTargetsByComponent: map[string]int32{"worker": 2},
				NewWorkerReplicaTargetsByComponent: map[string]int32{"worker": 1},
			}
			reserved, err := newDGDWorkerRolloutReconciler(kubeClient, nil).
				reserveTransactionalRolloutExtras(
					context.Background(),
					dgd,
					rollingUpdateCtx,
					map[string]int32{"worker": 2},
					map[string]int32{"worker": 0},
				)
			if err != nil || reserved != tc.wantReserved {
				t.Fatalf("rollout reservation = (%v, %v), want (%v, nil)", reserved, err, tc.wantReserved)
			}
			stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
			if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
				t.Fatalf("read rollout reservation: %v", err)
			}
			if tc.wantReserved {
				if stored.Status.Ledger.RolloutExtraWatts != 300 || stored.Status.Ledger.TotalChargedWatts != 900 {
					t.Fatalf("persisted rollout ledger = %+v, want 300W extra and 900W total", stored.Status.Ledger)
				}
			} else if stored.Status.Ledger.RolloutExtraWatts != 0 || stored.Status.Ledger.TotalChargedWatts != 600 {
				t.Fatalf("rejected rollout changed ledger: %+v", stored.Status.Ledger)
			}
		})
	}

	t.Log("Keep a declared DCD surge charged before its Pod exists")
	dgd, dgpb, qualification := statusTestObjects(1)
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 2}
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying
	inventory := dgdPowerBudgetInventory{
		RolloutInProgress: true,
		DCDs: []nvidiacomv1beta1.DynamoComponentDeployment{{
			ObjectMeta: metav1.ObjectMeta{Name: "graph-worker-00-new"},
			Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker-00",
					Replicas:      ptr.To(int32(3)),
				},
			},
		}},
	}
	status, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		inventory,
		powerReportHistory{},
		qualification,
		time.Now(),
		time.Minute,
		false,
	)
	if err != nil {
		t.Fatalf("build declared rollout inventory: %v", err)
	}
	if status.Ledger.InGateReservedWatts != 700 || status.Ledger.RolloutExtraWatts != 350 ||
		status.Ledger.TotalChargedWatts != 1050 {
		t.Fatalf("declared rollout ledger = %+v, want 700W committed + 350W rollout extra", status.Ledger)
	}
}

func transactionalReplicaTestDGD(replicas int32) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "graph",
			UID:       types.UID("dgd-uid"),
			Annotations: map[string]string{
				nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName:  "worker",
				ComponentType:  nvidiacomv1beta1.ComponentTypeWorker,
				Replicas:       ptr.To(replicas),
				ScalingAdapter: &nvidiacomv1beta1.ScalingAdapter{},
			}},
		},
	}
}

func transactionalReplicaTestDGPB(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	budgetWatts int64,
	committed int32,
	phase nvidiacomv1beta1.DynamoGraphPowerBudgetPhase,
) *nvidiacomv1beta1.DynamoGraphPowerBudget {
	committedTargets := map[string]int32(nil)
	component := nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
		Name:                       "worker",
		RequestedCapWattsPerGPU:    300,
		InGateBoundWattsPerGPU:     300,
		UnenforcedBoundWattsPerGPU: 700,
		PhysicalGPUsPerReplica:     1,
	}
	ledger := nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{}
	if committed > 0 {
		committedTargets = map[string]int32{"worker": committed}
		component.EnforcedPhysicalGPUs = committed
		ledger.EnforcedWatts = int64(committed) * 300
		ledger.TotalChargedWatts = ledger.EnforcedWatts
	}
	return &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name},
		Spec: nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
			BudgetWatts: budgetWatts,
			Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
		},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID:                  string(dgd.UID),
			Phase:                   phase,
			CommittedReplicaTargets: committedTargets,
			Components:              []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{component},
			Ledger:                  ledger,
		},
	}
}

func transactionalReplicaTestAdapter(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	replicas int32,
) *nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter {
	controller := true
	return &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: dgd.Namespace,
			Name:      generateAdapterName(dgd.Name, componentName),
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           componentName,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(),
				Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
				Name:       dgd.Name,
				UID:        dgd.UID,
				Controller: &controller,
			}},
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
			Replicas: replicas,
			DGDRef: nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        dgd.Name,
				ServiceName: componentName,
			},
		},
	}
}

func transactionalReplicaStatusComponent(
	componentName string,
	committed int32,
) nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus {
	return nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
		Name:                       componentName,
		RequestedCapWattsPerGPU:    300,
		InGateBoundWattsPerGPU:     300,
		UnenforcedBoundWattsPerGPU: 700,
		PhysicalGPUsPerReplica:     1,
		EnforcedPhysicalGPUs:       committed,
	}
}

func bindTransactionalReplicaTestDGPB(
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	controller := true
	dgpb.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
		Name:       dgd.Name,
		UID:        dgd.UID,
		Controller: &controller,
	}}
}

func TestDGPBCopyOnce(t *testing.T) {
	t.Log("Materialize immutable budget policy and DGD UID binding once")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "graph",
			UID:       types.UID("dgd-uid"),
			Annotations: map[string]string{
				nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
				nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation:   "2400",
				nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation: "2",
			},
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()

	wait, err := reconcileDGDPowerBudget(context.Background(), kubeClient, dgd)
	if err != nil {
		t.Fatalf("reconcileDGDPowerBudget() create error = %v", err)
	}
	if !wait {
		t.Fatal("reconcileDGDPowerBudget() create wait = false, want true")
	}

	key := types.NamespacedName{Namespace: dgd.Namespace, Name: dgd.Name}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), key, dgpb); err != nil {
		t.Fatalf("get created DGPB: %v", err)
	}
	if dgpb.Spec.BudgetWatts != 2400 || dgpb.Spec.Policy.MinEndpoint != 2 {
		t.Fatalf("created DGPB spec = %#v, want budget 2400 and minEndpoint 2", dgpb.Spec)
	}
	if dgpb.Status.DGDUID != string(dgd.UID) || dgpb.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing {
		t.Fatalf("created DGPB status = %#v, want UID binding and Initializing", dgpb.Status)
	}
	owner := metav1.GetControllerOf(dgpb)
	if owner == nil || owner.UID != dgd.UID {
		t.Fatalf("created DGPB controller = %#v, want DGD UID %q", owner, dgd.UID)
	}

	t.Log("Fail closed rather than rewrite the copied spec when annotation admission is bypassed")
	dgd.Annotations[nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation] = "9999"
	dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation] = "9"
	wait, err = reconcileDGDPowerBudget(context.Background(), kubeClient, dgd)
	if err == nil || wait {
		t.Fatalf("reconcileDGDPowerBudget() mismatched policy = (%v, %v), want (false, error)", wait, err)
	}
	if err := kubeClient.Get(context.Background(), key, dgpb); err != nil {
		t.Fatalf("get existing DGPB: %v", err)
	}
	if dgpb.Spec.BudgetWatts != 2400 || dgpb.Spec.Policy.MinEndpoint != 2 {
		t.Fatalf("existing DGPB spec was rewritten: %#v", dgpb.Spec)
	}
}

func TestDGPBPrecreatedPolicyMismatchDoesNotBind(t *testing.T) {
	t.Log("Precreate a same-name DGPB with a forged matching controller UID and divergent policy")
	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 scheme: %v", err)
	}
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "graph",
			UID:       types.UID("dgd-uid"),
			Annotations: map[string]string{
				nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
				nvidiacomv1beta1.DynamoGraphGPUPowerBudgetAnnotation:   "2400",
				nvidiacomv1beta1.DynamoGraphPowerMinEndpointAnnotation: "2",
			},
		},
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{Namespace: dgd.Namespace, Name: dgd.Name},
		Spec: nvidiacomv1beta1.DynamoGraphPowerBudgetSpec{
			BudgetWatts: 9999,
			Policy:      nvidiacomv1beta1.DynamoGraphPowerBudgetPolicy{MinEndpoint: 1},
		},
	}
	bindTransactionalReplicaTestDGPB(dgpb, dgd)
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgpb).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()

	wait, err := reconcileDGDPowerBudget(context.Background(), kubeClient, dgd)
	if err == nil || wait {
		t.Fatalf("reconcileDGDPowerBudget() forged policy = (%v, %v), want (false, error)", wait, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	if err := kubeClient.Get(context.Background(), client.ObjectKeyFromObject(dgpb), stored); err != nil {
		t.Fatalf("read forged DGPB after rejection: %v", err)
	}
	if stored.Status.DGDUID != "" || stored.Status.Phase != "" {
		t.Fatalf("forged DGPB was bound before policy validation: %+v", stored.Status)
	}
}
