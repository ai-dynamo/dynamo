/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"maps"
	"strings"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestPowerInfeasibleReportsFloorOverflowWithoutCapMutation(t *testing.T) {
	dgd, dgpb, _ := statusTestObjects(1)
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	dgpb.Generation = 7
	dgpb.Spec.BudgetWatts = 600
	dgpb.Spec.Policy.MinEndpoint = 2
	desired := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		CommittedReplicaTargets: map[string]int32{"worker-00": 2},
		Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			UnknownWatts: 700, TotalChargedWatts: 700,
		},
	}
	inventory := dgdPowerBudgetInventory{DCDs: []nvidiacomv1beta1.DynamoComponentDeployment{
		recoveryTestDCD("worker-00", 2),
	}}
	before := dgd.DeepCopy()

	t.Log("Report the immutable floor as infeasible without changing the workload or cap intent")
	changed, err := applyRecoveryScaleDown(dgd, dgpb, inventory, &desired)
	if err != nil {
		t.Fatalf("applyRecoveryScaleDown() error = %v", err)
	}
	if !changed || desired.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible {
		t.Fatalf("recovery result = (%v, %q), want changed Infeasible", changed, desired.Phase)
	}
	if desired.RequiredWatts != 700 || desired.AvailableWatts != 600 {
		t.Fatalf("infeasible watts = required %d available %d, want 700/600", desired.RequiredWatts, desired.AvailableWatts)
	}
	condition := apiMeta.FindStatusCondition(
		desired.Conditions,
		nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible,
	)
	if condition == nil || condition.Status != metav1.ConditionTrue ||
		condition.Reason != "MinimumFootprintExceedsBudget" || condition.ObservedGeneration != 7 {
		t.Fatalf("PowerInfeasible condition = %#v", condition)
	}
	if !maps.Equal(desired.CommittedReplicaTargets, map[string]int32{"worker-00": 2}) {
		t.Fatalf("floor commitment changed: %#v", desired.CommittedReplicaTargets)
	}
	if !equality.Semantic.DeepEqual(dgd, before) {
		t.Fatal("recovery mutated the DGD or its immutable cap intent")
	}
}

func TestRecoveryStabilityWindowBlocksImmediateReopen(t *testing.T) {
	now := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	window := 30 * time.Second

	t.Log("Start a durable stability interval when a recovering ledger first fits")
	desired := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		Phase:  nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle,
		Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{TotalChargedWatts: 500},
	}
	fitSince, requeueAfter := applyRecoveryStabilityWindow(
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		&desired,
		0,
		now,
		window,
	)
	if fitSince != now.UnixNano() || requeueAfter != window ||
		desired.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering {
		t.Fatalf("first stability hold = fitSince %d, requeue %s, phase %q", fitSince, requeueAfter, desired.Phase)
	}

	t.Log("Keep the fence closed until the entire window has elapsed")
	desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	nextFitSince, requeueAfter := applyRecoveryStabilityWindow(
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		&desired,
		fitSince,
		now.Add(window-time.Nanosecond),
		window,
	)
	if nextFitSince != fitSince || requeueAfter != time.Nanosecond ||
		desired.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering {
		t.Fatalf("pre-expiry hold = fitSince %d, requeue %s, phase %q", nextFitSince, requeueAfter, desired.Phase)
	}

	t.Log("Clear the durable marker and permit the healthy Idle phase at expiry")
	desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	nextFitSince, requeueAfter = applyRecoveryStabilityWindow(
		nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		&desired,
		fitSince,
		now.Add(window),
		window,
	)
	if nextFitSince != 0 || requeueAfter != 0 ||
		desired.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle {
		t.Fatalf("expired stability hold = fitSince %d, requeue %s, phase %q", nextFitSince, requeueAfter, desired.Phase)
	}
}

func TestRecoveryStabilityCacheLagClosesFence(t *testing.T) {
	dgd, dgpb, _ := statusTestObjects(1)
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	dgpb.Status.InventoryEpoch = 4
	dgpb.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 350, TotalChargedWatts: 350,
	}
	controller := true
	dcd := recoveryTestDCD("worker-00", 1)
	dcd.Namespace = dgd.Namespace
	dcd.OwnerReferences = []metav1.OwnerReference{mapOwnerReference(dgd, &controller)}

	scheme := runtime.NewScheme()
	for name, add := range map[string]func(*runtime.Scheme) error{
		"core":     corev1.AddToScheme,
		"v1alpha1": nvidiacomv1alpha1.AddToScheme,
		"v1beta1":  nvidiacomv1beta1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatalf("add %s scheme: %v", name, err)
		}
	}
	cachedClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd.DeepCopy(), dgpb.DeepCopy()).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	liveClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd.DeepCopy(), dgpb.DeepCopy(), &dcd).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:               cachedClient,
		PowerInventoryReader: liveClient,
	}

	t.Log("Detect that the cached DCD inventory is behind the authoritative API view")
	cached, err := reconciler.observeDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil {
		t.Fatalf("observe cached inventory: %v", err)
	}
	current, err := reconciler.powerInventoryCacheCurrent(context.Background(), dgd, cached)
	if err != nil {
		t.Fatalf("compare cached inventory: %v", err)
	}
	if current {
		t.Fatal("stale cached inventory was accepted as current")
	}

	t.Log("Persist Stale before any transactional admission can consume the incomplete cache")
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgpb.Namespace, Name: dgpb.Name}
	if err := cachedClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read cached DGPB: %v", err)
	}
	if err := persistPowerInventoryState(context.Background(), cachedClient, stored, powerInventoryState{
		Version:     powerInventoryStateVersion,
		TargetEpoch: 4,
		Fingerprint: strings.Repeat("a", 64),
	}); err != nil {
		t.Fatalf("seed durable inventory state: %v", err)
	}
	updated, requeueAfter, err := closePowerInventoryForCacheLag(context.Background(), cachedClient, stored)
	if err != nil || !updated || requeueAfter != 5*time.Second {
		t.Fatalf("closePowerInventoryForCacheLag() = (%v, %s, %v)", updated, requeueAfter, err)
	}
	if err := cachedClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read closed DGPB: %v", err)
	}
	if stored.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale ||
		stored.Status.InventoryEpoch != 5 || stored.Status.Ledger.TotalChargedWatts != 350 {
		t.Fatalf("closed cache-lag status = %#v", stored.Status)
	}
	state, valid := loadPowerInventoryState(stored)
	if !valid || state.TargetEpoch != 5 || state.Fingerprint != powerInventoryCacheStaleFingerprint() {
		t.Fatalf("closed cache-lag marker = %#v, valid %v", state, valid)
	}
}

func TestRecoveryStabilityCachedListErrorClosesFence(t *testing.T) {
	dgd, dgpb, _ := statusTestObjects(1)
	dgd.Annotations = map[string]string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
	}
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	dgpb.Status.InventoryEpoch = 2
	dgpb.Status.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
		EnforcedWatts: 350, TotalChargedWatts: 350,
	}

	scheme := runtime.NewScheme()
	for name, add := range map[string]func(*runtime.Scheme) error{
		"core":     corev1.AddToScheme,
		"v1alpha1": nvidiacomv1alpha1.AddToScheme,
		"v1beta1":  nvidiacomv1beta1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatalf("add %s scheme: %v", name, err)
		}
	}
	cachedClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd.DeepCopy(), dgpb.DeepCopy()).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		WithInterceptorFuncs(interceptor.Funcs{
			List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
				return errors.New("informer list unavailable")
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: cachedClient}

	t.Log("Close the durable fence when the cache cannot list a safety-critical inventory dependency")
	updated, requeueAfter, err := reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || !updated || requeueAfter != 5*time.Second {
		t.Fatalf("reconcileDGDPowerBudgetInventory() = (%v, %s, %v)", updated, requeueAfter, err)
	}
	stored := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgpb.Namespace, Name: dgpb.Name}
	if err := cachedClient.Get(context.Background(), key, stored); err != nil {
		t.Fatalf("read closed DGPB: %v", err)
	}
	if stored.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale ||
		stored.Status.InventoryEpoch != 3 || stored.Status.Ledger.TotalChargedWatts != 350 {
		t.Fatalf("cached-list failure status = %#v", stored.Status)
	}
}

func mapOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment, controller *bool) metav1.OwnerReference {
	return metav1.OwnerReference{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
		Name:       dgd.Name,
		UID:        dgd.UID,
		Controller: controller,
	}
}

func TestRecoveryScaleDownCommitsDecodeFirstAndHoldsTerminatingCharge(t *testing.T) {
	dgd, dgpb, _ := statusTestObjects(0)
	dgd.Spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "prefill", ComponentType: nvidiacomv1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(3))},
		{ComponentName: "decode", ComponentType: nvidiacomv1beta1.ComponentTypeDecode, Replicas: ptr.To(int32(3))},
	}
	dgpb.Spec.BudgetWatts = 1400
	dgpb.Spec.Policy.MinEndpoint = 2
	desired := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		CommittedReplicaTargets: map[string]int32{"prefill": 3, "decode": 3},
		Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			UnknownWatts: 2100, TotalChargedWatts: 2100,
		},
	}
	inventory := dgdPowerBudgetInventory{DCDs: []nvidiacomv1beta1.DynamoComponentDeployment{
		recoveryTestDCD("prefill", 3), recoveryTestDCD("decode", 3),
	}}

	changed, err := applyRecoveryScaleDown(dgd, dgpb, inventory, &desired)
	if err != nil {
		t.Fatalf("applyRecoveryScaleDown() error = %v", err)
	}
	want := map[string]int32{"prefill": 3, "decode": 2}
	if !changed || !maps.Equal(desired.CommittedReplicaTargets, want) {
		t.Fatalf("first recovery step = %#v, want decode-first %v", desired, want)
	}

	t.Log("Keep the charged terminating replica and block the next reduction")
	dgd.Spec.Components[1].Replicas = ptr.To(int32(2))
	desired.RolloutInProgress = true
	desired.Ledger.RolloutExtraWatts = 700
	desired.Ledger.TotalChargedWatts = 2100
	desired.Components = []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{{
		Name: "decode", TerminatingReplicas: 1,
	}}
	inventory = dgdPowerBudgetInventory{
		DCDs: []nvidiacomv1beta1.DynamoComponentDeployment{
			recoveryTestDCD("prefill", 3), recoveryTestDCD("decode", 2),
		},
		RolloutInProgress: true,
	}

	changed, err = applyRecoveryScaleDown(dgd, dgpb, inventory, &desired)
	if err != nil {
		t.Fatalf("terminating applyRecoveryScaleDown() error = %v", err)
	}
	if changed || !maps.Equal(desired.CommittedReplicaTargets, want) {
		t.Fatalf("terminating recovery changed vector: %#v", desired.CommittedReplicaTargets)
	}
}

func TestRecoveryScaleDownWaitsForPriorDCDTarget(t *testing.T) {
	dgd, dgpb, _ := statusTestObjects(1)
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))
	dgpb.Spec.BudgetWatts = 350
	desired := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		Phase:                   nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		CommittedReplicaTargets: map[string]int32{"worker-00": 2},
		Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			UnknownWatts: 700, TotalChargedWatts: 700,
		},
	}
	inventory := dgdPowerBudgetInventory{DCDs: []nvidiacomv1beta1.DynamoComponentDeployment{
		recoveryTestDCD("worker-00", 3),
	}}

	changed, err := applyRecoveryScaleDown(dgd, dgpb, inventory, &desired)
	if err != nil {
		t.Fatalf("applyRecoveryScaleDown() error = %v", err)
	}
	if changed || desired.CommittedReplicaTargets["worker-00"] != 2 {
		t.Fatalf("reduced again before prior DCD target converged: %#v", desired)
	}

	t.Log("Treat a missing component DCD as cache lag rather than a settled zero target")
	inventory.DCDs = nil
	changed, err = applyRecoveryScaleDown(dgd, dgpb, inventory, &desired)
	if err != nil {
		t.Fatalf("missing-DCD applyRecoveryScaleDown() error = %v", err)
	}
	if changed || desired.CommittedReplicaTargets["worker-00"] != 2 {
		t.Fatalf("missing-DCD recovery changed vector: %#v", desired)
	}
}

func TestRecoveryScaleDownPersistsCommitBeforeDGDReplicaPatch(t *testing.T) {
	dgd, dgpb, qualification := statusTestObjects(2)
	dgd.Annotations = map[string]string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
	}
	dgd.Spec.Components[0].ComponentType = nvidiacomv1beta1.ComponentTypePrefill
	dgd.Spec.Components[1].ComponentType = nvidiacomv1beta1.ComponentTypeDecode
	for i := range dgd.Spec.Components {
		dgd.Spec.Components[i].Replicas = ptr.To(int32(3))
	}
	dgpb.Spec.BudgetWatts = 1750
	dgpb.Spec.Policy.MinEndpoint = 2
	dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	dgpb.Status.CommittedReplicaTargets = map[string]int32{"worker-00": 3, "worker-01": 3}

	controller := true
	dcds := []nvidiacomv1beta1.DynamoComponentDeployment{
		recoveryTestDCD("worker-00", 3), recoveryTestDCD("worker-01", 3),
	}
	for i := range dcds {
		dcds[i].Namespace = dgd.Namespace
		dcds[i].OwnerReferences = []metav1.OwnerReference{{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
			Name:       dgd.Name,
			UID:        dgd.UID,
			Controller: &controller,
		}}
	}

	scheme := runtime.NewScheme()
	for name, add := range map[string]func(*runtime.Scheme) error{
		"core":     corev1.AddToScheme,
		"v1alpha1": nvidiacomv1alpha1.AddToScheme,
		"v1beta1":  nvidiacomv1beta1.AddToScheme,
	} {
		if err := add(scheme); err != nil {
			t.Fatalf("add %s scheme: %v", name, err)
		}
	}
	objects := []runtime.Object{dgd, dgpb, &dcds[0], &dcds[1]}
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(objects...).
		WithStatusSubresource(&nvidiacomv1beta1.DynamoGraphPowerBudget{}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:             kubeClient,
		PowerQualification: qualification,
		PowerNow: func() time.Time {
			return time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
		},
	}

	updated, _, err := reconciler.reconcileDGDPowerBudgetInventory(context.Background(), dgd)
	if err != nil || !updated {
		t.Fatalf("reconcileDGDPowerBudgetInventory() = (%v, %v), want (true, nil)", updated, err)
	}
	storedBudget := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgd.Namespace, Name: dgd.Name}
	if err := kubeClient.Get(context.Background(), key, storedBudget); err != nil {
		t.Fatalf("read stored DGPB: %v", err)
	}
	wantCommitted := map[string]int32{"worker-00": 3, "worker-01": 2}
	if !maps.Equal(storedBudget.Status.CommittedReplicaTargets, wantCommitted) ||
		storedBudget.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering {
		t.Fatalf("stored recovery status = %#v, want committed %v in Recovering", storedBudget.Status, wantCommitted)
	}
	storedDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := kubeClient.Get(context.Background(), key, storedDGD); err != nil {
		t.Fatalf("read stored DGD: %v", err)
	}
	if got := ptr.Deref(storedDGD.Spec.Components[1].Replicas, int32(1)); got != 3 {
		t.Fatalf("decode DGD replicas = %d, want unchanged 3 before the next mirror boundary", got)
	}
}

func recoveryTestDCD(component string, replicas int32) nvidiacomv1beta1.DynamoComponentDeployment {
	return nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: component + "-dcd"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: component,
				Replicas:      ptr.To(replicas),
			},
		},
	}
}
