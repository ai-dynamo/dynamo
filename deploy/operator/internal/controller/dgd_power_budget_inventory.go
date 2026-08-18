/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type dgdPowerBudgetInventory struct {
	DCDs                      []nvidiacomv1beta1.DynamoComponentDeployment
	ScalingAdapters           []nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter
	Pods                      []corev1.Pod
	RolloutInProgress         bool
	RecordAcceptedPowerReport func(*corev1.Pod, powerbudget.AgentReport)
}

func (r *DynamoGraphDeploymentReconciler) observeDGDPowerBudgetInventory(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dgdPowerBudgetInventory, error) {
	return observeDGDPowerBudgetInventoryWithReader(ctx, r.Client, dgd)
}

func observeDGDPowerBudgetInventoryWithReader(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dgdPowerBudgetInventory, error) {
	inventory := dgdPowerBudgetInventory{RolloutInProgress: rollingUpdateInProgress(dgd.Status.RollingUpdate)}

	dcds := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := reader.List(ctx, dcds, client.InNamespace(dgd.Namespace)); err != nil {
		return inventory, fmt.Errorf("list DynamoComponentDeployments: %w", err)
	}
	for i := range dcds.Items {
		if owner := controllerOwnerUID(&dcds.Items[i]); owner == dgd.UID {
			inventory.DCDs = append(inventory.DCDs, *dcds.Items[i].DeepCopy())
		}
	}
	sort.Slice(inventory.DCDs, func(i, j int) bool { return inventory.DCDs[i].Name < inventory.DCDs[j].Name })

	adapters := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	if err := reader.List(
		ctx,
		adapters,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		return inventory, fmt.Errorf("list DynamoGraphDeploymentScalingAdapters: %w", err)
	}
	inventory.ScalingAdapters = append(inventory.ScalingAdapters, adapters.Items...)
	sort.Slice(inventory.ScalingAdapters, func(i, j int) bool {
		return inventory.ScalingAdapters[i].Name < inventory.ScalingAdapters[j].Name
	})

	pods := &corev1.PodList{}
	if err := reader.List(
		ctx,
		pods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		return inventory, fmt.Errorf("list DGD worker Pods: %w", err)
	}
	for i := range pods.Items {
		if isDGDManagedWorkerPod(&pods.Items[i]) {
			inventory.Pods = append(inventory.Pods, *pods.Items[i].DeepCopy())
		}
	}
	sort.Slice(inventory.Pods, func(i, j int) bool { return inventory.Pods[i].Name < inventory.Pods[j].Name })
	return inventory, nil
}

func (r *DynamoGraphDeploymentReconciler) powerInventoryCacheCurrent(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	cached dgdPowerBudgetInventory,
) (bool, error) {
	if r.PowerInventoryReader == nil {
		// Unit reconcilers and static-mode callers do not install the production
		// API reader. SetupDynamoGraphDeployment always does.
		return true, nil
	}
	live, err := observeDGDPowerBudgetInventoryWithReader(ctx, r.PowerInventoryReader, dgd)
	if err != nil {
		return false, fmt.Errorf("read authoritative power inventory: %w", err)
	}
	zeroStatus := nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{}
	zeroHistory := powerReportHistory{}
	cachedFingerprint, err := calculatePowerInventoryFingerprint(dgd, cached, zeroStatus, zeroHistory)
	if err != nil {
		return false, fmt.Errorf("fingerprint cached power inventory: %w", err)
	}
	liveFingerprint, err := calculatePowerInventoryFingerprint(dgd, live, zeroStatus, zeroHistory)
	if err != nil {
		return false, fmt.Errorf("fingerprint authoritative power inventory: %w", err)
	}
	return cachedFingerprint == liveFingerprint, nil
}

func controllerOwnerUID(object client.Object) types.UID {
	for _, owner := range object.GetOwnerReferences() {
		if owner.Controller != nil && *owner.Controller {
			return owner.UID
		}
	}
	return ""
}

func (r *DynamoGraphDeploymentReconciler) reconcileDGDPowerBudgetInventory(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (updated bool, requeueAfter time.Duration, err error) {
	if dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation] !=
		nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence {
		return false, 0, nil
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgd.Namespace, Name: dgd.Name}
	if err := r.Get(ctx, key, dgpb); err != nil {
		return false, 0, fmt.Errorf("read DynamoGraphPowerBudget %s inventory: %w", key, err)
	}
	inventory, err := r.observeDGDPowerBudgetInventory(ctx, dgd)
	if err != nil {
		log.FromContext(ctx).Error(err, "cached power inventory read failed; closing replica fence")
		return closePowerInventoryForCacheLag(ctx, r.Client, dgpb)
	}
	r.emitPowerGateFailureEvents(string(dgd.UID), inventory.Pods)
	r.emitPowerCapClampFeedback(ctx, dgd)
	cacheCurrent, cacheErr := r.powerInventoryCacheCurrent(ctx, dgd, inventory)
	if !cacheCurrent {
		if cacheErr != nil {
			log.FromContext(ctx).Error(cacheErr, "authoritative power inventory read failed; closing replica fence")
		}
		return closePowerInventoryForCacheLag(ctx, r.Client, dgpb)
	}
	inventory.RecordAcceptedPowerReport = func(pod *corev1.Pod, report powerbudget.AgentReport) {
		finishedAt := time.Time{}
		for _, gpu := range report.GPUs {
			if gpu.ObservedAt.After(finishedAt) {
				finishedAt = gpu.ObservedAt
			}
		}
		elapsed, known, unseen := r.powerGateEvents.observeSuccess(string(dgd.UID), pod, finishedAt)
		if unseen {
			recordPowerGateWait("success", elapsed, known)
		}
	}
	now := time.Now()
	if r.PowerNow != nil {
		now = r.PowerNow()
	}
	freshnessLimit := r.PowerReportFreshness
	if freshnessLimit <= 0 {
		freshnessLimit = time.Minute
	}
	grovePathway := dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderGrove
	state, stateValid := loadPowerInventoryState(dgpb)
	reportHistory := mergeReportedPowerPods(dgpb, inventory, state, stateValid)
	desired, err := buildDGPBInventoryStatus(
		dgd,
		dgpb,
		inventory,
		reportHistory,
		r.PowerQualification,
		now,
		freshnessLimit,
		grovePathway,
	)
	if err != nil {
		return false, 0, err
	}
	if _, err := applyRecoveryScaleDown(dgd, dgpb, inventory, &desired); err != nil {
		return false, 0, err
	}
	recoveryFitSince, recoveryRequeueAfter := applyRecoveryStabilityWindow(
		dgpb.Status.Phase,
		&desired,
		state.RecoveryFitSinceUnixNano,
		now,
		r.PowerRecoveryStability,
	)
	fingerprint, err := calculatePowerInventoryFingerprint(dgd, inventory, desired, reportHistory)
	if err != nil {
		return false, 0, err
	}
	if _, err := powerbudget.EncodeStatusSnapshot(desired); err != nil {
		return false, 0, err
	}
	recordPowerBudgetStatus(desired)
	requeueAfter = nextPowerReportRequeue(inventory.Pods, now, freshnessLimit)
	if recoveryRequeueAfter > 0 && (requeueAfter == 0 || recoveryRequeueAfter < requeueAfter) {
		requeueAfter = recoveryRequeueAfter
	}
	baseEpoch := dgpb.Status.InventoryEpoch
	if stateValid && state.TargetEpoch > baseEpoch {
		baseEpoch = state.TargetEpoch
	}
	historyMatches := stateValid && powerReportHistoryEqual(state, reportHistory)
	if !historyMatches && (reportHistory.All || len(reportHistory.PodUIDs) != 0) {
		// Make the one-way B_c -> U_c history durable before aggregate status
		// consumes the report. If this patch fails, no accepted status can lose
		// the only record that the Pod has reported.
		if baseEpoch == math.MaxInt64 {
			return false, requeueAfter, fmt.Errorf("DynamoGraphPowerBudget inventoryEpoch overflow")
		}
		state = powerInventoryState{
			Version:                  powerInventoryStateVersion,
			TargetEpoch:              baseEpoch + 1,
			Fingerprint:              fingerprint,
			RecoveryFitSinceUnixNano: recoveryFitSince,
			AllPodsReported:          reportHistory.All,
			ReportedPodUIDs:          reportHistory.PodUIDs,
		}
		if err := persistPowerInventoryState(ctx, r.Client, dgpb, state); err != nil {
			return false, requeueAfter, err
		}
		return true, requeueAfter, nil
	}

	stateMatches := historyMatches && state.Fingerprint == fingerprint &&
		state.RecoveryFitSinceUnixNano == recoveryFitSince
	if !stateMatches {
		// A status-first retry can observe the new status with the old marker.
		// Reuse that epoch instead of advancing twice for one semantic input.
		statusFirstRetry := powerInventoryStatusSemanticallyEqual(dgpb.Status, desired) &&
			((stateValid && dgpb.Status.InventoryEpoch > state.TargetEpoch) ||
				(!stateValid && dgpb.Status.InventoryEpoch > 0))
		targetEpoch := dgpb.Status.InventoryEpoch
		if !statusFirstRetry {
			if baseEpoch == math.MaxInt64 {
				return false, requeueAfter, fmt.Errorf("DynamoGraphPowerBudget inventoryEpoch overflow")
			}
			targetEpoch = baseEpoch + 1
			desired.InventoryEpoch = targetEpoch
			if _, err := persistDGPBInventoryStatus(ctx, r.Client, dgpb, desired); err != nil {
				return false, requeueAfter, err
			}
		}
		state = powerInventoryState{
			Version:                  powerInventoryStateVersion,
			TargetEpoch:              targetEpoch,
			Fingerprint:              fingerprint,
			RecoveryFitSinceUnixNano: recoveryFitSince,
			AllPodsReported:          reportHistory.All,
			ReportedPodUIDs:          reportHistory.PodUIDs,
		}
		if err := persistPowerInventoryState(ctx, r.Client, dgpb, state); err != nil {
			return false, requeueAfter, err
		}
		return true, requeueAfter, nil
	}

	desired.InventoryEpoch = dgpb.Status.InventoryEpoch
	if state.TargetEpoch > desired.InventoryEpoch {
		desired.InventoryEpoch = state.TargetEpoch
	}
	updated, err = persistDGPBInventoryStatus(ctx, r.Client, dgpb, desired)
	return updated, requeueAfter, err
}

func closePowerInventoryForCacheLag(
	ctx context.Context,
	kubeClient client.Client,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
) (bool, time.Duration, error) {
	const retryAfter = 5 * time.Second
	desired := dgpb.Status
	desired.ObservedGeneration = dgpb.Generation
	desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
	if !equality.Semantic.DeepEqual(dgpb.Status, desired) {
		if dgpb.Status.InventoryEpoch == math.MaxInt64 {
			return false, retryAfter, fmt.Errorf("DynamoGraphPowerBudget inventoryEpoch overflow while closing stale cache fence")
		}
		desired.InventoryEpoch = dgpb.Status.InventoryEpoch + 1
	}
	if _, err := powerbudget.EncodeStatusSnapshot(desired); err != nil {
		return false, retryAfter, err
	}
	recordPowerBudgetStatus(desired)
	updated, err := persistDGPBInventoryStatus(ctx, kubeClient, dgpb, desired)
	if err != nil {
		return false, retryAfter, err
	}
	state, stateValid := loadPowerInventoryState(dgpb)
	staleFingerprint := powerInventoryCacheStaleFingerprint()
	if stateValid && (state.TargetEpoch != desired.InventoryEpoch || state.Fingerprint != staleFingerprint) {
		state.TargetEpoch = desired.InventoryEpoch
		state.Fingerprint = staleFingerprint
		if err := persistPowerInventoryState(ctx, kubeClient, dgpb, state); err != nil {
			return updated, retryAfter, err
		}
		updated = true
	}
	return updated, retryAfter, nil
}

func powerInventoryCacheStaleFingerprint() string {
	digest := sha256.Sum256([]byte("power-inventory-cache-stale"))
	return hex.EncodeToString(digest[:])
}

func nextPowerReportRequeue(pods []corev1.Pod, now time.Time, freshnessLimit time.Duration) time.Duration {
	if now.IsZero() || freshnessLimit <= 0 {
		return 0
	}
	var next time.Duration
	for i := range pods {
		report, err := powerbudget.DecodeAgentReport(
			[]byte(pods[i].Annotations[powerbudget.AgentReportAnnotation]),
		)
		if err != nil {
			continue
		}
		for _, gpu := range report.GPUs {
			expiresIn := gpu.ObservedAt.Add(freshnessLimit).Sub(now)
			if expiresIn < 0 {
				continue
			}
			// ClassifyGPUCharge accepts evidence exactly at the freshness
			// boundary, so ensure one final reconciliation occurs just after it.
			if expiresIn == 0 {
				expiresIn = time.Nanosecond
			}
			if next == 0 || expiresIn < next {
				next = expiresIn
			}
		}
	}
	return next
}

func persistDGPBInventoryStatus(
	ctx context.Context,
	kubeClient client.Client,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	desired nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
) (bool, error) {
	if equality.Semantic.DeepEqual(dgpb.Status, desired) {
		return false, nil
	}
	if desired.InventoryEpoch < dgpb.Status.InventoryEpoch {
		return false, fmt.Errorf(
			"DynamoGraphPowerBudget inventoryEpoch cannot decrease from %d to %d",
			dgpb.Status.InventoryEpoch,
			desired.InventoryEpoch,
		)
	}
	before := dgpb.DeepCopy()
	dgpb.Status = desired
	if err := kubeClient.Status().Patch(
		ctx,
		dgpb,
		client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{}),
	); err != nil {
		return false, fmt.Errorf("patch DynamoGraphPowerBudget inventory status: %w", err)
	}
	return true, nil
}
