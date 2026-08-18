/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"maps"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const defaultPowerRecoveryStability = time.Minute

// applyRecoveryScaleDown commits at most one replica-only recovery step to the
// desired DGPB status. The inventory writer persists this lower vector before
// ReconcileTransactionalReplicas can mirror it into the DGD.
func applyRecoveryScaleDown(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	inventory dgdPowerBudgetInventory,
	desired *nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
) (bool, error) {
	if desired == nil {
		return false, nil
	}
	if desired.Ledger.TotalChargedWatts <= dgpb.Spec.BudgetWatts {
		clearPowerInfeasible(desired)
		return false, nil
	}
	if !recoveryInventorySettled(dgd, inventory, *desired) {
		clearPowerInfeasible(desired)
		return false, nil
	}

	committed := powerbudget.ReplicaVector(maps.Clone(desired.CommittedReplicaTargets))
	roles, err := recoveryComponentRoles(dgd, committed)
	if err != nil {
		return false, err
	}
	decision, err := powerbudget.NextRecoveryReduction(
		committed,
		roles,
		dgpb.Spec.Policy.MinEndpoint,
		desired.Ledger.TotalChargedWatts,
		dgpb.Spec.BudgetWatts,
	)
	if err != nil {
		return false, fmt.Errorf("compute replica-only power recovery: %w", err)
	}
	if decision.ReducedComponent == "" {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible
		desired.RequiredWatts = decision.RequiredWatts
		desired.AvailableWatts = decision.AvailableWatts
		apiMeta.SetStatusCondition(&desired.Conditions, metav1.Condition{
			Type:               nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: dgpb.Generation,
			Reason:             "MinimumFootprintExceedsBudget",
			Message: fmt.Sprintf(
				"minimum endpoint footprint requires %dW; budget provides %dW",
				decision.RequiredWatts,
				decision.AvailableWatts,
			),
		})
		recordPowerRecoveryAction(powerRecoveryActionInfeasible)
		return true, nil
	}

	clearPowerInfeasible(desired)
	desired.CommittedReplicaTargets = maps.Clone(decision.Committed)
	desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering
	recordPowerRecoveryAction(powerRecoveryActionScaleDown)
	log.Log.Info(
		"Committed one replica-only power recovery step",
		"dgd", dgd.Name,
		"component", decision.ReducedComponent,
		"committed", desired.CommittedReplicaTargets,
		"requiredWatts", decision.RequiredWatts,
		"availableWatts", decision.AvailableWatts,
	)
	return true, nil
}

func clearPowerInfeasible(status *nvidiacomv1beta1.DynamoGraphPowerBudgetStatus) {
	status.RequiredWatts = 0
	status.AvailableWatts = 0
	apiMeta.RemoveStatusCondition(
		&status.Conditions,
		nvidiacomv1beta1.DynamoGraphPowerBudgetConditionTypePowerInfeasible,
	)
}

// applyRecoveryStabilityWindow keeps the fence closed after a recovering
// ledger first fits. The private timestamp survives controller replacement;
// reopening remains subject to the ordinary healthy-baseline checks.
func applyRecoveryStabilityWindow(
	previousPhase nvidiacomv1beta1.DynamoGraphPowerBudgetPhase,
	desired *nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
	fitSinceUnixNano int64,
	now time.Time,
	window time.Duration,
) (nextFitSinceUnixNano int64, requeueAfter time.Duration) {
	if desired == nil || desired.Ledger.TotalChargedWatts < 0 {
		return 0, 0
	}
	if desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering ||
		desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible {
		return 0, 0
	}
	if previousPhase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering &&
		previousPhase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible {
		return 0, 0
	}
	if window <= 0 {
		window = defaultPowerRecoveryStability
	}
	if now.IsZero() {
		return 0, window
	}
	if fitSinceUnixNano == 0 {
		fitSinceUnixNano = now.UnixNano()
	}
	fitSince := time.Unix(0, fitSinceUnixNano)
	stableAt := fitSince.Add(window)
	if !now.Before(stableAt) {
		recordPowerRecoveryAction(powerRecoveryActionReopen)
		return 0, 0
	}

	desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering
	recordPowerRecoveryAction(powerRecoveryActionStabilityHold)
	return fitSinceUnixNano, stableAt.Sub(now)
}

// recoveryInventorySettled prevents a second reduction until the prior lower
// commitment has reached the DGD and no old/surge/terminating physical-GPU
// obligation remains. DCD targets below the commitment are safe; targets above
// it still represent capacity that the workload program must scale down.
func recoveryInventorySettled(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inventory dgdPowerBudgetInventory,
	status nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
) bool {
	if !committedReplicaVectorApplied(dgd, status.CommittedReplicaTargets) ||
		inventory.RolloutInProgress || status.RolloutInProgress ||
		status.Ledger.RolloutExtraWatts > 0 {
		return false
	}
	for _, component := range status.Components {
		if component.TerminatingReplicas > 0 {
			return false
		}
	}
	targets, err := desiredDCDReplicaTargets(inventory.DCDs)
	if err != nil {
		return false
	}
	for component, committed := range status.CommittedReplicaTargets {
		target, observed := targets[component]
		if !observed || target > int64(committed) {
			return false
		}
	}
	return true
}

func recoveryComponentRoles(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	committed powerbudget.ReplicaVector,
) (map[string]nvidiacomv1beta1.ComponentType, error) {
	roles := make(map[string]nvidiacomv1beta1.ComponentType, len(committed))
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		if _, found := committed[component.ComponentName]; !found {
			continue
		}
		role := component.ComponentType
		if role == nvidiacomv1beta1.ComponentTypeWorker {
			role = nvidiacomv1beta1.ComponentTypeDecode
		}
		roles[component.ComponentName] = role
	}
	if len(roles) != len(committed) {
		return nil, fmt.Errorf("recovery roles do not match the committed worker vector")
	}
	return roles, nil
}
