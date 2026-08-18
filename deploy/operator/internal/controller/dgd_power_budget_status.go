/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"fmt"
	"math"
	"sort"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const qualifiedGPUProductLabel = "nvidia.com/gpu.product"

func buildDGPBInventoryStatus(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	inventory dgdPowerBudgetInventory,
	reportHistory powerReportHistory,
	qualification powerbudget.QualificationIndex,
	now time.Time,
	freshnessLimit time.Duration,
	grovePathway bool,
) (nvidiacomv1beta1.DynamoGraphPowerBudgetStatus, error) {
	desired := dgpb.Status
	desired.ObservedGeneration = dgpb.Generation
	desired.Components = nil
	desired.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{}

	if grovePathway || dgd.HasAnyMultinodeComponent() || dgdHasCheckpointConfiguration(dgd) {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
		return desired, nil
	}

	workers := make([]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, 0, len(dgd.Spec.Components))
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			workers = append(workers, component)
		}
	}
	if len(workers) > powerbudget.MaxPowerManagedComponents {
		return desired, fmt.Errorf(
			"DGD has %d power-managed components, limit is %d",
			len(workers),
			powerbudget.MaxPowerManagedComponents,
		)
	}
	if len(workers) == 0 {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
		return desired, nil
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].ComponentName < workers[j].ComponentName })
	dcdReplicaTargets, err := desiredDCDReplicaTargets(inventory.DCDs)
	if err != nil {
		return desired, err
	}

	var ledger powerbudget.Ledger
	configurationQualified := true
	liveHardwareQualified := true
	healthyEvidence := true
	for _, component := range workers {
		config, err := componentPowerConfig(component, qualification)
		if err != nil {
			configurationQualified = false
			continue
		}
		row := nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{
			Name:                       component.ComponentName,
			ReplicaStatus:              dgd.Status.Components[component.ComponentName],
			RequestedCapWattsPerGPU:    config.requestedCapWatts,
			InGateBoundWattsPerGPU:     config.bounds.InGateWatts,
			UnenforcedBoundWattsPerGPU: config.bounds.UnenforcedWatts,
			PhysicalGPUsPerReplica:     int32(config.physicalGPUsPerReplica),
		}

		componentPods := make([]*corev1.Pod, 0)
		for i := range inventory.Pods {
			if inventory.Pods[i].Labels[consts.KubeLabelDynamoComponent] == component.ComponentName {
				componentPods = append(componentPods, &inventory.Pods[i])
			}
		}
		committedReplicas := max(int64(0), int64(desired.CommittedReplicaTargets[component.ComponentName]))
		rolloutExtras := rolloutExtraPodIndexes(componentPods, committedReplicas)

		observedCommittedSlots := int64(0)
		observedRolloutExtraSlots := int64(0)
		for i, pod := range componentPods {
			if !pod.DeletionTimestamp.IsZero() {
				row.TerminatingReplicas++
			}
			encoded := pod.Annotations[powerbudget.AgentReportAnnotation]
			report, decodeErr := powerbudget.DecodeAgentReport([]byte(encoded))
			reportedGPUUUIDs := []string(nil)
			allocationID := ""
			if decodeErr == nil {
				reportedGPUUUIDs = make([]string, 0, len(report.GPUs))
				for _, gpu := range report.GPUs {
					reportedGPUUUIDs = append(reportedGPUUUIDs, gpu.UUID)
					if err := config.bounds.ValidateLiveRange(
						gpu.ConstraintMinWatts,
						gpu.ConstraintMaxWatts,
					); err != nil {
						liveHardwareQualified = false
					}
				}
				// The trusted node Agent resolves this UUID set from kubelet
				// PodResources. Recomputing the allocation ID here proves that
				// the report is internally bound to this Pod/container/set; P2.3
				// supplies the authoritative producer-side PodResources lookup.
				allocationID = expectedPowerAllocationID(
					string(pod.UID),
					nvidiacomv1beta1.MainContainerName,
					reportedGPUUUIDs,
				)
			}
			evidence, err := evaluatePodPowerReport(encoded, podPowerReportExpectation{
				DGDUID:           string(dgd.UID),
				Component:        component.ComponentName,
				PodUID:           string(pod.UID),
				Node:             pod.Spec.NodeName,
				AllocationID:     allocationID,
				ExpectedGPUCount: config.physicalGPUsPerReplica,
				// The operator has no independent device-plugin UUID source;
				// P2.3's trusted Agent supplies the PodResources UUID set.
				ExpectedGPUUUIDs:          nil,
				ExpectedRequestedCapWatts: config.requestedCapWatts,
				ReportExisted:             reportHistory.Contains(string(pod.UID)),
				Bounds:                    config.bounds,
				Now:                       now,
				FreshnessLimit:            freshnessLimit,
				StructurallyInGated:       true,
				RolloutExtra:              rolloutExtras[i],
			})
			if err != nil {
				return desired, fmt.Errorf("evaluate Pod %s power report: %w", pod.Name, err)
			}
			recordPowerReportEvidence(evidence)
			if !evidence.Accepted {
				healthyEvidence = false
			} else if inventory.RecordAcceptedPowerReport != nil {
				inventory.RecordAcceptedPowerReport(pod, report)
			}
			if rolloutExtras[i] {
				observedRolloutExtraSlots += int64(len(evidence.Charges))
			} else {
				observedCommittedSlots += int64(len(evidence.Charges))
			}
			for chargeIndex, charge := range evidence.Charges {
				if err := ledger.AddCharge(
					fmt.Sprintf("pod/%s/%d", pod.UID, chargeIndex),
					charge,
				); err != nil {
					return desired, err
				}
				if err := addComponentChargeCount(&row, charge.Class(), 1); err != nil {
					return desired, err
				}
			}
		}

		committedSlots := committedReplicas *
			int64(config.physicalGPUsPerReplica)
		missingCommittedSlots := committedSlots - observedCommittedSlots
		if missingCommittedSlots > 0 {
			charge, err := powerbudget.ClassifyGPUCharge(powerbudget.GPUChargeInput{
				Bounds:       config.bounds,
				InjectedGate: true,
			})
			if err != nil {
				return desired, err
			}
			if err := ledger.AddAggregateCharge(missingCommittedSlots, charge); err != nil {
				return desired, err
			}
			if err := addComponentChargeCount(&row, charge.Class(), missingCommittedSlots); err != nil {
				return desired, err
			}
			healthyEvidence = false
		}

		// Reserve declared old/new DCD targets before their Pods exist. A
		// replacement within the committed total consumes an ordinary slot;
		// only aggregate target capacity above committed becomes rollout-extra.
		desiredReplicas := max(committedReplicas, dcdReplicaTargets[component.ComponentName])
		requiredRolloutExtraSlots := (desiredReplicas - committedReplicas) *
			int64(config.physicalGPUsPerReplica)
		missingRolloutExtraSlots := requiredRolloutExtraSlots - observedRolloutExtraSlots
		if missingRolloutExtraSlots > 0 {
			charge, err := powerbudget.ClassifyGPUCharge(powerbudget.GPUChargeInput{
				Bounds:       config.bounds,
				InjectedGate: true,
				RolloutExtra: true,
			})
			if err != nil {
				return desired, err
			}
			if err := ledger.AddAggregateCharge(missingRolloutExtraSlots, charge); err != nil {
				return desired, err
			}
			healthyEvidence = false
		}
		desired.Components = append(desired.Components, row)
	}

	if !configurationQualified || len(desired.Components) != len(workers) {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified
		desired.Components = nil
		desired.Ledger = nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{}
		return desired, nil
	}
	desired.Ledger = powerbudget.NewLedgerStatus(ledger)
	previousRolloutWatts := dgpb.Status.Ledger.RolloutExtraWatts
	computedRolloutWatts := desired.Ledger.RolloutExtraWatts
	retainPendingReservation := previousRolloutWatts > computedRolloutWatts &&
		(!dgpb.Status.RolloutInProgress || inventory.RolloutInProgress || computedRolloutWatts > 0)
	if retainPendingReservation {
		desired.Ledger, err = powerbudget.PreserveRolloutExtraFloor(desired.Ledger, previousRolloutWatts)
		if err != nil {
			return desired, fmt.Errorf("preserve pending rollout reservation: %w", err)
		}
	}
	// False plus retained extra watts means the reservation is waiting for the
	// child write to become visible. Once inventory has observed an active
	// rollout, True is retained until DCD/Pod obligations are also gone.
	desired.RolloutInProgress = inventory.RolloutInProgress || computedRolloutWatts > 0 ||
		(dgpb.Status.RolloutInProgress && (computedRolloutWatts > 0 || retainPendingReservation))
	healthyBaseline := healthyEvidence &&
		desired.Ledger.UnknownWatts == 0 &&
		desired.Ledger.InGateReservedWatts == 0 &&
		desired.Ledger.RolloutExtraWatts == 0 &&
		committedReplicaVectorApplied(dgd, desired.CommittedReplicaTargets)
	if !liveHardwareQualified {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified
	} else if desired.Ledger.TotalChargedWatts > dgpb.Spec.BudgetWatts {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering
	} else if inventory.RolloutInProgress || desired.Ledger.RolloutExtraWatts > 0 {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying
	} else if healthyBaseline {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
	} else if desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle ||
		desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering ||
		desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInfeasible {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale
	} else if desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified ||
		desired.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseStale {
		desired.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing
	}
	if _, err := powerbudget.EncodeStatusSnapshot(desired); err != nil {
		return desired, err
	}
	return desired, nil
}

func committedReplicaVectorApplied(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	committed map[string]int32,
) bool {
	if len(committed) == 0 {
		return false
	}
	workers := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		workers++
		target, found := committed[component.ComponentName]
		if !found || target < 1 || ptr.Deref(component.Replicas, int32(1)) != target {
			return false
		}
	}
	return workers == len(committed)
}

func desiredDCDReplicaTargets(
	dcds []nvidiacomv1beta1.DynamoComponentDeployment,
) (map[string]int64, error) {
	targets := make(map[string]int64)
	for i := range dcds {
		componentName := dynamo.GetDCDComponentName(&dcds[i])
		if componentName == "" {
			return nil, fmt.Errorf("DynamoComponentDeployment %q has no component identity", dcds[i].Name)
		}
		replicas := ptr.Deref(dcds[i].Spec.Replicas, int32(1))
		if replicas < 0 || targets[componentName] > math.MaxInt64-int64(replicas) {
			return nil, fmt.Errorf("DynamoComponentDeployment targets overflow for component %q", componentName)
		}
		targets[componentName] += int64(replicas)
	}
	return targets, nil
}

func rolloutExtraPodIndexes(pods []*corev1.Pod, committedReplicas int64) []bool {
	extra := make([]bool, len(pods))
	extraCount := int64(len(pods)) - committedReplicas
	if extraCount <= 0 {
		return extra
	}

	// Terminating Pods are the clearest old-capacity obligation, so classify
	// them outside the committed vector first. Fill any remaining extras from
	// the deterministic end of the name-sorted inventory.
	for i := range pods {
		if extraCount == 0 {
			return extra
		}
		if !pods[i].DeletionTimestamp.IsZero() {
			extra[i] = true
			extraCount--
		}
	}
	for i := len(pods) - 1; i >= 0 && extraCount > 0; i-- {
		if extra[i] {
			continue
		}
		extra[i] = true
		extraCount--
	}
	return extra
}

type dgdComponentPowerConfig struct {
	requestedCapWatts      int64
	physicalGPUsPerReplica int
	qualifiedProduct       string
	bounds                 powerbudget.ComponentBounds
}

func componentPowerConfig(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	qualification powerbudget.QualificationIndex,
) (dgdComponentPowerConfig, error) {
	if component.PodTemplate == nil {
		return dgdComponentPowerConfig{}, fmt.Errorf("component %q has no Pod template", component.ComponentName)
	}
	requestedCap, err := parsePositivePowerAnnotation(
		component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit],
		64,
	)
	if err != nil {
		return dgdComponentPowerConfig{}, fmt.Errorf("component %q has no valid immutable power cap", component.ComponentName)
	}
	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(component))
	if err != nil || gpuCount < 1 {
		return dgdComponentPowerConfig{}, fmt.Errorf("component %q has no valid physical GPU count", component.ComponentName)
	}
	sku := component.PodTemplate.Spec.NodeSelector[qualifiedGPUProductLabel]
	bounds, err := qualification.QualifiedBounds(requestedCap, []string{sku})
	if err != nil {
		return dgdComponentPowerConfig{}, fmt.Errorf("component %q: %w", component.ComponentName, err)
	}
	return dgdComponentPowerConfig{
		requestedCapWatts:      requestedCap,
		physicalGPUsPerReplica: gpuCount,
		qualifiedProduct:       sku,
		bounds:                 bounds,
	}, nil
}

func addComponentChargeCount(
	row *nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus,
	class powerbudget.ChargeClass,
	count int64,
) error {
	if count < 0 || count > math.MaxInt32 {
		return fmt.Errorf("component charge count %d is outside int32 status bounds", count)
	}
	var destination *int32
	switch class {
	case powerbudget.ChargeClassEnforced:
		destination = &row.EnforcedPhysicalGPUs
	case powerbudget.ChargeClassUnknown:
		destination = &row.UnknownPhysicalGPUs
	case powerbudget.ChargeClassInGate:
		destination = &row.InGatePhysicalGPUs
	default:
		return nil
	}
	if int64(*destination) > math.MaxInt32-count {
		return fmt.Errorf("component %q charge count overflows int32 status", row.Name)
	}
	*destination += int32(count)
	return nil
}

func dgdHasCheckpointConfiguration(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	for i := range dgd.Spec.Components {
		experimental := dgd.Spec.Components[i].Experimental
		if experimental == nil || experimental.Checkpoint == nil {
			continue
		}
		checkpoint := experimental.Checkpoint
		if checkpoint.Enabled || ptr.Deref(checkpoint.CheckpointRef, "") != "" {
			return true
		}
	}
	return false
}
