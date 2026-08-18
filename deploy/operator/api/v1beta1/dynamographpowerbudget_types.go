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

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// These types define the complete Phase 2 API contract; later implementation
// changes populate the status incrementally without changing field ownership.
// DGPB is v1beta1-only, so it has no spoke-version conversion contract.
// DynamoGraphPowerBudgetPhase describes the aggregate power-budget lifecycle.
// +kubebuilder:validation:Enum=Initializing;Idle;Applying;Recovering;Stale;Infeasible;Unqualified
type DynamoGraphPowerBudgetPhase string

const (
	// DynamoGraphPowerBudgetPhaseInitializing holds workloads while the initial
	// component-complete replica vector is evaluated.
	DynamoGraphPowerBudgetPhaseInitializing DynamoGraphPowerBudgetPhase = "Initializing"
	// DynamoGraphPowerBudgetPhaseIdle indicates that enforcement is healthy and
	// the replica fence is open.
	DynamoGraphPowerBudgetPhaseIdle DynamoGraphPowerBudgetPhase = "Idle"
	// DynamoGraphPowerBudgetPhaseApplying indicates that a reserved transition is
	// waiting for workload mirroring or fresh enforcement reports.
	DynamoGraphPowerBudgetPhaseApplying DynamoGraphPowerBudgetPhase = "Applying"
	// DynamoGraphPowerBudgetPhaseRecovering indicates replica-only over-budget recovery.
	DynamoGraphPowerBudgetPhaseRecovering DynamoGraphPowerBudgetPhase = "Recovering"
	// DynamoGraphPowerBudgetPhaseStale indicates that required observed state is stale.
	DynamoGraphPowerBudgetPhaseStale DynamoGraphPowerBudgetPhase = "Stale"
	// DynamoGraphPowerBudgetPhaseInfeasible indicates that the immutable minimum
	// endpoint footprint cannot fit within the budget.
	DynamoGraphPowerBudgetPhaseInfeasible DynamoGraphPowerBudgetPhase = "Infeasible"
	// DynamoGraphPowerBudgetPhaseUnqualified indicates that watched hardware falls
	// outside the qualified SKU bounds.
	DynamoGraphPowerBudgetPhaseUnqualified DynamoGraphPowerBudgetPhase = "Unqualified"
)

const (
	// DynamoGraphPowerBudgetMaxComponents is the supported per-DGD component limit.
	DynamoGraphPowerBudgetMaxComponents = 25

	// DynamoGraphPowerControlModeAnnotation opts a DGD into transactional
	// replica-fence power control.
	DynamoGraphPowerControlModeAnnotation = "dynamo.nvidia.com/power-control-mode"
	// DynamoGraphGPUPowerBudgetAnnotation supplies the immutable aggregate
	// physical-GPU budget copied into the DGPB.
	DynamoGraphGPUPowerBudgetAnnotation = "dynamo.nvidia.com/gpu-power-budget"
	// DynamoGraphPowerMinEndpointAnnotation supplies the immutable per-component
	// replica floor copied into the DGPB.
	DynamoGraphPowerMinEndpointAnnotation = "dynamo.nvidia.com/power-min-endpoint"
	// DynamoGraphPowerControlModeTransactionalReplicaFence is the only supported
	// Phase 2 power-control mode.
	DynamoGraphPowerControlModeTransactionalReplicaFence = "transactional-replica-fence"

	// DynamoGraphPowerBudgetConditionTypePowerInfeasible reports that the immutable
	// minimum endpoint footprint cannot fit within the available budget.
	DynamoGraphPowerBudgetConditionTypePowerInfeasible = "PowerInfeasible"
)

// DynamoGraphPowerBudgetPolicy contains immutable replica-admission policy.
type DynamoGraphPowerBudgetPolicy struct {
	// minEndpoint is the minimum replica count for every power-managed component.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	MinEndpoint int32 `json:"minEndpoint"`
}

// DynamoGraphPowerBudgetSpec defines the immutable power policy for one
// DynamoGraphDeployment.
// +kubebuilder:validation:XValidation:rule="!oldSelf.hasValue() || self == oldSelf.value()",message="spec is immutable",optionalOldSelf=true
type DynamoGraphPowerBudgetSpec struct {
	// budgetWatts is the aggregate physical-GPU power budget for the owning DGD.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	BudgetWatts int64 `json:"budgetWatts"`

	// policy contains immutable replica-admission policy.
	// +kubebuilder:validation:Required
	Policy DynamoGraphPowerBudgetPolicy `json:"policy"`
}

// DynamoGraphPowerBudgetComponentStatus contains bounded aggregate state for
// one power-managed component. Exact per-GPU evidence remains on Pods.
type DynamoGraphPowerBudgetComponentStatus struct {
	// name is the DGD component name.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// replicaStatus reuses the standard aggregate Dynamo component inventory.
	ReplicaStatus ComponentReplicaStatus `json:"replicaStatus"`

	// requestedCapWattsPerGPU is the immutable requested cap R_c.
	// +kubebuilder:validation:Minimum=1
	RequestedCapWattsPerGPU int64 `json:"requestedCapWattsPerGPU"`

	// inGateBoundWattsPerGPU is the qualified reservation bound B_c.
	// +kubebuilder:validation:Minimum=1
	InGateBoundWattsPerGPU int64 `json:"inGateBoundWattsPerGPU"`

	// unenforcedBoundWattsPerGPU is the qualified unknown-evidence bound U_c.
	// +kubebuilder:validation:Minimum=1
	UnenforcedBoundWattsPerGPU int64 `json:"unenforcedBoundWattsPerGPU"`

	// physicalGPUsPerReplica is the immutable physical-GPU count per replica.
	// +kubebuilder:validation:Minimum=1
	PhysicalGPUsPerReplica int32 `json:"physicalGPUsPerReplica"`

	// inGatePhysicalGPUs is the aggregate count charged at B_c.
	// +kubebuilder:validation:Minimum=0
	InGatePhysicalGPUs int32 `json:"inGatePhysicalGPUs"`

	// terminatingReplicas remain charged until deletion is observed.
	// +kubebuilder:validation:Minimum=0
	TerminatingReplicas int32 `json:"terminatingReplicas"`

	// enforcedPhysicalGPUs is the aggregate count with accepted exact readback.
	// +kubebuilder:validation:Minimum=0
	EnforcedPhysicalGPUs int32 `json:"enforcedPhysicalGPUs"`

	// unknownPhysicalGPUs is the aggregate count charged at U_c.
	// +kubebuilder:validation:Minimum=0
	UnknownPhysicalGPUs int32 `json:"unknownPhysicalGPUs"`
}

// DynamoGraphPowerBudgetLedgerStatus contains the four disjoint aggregate watt
// classes and their checked total.
type DynamoGraphPowerBudgetLedgerStatus struct {
	// +kubebuilder:validation:Minimum=0
	EnforcedWatts int64 `json:"enforcedWatts"`
	// +kubebuilder:validation:Minimum=0
	UnknownWatts int64 `json:"unknownWatts"`
	// +kubebuilder:validation:Minimum=0
	InGateReservedWatts int64 `json:"inGateReservedWatts"`
	// +kubebuilder:validation:Minimum=0
	RolloutExtraWatts int64 `json:"rolloutExtraWatts"`
	// +kubebuilder:validation:Minimum=0
	TotalChargedWatts int64 `json:"totalChargedWatts"`
}

// DynamoGraphPowerBudgetStatus defines the bounded durable power-budget state.
type DynamoGraphPowerBudgetStatus struct {
	// dgdUID binds the status to one DGD enrollment generation.
	// +optional
	DGDUID string `json:"dgdUID,omitempty"`

	// observedGeneration is the DGPB generation used for this status.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// inventoryEpoch advances only when a semantic ledger input changes.
	// +optional
	// +kubebuilder:validation:Minimum=0
	InventoryEpoch int64 `json:"inventoryEpoch,omitempty"`

	// phase is the sole persisted replica-fence authority. Only Idle is open.
	// +optional
	Phase DynamoGraphPowerBudgetPhase `json:"phase,omitempty"`

	// committedReplicaTargets is the durable all-role reservation written before
	// DGD replica mirroring.
	// +optional
	// +kubebuilder:validation:MaxProperties=25
	// +kubebuilder:validation:XValidation:rule="self.all(component, self[component] >= 0)",message="committed replica targets must be nonnegative"
	CommittedReplicaTargets map[string]int32 `json:"committedReplicaTargets,omitempty"`

	// components contains one aggregate row per power-managed component.
	// +optional
	// +kubebuilder:validation:MaxItems=25
	// +listType=map
	// +listMapKey=name
	Components []DynamoGraphPowerBudgetComponentStatus `json:"components,omitempty"`

	// ledger contains aggregate conservative watts only.
	// +optional
	Ledger DynamoGraphPowerBudgetLedgerStatus `json:"ledger,omitempty"`

	// rolloutInProgress is the deployment-wide rollout hold observed by Planner.
	// +optional
	RolloutInProgress bool `json:"rolloutInProgress,omitempty"`

	// requiredWatts and availableWatts are set for PowerInfeasible diagnostics.
	// +optional
	// +kubebuilder:validation:Minimum=0
	RequiredWatts int64 `json:"requiredWatts,omitempty"`
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableWatts int64 `json:"availableWatts,omitempty"`

	// conditions contains bounded aggregate conditions, currently PowerInfeasible.
	// +optional
	// +kubebuilder:validation:MaxItems=1
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:resource:shortName=dgpb

// DynamoGraphPowerBudget materializes immutable power policy and aggregate
// enforcement state for one DynamoGraphDeployment.
type DynamoGraphPowerBudget struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphPowerBudgetSpec   `json:"spec"`
	Status DynamoGraphPowerBudgetStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphPowerBudgetList contains a list of DynamoGraphPowerBudget objects.
type DynamoGraphPowerBudgetList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphPowerBudget `json:"items"`
}
