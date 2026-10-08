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

// EngineGroupOperationStatus is a user-facing projection of one immutable membership operation.
// Recovery authority lives in the controller's private, versioned checkpoint, not this projection.
type EngineGroupOperationStatus struct {
	// id identifies the same operation across retries and controller restarts.
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// intent distinguishes planned growth, planned shrink, failure recovery, and deletion-driven retirement.
	// +kubebuilder:validation:Enum=Grow;Shrink;Recover;Retire
	Intent string `json:"intent"`

	// shape records the resolved backend operation semantics.
	// +kubebuilder:validation:Enum=Grow;Retire;ReduceToSurvivors;Restore;Remap
	Shape string `json:"shape"`

	// specGeneration is the object generation from which this operation was planned.
	// It does not change when a newer scale target arrives.
	// +kubebuilder:validation:Minimum=1
	SpecGeneration int64 `json:"specGeneration"`

	// baseTopology is the exact committed membership from which the operation starts.
	BaseTopology EngineGroupTopologyStatus `json:"baseTopology"`

	// targetReplicas is this operation's immutable logical allocation target.
	// It may differ from the latest spec.replicas while a multi-step resize progresses.
	// +kubebuilder:validation:Minimum=0
	TargetReplicas int32 `json:"targetReplicas"`

	// joiningReplicas names logical allocations introduced or restored by this operation.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	JoiningReplicas []string `json:"joiningReplicas,omitempty"`

	// nominatedReplicas names allocations selected for planned retirement or recovery.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	NominatedReplicas []string `json:"nominatedReplicas,omitempty"`

	// targetNativeMembers names the exact desired native membership for this operation.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	TargetNativeMembers []string `json:"targetNativeMembers,omitempty"`

	// queuedTargetReplicas reports a newer absolute target without mutating the active plan.
	// +optional
	// +kubebuilder:validation:Minimum=1
	QueuedTargetReplicas *int32 `json:"queuedTargetReplicas,omitempty"`

	// trafficRequirement records whether retained members may keep serving during the change.
	TrafficRequirement EngineGroupTrafficRequirement `json:"trafficRequirement"`

	// servingVerification records whether commit requires serving-progress proof before admission.
	ServingVerification EngineGroupVerificationRequirement `json:"servingVerification"`

	// phase summarizes membership and cross-subsystem progress, not a second transaction protocol.
	Phase EngineGroupOperationPhase `json:"phase"`

	// startedAt records when the immutable plan was persisted.
	StartedAt metav1.Time `json:"startedAt"`

	// lastTransitionTime changes only when phase changes.
	LastTransitionTime metav1.Time `json:"lastTransitionTime"`

	// committedTopology is the result correlated with this exact operation, when observed.
	// +optional
	CommittedTopology *EngineGroupTopologyStatus `json:"committedTopology,omitempty"`

	// verification contains topology-bound serving evidence, independently from membership commit.
	// +optional
	Verification *EngineGroupVerificationStatus `json:"verification,omitempty"`

	// error is the current structured operation failure, if any.
	// +optional
	Error *EngineGroupFailureStatus `json:"error,omitempty"`
}

// EngineGroupOperationPhase summarizes progress without replacing engine transaction authority.
// +kubebuilder:validation:Enum=Pending;Submitting;Committing;Committed;Failed;Unknown;Aborting;Aborted
type EngineGroupOperationPhase string

const (
	// EngineGroupOperationPhasePending means preparatory capacity or traffic work remains.
	EngineGroupOperationPhasePending EngineGroupOperationPhase = "Pending"
	// EngineGroupOperationPhaseSubmitting means a durable membership target has no conclusive observation yet.
	EngineGroupOperationPhaseSubmitting EngineGroupOperationPhase = "Submitting"
	// EngineGroupOperationPhaseCommitting means the engine accepted its membership transition.
	EngineGroupOperationPhaseCommitting EngineGroupOperationPhase = "Committing"
	// EngineGroupOperationPhaseCommitted means membership committed; verification or admission may still remain.
	EngineGroupOperationPhaseCommitted EngineGroupOperationPhase = "Committed"
	// EngineGroupOperationPhaseFailed means a definitive failure blocks or rejects progress.
	EngineGroupOperationPhaseFailed EngineGroupOperationPhase = "Failed"
	// EngineGroupOperationPhaseUnknown means the membership outcome cannot be established conclusively.
	EngineGroupOperationPhaseUnknown EngineGroupOperationPhase = "Unknown"
	// EngineGroupOperationPhaseAborting means a provably uncommitted change is restoring preparatory state.
	EngineGroupOperationPhaseAborting EngineGroupOperationPhase = "Aborting"
	// EngineGroupOperationPhaseAborted means preparatory state has been restored after rejection.
	EngineGroupOperationPhaseAborted EngineGroupOperationPhase = "Aborted"
)

// EngineGroupVerificationStatus records serving progress independently from membership commit.
type EngineGroupVerificationStatus struct {
	// phase is the durable serving-verification state.
	Phase EngineGroupVerificationPhase `json:"phase"`

	// proof is a positive result bound to one immutable topology.
	// +optional
	Proof *EngineGroupServingProofStatus `json:"proof,omitempty"`

	// failure is a conclusive verification failure.
	// +optional
	Failure *EngineGroupFailureStatus `json:"failure,omitempty"`
}

// EngineGroupServingProofStatus binds successful verification to an immutable topology.
type EngineGroupServingProofStatus struct {
	// topologyGeneration identifies the verified topology.
	// +kubebuilder:validation:Minimum=1
	TopologyGeneration int64 `json:"topologyGeneration"`

	// runtimeDigest identifies the runtime-observed membership and serving path.
	// +kubebuilder:validation:MinLength=1
	RuntimeDigest string `json:"runtimeDigest"`

	// observedAt records when serving progress was proven.
	ObservedAt metav1.Time `json:"observedAt"`
}

// EngineGroupFailureStatus is one structured subsystem or transition failure.
type EngineGroupFailureStatus struct {
	// classification states whether the same external intent may be retried.
	Classification EngineGroupFailureClassification `json:"classification"`

	// reason is a stable machine-readable failure reason.
	// +kubebuilder:validation:MinLength=1
	Reason string `json:"reason"`

	// message explains the failure for a human reader.
	// +optional
	Message string `json:"message,omitempty"`
}

// EngineGroupTrafficRequirement describes the traffic boundary around membership mutation.
// +kubebuilder:validation:Enum=KeepServing;QuiesceGroup
type EngineGroupTrafficRequirement string

const (
	// EngineGroupTrafficRequirementKeepServing permits retained replicas to continue serving.
	EngineGroupTrafficRequirementKeepServing EngineGroupTrafficRequirement = "KeepServing"
	// EngineGroupTrafficRequirementQuiesceGroup drains the complete base topology before mutation.
	EngineGroupTrafficRequirementQuiesceGroup EngineGroupTrafficRequirement = "QuiesceGroup"
)

// EngineGroupVerificationRequirement states whether commit needs a serving-progress proof.
// +kubebuilder:validation:Enum=None;Required
type EngineGroupVerificationRequirement string

const (
	// EngineGroupVerificationRequirementNone permits admission after authoritative membership commit.
	EngineGroupVerificationRequirementNone EngineGroupVerificationRequirement = "None"
	// EngineGroupVerificationRequirementRequired requires a matching serving proof before admission.
	EngineGroupVerificationRequirementRequired EngineGroupVerificationRequirement = "Required"
)

// EngineGroupFailureClassification states whether reconciliation may retry the same external intent.
// +kubebuilder:validation:Enum=Retryable;Terminal
type EngineGroupFailureClassification string

const (
	// EngineGroupFailureClassificationRetryable permits retrying the same intent.
	EngineGroupFailureClassificationRetryable EngineGroupFailureClassification = "Retryable"
	// EngineGroupFailureClassificationTerminal means the same intent cannot safely make progress.
	EngineGroupFailureClassificationTerminal EngineGroupFailureClassification = "Terminal"
)

// EngineGroupVerificationPhase is the independent serving-verification state.
// +kubebuilder:validation:Enum=Pending;Passed;Failed
type EngineGroupVerificationPhase string

const (
	// EngineGroupVerificationPhasePending means a proof is required for the committed topology.
	EngineGroupVerificationPhasePending EngineGroupVerificationPhase = "Pending"
	// EngineGroupVerificationPhasePassed means the exact topology produced serving progress.
	EngineGroupVerificationPhasePassed EngineGroupVerificationPhase = "Passed"
	// EngineGroupVerificationPhaseFailed means a conclusive serving check failed.
	EngineGroupVerificationPhaseFailed EngineGroupVerificationPhase = "Failed"
)
