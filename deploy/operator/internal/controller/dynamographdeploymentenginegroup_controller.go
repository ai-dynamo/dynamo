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
	"errors"
	"fmt"
	"reflect"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	engineGroupFinalizer    = "nvidia.com/dynamographdeploymentenginegroup-finalizer"
	engineGroupRequeueAfter = time.Second

	engineGroupConditionAvailable     = "Available"
	engineGroupConditionProgressing   = "Progressing"
	engineGroupConditionDegraded      = "Degraded"
	engineGroupConditionTargetReached = "TargetReached"
	engineGroupConditionTargetValid   = "TargetValid"
	engineGroupConditionTopologyKnown = "TopologyKnown"
	engineGroupConditionRuntimeReady  = "RuntimeReady"
)

// ErrEngineGroupRuntimeUnavailable means no production control-plane adapter can own this group.
var ErrEngineGroupRuntimeUnavailable = errors.New("Engine Group runtime is unavailable")

// ScalePlanResolution distinguishes executable work, a definitive target rejection, and no required operation.
// An ordinary method error means resolution was inconclusive and none of these fields is authoritative.
type ScalePlanResolution struct {
	Plan      *enginegroup.ResolvedPlan
	Rejection *enginegroup.Failure
}

// EngineGroupScalePlanResolver resolves an absolute replica target into one immutable semantic plan.
type EngineGroupScalePlanResolver interface {
	ResolveScalePlan(
		ctx context.Context,
		groupID enginegroup.GroupID,
		targetReplicas int32,
		status enginegroup.GroupStatus,
	) (ScalePlanResolution, error)
}

// EngineGroupRuntime contains the typed adapters and immutable profile selected for one group.
type EngineGroupRuntime struct {
	Profile    nvidiacomv1beta1.EngineGroupProfileStatus
	Capacity   enginegroup.CapacityAdapter
	Membership enginegroup.MembershipAdapter
	Traffic    enginegroup.TrafficAdapter
	Verifier   enginegroup.ServingVerifier
	Planner    EngineGroupScalePlanResolver
}

// EngineGroupRuntimeProvider resolves the production control plane for one Engine Group resource.
type EngineGroupRuntimeProvider interface {
	Resolve(ctx context.Context, group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) (EngineGroupRuntime, error)
}

// DynamoGraphDeploymentEngineGroupReconciler persists and resumes one Engine Group workflow.
type DynamoGraphDeploymentEngineGroupReconciler struct {
	client.Client
	RuntimeProvider EngineGroupRuntimeProvider
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups/finalizers,verbs=update

// Reconcile implements the level-based control loop for one independently resizable engine world.
func (r *DynamoGraphDeploymentEngineGroupReconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}
	if err := r.Get(ctx, req.NamespacedName, group); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Resolve one coherent adapter/profile bundle for every observation and effect in this reconcile.
	runtime, err := r.RuntimeProvider.Resolve(ctx, group)
	if err != nil {
		if !group.DeletionTimestamp.IsZero() && !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
			return ctrl.Result{}, nil
		}
		return r.reconcileUnavailableRuntime(ctx, group, err)
	}
	if err := validateEngineGroupRuntime(runtime); err != nil {
		if !group.DeletionTimestamp.IsZero() && !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
			return ctrl.Result{}, nil
		}
		return r.reconcileUnavailableRuntime(ctx, group, err)
	}

	// Install deletion protection only after a runtime can own effects for this resource.
	if group.DeletionTimestamp.IsZero() && !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
		controllerutil.AddFinalizer(group, engineGroupFinalizer)
		if err := r.Update(ctx, group); err != nil {
			return ctrl.Result{}, fmt.Errorf("add Engine Group finalizer: %w", err)
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Deletion remains fail-closed until fresh adapters and, after initialization, accepted targets prove emptiness.
	if !group.DeletionTimestamp.IsZero() {
		return r.reconcileDeletion(ctx, group, runtime)
	}

	// Freeze the immutable runtime profile before initializing or advancing the workflow journal.
	if group.Status.Profile == nil {
		before := group.Status.DeepCopy()
		group.Status.Profile = runtime.Profile.DeepCopy()
		group.Status.ScaleUnit = nvidiacomv1beta1.EngineGroupScaleUnitReplicas
		group.Status.ObservedGeneration = group.Generation
		setEngineGroupCondition(group, engineGroupConditionRuntimeReady, metav1.ConditionTrue, "RuntimeResolved",
			"The Engine Group runtime and immutable profile are resolved")
		if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
	if group.Status.Profile.Fingerprint != runtime.Profile.Fingerprint {
		return r.reconcileUnavailableRuntime(ctx, group, fmt.Errorf(
			"resolved profile fingerprint changed from %q to %q",
			group.Status.Profile.Fingerprint,
			runtime.Profile.Fingerprint,
		))
	}

	// Bootstrap the durable journal from authoritative subsystem observations before planning a change.
	if group.Status.Reconciliation == nil {
		return r.initializeEngineGroupStatus(ctx, group, runtime)
	}

	status := engineGroupStatusFromAPI(group.Status.Reconciliation)
	before := group.Status.DeepCopy()
	desiredPlan, validation, planningErr := r.resolveDesiredEngineGroupPlan(ctx, group, runtime, status)
	group.Status.TargetValidation = validation

	// Let the pure coordinator derive the next durable level or safely repeat an already persisted effect.
	coordinator := enginegroup.NewCoordinator(runtime.Capacity, runtime.Membership, runtime.Traffic, runtime.Verifier)
	result, reconcileErr := coordinator.Reconcile(ctx, engineGroupID(group), desiredPlan, status)
	var observationErr *enginegroup.ObservationError
	errors.As(reconcileErr, &observationErr)
	r.projectEngineGroupStatus(group, runtime.Profile, result.Status, observationErr, planningErr)
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	if reconcileErr != nil || planningErr != nil {
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, errors.Join(reconcileErr, planningErr)
	}
	if result.Requeue {
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, nil
	}
	return ctrl.Result{}, nil
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) initializeEngineGroupStatus(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime EngineGroupRuntime,
) (ctrl.Result, error) {
	before := group.Status.DeepCopy()
	observations, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil {
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, fmt.Errorf("observe initial Engine Group state: %w", err)
	}

	// Validate and persist one complete already-running base before any membership transition is planned.
	status, err := enginegroup.NewGroupStatus(
		observations.membership.CommittedTopology,
		observations.capacity,
		observations.traffic,
	)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("initialize Engine Group journal: %w", err)
	}
	r.projectEngineGroupStatus(group, runtime.Profile, status, nil, nil)
	if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) resolveDesiredEngineGroupPlan(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime EngineGroupRuntime,
	status enginegroup.GroupStatus,
) (*enginegroup.ResolvedPlan, *nvidiacomv1beta1.EngineGroupTargetValidationStatus, error) {
	if validation := validateEngineGroupReplicaTarget(group.Spec.Replicas, runtime.Profile, status); validation != nil {
		return nil, validation, nil
	}

	// Cardinality alone cannot prove convergence: recovery and remap plans may preserve replica count.
	current, _ := status.Topologies.Current()
	resolution, err := runtime.Planner.ResolveScalePlan(ctx, engineGroupID(group), group.Spec.Replicas, status)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve Engine Group scale plan: %w", err)
	}
	if resolution.Plan != nil && resolution.Rejection != nil {
		return nil, nil, errors.New("scale plan resolution contains both a plan and a rejection")
	}
	if resolution.Rejection != nil {
		if resolution.Rejection.Classification != enginegroup.FailureClassificationTerminal ||
			resolution.Rejection.Reason == "" {
			return nil, nil, errors.New("scale plan rejection must be terminal and contain a reason")
		}
		return nil, &nvidiacomv1beta1.EngineGroupTargetValidationStatus{
			RequestedReplicas: group.Spec.Replicas,
			EffectiveReplicas: current.ReplicaCount(),
			MinReplicas:       runtime.Profile.MinReplicas,
			MaxReplicas:       runtime.Profile.MaxReplicas,
			Reason:            resolution.Rejection.Reason,
			Message:           resolution.Rejection.Message,
		}, nil
	}
	return resolution.Plan, nil, nil
}

func validateEngineGroupReplicaTarget(
	target int32,
	profile nvidiacomv1beta1.EngineGroupProfileStatus,
	status enginegroup.GroupStatus,
) *nvidiacomv1beta1.EngineGroupTargetValidationStatus {
	if target >= profile.MinReplicas && target <= profile.MaxReplicas {
		return nil
	}
	current, _ := status.Topologies.Current()
	return &nvidiacomv1beta1.EngineGroupTargetValidationStatus{
		RequestedReplicas: target,
		EffectiveReplicas: current.ReplicaCount(),
		MinReplicas:       profile.MinReplicas,
		MaxReplicas:       profile.MaxReplicas,
		Reason:            "OutsideProfileBounds",
		Message: fmt.Sprintf(
			"requested replicas %d are outside profile bounds [%d,%d]",
			target,
			profile.MinReplicas,
			profile.MaxReplicas,
		),
	}
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) reconcileDeletion(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime EngineGroupRuntime,
) (ctrl.Result, error) {
	before := group.Status.DeepCopy()
	transitionID := ""
	var status enginegroup.GroupStatus
	if group.Status.Reconciliation != nil {
		status = engineGroupStatusFromAPI(group.Status.Reconciliation)
		if status.Membership.Desired != nil {
			transitionID = status.Membership.Desired.TransitionID
		}
	}
	observations, observationErr := observeEngineGroupRuntime(
		ctx,
		runtime,
		engineGroupID(group),
		transitionID,
	)
	if observationErr != nil {
		setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "DeletionBlocked",
			"Deletion is blocked because current capacity, traffic, or membership cannot be established")
		if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
			return ctrl.Result{}, updateErr
		}
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, observationErr
	}

	if group.Status.Reconciliation == nil {
		ready, evidenceErr := enginegroup.UninitializedDeletionEvidenceReady(
			observations.capacity,
			observations.traffic,
			observations.membership,
		)
		if evidenceErr != nil {
			return r.blockEngineGroupDeletion(ctx, group, before,
				"Deletion is blocked because uninitialized external state is invalid", evidenceErr)
		}
		if !ready {
			return r.blockEngineGroupDeletion(ctx, group, before,
				"Deletion is blocked because the uninitialized external world is not empty", nil)
		}
		return r.removeEngineGroupFinalizer(ctx, group)
	}

	ready, evidenceErr := enginegroup.TerminalDeletionEvidenceReady(
		status,
		observations.capacity,
		observations.traffic,
		observations.membership,
	)
	if evidenceErr != nil {
		return r.blockEngineGroupDeletion(ctx, group, before,
			"Deletion is blocked because terminal evidence is invalid or inconclusive", evidenceErr)
	}
	if !ready {
		return r.blockEngineGroupDeletion(ctx, group, before,
			"Terminal empty capacity, traffic, and membership targets must be accepted and converged before deletion", nil)
	}

	// Release the finalizer only after every external authority and accepted target prove an empty world.
	return r.removeEngineGroupFinalizer(ctx, group)
}

type engineGroupRuntimeObservations struct {
	capacity   enginegroup.CapacityObservation
	traffic    enginegroup.TrafficObservation
	membership enginegroup.MembershipObservation
}

func observeEngineGroupRuntime(
	ctx context.Context,
	runtime EngineGroupRuntime,
	groupID enginegroup.GroupID,
	transitionID string,
) (engineGroupRuntimeObservations, error) {
	capacity, capacityErr := runtime.Capacity.Observe(ctx, groupID)
	traffic, trafficErr := runtime.Traffic.Observe(ctx, groupID)
	membership, membershipErr := runtime.Membership.Observe(ctx, groupID, transitionID)
	if capacityErr != nil {
		capacityErr = fmt.Errorf("observe capacity: %w", capacityErr)
	}
	if trafficErr != nil {
		trafficErr = fmt.Errorf("observe traffic: %w", trafficErr)
	}
	if membershipErr != nil {
		membershipErr = fmt.Errorf("observe membership: %w", membershipErr)
	}
	if err := errors.Join(capacityErr, trafficErr, membershipErr); err != nil {
		return engineGroupRuntimeObservations{}, err
	}
	return engineGroupRuntimeObservations{
		capacity:   capacity,
		traffic:    traffic,
		membership: membership,
	}, nil
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) blockEngineGroupDeletion(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	before *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus,
	message string,
	err error,
) (ctrl.Result, error) {
	setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "DeletionBlocked", message)
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, err
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) removeEngineGroupFinalizer(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (ctrl.Result, error) {
	controllerutil.RemoveFinalizer(group, engineGroupFinalizer)
	if err := r.Update(ctx, group); err != nil {
		return ctrl.Result{}, fmt.Errorf("remove Engine Group finalizer: %w", err)
	}
	return ctrl.Result{}, nil
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) reconcileUnavailableRuntime(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	err error,
) (ctrl.Result, error) {
	before := group.Status.DeepCopy()
	group.Status.ObservedGeneration = group.Generation
	group.Status.ScaleUnit = nvidiacomv1beta1.EngineGroupScaleUnitReplicas
	setEngineGroupCondition(group, engineGroupConditionRuntimeReady, metav1.ConditionFalse, "RuntimeUnavailable", err.Error())
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	if errors.Is(err, ErrEngineGroupRuntimeUnavailable) {
		return ctrl.Result{}, nil
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, err
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) updateEngineGroupStatus(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	before *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus,
) error {
	if reflect.DeepEqual(*before, group.Status) {
		return nil
	}
	if err := r.Status().Update(ctx, group); err != nil {
		return fmt.Errorf("update Engine Group status: %w", err)
	}
	return nil
}

func (r *DynamoGraphDeploymentEngineGroupReconciler) projectEngineGroupStatus(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	profile nvidiacomv1beta1.EngineGroupProfileStatus,
	status enginegroup.GroupStatus,
	observationErr *enginegroup.ObservationError,
	planningErr error,
) {
	group.Status.ObservedGeneration = group.Generation
	group.Status.ScaleUnit = nvidiacomv1beta1.EngineGroupScaleUnitReplicas
	group.Status.Profile = profile.DeepCopy()
	group.Status.Reconciliation = engineGroupStatusToAPI(status)
	group.Status.Replicas = int32(len(status.Capacity.Observed.Allocations))
	group.Status.AvailableReplicas = countAvailableEngineGroupAllocations(status.Capacity.Observed)
	group.Status.ReleaseAuthorizations = engineGroupReleaseFencesToAPI(status.Capacity.Observed.ReleaseFences)

	// Project fresh engine truth independently from the coordinator's accepted topology history.
	observedTopology := status.Membership.Observed.CommittedTopology
	topologyKnown := observedTopology.Generation > 0
	if topologyKnown {
		topology := engineGroupTopologyToAPI(observedTopology)
		group.Status.Topology = &topology
		group.Status.ActiveReplicas = observedTopology.ReplicaCount()
	} else {
		group.Status.Topology = nil
		group.Status.ActiveReplicas = 0
	}
	trafficTopologyGeneration := observedTopology.Generation
	trafficOperationID := ""
	if status.Traffic.Accepted != nil {
		trafficTopologyGeneration = status.Traffic.Accepted.TopologyGeneration
		trafficOperationID = status.Traffic.Accepted.TransitionID
	}
	group.Status.Traffic = &nvidiacomv1beta1.EngineGroupTrafficStatus{
		OperationID:        trafficOperationID,
		TopologyGeneration: trafficTopologyGeneration,
		Admitted:           engineGroupMembershipsToAPI(status.Traffic.Observed.Admitted),
		Draining:           engineGroupMembershipsToAPI(status.Traffic.Observed.Draining),
		Drained:            engineGroupMembershipsToAPI(status.Traffic.Observed.Drained),
	}
	group.Status.ReplicaStates = projectEngineGroupReplicaStates(status)
	recognizedTopology, found := status.Topologies.Current()
	topologyRecognized := found && enginegroup.SameTopology(recognizedTopology, observedTopology)
	projectEngineGroupConditions(
		group,
		status,
		topologyKnown,
		topologyRecognized,
		observationErr,
		planningErr,
	)
}

func projectEngineGroupConditions(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	status enginegroup.GroupStatus,
	topologyKnown bool,
	topologyRecognized bool,
	observationErr *enginegroup.ObservationError,
	planningErr error,
) {
	setEngineGroupCondition(group, engineGroupConditionRuntimeReady, metav1.ConditionTrue, "RuntimeResolved",
		"The Engine Group runtime and immutable profile are resolved")
	switch {
	case group.Status.TargetValidation != nil:
		setEngineGroupCondition(group, engineGroupConditionTargetValid, metav1.ConditionFalse, "TargetRejected",
			"The requested replica target is rejected")
	case planningErr != nil:
		setEngineGroupCondition(group, engineGroupConditionTargetValid, metav1.ConditionUnknown, "PlanResolutionInconclusive",
			"The requested replica target could not be resolved conclusively")
	default:
		setEngineGroupCondition(group, engineGroupConditionTargetValid, metav1.ConditionTrue, "TargetValid",
			"The requested replica target is valid")
	}
	progressing := status.Transition != nil && (status.Transition.Outcome == enginegroup.TransitionOutcomeProgressing ||
		status.Transition.Outcome == enginegroup.TransitionOutcomeReverting)
	setEngineGroupCondition(group, engineGroupConditionProgressing, conditionStatus(progressing),
		chooseEngineGroupString(progressing, "TransitionInProgress", "NoTransitionInProgress"),
		chooseEngineGroupString(progressing, "An Engine Group transition is progressing", "No Engine Group transition is progressing"))
	if observationErr != nil {
		message := fmt.Sprintf("Current %s state could not be established", observationErr.Authority)
		setEngineGroupCondition(group, engineGroupConditionTopologyKnown, metav1.ConditionUnknown,
			"ObservationFailed", message)
		setEngineGroupCondition(group, engineGroupConditionTargetReached, metav1.ConditionUnknown,
			"ObservationFailed", message)
		setEngineGroupCondition(group, engineGroupConditionAvailable, metav1.ConditionUnknown,
			"ObservationFailed", message)
		setEngineGroupCondition(group, engineGroupConditionDegraded, metav1.ConditionUnknown,
			"ObservationFailed", message)
		return
	}

	setEngineGroupCondition(group, engineGroupConditionTopologyKnown, conditionStatus(topologyKnown),
		chooseEngineGroupString(topologyKnown, "TopologyObserved", "TopologyUnknown"),
		chooseEngineGroupString(topologyKnown, "Authoritative engine membership is known", "Authoritative engine membership is unknown"))
	available := topologyKnown && group.Status.ActiveReplicas >= group.Status.Profile.MinSafeServingReplicas &&
		allEngineGroupMembersAvailable(
			status.Membership.Observed.CommittedTopology,
			status.Registry,
			status.Capacity.Observed,
		) &&
		allEngineGroupMembersAdmitted(status.Membership.Observed.CommittedTopology, status.Traffic.Observed)
	setEngineGroupCondition(group, engineGroupConditionAvailable, conditionStatus(available),
		chooseEngineGroupString(available, "ServingAvailable", "ServingUnavailable"),
		chooseEngineGroupString(available, "The committed topology is available and admitted", "The committed topology is not fully available and admitted"))

	targetReached := topologyKnown && group.Status.ActiveReplicas == group.Spec.Replicas
	setEngineGroupCondition(group, engineGroupConditionTargetReached, conditionStatus(targetReached),
		chooseEngineGroupString(targetReached, "TargetReached", "TargetNotReached"),
		chooseEngineGroupString(targetReached, "Engine membership reached the desired target", "Engine membership has not reached the desired target"))
	transitionSettled := status.Transition == nil ||
		status.Transition.Outcome == enginegroup.TransitionOutcomeCompleted ||
		status.Transition.Outcome == enginegroup.TransitionOutcomeRolledBack
	if targetReached && available && topologyRecognized && transitionSettled {
		group.Status.LastStableReplicas = group.Status.ActiveReplicas
	}
	degraded := group.Status.LastStableReplicas > 0 && group.Status.ActiveReplicas < group.Status.LastStableReplicas &&
		!exactPlannedRetirementObserved(status)
	setEngineGroupCondition(group, engineGroupConditionDegraded, conditionStatus(degraded),
		chooseEngineGroupString(degraded, "MembershipBelowLastStable", "MembershipStable"),
		chooseEngineGroupString(degraded, "Engine membership is below its last stable size", "Engine membership is not degraded"))
}

func exactPlannedRetirementObserved(status enginegroup.GroupStatus) bool {
	if status.Transition == nil || status.Transition.Spec.Plan.Change.Kind != enginegroup.PlanKindRetire ||
		status.Transition.Spec.Plan.Change.Retire == nil {
		return false
	}
	base, found := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
	if !found {
		return false
	}
	retiring := make(map[enginegroup.ReplicaID]struct{}, len(status.Transition.Spec.Plan.Change.Retire.Replicas))
	for _, replicaID := range status.Transition.Spec.Plan.Change.Retire.Replicas {
		retiring[replicaID] = struct{}{}
	}
	expected := make([]enginegroup.ReplicaMembership, 0, len(base.Replicas)-len(retiring))
	for _, member := range base.Replicas {
		if _, found := retiring[member.ReplicaID]; !found {
			expected = append(expected, member)
		}
	}
	return enginegroup.SameMembershipSet(expected, status.Membership.Observed.CommittedTopology.Replicas)
}

func projectEngineGroupReplicaStates(status enginegroup.GroupStatus) []nvidiacomv1beta1.EngineGroupReplicaStatus {
	replicas := make([]nvidiacomv1beta1.EngineGroupReplicaStatus, 0, len(status.Registry.Replicas))
	for _, record := range status.Registry.Replicas {
		projected := nvidiacomv1beta1.EngineGroupReplicaStatus{
			ReplicaID:    string(record.ReplicaID),
			SlotID:       string(record.SlotID),
			Availability: nvidiacomv1beta1.EngineGroupReplicaAvailabilityUnknown,
			Membership:   nvidiacomv1beta1.EngineGroupReplicaMembershipUnknown,
		}
		if record.Current != nil {
			projected.Current = &nvidiacomv1beta1.EngineGroupReplicaIncarnation{
				RuntimeIncarnation: string(record.Current.RuntimeIncarnation),
				CapacityRefs:       engineGroupIncarnationToAPI(*record.Current).CapacityRefs,
			}
		}
		for _, previous := range record.History {
			projected.PreviousIncarnations = append(
				projected.PreviousIncarnations,
				nvidiacomv1beta1.EngineGroupReplicaIncarnation{
					RuntimeIncarnation: string(previous.Incarnation.RuntimeIncarnation),
					CapacityRefs:       engineGroupIncarnationToAPI(previous.Incarnation).CapacityRefs,
				},
			)
		}
		if allocation, found := engineGroupAllocationByReplica(status.Capacity.Observed, record.ReplicaID); found {
			if allocation.Available {
				projected.Availability = nvidiacomv1beta1.EngineGroupReplicaAvailabilityAvailable
			} else {
				projected.Availability = nvidiacomv1beta1.EngineGroupReplicaAvailabilityUnavailable
			}
		}
		if membership, found := engineGroupTopologyMembership(status.Membership.Observed.CommittedTopology, record.ReplicaID); found {
			projected.Membership = nvidiacomv1beta1.EngineGroupReplicaMembershipActive
			projected.NativeMembers = engineGroupNativeMembersToAPI(membership.NativeMembers)
		} else if len(record.History) > 0 {
			projected.Membership = nvidiacomv1beta1.EngineGroupReplicaMembershipMasked
			projected.NativeMembers = engineGroupNativeMembersToAPI(record.History[len(record.History)-1].NativeMembers)
		}
		applyEngineGroupTransitionIntent(&projected, status.Transition)
		replicas = append(replicas, projected)
	}
	return replicas
}

func applyEngineGroupTransitionIntent(
	replica *nvidiacomv1beta1.EngineGroupReplicaStatus,
	transition *enginegroup.TransitionStatus,
) {
	if transition == nil || transition.Outcome != enginegroup.TransitionOutcomeProgressing {
		return
	}
	switch transition.Spec.Plan.Change.Kind {
	case enginegroup.PlanKindGrow:
		for _, target := range transition.Spec.Plan.Change.Grow.Replicas {
			if string(target.ReplicaID) == replica.ReplicaID {
				replica.Membership = nvidiacomv1beta1.EngineGroupReplicaMembershipJoining
			}
		}
	case enginegroup.PlanKindRestore:
		for _, target := range transition.Spec.Plan.Change.Restore.Replicas {
			if string(target.ReplicaID) == replica.ReplicaID {
				replica.Membership = nvidiacomv1beta1.EngineGroupReplicaMembershipJoining
			}
		}
	case enginegroup.PlanKindRetire:
		for _, replicaID := range transition.Spec.Plan.Change.Retire.Replicas {
			if string(replicaID) == replica.ReplicaID {
				replica.Membership = nvidiacomv1beta1.EngineGroupReplicaMembershipRetiring
			}
		}
	}
}

func countAvailableEngineGroupAllocations(observation enginegroup.CapacityObservation) int32 {
	var count int32
	for _, allocation := range observation.Allocations {
		if allocation.Available {
			count++
		}
	}
	return count
}

func allEngineGroupMembersAdmitted(
	topology enginegroup.MembershipTopology,
	traffic enginegroup.TrafficObservation,
) bool {
	return enginegroup.SameMembershipSet(topology.Replicas, traffic.Admitted)
}

func allEngineGroupMembersAvailable(
	topology enginegroup.MembershipTopology,
	registry enginegroup.ReplicaRegistry,
	capacity enginegroup.CapacityObservation,
) bool {
	for _, member := range topology.Replicas {
		record, found := engineGroupReplicaRecord(registry, member.ReplicaID)
		if !found || record.Current == nil || record.Current.RuntimeIncarnation != member.RuntimeIncarnation {
			return false
		}
		allocation, found := engineGroupAllocationByReplica(capacity, member.ReplicaID)
		if !found || !allocation.Available || !enginegroup.SameIncarnation(allocation.Incarnation, *record.Current) {
			return false
		}
	}
	return true
}

func engineGroupAllocationByReplica(
	observation enginegroup.CapacityObservation,
	replicaID enginegroup.ReplicaID,
) (enginegroup.CapacityAllocation, bool) {
	for _, allocation := range observation.Allocations {
		if allocation.Incarnation.ReplicaID == replicaID {
			return allocation, true
		}
	}
	return enginegroup.CapacityAllocation{}, false
}

func engineGroupReplicaRecord(
	registry enginegroup.ReplicaRegistry,
	replicaID enginegroup.ReplicaID,
) (enginegroup.ReplicaRecord, bool) {
	for _, record := range registry.Replicas {
		if record.ReplicaID == replicaID {
			return record, true
		}
	}
	return enginegroup.ReplicaRecord{}, false
}

func engineGroupTopologyMembership(
	topology enginegroup.MembershipTopology,
	replicaID enginegroup.ReplicaID,
) (enginegroup.ReplicaMembership, bool) {
	for _, membership := range topology.Replicas {
		if membership.ReplicaID == replicaID {
			return membership, true
		}
	}
	return enginegroup.ReplicaMembership{}, false
}

func validateEngineGroupRuntime(runtime EngineGroupRuntime) error {
	if runtime.Profile.Fingerprint == "" {
		return errors.New("Engine Group runtime profile fingerprint is empty")
	}
	if runtime.Capacity == nil || runtime.Membership == nil || runtime.Traffic == nil ||
		runtime.Verifier == nil || runtime.Planner == nil {
		return errors.New("Engine Group runtime has incomplete adapters")
	}
	return nil
}

func engineGroupID(group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) enginegroup.GroupID {
	return enginegroup.GroupID(group.UID)
}

func setEngineGroupCondition(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	conditionType string,
	status metav1.ConditionStatus,
	reason string,
	message string,
) {
	meta.SetStatusCondition(&group.Status.Conditions, metav1.Condition{
		Type:               conditionType,
		Status:             status,
		ObservedGeneration: group.Generation,
		Reason:             reason,
		Message:            message,
	})
}

func conditionStatus(value bool) metav1.ConditionStatus {
	if value {
		return metav1.ConditionTrue
	}
	return metav1.ConditionFalse
}

func chooseEngineGroupString(value bool, whenTrue string, whenFalse string) string {
	if value {
		return whenTrue
	}
	return whenFalse
}

// SetupWithManager registers the Engine Group controller and its status/finalizer writes.
func (r *DynamoGraphDeploymentEngineGroupReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}).
		Named("dynamographdeploymentenginegroup").
		Complete(r)
}

type unavailableEngineGroupRuntimeProvider struct{}

func (unavailableEngineGroupRuntimeProvider) Resolve(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (EngineGroupRuntime, error) {
	return EngineGroupRuntime{}, ErrEngineGroupRuntimeUnavailable
}
