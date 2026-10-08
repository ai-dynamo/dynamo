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

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	engineGroupFinalizer           = "nvidia.com/dynamographdeploymentenginegroup-finalizer"
	engineGroupRequeueAfter        = time.Second
	engineGroupObservationInterval = 10 * time.Second

	engineGroupConditionAvailable     = "Available"
	engineGroupConditionProgressing   = "Progressing"
	engineGroupConditionDegraded      = "Degraded"
	engineGroupConditionTargetReached = "TargetReached"
	engineGroupConditionTargetValid   = "TargetValid"
	engineGroupConditionTopologyKnown = "TopologyKnown"
	engineGroupConditionRuntimeReady  = "RuntimeReady"
)

// ErrRuntimeUnavailable means no production control-plane adapter can own this group.
var ErrRuntimeUnavailable = errors.New("Engine Group runtime is unavailable")

// ScalePlanResolution distinguishes executable work, a definitive target rejection, and no required operation.
// An ordinary method error means resolution was inconclusive and none of these fields is authoritative.
type ScalePlanResolution struct {
	Plan      *domain.ResolvedPlan
	Rejection *domain.Failure
}

// ScalePlanResolver resolves an absolute replica target into one immutable semantic plan.
// Zero is reserved for deletion-driven whole-world retirement, never ordinary scaling.
// Unsupported retirement must return a definitive rejection, not silently force capacity removal.
type ScalePlanResolver interface {
	ResolveScalePlan(
		ctx context.Context,
		groupID domain.GroupID,
		targetReplicas int32,
		status domain.GroupStatus,
	) (ScalePlanResolution, error)
}

// Runtime contains the typed adapters and immutable profile selected for one group.
type Runtime struct {
	Profile    nvidiacomv1beta1.EngineGroupProfileStatus
	Capacity   domain.CapacityAdapter
	Membership domain.MembershipAdapter
	Traffic    domain.TrafficAdapter
	Verifier   domain.ServingVerifier
	Planner    ScalePlanResolver
}

// RuntimeProvider resolves the production control plane for one Engine Group resource.
type RuntimeProvider interface {
	Resolve(ctx context.Context, group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) (Runtime, error)
}

// Reconciler persists and resumes one Engine Group workflow.
type Reconciler struct {
	client.Client
	RuntimeProvider RuntimeProvider
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentenginegroups/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=pods,verbs=patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch

// Reconcile implements the level-based control loop for one independently resizable engine world.
func (r *Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}
	if err := r.Get(ctx, req.NamespacedName, group); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// A child deleted before this controller owns effects needs no engine finalization.
	if !group.DeletionTimestamp.IsZero() && !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
		return ctrl.Result{}, nil
	}

	// Resolve without effects, and protect the owner before adding any dependent Pod finalizer.
	runtime, runtimeErr := r.RuntimeProvider.Resolve(ctx, group)
	if runtimeErr == nil {
		runtimeErr = validateEngineGroupRuntime(runtime)
	}
	if runtimeErr == nil && group.DeletionTimestamp.IsZero() && !controllerutil.ContainsFinalizer(group, engineGroupFinalizer) {
		controllerutil.AddFinalizer(group, engineGroupFinalizer)
		if err := r.Update(ctx, group); err != nil {
			return ctrl.Result{}, fmt.Errorf("add Engine Group finalizer: %w", err)
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Fence physical teardown even when the old runtime can no longer resolve its deleted clique.
	pending, bindingErr := r.reconcileGroveWorld(ctx, group)
	if pending || bindingErr != nil {
		before := group.Status.DeepCopy()
		invalidateEngineGroupHealth(group, "WorldRestartPending", "Physical world teardown or restart has not yet established fresh membership")
		setEngineGroupCondition(group, engineGroupConditionRuntimeReady, metav1.ConditionFalse, "WorldRestartPending",
			"Normal membership effects are paused until the physical world binding is fenced")
		if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
			return ctrl.Result{}, errors.Join(bindingErr, updateErr)
		}
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, bindingErr
	}

	// A failed binding may progress only through the fenced handoff, never through ordinary effects.
	if runtimeErr != nil {
		return r.reconcileUnavailableRuntime(ctx, group, runtimeErr)
	}

	// Deletion remains fail-closed until fresh adapters and, after initialization, accepted targets prove emptiness.
	if !group.DeletionTimestamp.IsZero() {
		return r.reconcileDeletion(ctx, group, runtime)
	}

	// Freeze the immutable runtime profile before initializing or advancing the workflow journal.
	if group.Status.Profile == nil {
		before := group.Status.DeepCopy()
		group.Status.Profile = runtime.Profile.DeepCopy()
		projectEngineGroupScaleIdentity(group)
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

	// Load private recovery authority; a missing or mismatched checkpoint is never adopted as a new world.
	store := engineGroupCheckpointStore(r, group)
	checkpoint, snapshot, err := loadEngineGroupCheckpoint(ctx, store, group)
	if err != nil {
		return r.reconcileCheckpointLoadFailure(ctx, group, runtime, store, snapshot, err)
	}
	if !snapshot.Exists() {
		return r.initializeEngineGroupStatus(ctx, group, runtime, store, snapshot)
	}
	before := group.Status.DeepCopy()
	checkpoint.restoreProjectionInputs(group)

	// Publish initial checkpoint existence before allowing effects, including after a failed status write.
	if group.Status.Topology == nil {
		topology := engineGroupTopologyToAPI(checkpoint.State.Membership.Observed.CommittedTopology)
		group.Status.Topology = &topology
		invalidateEngineGroupHealth(group, "CheckpointRestored", "Historical membership is restored; fresh runtime observation is required")
		if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}
	status := checkpoint.State
	desiredPlan, validation, planningErr := r.resolveDesiredEngineGroupPlan(ctx, group, runtime, status)
	group.Status.TargetValidation = validation

	// Let the pure coordinator derive the next durable level or safely repeat an already persisted effect.
	coordinator := domain.NewCoordinator(runtime.Capacity, runtime.Membership, runtime.Traffic, runtime.Verifier)
	result, reconcileErr := coordinator.Reconcile(ctx, engineGroupID(group), desiredPlan, status)
	var observationErr *domain.ObservationError
	errors.As(reconcileErr, &observationErr)
	if planningErr == nil && validation == nil && observationErr == nil {
		reconcileEngineGroupDesiredAssignment(group, runtime.Profile, result.Status, desiredPlan)
	}
	r.projectEngineGroupStatus(group, runtime.Profile, result.Status, observationErr, planningErr)
	if err := projectEngineGroupOperation(group, result.Status); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}

	// Persist every new desired level before a subsequent invocation can apply it externally.
	next := newEngineGroupCheckpoint(group, result.Status)
	if err := persistEngineGroupCheckpoint(ctx, store, snapshot, checkpoint, next); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	if reconcileErr != nil || planningErr != nil {
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, errors.Join(reconcileErr, planningErr)
	}
	if result.Requeue {
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, nil
	}

	// Engine-only failures and drift need fresh observation even when Kubernetes objects and targets are unchanged.
	return ctrl.Result{RequeueAfter: engineGroupObservationInterval}, nil
}

func (r *Reconciler) initializeEngineGroupStatus(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime Runtime,
	store kubejournal.Store,
	snapshot kubejournal.Snapshot,
) (ctrl.Result, error) {
	// Even an interrupted first formation needs a verified fresh lifetime after a fenced rebind.
	uid := group.Annotations[consts.KubeAnnotationDynamoEngineGroupPodCliqueUID]
	if uid != "" && group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] == consts.KubeLabelDynamoEngineGroupSGLang {
		var fence groveWorldFence
		fenceSnapshot, err := groveWorldFenceStore(r.Client, group, types.UID(uid)).Load(ctx, &fence)
		if err != nil {
			return r.reconcileCheckpointFailure(ctx, group, err)
		}
		if fenceSnapshot.Exists() && fence.PreviousUID != "" {
			if err := r.authorizeGroveCheckpointRestart(ctx, group, fence.PreviousUID); err != nil {
				return r.reconcileCheckpointFailure(ctx, group, err)
			}
			return r.initializeRestartedEngineGroup(ctx, group, runtime, store, snapshot)
		}
	}

	before := group.Status.DeepCopy()
	observations, err := observeEngineGroupRuntime(ctx, runtime, engineGroupID(group), "")
	if err != nil {
		invalidateEngineGroupHealth(group, "ObservationFailed", "Initial external state could not be established")
		if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
			return ctrl.Result{}, updateErr
		}
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, fmt.Errorf("observe initial Engine Group state: %w", err)
	}

	// Validate and persist one complete already-running base before any membership transition is planned.
	status, err := domain.NewGroupStatus(
		observations.membership.CommittedTopology,
		observations.capacity,
		observations.traffic,
	)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("initialize Engine Group journal: %w", err)
	}
	reconcileEngineGroupDesiredAssignment(group, runtime.Profile, status, nil)
	r.projectEngineGroupStatus(group, runtime.Profile, status, nil, nil)
	if err := persistEngineGroupCheckpoint(ctx, store, snapshot, engineGroupCheckpoint{}, newEngineGroupCheckpoint(group, status)); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

func (r *Reconciler) resolveDesiredEngineGroupPlan(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime Runtime,
	status domain.GroupStatus,
) (*domain.ResolvedPlan, *nvidiacomv1beta1.EngineGroupTargetValidationStatus, error) {
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
		if resolution.Rejection.Classification != domain.FailureClassificationTerminal ||
			resolution.Rejection.Reason == "" {
			return nil, nil, errors.New("scale plan rejection must be terminal and contain a reason")
		}
		return nil, &nvidiacomv1beta1.EngineGroupTargetValidationStatus{
			RequestedReplicas: group.Spec.Replicas,
			EffectiveReplicas: current.ReplicaCount(),
			MinReplicas:       runtime.Profile.MinSupportedReplicas,
			MaxReplicas:       runtime.Profile.MaxSupportedReplicas,
			Reason:            resolution.Rejection.Reason,
			Message:           resolution.Rejection.Message,
		}, nil
	}
	return resolution.Plan, nil, nil
}

func validateEngineGroupReplicaTarget(
	target int32,
	profile nvidiacomv1beta1.EngineGroupProfileStatus,
	status domain.GroupStatus,
) *nvidiacomv1beta1.EngineGroupTargetValidationStatus {
	if target >= profile.MinSupportedReplicas && target <= profile.MaxSupportedReplicas {
		return nil
	}
	current, _ := status.Topologies.Current()
	return &nvidiacomv1beta1.EngineGroupTargetValidationStatus{
		RequestedReplicas: target,
		EffectiveReplicas: current.ReplicaCount(),
		MinReplicas:       profile.MinSupportedReplicas,
		MaxReplicas:       profile.MaxSupportedReplicas,
		Reason:            "OutsideProfileBounds",
		Message: fmt.Sprintf(
			"requested replicas %d are outside profile bounds [%d,%d]",
			target,
			profile.MinSupportedReplicas,
			profile.MaxSupportedReplicas,
		),
	}
}

func (r *Reconciler) reconcileDeletion(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	runtime Runtime,
) (ctrl.Result, error) {
	// Retirement uses the same immutable profile as the live world, never a newly resolved substitute.
	if group.Status.Profile != nil && group.Status.Profile.Fingerprint != runtime.Profile.Fingerprint {
		return r.reconcileCheckpointFailure(ctx, group, errors.New("retirement runtime has a different immutable profile"))
	}
	before := group.Status.DeepCopy()
	transitionID := ""
	checkpoint, snapshot, err := loadEngineGroupCheckpoint(ctx, engineGroupCheckpointStore(r, group), group)
	if err != nil {
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	status := checkpoint.State
	if snapshot.Exists() {
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
		invalidateEngineGroupHealth(group, "ObservationFailed", "Current external state could not be established during deletion")
		setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "DeletionBlocked",
			"Deletion is blocked because current capacity, traffic, or membership cannot be established")
		if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
			return ctrl.Result{}, updateErr
		}
		return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, observationErr
	}

	if !snapshot.Exists() {
		ready, evidenceErr := domain.UninitializedDeletionEvidenceReady(
			observations.capacity,
			observations.traffic,
			observations.membership,
		)
		if evidenceErr != nil {
			return r.blockEngineGroupDeletion(ctx, group, before,
				"Deletion is blocked because uninitialized external state is invalid", evidenceErr)
		}
		if !ready {
			// Persist the profile before the first checkpoint, including deletion during initial formation.
			if group.Status.Profile == nil {
				group.Status.Profile = runtime.Profile.DeepCopy()
				projectEngineGroupScaleIdentity(group)
				if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
					return ctrl.Result{}, err
				}
				return ctrl.Result{Requeue: true}, nil
			}

			// Capture already-formed startup membership without inventing a Bootstrap operation.
			return r.initializeEngineGroupStatus(ctx, group, runtime, engineGroupCheckpointStore(r, group), snapshot)
		}
		return r.removeEngineGroupFinalizer(ctx, group)
	}

	// An observation-only checkpoint has no effect that could complete later. Empty startup may finish directly.
	if status.Transition == nil && status.Membership.Desired == nil && status.Capacity.Desired == nil && status.Traffic.Desired == nil {
		ready, evidenceErr := domain.UninitializedDeletionEvidenceReady(observations.capacity, observations.traffic, observations.membership)
		if evidenceErr != nil {
			return r.blockEngineGroupDeletion(ctx, group, before, "Deletion is blocked because startup evidence is invalid", evidenceErr)
		}
		if ready {
			return r.removeEngineGroupFinalizer(ctx, group)
		}
	}

	ready, evidenceErr := domain.TerminalDeletionEvidenceReady(
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
		return r.reconcileEngineGroupRetirement(ctx, group, runtime, checkpoint, snapshot)
	}

	// Release the finalizer only after every external authority and accepted target prove an empty world.
	return r.removeEngineGroupFinalizer(ctx, group)
}

type engineGroupRuntimeObservations struct {
	capacity   domain.CapacityObservation
	traffic    domain.TrafficObservation
	membership domain.MembershipObservation
}

func observeEngineGroupRuntime(
	ctx context.Context,
	runtime Runtime,
	groupID domain.GroupID,
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

func (r *Reconciler) blockEngineGroupDeletion(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	before *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus,
	message string,
	err error,
) (ctrl.Result, error) {
	// Invalid fresh evidence cannot leave historical health advertised as current during deletion.
	if err != nil {
		invalidateEngineGroupHealth(group, "ObservationFailed", message)
	}
	setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "DeletionBlocked", message)
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, err
}

func (r *Reconciler) removeEngineGroupFinalizer(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (ctrl.Result, error) {
	before := group.DeepCopy()
	controllerutil.RemoveFinalizer(group, engineGroupFinalizer)
	if err := r.Patch(ctx, group, client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})); err != nil {
		return ctrl.Result{}, fmt.Errorf("remove Engine Group finalizer: %w", err)
	}
	return ctrl.Result{}, nil
}

func (r *Reconciler) reconcileUnavailableRuntime(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	err error,
) (ctrl.Result, error) {
	before := group.Status.DeepCopy()
	projectEngineGroupScaleIdentity(group)
	setEngineGroupCondition(group, engineGroupConditionRuntimeReady, metav1.ConditionFalse, "RuntimeUnavailable", err.Error())
	invalidateEngineGroupHealth(group, "RuntimeUnavailable", "Current external state cannot be established without a usable runtime")
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, updateErr
	}
	if errors.Is(err, ErrRuntimeUnavailable) {
		return ctrl.Result{}, nil
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, err
}

func (r *Reconciler) reconcileCheckpointFailure(
	ctx context.Context,
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	err error,
) (ctrl.Result, error) {
	before := group.Status.DeepCopy()
	invalidateEngineGroupHealth(group, "CheckpointUnavailable", "Recovery authority cannot be established: "+err.Error())
	setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "CheckpointUnavailable",
		"No new operation may proceed without its durable recovery checkpoint")
	if updateErr := r.updateEngineGroupStatus(ctx, group, before); updateErr != nil {
		return ctrl.Result{}, errors.Join(err, updateErr)
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, err
}

func (r *Reconciler) updateEngineGroupStatus(
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

func (r *Reconciler) projectEngineGroupStatus(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	profile nvidiacomv1beta1.EngineGroupProfileStatus,
	status domain.GroupStatus,
	observationErr *domain.ObservationError,
	planningErr error,
) {
	if observationErr == nil {
		group.Status.ObservedGeneration = group.Generation
	}
	projectEngineGroupScaleIdentity(group)
	group.Status.Profile = profile.DeepCopy()
	group.Status.Replicas = int32(len(status.Capacity.Observed.Allocations))
	group.Status.ReleaseAuthorizations = engineGroupReleaseFencesToAPI(status.Capacity.Observed.ReleaseFences)

	// Project fresh engine truth independently from the coordinator's accepted topology history.
	observedTopology := status.Membership.Observed.CommittedTopology
	topologyKnown := observedTopology.Generation > 0
	if topologyKnown {
		topology := engineGroupTopologyToAPI(observedTopology)
		group.Status.Topology = &topology
		group.Status.ActiveNativeMemberCount = observedTopology.NativeMemberCount()
	} else {
		group.Status.Topology = nil
		group.Status.ActiveNativeMemberCount = 0
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
	group.Status.DesiredNativeMemberCount = int32(len(group.Status.DesiredNativeMembers))
	group.Status.ReplicaStates = projectEngineGroupReplicaStates(status, group.Status.ReplicaStates)
	group.Status.AvailableReplicas = countAvailableEngineGroupReplicas(group.Status.ReplicaStates, profile.NativeMembersPerReplica)
	recognizedTopology, found := status.Topologies.Current()
	topologyRecognized := found && domain.SameTopology(recognizedTopology, observedTopology)
	projectEngineGroupConditions(
		group,
		status,
		topologyKnown,
		topologyRecognized,
		observationErr,
		planningErr,
	)
}

func projectEngineGroupScaleIdentity(group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) {
	group.Status.ScaleUnit = nvidiacomv1beta1.EngineGroupScaleUnitReplicas
	group.Status.Selector = strings.Join([]string{
		consts.KubeLabelDynamoEngineGroup + "=" + group.Name,
		consts.KubeLabelDynamoScaleRepresentative + "=" + consts.KubeLabelDynamoScaleRepresentativeYes,
	}, ",")
}

func projectEngineGroupConditions(
	group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
	status domain.GroupStatus,
	topologyKnown bool,
	topologyRecognized bool,
	observationErr *domain.ObservationError,
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
	progressing := status.Transition != nil && (status.Transition.Outcome == domain.TransitionOutcomeProgressing ||
		status.Transition.Outcome == domain.TransitionOutcomeReverting)
	setEngineGroupCondition(group, engineGroupConditionProgressing, conditionStatus(progressing),
		chooseEngineGroupString(progressing, "TransitionInProgress", "NoTransitionInProgress"),
		chooseEngineGroupString(progressing, "An Engine Group transition is progressing", "No Engine Group transition is progressing"))
	if observationErr != nil {
		message := fmt.Sprintf("Current %s state could not be established", observationErr.Authority)
		invalidateEngineGroupHealth(group, "ObservationFailed", message)
		return
	}

	setEngineGroupCondition(group, engineGroupConditionTopologyKnown, conditionStatus(topologyKnown),
		chooseEngineGroupString(topologyKnown, "TopologyObserved", "TopologyUnknown"),
		chooseEngineGroupString(topologyKnown, "Authoritative engine membership is known", "Authoritative engine membership is unknown"))
	available := topologyKnown && group.Status.ActiveNativeMemberCount >= group.Status.Profile.MinSafeServingNativeMembers &&
		allEngineGroupMembersAvailable(
			status.Membership.Observed.CommittedTopology,
			status.Registry,
			status.Capacity.Observed,
		) &&
		allEngineGroupMembersAdmitted(status.Membership.Observed.CommittedTopology, status.Traffic.Observed)
	setEngineGroupCondition(group, engineGroupConditionAvailable, conditionStatus(available),
		chooseEngineGroupString(available, "ServingAvailable", "ServingUnavailable"),
		chooseEngineGroupString(available, "The committed topology is available and admitted", "The committed topology is not fully available and admitted"))

	targetReached := topologyKnown && group.Status.DesiredAssignmentGeneration == group.Generation &&
		group.Status.Replicas == group.Spec.Replicas &&
		int64(group.Status.DesiredNativeMemberCount) == int64(group.Spec.Replicas)*int64(group.Status.Profile.NativeMembersPerReplica) &&
		slices.Equal(engineGroupTopologyNativeMembers(status.Membership.Observed.CommittedTopology), group.Status.DesiredNativeMembers)
	setEngineGroupCondition(group, engineGroupConditionTargetReached, conditionStatus(targetReached),
		chooseEngineGroupString(targetReached, "TargetReached", "TargetNotReached"),
		chooseEngineGroupString(targetReached, "Engine membership reached the desired target", "Engine membership has not reached the desired target"))
	transitionSettled := status.Transition == nil ||
		status.Transition.Outcome == domain.TransitionOutcomeCompleted ||
		status.Transition.Outcome == domain.TransitionOutcomeRolledBack
	allocationDegraded := engineGroupHasDegradedAllocation(group.Status.ReplicaStates)
	if targetReached && available && topologyRecognized && transitionSettled && !allocationDegraded {
		group.Status.LastStableReplicas = group.Status.Replicas
		group.Status.LastStableTopologyGeneration = status.Membership.Observed.CommittedTopology.Generation
	}
	degraded := allocationDegraded ||
		(engineGroupUnexpectedMembershipLoss(group, status) && !exactPlannedRetirementObserved(status))
	setEngineGroupCondition(group, engineGroupConditionDegraded, conditionStatus(degraded),
		chooseEngineGroupString(degraded, "UnexpectedMembershipLossOrDegradation", "MembershipStable"),
		chooseEngineGroupString(degraded, "An allocation is degraded or membership has unexpected losses", "Engine membership is not degraded"))
}

// invalidateEngineGroupHealth retains historical evidence without advertising current authority.
func invalidateEngineGroupHealth(group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup, reason, message string) {
	for _, conditionType := range []string{
		engineGroupConditionTopologyKnown, engineGroupConditionTargetReached,
		engineGroupConditionAvailable, engineGroupConditionDegraded,
	} {
		setEngineGroupCondition(group, conditionType, metav1.ConditionUnknown, reason, message)
	}
}

func exactPlannedRetirementObserved(status domain.GroupStatus) bool {
	if status.Transition == nil || status.Transition.Spec.Plan.Change.Kind != domain.PlanKindRetire ||
		status.Transition.Spec.Plan.Change.Retire == nil {
		return false
	}
	base, found := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
	if !found {
		return false
	}
	retiring := make(map[domain.ReplicaID]struct{}, len(status.Transition.Spec.Plan.Change.Retire.Replicas))
	for _, replicaID := range status.Transition.Spec.Plan.Change.Retire.Replicas {
		retiring[replicaID] = struct{}{}
	}
	expected := make([]domain.ReplicaMembership, 0, len(base.Replicas)-len(retiring))
	for _, member := range base.Replicas {
		if _, found := retiring[member.ReplicaID]; !found {
			expected = append(expected, member)
		}
	}
	return domain.SameMembershipSet(expected, status.Membership.Observed.CommittedTopology.Replicas)
}

func allEngineGroupMembersAdmitted(
	topology domain.MembershipTopology,
	traffic domain.TrafficObservation,
) bool {
	return domain.SameMembershipSet(topology.Replicas, traffic.Admitted)
}

func allEngineGroupMembersAvailable(
	topology domain.MembershipTopology,
	registry domain.ReplicaRegistry,
	capacity domain.CapacityObservation,
) bool {
	for _, member := range topology.Replicas {
		record, found := engineGroupReplicaRecord(registry, member.ReplicaID)
		if !found || record.Current == nil || !domain.MembershipMatchesIncarnation(member, *record.Current) {
			return false
		}
		allocation, found := engineGroupAllocationByReplica(capacity, member.ReplicaID)
		if !found || !allocation.Available || !domain.SameIncarnation(allocation.Incarnation, *record.Current) {
			return false
		}
	}
	return true
}

func engineGroupAllocationByReplica(
	observation domain.CapacityObservation,
	replicaID domain.ReplicaID,
) (domain.CapacityAllocation, bool) {
	for _, allocation := range observation.Allocations {
		if allocation.Incarnation.ReplicaID == replicaID {
			return allocation, true
		}
	}
	return domain.CapacityAllocation{}, false
}

func engineGroupReplicaRecord(
	registry domain.ReplicaRegistry,
	replicaID domain.ReplicaID,
) (domain.ReplicaRecord, bool) {
	for _, record := range registry.Replicas {
		if record.ReplicaID == replicaID {
			return record, true
		}
	}
	return domain.ReplicaRecord{}, false
}

func engineGroupTopologyMembership(
	topology domain.MembershipTopology,
	replicaID domain.ReplicaID,
) (domain.ReplicaMembership, bool) {
	for _, membership := range topology.Replicas {
		if membership.ReplicaID == replicaID {
			return membership, true
		}
	}
	return domain.ReplicaMembership{}, false
}

func validateEngineGroupRuntime(runtime Runtime) error {
	if runtime.Profile.Fingerprint == "" {
		return errors.New("Engine Group runtime profile fingerprint is empty")
	}
	// Reject incomplete geometry rather than silently assuming one native member per allocation.
	if runtime.Profile.NativeMembersPerReplica < 1 || runtime.Profile.PodsPerReplica < 1 ||
		runtime.Profile.MinSafeServingNativeMembers < 1 || runtime.Profile.MinSupportedReplicas < 1 ||
		runtime.Profile.MaxSupportedReplicas < runtime.Profile.MinSupportedReplicas {
		return errors.New("Engine Group runtime profile has invalid allocation or native-member geometry")
	}
	if runtime.Capacity == nil || runtime.Membership == nil || runtime.Traffic == nil ||
		runtime.Verifier == nil || runtime.Planner == nil {
		return errors.New("Engine Group runtime has incomplete adapters")
	}
	return nil
}

func engineGroupID(group *nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup) domain.GroupID {
	return domain.GroupID(group.UID)
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
func (r *Reconciler) SetupWithManager(mgr ctrl.Manager, groveEnabled bool) error {
	// Native clique deletion and recreation must wake the same logical world, even without Pod updates.
	controller := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{}).
		Owns(&corev1.ConfigMap{}).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(r.engineGroupsForCapacity)).
		Named("dynamographdeploymentenginegroup")
	if groveEnabled {
		controller = controller.Watches(&grove.PodClique{}, handler.EnqueueRequestsFromMapFunc(r.engineGroupsForCapacity))
	}
	return controller.Complete(r)
}

func (r *Reconciler) engineGroupsForCapacity(
	_ context.Context,
	object client.Object,
) []reconcile.Request {
	groupName := object.GetLabels()[consts.KubeLabelDynamoEngineGroup]
	if groupName == "" {
		return nil
	}
	return []reconcile.Request{{NamespacedName: client.ObjectKey{
		Namespace: object.GetNamespace(),
		Name:      groupName,
	}}}
}

type unavailableEngineGroupRuntimeProvider struct{}

func (unavailableEngineGroupRuntimeProvider) Resolve(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup,
) (Runtime, error) {
	return Runtime{}, ErrRuntimeUnavailable
}
