/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"sort"
	"strings"

	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/imdario/mergo"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"

	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	rbacv1 "k8s.io/api/rbac/v1"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

type Reason string
type Message string

const (
	reasonFailedToInitializeWorkerHash Reason = "failed_to_initialize_worker_hash"
	reasonRollingUpdateFailed          Reason = "rolling_update_failed"

	dgdComponentPodIndex = ".metadata.dgdComponent"
)

// rbacManager interface for managing RBAC resources
type rbacManager interface {
	EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

// DynamoGraphDeploymentReconciler reconciles a DynamoGraphDeployment object
type DynamoGraphDeploymentReconciler struct {
	client.Client
	Config                *configv1alpha1.OperatorConfiguration
	RuntimeConfig         *commoncontroller.RuntimeConfig
	RestConfig            *rest.Config
	Recorder              record.EventRecorder
	DockerSecretRetriever dockerSecretRetriever
	ScaleClient           scale.ScalesGetter
	SSHKeyManager         *secret.SSHKeyManager
	RBACManager           rbacManager
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=grove.io,resources=podcliques,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquescalinggroups,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquescalinggroups/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=grove.io,resources=clustertopologybindings,verbs=get;list;watch
// +kubebuilder:rbac:groups=scheduling.run.ai,resources=queues,verbs=get;list
// +kubebuilder:rbac:groups=inference.networking.k8s.io,resources=inferencepools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.istio.io,resources=destinationrules,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaimtemplates,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=deviceclasses,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=rolebindings,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoGraphDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *DynamoGraphDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logger := log.FromContext(ctx)

	reason := Reason("undefined")
	message := Message("")
	state := nvidiacomv1beta1.DGDStatePending
	// retrieve the CRD
	dynamoDeployment := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err = r.Get(ctx, req.NamespacedName, dynamoDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	defer func() {
		// Skip status update if DGD is being deleted
		if !dynamoDeployment.GetDeletionTimestamp().IsZero() {
			logger.Info("Reconciliation done - skipping status update for deleted resource")
			return
		}

		if err != nil {
			state = nvidiacomv1beta1.DGDStateFailed
			message = Message(err.Error())
			logger.Error(err, "Reconciliation failed")
		}
		dynamoDeployment.SetState(state)

		readyStatus := metav1.ConditionFalse
		if state == nvidiacomv1beta1.DGDStateSuccessful {
			readyStatus = metav1.ConditionTrue
		}

		// Update Ready condition
		dynamoDeployment.AddStatusCondition(metav1.Condition{
			Type:    "Ready",
			Status:  readyStatus,
			Reason:  string(reason),
			Message: string(message),
		})

		// Only set ObservedGeneration when reconciliation succeeded (no error),
		// so it accurately reflects the last successfully processed generation.
		if err == nil {
			dynamoDeployment.Status.ObservedGeneration = dynamoDeployment.Generation
		}
		// Propagate topology condition from framework (e.g., Grove PCS) to DGD status
		r.propagateTopologyCondition(ctx, dynamoDeployment)

		updateErr := r.Status().Update(ctx, dynamoDeployment)
		if updateErr != nil {
			logger.Error(updateErr, "Unable to update the CRD status", "crd", req.NamespacedName, "state", state, "reason", reason, "message", message)
			// Set err to trigger requeue
			if err == nil {
				err = updateErr
			}
		}
		logger.Info("Reconciliation done")
	}()

	// Handle finalizer
	deleted, err := commoncontroller.HandleFinalizer(ctx, dynamoDeployment, r.Client, r)
	if err != nil {
		logger.Error(err, "failed to handle the finalizer")
		reason = "failed_to_handle_the_finalizer"
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	if err = r.migrateCurrentWorkerHashIfNeeded(ctx, dynamoDeployment); err != nil {
		logger.Error(err, "Failed to migrate worker hash")
		reason = "failed_to_migrate_worker_hash"
		return ctrl.Result{}, err
	}

	if r.supportsManagedRollingUpdate(dynamoDeployment) {
		if err = r.initializeWorkerHashIfNeeded(ctx, dynamoDeployment); err != nil {
			logger.Error(err, "Failed to initialize worker hash")
			reason = reasonFailedToInitializeWorkerHash
			message = Message(err.Error())
			return ctrl.Result{}, err
		}

		rollingUpdateInProgress := r.isRollingUpdateInProgress(dynamoDeployment)
		triggerRollingUpdate := false
		if !rollingUpdateInProgress {
			triggerRollingUpdate, err = r.shouldTriggerRollingUpdate(dynamoDeployment)
			if err != nil {
				logger.Error(err, "Failed to check rolling update trigger")
				state = nvidiacomv1beta1.DGDStateFailed
				reason = reasonRollingUpdateFailed
				message = Message(err.Error())
				return ctrl.Result{}, err
			}
		}
		if rollingUpdateInProgress || triggerRollingUpdate {
			if err = r.reconcileRollingUpdate(ctx, dynamoDeployment); err != nil {
				logger.Error(err, "Failed to reconcile rolling update")
				state = nvidiacomv1beta1.DGDStateFailed
				reason = reasonRollingUpdateFailed
				message = Message(err.Error())
				return ctrl.Result{}, err
			}
		}
	} else {
		if r.currentWorkerHashes(dynamoDeployment).empty() {
			hashes, err := r.desiredWorkerHashes(dynamoDeployment)
			if err != nil {
				logger.Error(err, "Failed to compute worker hash for unsupported pathway")
				reason = reasonFailedToInitializeWorkerHash
				message = Message(err.Error())
				return ctrl.Result{}, err
			}
			r.setCurrentWorkerHashes(dynamoDeployment, workerHashesForCompletedGeneration(hashes.v2, hashes))
			if updateErr := r.Update(ctx, dynamoDeployment); updateErr != nil {
				logger.Error(updateErr, "Failed to initialize worker hash for unsupported pathway")
				reason = reasonFailedToInitializeWorkerHash
				message = Message(updateErr.Error())
				return ctrl.Result{}, updateErr
			}
		}

		// For unsupported pathways, log if a rolling update would have been triggered
		triggerRollingUpdate, err := r.shouldTriggerRollingUpdate(dynamoDeployment)
		if err != nil {
			logger.Error(err, "Failed to check rolling update trigger for unsupported pathway")
			state = nvidiacomv1beta1.DGDStateFailed
			reason = reasonRollingUpdateFailed
			message = Message(err.Error())
			return ctrl.Result{}, err
		}
		if triggerRollingUpdate {
			logger.Info("Worker spec change detected but rolling update not supported for this pathway",
				"isGrove", r.isGrovePathway(dynamoDeployment),
				"hasMultinode", dynamoDeployment.HasAnyMultinodeComponent())
			r.Recorder.Event(dynamoDeployment, corev1.EventTypeWarning, "RollingUpdateNotSupported",
				"Worker spec changed but custom rolling updates are not supported for Grove/multinode deployments")

			// Update the hash to prevent repeated warnings. If the unsupported
			// path is processing a v2-only worker change, preserve the migrated
			// v2-only state instead of resurrecting the downgrade-compatible v1
			// annotation for pod contents it no longer represents.
			hashes, err := r.desiredWorkerHashes(dynamoDeployment)
			if err != nil {
				logger.Error(err, "Failed to compute worker hash for unsupported pathway")
				state = nvidiacomv1beta1.DGDStateFailed
				reason = reasonRollingUpdateFailed
				message = Message(err.Error())
				return ctrl.Result{}, err
			}
			r.setCurrentWorkerHashes(dynamoDeployment, r.workerHashesForUnsupportedPathway(dynamoDeployment, hashes))
			if updateErr := r.Update(ctx, dynamoDeployment); updateErr != nil {
				logger.Error(updateErr, "Failed to update worker hash for unsupported pathway")
			}
		}
	}

	reconcileResult, err := r.reconcileResources(ctx, dynamoDeployment)

	if err != nil {
		logger.Error(err, "failed to reconcile the resources")
		reason = "failed_to_reconcile_the_resources"
		return ctrl.Result{}, err
	}

	// Assign status only after confirming the reconcile succeeded. On error,
	// reconcileResources returns an empty ReconcileResult; assigning it here
	// would wipe Components/Restart, which the deferred status update would then
	// persist. Deferring the assignment past the error check preserves the
	// last-known-good status on a transient failure.
	state = reconcileResult.State
	reason = reconcileResult.Reason
	message = reconcileResult.Message
	dynamoDeployment.Status.Components = reconcileResult.ComponentStatus
	dynamoDeployment.Status.Restart = reconcileResult.RestartStatus

	// Override state based on rolling update status if a rolling update is in progress
	if dynamoDeployment.Status.RollingUpdate != nil {
		switch dynamoDeployment.Status.RollingUpdate.Phase {
		case nvidiacomv1beta1.RollingUpdatePhaseCompleted:
			// Keep the reconcileResult state (should be Ready if resources are ready)
		case nvidiacomv1beta1.RollingUpdatePhasePending, nvidiacomv1beta1.RollingUpdatePhaseInProgress:
			// Rolling update in progress - resources are being transitioned
			if state != nvidiacomv1beta1.DGDStateFailed {
				state = nvidiacomv1beta1.DGDStatePending
				reason = "rolling_update_in_progress"
				message = "Rolling update in progress"
			}
		}
	}

	return ctrl.Result{}, nil
}

type Resource interface {
	IsReady() (ready bool, reason string)
	GetName() string
	GetComponentStatuses() map[string]nvidiacomv1beta1.ComponentReplicaStatus
}

type ReconcileResult struct {
	State           nvidiacomv1beta1.DGDState
	Reason          Reason
	Message         Message
	ComponentStatus map[string]nvidiacomv1beta1.ComponentReplicaStatus
	RestartStatus   *nvidiacomv1beta1.RestartStatus
}

func (r *DynamoGraphDeploymentReconciler) reconcileResources(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) (ReconcileResult, error) {
	logger := log.FromContext(ctx)

	// Ensure planner RBAC exists in cluster-wide mode
	if r.Config.Namespace.Restricted == "" {
		if r.RBACManager == nil {
			return ReconcileResult{}, fmt.Errorf("RBAC manager not initialized in cluster-wide mode")
		}
		if r.Config.RBAC.PlannerClusterRoleName == "" {
			return ReconcileResult{}, fmt.Errorf("planner ClusterRole name is required in cluster-wide mode")
		}
		if err := r.RBACManager.EnsureServiceAccountWithRBAC(
			ctx,
			dynamoDeployment.Namespace,
			consts.PlannerServiceAccountName,
			r.Config.RBAC.PlannerClusterRoleName,
		); err != nil {
			logger.Error(err, "Failed to ensure planner RBAC")
			return ReconcileResult{}, fmt.Errorf("failed to ensure planner RBAC: %w", err)
		}

		// Ensure EPP RBAC exists in cluster-wide mode if EPP service is present
		if dynamoDeployment.HasEPPComponent() {
			if r.Config.RBAC.EPPClusterRoleName == "" {
				return ReconcileResult{}, fmt.Errorf("EPP ClusterRole name is required in cluster-wide mode when EPP service is present")
			}
			if err := r.RBACManager.EnsureServiceAccountWithRBAC(
				ctx,
				dynamoDeployment.Namespace,
				consts.EPPServiceAccountName,
				r.Config.RBAC.EPPClusterRoleName,
			); err != nil {
				logger.Error(err, "Failed to ensure EPP RBAC")
				return ReconcileResult{}, fmt.Errorf("failed to ensure EPP RBAC: %w", err)
			}
		}
	}

	// Reconcile top-level PVCs first
	err := r.reconcilePVCs(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "Failed to reconcile top-level PVCs")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile top-level PVCs: %w", err)
	}

	// Reconcile discovery RBAC before checkpoints so auto-created checkpoint
	// Jobs can reference the DGD discovery service account in their prepared
	// pod template.
	err = r.reconcileK8sDiscoveryResources(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "Failed to reconcile K8s discovery resources")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile K8s discovery resources: %w", err)
	}

	if err := r.reconcileGMSResourceClaimTemplates(ctx, dynamoDeployment); err != nil {
		return ReconcileResult{}, err
	}

	// Reconcile checkpoints for components with checkpointing enabled.
	checkpointStatuses, checkpointInfos, err := r.reconcileCheckpoints(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "Failed to reconcile checkpoints")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile checkpoints: %w", err)
	}
	dynamoDeployment.Status.Checkpoints = checkpointStatuses

	// Reconcile EPP resources (ConfigMaps, Services, InferencePools) if EPP service exists
	err = r.reconcileEPPResources(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "Failed to reconcile EPP resources")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile EPP resources: %w", err)
	}

	// Reconcile the wait-for-leader ConfigMap for multinode mp deployments
	err = r.reconcileWaitLeaderConfigMap(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "Failed to reconcile wait-leader ConfigMap")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile wait-leader ConfigMap: %w", err)
	}

	// Determine if any component is multinode.
	hasMultinode := dynamoDeployment.HasAnyMultinodeComponent()

	if r.SSHKeyManager != nil && hasMultinode {
		if err := r.SSHKeyManager.EnsureAndReplicate(ctx, dynamoDeployment.Namespace); err != nil {
			logger.Error(err, "Failed to ensure MPI SSH key secret", "namespace", dynamoDeployment.Namespace)
			return ReconcileResult{}, fmt.Errorf("failed to ensure MPI SSH key secret: %w", err)
		}
	}

	// return error early if Grove and LWS is not available for multinode
	if !r.isGrovePathway(dynamoDeployment) && hasMultinode && !r.RuntimeConfig.Gate.Enabled(features.LWS) {
		err := fmt.Errorf("no multinode orchestrator available")
		logger.Error(err, err.Error(), "hasMultinode", hasMultinode, "lwsEnabled", r.RuntimeConfig.Gate.Enabled(features.LWS))
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Dynamo components deployments: %w", err)
	}

	restartStatus := r.computeRestartStatus(ctx, dynamoDeployment)
	restartState := dynamo.DetermineRestartState(dynamoDeployment, restartStatus)

	var result ReconcileResult
	if r.isGrovePathway(dynamoDeployment) {
		logger.Info("Reconciling Grove resources", "hasMultinode", hasMultinode, "lwsEnabled", r.RuntimeConfig.Gate.Enabled(features.LWS))
		result, err = r.reconcileGroveResources(ctx, dynamoDeployment, restartState, checkpointInfos)
	} else {
		logger.Info("Reconciling Dynamo components deployments", "hasMultinode", hasMultinode, "lwsEnabled", r.RuntimeConfig.Gate.Enabled(features.LWS))
		result, err = r.reconcileDynamoComponentsDeployments(ctx, dynamoDeployment, restartState, checkpointInfos)
	}
	if err != nil {
		logger.Error(err, "Failed to reconcile Dynamo components deployments")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Dynamo components deployments: %w", err)
	}
	result.RestartStatus = restartStatus
	result = applyCheckpointStartupReadiness(result, checkpointInfos)

	// Reconcile DynamoGraphDeploymentScalingAdapters after applying checkpoint
	// startup readiness. In WaitForCheckpoint, the DGD deliberately reports
	// pending before regular workers exist; letting a scaling adapter write
	// replicas while gated would create an avoidable update loop.
	if result.State != nvidiacomv1beta1.DGDStatePending || result.Reason != "waiting_for_checkpoint" {
		err = r.reconcileScalingAdapters(ctx, dynamoDeployment)
		if err != nil {
			logger.Error(err, "Failed to reconcile scaling adapters")
			return ReconcileResult{}, fmt.Errorf("failed to reconcile scaling adapters: %w", err)
		}
	}

	return result, nil
}

func (r *DynamoGraphDeploymentReconciler) isGrovePathway(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return r.RuntimeConfig.Gate.Enabled(features.Grove) && (dgd.Annotations == nil ||
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse)
}

func (r *DynamoGraphDeploymentReconciler) getUpdatedInProgress(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
	if r.isGrovePathway(dgd) {
		return r.getUpdatedInProgressForGrove(ctx, dgd, inProgress)
	}
	return r.getUpdatedInProgressForComponent(ctx, dgd, inProgress)
}

// getUpdatedInProgressForGrove checks which components are still in progress.
func (r *DynamoGraphDeploymentReconciler) getUpdatedInProgressForGrove(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
	logger := log.FromContext(ctx)

	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsName := dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components)
	err := r.Client.Get(ctx, types.NamespacedName{Name: pcsName, Namespace: dgd.Namespace}, pcs)
	if err != nil {
		logger.Error(err, "failed to get PodCliqueSet")
		return inProgress
	}

	if pcs.Status.ObservedGeneration == nil {
		logger.Info("PodCliqueSet observedGeneration is nil", "name", dgd.Name)
		return inProgress
	}

	if *pcs.Status.ObservedGeneration < pcs.Generation {
		logger.Info("PodCliqueSet not yet reconciled", "name", dgd.Name, "generation", pcs.Generation, "observedGeneration", *pcs.Status.ObservedGeneration)
		return inProgress
	}

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		component := dgd.GetComponentByName(componentName)
		if component == nil {
			logger.V(1).Info("component not found in DGD", "componentName", componentName)
			continue
		}
		resourceName := fmt.Sprintf("%s-0-%s", pcsName, strings.ToLower(componentName))

		var isReady bool
		var reason string
		// Keep in sync with reconcileGroveScaling and Grove status aggregation:
		// any component that requires a PodCliqueScalingGroup (multinode or
		// inter-pod GMS) must be queried via CheckPCSGReady, otherwise
		// single-node GMS components stall in the in-progress list because the
		// corresponding PodClique never exists.
		usesPCSG := component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled()
		// A transient read error surfaces here as isReady=false, so the component
		// conservatively remains in-progress (we never mark a restart complete on
		// a component we could not read). The reconcile that owns readiness
		// classification propagates the error and retries, so it is safe to
		// discard it in this progress-tracking helper.
		if usesPCSG {
			isReady, reason, _, _, _ = dynamo.CheckPCSGReady(ctx, r.Client, resourceName, dgd.Namespace, logger)
		} else {
			isReady, reason, _, _, _ = dynamo.CheckPodCliqueReady(ctx, r.Client, resourceName, dgd.Namespace, logger)
		}
		if !isReady {
			logger.V(1).Info("component not ready", "componentName", componentName, "resourceName", resourceName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}

	return updatedInProgress
}

// propagateTopologyCondition reads the PCS topology condition from Grove and maps it
// to a TopologyLevelsAvailable condition on the DGD. This is a no-op when no
// topology constraints are set or when the Grove pathway is not in use.
func (r *DynamoGraphDeploymentReconciler) propagateTopologyCondition(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	if !dgd.HasAnyTopologyConstraint() || !r.isGrovePathway(dgd) {
		return
	}
	logger := log.FromContext(ctx)

	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.Client.Get(ctx, types.NamespacedName{Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components), Namespace: dgd.Namespace}, pcs); err != nil {
		if errors.IsNotFound(err) {
			return
		}
		logger.V(1).Info("failed to read PCS for topology condition propagation", "error", err)
		return
	}

	// Look for Grove's TopologyLevelsUnavailable condition on the PCS.
	var groveTopoCond *metav1.Condition
	for i := range pcs.Status.Conditions {
		if pcs.Status.Conditions[i].Type == groveconstants.ConditionTopologyLevelsUnavailable {
			groveTopoCond = &pcs.Status.Conditions[i]
			break
		}
	}

	var dynamoCond metav1.Condition
	if groveTopoCond == nil {
		// No topology condition from Grove yet — don't assume healthy.
		dynamoCond = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionUnknown,
			Reason:  nvidiacomv1beta1.ConditionReasonTopologyConditionPending,
			Message: "Waiting for topology condition from the scheduling framework",
		}
	} else if groveTopoCond.Status == metav1.ConditionTrue {
		// Grove reports topology levels are unavailable.
		reason := nvidiacomv1beta1.ConditionReasonTopologyLevelsUnavailable
		if groveTopoCond.Reason == groveconstants.ConditionReasonClusterTopologyNotFound {
			reason = nvidiacomv1beta1.ConditionReasonTopologyDefinitionNotFound
		}
		dynamoCond = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionFalse,
			Reason:  reason,
			Message: groveTopoCond.Message,
		}
		prev := meta.FindStatusCondition(dgd.Status.Conditions, nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable)
		if prev == nil || prev.Status != metav1.ConditionFalse || prev.Reason != reason || prev.Message != groveTopoCond.Message {
			logger.Info("Topology constraints no longer enforced", "reason", reason, "message", groveTopoCond.Message)
			r.Recorder.Eventf(dgd, corev1.EventTypeWarning, reason, "Topology constraints no longer enforced: %s", groveTopoCond.Message)
		}
	} else {
		// Grove's TopologyLevelsUnavailable is False → all levels available.
		dynamoCond = metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeTopologyLevelsAvailable,
			Status:  metav1.ConditionTrue,
			Reason:  nvidiacomv1beta1.ConditionReasonAllTopologyLevelsAvailable,
			Message: "All required topology levels are available in the cluster topology",
		}
	}

	dgd.AddStatusCondition(dynamoCond)
}

func isRestartAlreadyProcessed(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if dgd.Spec.Restart == nil || dgd.Spec.Restart.ID == "" {
		return true
	}

	if dgd.Status.Restart == nil || dgd.Status.Restart.ObservedID == "" {
		return false
	}

	if dgd.Spec.Restart.ID == dgd.Status.Restart.ObservedID &&
		(dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseCompleted ||
			dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseFailed ||
			dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseSuperseded) {
		return true
	}

	return false
}

// scaleGroveResource scales a Grove resource using the generic scaling function
func (r *DynamoGraphDeploymentReconciler) scaleGroveResource(ctx context.Context, resourceName, namespace string, newReplicas int32, resourceType string) error {
	logger := log.FromContext(ctx)
	// Determine the GroupVersionResource based on resource type
	var gvr schema.GroupVersionResource
	switch resourceType {
	case "PodClique":
		gvr = consts.PodCliqueGVR
	case "PodCliqueScalingGroup":
		gvr = consts.PodCliqueScalingGroupGVR
	default:
		return fmt.Errorf("unsupported Grove resource type: %s", resourceType)
	}

	// Use the generic scaling function
	err := commoncontroller.ScaleResource(ctx, r.ScaleClient, gvr, namespace, resourceName, newReplicas)
	if err != nil {
		if errors.IsNotFound(err) {
			// Resource doesn't exist yet - this is normal during initial creation when Grove is still creating the resources asynchronously
			logger.V(1).Info("Grove resource not found yet, skipping scaling for now - will retry on next reconciliation", "gvr", gvr, "name", resourceName, "namespace", namespace)
			return nil
		}
	}
	return err
}

func (r *DynamoGraphDeploymentReconciler) reconcileGrovePodCliqueSet(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	renderDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	existingPodCliqueSet *grovev1alpha1.PodCliqueSet,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (*commoncontroller.Resource, error) {
	logger := log.FromContext(ctx)
	if renderDeployment == nil {
		renderDeployment = dynamoDeployment
	}

	existingRestartAnnotations := restartAnnotationsFromPodCliqueSet(existingPodCliqueSet)

	// generate the dynamoComponentsDeployments from the config
	grovePodCliqueSet, err := dynamo.GenerateGrovePodCliqueSet(ctx, renderDeployment, r.Config, r.RuntimeConfig, r.Client, r.DockerSecretRetriever, restartState, existingRestartAnnotations, checkpointInfos)
	if err != nil {
		logger.Error(err, "failed to generate the Grove GangSet")
		return nil, fmt.Errorf("failed to generate the Grove GangSet: %w", err)
	}
	prepareGroveTopologyConstraintUpgrade(grovePodCliqueSet, existingPodCliqueSet)
	preserveGrovePodCliqueSetOrder(grovePodCliqueSet, existingPodCliqueSet)
	preserveGrovePodCliqueSetReplicas(grovePodCliqueSet, existingPodCliqueSet, checkpointInfos)
	_, syncedGrovePodCliqueSet, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*grovev1alpha1.PodCliqueSet, bool, error) {
		return grovePodCliqueSet, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the Grove GangSet")
		return nil, fmt.Errorf("failed to sync the Grove GangSet: %w", err)
	}
	syncedGrovePodCliqueSetAsResource, err := commoncontroller.NewResourceWithComponentStatuses(
		syncedGrovePodCliqueSet,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			// Grove readiness: all underlying PodCliques and PodCliqueScalingGroups have replicas == availableReplicas.
			// A transient (non-NotFound) read error is handled authoritatively by
			// reconcileGroveResources, which re-evaluates and returns the error so
			// the reconcile retries; here we defensively treat it as not-ready so a
			// read blip can never surface as "ready".
			allComponentsReady, reason, componentStatuses, readErr := dynamo.GetComponentReadinessAndServiceReplicaStatuses(ctx, r.Client, dynamoDeployment)
			if readErr != nil {
				return false, nvidiacomv1beta1.DGDReadyReasonSomeResourcesNotReady, nil
			}
			if !allComponentsReady {
				return false, reason, componentStatuses
			}
			return true, "", componentStatuses
		},
	)
	if err != nil {
		logger.Error(err, "failed to create the Grove PodClique Set resource")
		return nil, fmt.Errorf("failed to create the Grove PodClique Set resource: %w", err)
	}
	return syncedGrovePodCliqueSetAsResource, nil
}

func (r *DynamoGraphDeploymentReconciler) getExistingGrovePodCliqueSet(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) (*grovev1alpha1.PodCliqueSet, error) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	err := r.Client.Get(ctx, types.NamespacedName{Name: dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components), Namespace: dgd.Namespace}, pcs)
	if err != nil && !errors.IsNotFound(err) {
		return nil, fmt.Errorf("failed to get PodCliqueSet: %w", err)
	}
	if errors.IsNotFound(err) {
		return nil, nil
	}
	return pcs, nil
}

func restartAnnotationsFromPodCliqueSet(pcs *grovev1alpha1.PodCliqueSet) map[string]string {
	restartAnnotations := make(map[string]string)
	if pcs == nil {
		return restartAnnotations
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Annotations != nil {
			if timestamp, ok := clique.Annotations[consts.RestartAnnotation]; ok {
				if componentName, ok := clique.Labels[consts.KubeLabelDynamoComponent]; ok {
					restartAnnotations[componentName] = timestamp
				}
			}
		}
	}
	return restartAnnotations
}

func preserveGrovePodCliqueSetOrder(desired *grovev1alpha1.PodCliqueSet, existing *grovev1alpha1.PodCliqueSet) {
	if desired == nil || existing == nil {
		return
	}
	desired.Spec.Template.Cliques = orderLikeExisting(existing.Spec.Template.Cliques, desired.Spec.Template.Cliques, podCliqueTemplateName)
	desired.Spec.Template.PodCliqueScalingGroupConfigs = orderLikeExisting(existing.Spec.Template.PodCliqueScalingGroupConfigs, desired.Spec.Template.PodCliqueScalingGroupConfigs, podCliqueScalingGroupConfigName)
	desired.Spec.Template.ResourceClaimTemplates = orderLikeExisting(existing.Spec.Template.ResourceClaimTemplates, desired.Spec.Template.ResourceClaimTemplates, resourceClaimTemplateConfigName)
}

// prepareGroveTopologyConstraintUpgrade performs the first half of Grove's
// supported legacy topology migration. A pre-alpha.9 constraint has
// packDomain but no topologyName; Grove requires that object to be repaired by
// adding topologyName before packDomain can be migrated to pack.required.
// Keeping the legacy packing shape for this reconciliation lets the next
// reconciliation apply the generated modern shape without recreating the PCS.
func prepareGroveTopologyConstraintUpgrade(desired *grovev1alpha1.PodCliqueSet, existing *grovev1alpha1.PodCliqueSet) {
	if desired == nil || existing == nil {
		return
	}

	prepareLegacyGroveTopologyConstraintRepair(
		desired.Spec.Template.TopologyConstraint,
		existing.Spec.Template.TopologyConstraint,
	)

	existingCliqueConstraints := make(map[string]*grovev1alpha1.TopologyConstraint, len(existing.Spec.Template.Cliques))
	for _, clique := range existing.Spec.Template.Cliques {
		if clique != nil {
			existingCliqueConstraints[clique.Name] = clique.TopologyConstraint
		}
	}
	for _, clique := range desired.Spec.Template.Cliques {
		if clique != nil {
			prepareLegacyGroveTopologyConstraintRepair(clique.TopologyConstraint, existingCliqueConstraints[clique.Name])
		}
	}

	existingScalingGroupConstraints := make(map[string]*grovev1alpha1.TopologyConstraint, len(existing.Spec.Template.PodCliqueScalingGroupConfigs))
	for i := range existing.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &existing.Spec.Template.PodCliqueScalingGroupConfigs[i]
		existingScalingGroupConstraints[config.Name] = config.TopologyConstraint
	}
	for i := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &desired.Spec.Template.PodCliqueScalingGroupConfigs[i]
		prepareLegacyGroveTopologyConstraintRepair(config.TopologyConstraint, existingScalingGroupConstraints[config.Name])
	}
}

func prepareLegacyGroveTopologyConstraintRepair(desired *grovev1alpha1.TopologyConstraint, existing *grovev1alpha1.TopologyConstraint) {
	if desired == nil || existing == nil {
		return
	}
	// A constraint without an explicit desired name inherits from its repaired
	// parent and can migrate packDomain directly in the same update.
	if desired.TopologyName == "" || existing.TopologyName != "" || existing.PackDomain == "" {
		return
	}

	desired.PackDomain = existing.PackDomain
	if existing.Pack == nil {
		desired.Pack = nil
		return
	}
	pack := *existing.Pack
	desired.Pack = &pack
}

// Grove horizontal replicas are driven through scale subresources after creation;
// keep existing template values so DGD replica changes do not update the PCS spec.
func preserveGrovePodCliqueSetReplicas(
	desired *grovev1alpha1.PodCliqueSet,
	existing *grovev1alpha1.PodCliqueSet,
	checkpointInfoByComponent ...map[string]*checkpoint.CheckpointInfo,
) {
	if desired == nil || existing == nil {
		return
	}
	replicaPreserveSkips := map[string]struct{}{}
	if len(checkpointInfoByComponent) > 0 {
		for componentName, info := range checkpointInfoByComponent[0] {
			if info != nil &&
				info.Enabled &&
				info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
				!info.Ready {
				replicaPreserveSkips[strings.ToLower(componentName)] = struct{}{}
			}
		}
	}

	cliquesInScalingGroups := make(map[string]struct{})
	for _, config := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		for _, cliqueName := range config.CliqueNames {
			cliquesInScalingGroups[cliqueName] = struct{}{}
		}
	}

	cliqueReplicasByName := make(map[string]int32, len(existing.Spec.Template.Cliques))
	for _, clique := range existing.Spec.Template.Cliques {
		if clique == nil || clique.Name == "" {
			continue
		}
		cliqueReplicasByName[clique.Name] = clique.Spec.Replicas
	}
	for _, clique := range desired.Spec.Template.Cliques {
		if clique == nil {
			continue
		}
		if _, inScalingGroup := cliquesInScalingGroups[clique.Name]; inScalingGroup {
			continue
		}
		if componentName := clique.Labels[consts.KubeLabelDynamoComponent]; componentName != "" {
			if _, skip := replicaPreserveSkips[strings.ToLower(componentName)]; skip {
				continue
			}
		}
		if replicas, ok := cliqueReplicasByName[clique.Name]; ok {
			clique.Spec.Replicas = replicas
		}
	}

	scalingGroupReplicasByName := make(map[string]*int32, len(existing.Spec.Template.PodCliqueScalingGroupConfigs))
	for _, config := range existing.Spec.Template.PodCliqueScalingGroupConfigs {
		if config.Name == "" {
			// Defensive only; generated PCSG configs always have names.
			continue
		}
		scalingGroupReplicasByName[config.Name] = config.Replicas
	}
	for i := range desired.Spec.Template.PodCliqueScalingGroupConfigs {
		config := &desired.Spec.Template.PodCliqueScalingGroupConfigs[i]
		if _, skip := replicaPreserveSkips[strings.ToLower(config.Name)]; skip {
			continue
		}
		if replicas, ok := scalingGroupReplicasByName[config.Name]; ok {
			config.Replicas = replicas
		}
	}
}

func orderLikeExisting[T any](existing []T, desired []T, nameOf func(T) string) []T {
	if len(existing) == 0 || len(desired) < 2 {
		return desired
	}
	desiredByName := make(map[string]T, len(desired))
	for _, item := range desired {
		if name := nameOf(item); name != "" {
			desiredByName[name] = item
		}
	}
	ordered := make([]T, 0, len(desired))
	used := make(map[string]struct{}, len(desired))
	for _, existingItem := range existing {
		name := nameOf(existingItem)
		if desiredItem, ok := desiredByName[name]; ok {
			ordered = append(ordered, desiredItem)
			used[name] = struct{}{}
		}
	}
	for _, item := range desired {
		name := nameOf(item)
		if name == "" {
			ordered = append(ordered, item)
			continue
		}
		if _, ok := used[name]; !ok {
			ordered = append(ordered, item)
		}
	}
	return ordered
}

func podCliqueTemplateName(clique *grovev1alpha1.PodCliqueTemplateSpec) string {
	if clique == nil {
		return ""
	}
	return clique.Name
}

func podCliqueScalingGroupConfigName(config grovev1alpha1.PodCliqueScalingGroupConfig) string {
	return config.Name
}

func resourceClaimTemplateConfigName(config grovev1alpha1.ResourceClaimTemplateConfig) string {
	return config.Name
}

func (r *DynamoGraphDeploymentReconciler) prepareGroveRenderDeployment(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) (*nvidiacomv1beta1.DynamoGraphDeployment, *grovev1alpha1.PodCliqueSet, error) {
	existingPodCliqueSet, err := r.getExistingGrovePodCliqueSet(ctx, dgd)
	if err != nil {
		return nil, nil, err
	}

	renderDeployment := dgd.DeepCopy()
	for i := range renderDeployment.Spec.Components {
		component := &renderDeployment.Spec.Components[i]
		componentType := string(component.ComponentType)
		if !groveComponentTypeCanUseLegacyWorkerSelector(componentType) {
			continue
		}
		if podCliqueSetHasLegacyWorkerSelector(existingPodCliqueSet, component.ComponentName, componentType) {
			applyLegacyGroveWorkerComponentType(component, componentType)
		}
	}
	return renderDeployment, existingPodCliqueSet, nil
}

func groveComponentTypeCanUseLegacyWorkerSelector(componentType string) bool {
	return componentType == consts.ComponentTypePrefill || componentType == consts.ComponentTypeDecode
}

func podCliqueSetHasLegacyWorkerSelector(pcs *grovev1alpha1.PodCliqueSet, componentName string, componentType string) bool {
	if pcs == nil {
		return false
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique == nil || clique.Labels[consts.KubeLabelDynamoComponent] != componentName {
			continue
		}
		if hasLegacyWorkerSelector(clique.Labels, componentType) {
			return true
		}
	}
	return false
}

func applyLegacyGroveWorkerComponentType(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, subComponentType string) {
	component.ComponentType = nvidiacomv1beta1.ComponentTypeWorker
	if component.PodTemplate == nil {
		component.PodTemplate = &corev1.PodTemplateSpec{}
	}
	if component.PodTemplate.Labels == nil {
		component.PodTemplate.Labels = map[string]string{}
	}
	if _, ok := component.PodTemplate.Labels[consts.KubeLabelDynamoSubComponentType]; !ok {
		component.PodTemplate.Labels[consts.KubeLabelDynamoSubComponentType] = subComponentType
	}
}

// reconcileGroveScaling handles scaling operations for Grove resources based on component replica changes.
func (r *DynamoGraphDeploymentReconciler) reconcileGroveScaling(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) error {
	logger := log.FromContext(ctx)
	logger.V(1).Info("Reconciling Grove scaling operations")

	replicaIndex := 0
	pcsName := dynamo.PCSNameForDGD(dynamoDeployment.Name, dynamoDeployment.Spec.Components)
	for i := range dynamoDeployment.Spec.Components {
		component := &dynamoDeployment.Spec.Components[i]
		componentName := component.ComponentName
		info := checkpointInfos[componentName]
		gated := info != nil &&
			info.Enabled &&
			info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
			!info.Ready
		if component.Replicas == nil && !gated {
			continue
		}
		replicas := int32(1)
		if component.Replicas != nil {
			replicas = *component.Replicas
		}
		if gated {
			replicas = 0
		}

		usesPCSG := component.GetNumberOfNodes() > 1 || component.IsInterPodGMSEnabled()
		resourceName := fmt.Sprintf("%s-%d-%s", pcsName, replicaIndex, strings.ToLower(componentName))

		if usesPCSG {
			err := r.scaleGroveResource(ctx,
				resourceName,
				dynamoDeployment.Namespace,
				replicas,
				"PodCliqueScalingGroup")
			if err != nil {
				logger.Error(err, "Failed to scale PodCliqueScalingGroup", "componentName", componentName, "resourceName", resourceName, "replicas", replicas)
				return fmt.Errorf("failed to scale PodCliqueScalingGroup %s: %w", resourceName, err)
			}
		} else {
			err := r.scaleGroveResource(ctx,
				resourceName,
				dynamoDeployment.Namespace,
				replicas,
				"PodClique")
			if err != nil {
				logger.Error(err, "Failed to scale PodClique", "componentName", componentName, "resourceName", resourceName, "replicas", replicas)
				return fmt.Errorf("failed to scale PodClique %s: %w", resourceName, err)
			}
		}
	}

	logger.V(1).Info("Successfully reconciled Grove scaling operations")
	return nil
}

// reconcileGMSResourceClaimTemplates syncs ResourceClaimTemplates when DRA is
// available, including deleting stale templates for components that no longer
// use GMS. When DRA is unavailable, it fails fast if any component needs
// DRA-backed GPU allocation.
//
// Both the GMS sidecar and inter-pod GMS
// failover (failover.mode=interPod) allocate GPUs via DRA ResourceClaims.
// Without DRA, pods would be admitted by the webhook but silently reference
// ResourceClaimTemplates that reconcile never creates, producing a confusing
// "resourceclaim not found" at schedule time. We fail fast here so the user
// gets an actionable error instead.
func (r *DynamoGraphDeploymentReconciler) reconcileGMSResourceClaimTemplates(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)

	if !r.RuntimeConfig.Gate.Enabled(features.DRA) {
		for i := range dynamoDeployment.Spec.Components {
			component := &dynamoDeployment.Spec.Components[i]
			if dynamo.GetGPUMemoryService(component) != nil || component.IsInterPodFailoverEnabled() {
				return fmt.Errorf(
					"gpuMemoryService / inter-pod GMS failover requires DRA (Dynamic Resource Allocation), " +
						"but DRA is not available (either the resource.k8s.io/v1 API is not registered on this cluster, " +
						"which requires Kubernetes 1.34+, or DRA has been explicitly disabled in the operator configuration)")
			}
		}
		return nil
	}

	for i := range dynamoDeployment.Spec.Components {
		component := &dynamoDeployment.Spec.Components[i]
		gmsSpec := dynamo.GetGPUMemoryService(component)
		componentName := component.ComponentName
		gpuCount := 0
		deviceClassName := ""
		if gmsSpec != nil {
			var err error
			gpuCount, err = dra.ExtractGPUCountFromResourceRequirements(dynamo.GetMainContainerResources(component))
			if err != nil {
				return fmt.Errorf("invalid GPU resource requirements for GMS ResourceClaimTemplate for %s: %w", componentName, err)
			}
			deviceClassName = gmsSpec.DeviceClassName
			if deviceClassName == "" {
				deviceClassName = dra.DefaultDeviceClassName
			}
		}
		claimTemplateName := dra.ResourceClaimTemplateName(dynamoDeployment.Name, componentName)
		_, _, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*resourcev1.ResourceClaimTemplate, bool, error) {
			return dra.GenerateResourceClaimTemplate(ctx, r.Client, claimTemplateName, dynamoDeployment.Namespace, gpuCount, deviceClassName)
		})
		if err != nil {
			logger.Error(err, "failed to sync GMS ResourceClaimTemplate", "component", componentName)
			return fmt.Errorf("failed to sync GMS ResourceClaimTemplate for %s: %w", componentName, err)
		}
	}
	return nil
}

func findPodTemplateContainer(podTemplate *corev1.PodTemplateSpec, containerName string) (*corev1.Container, error) {
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == containerName {
			return &podTemplate.Spec.Containers[i], nil
		}
	}
	return nil, fmt.Errorf("checkpoint job pod template: pod spec has no container named %q", containerName)
}

func (r *DynamoGraphDeploymentReconciler) reconcileGroveResources(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment, restartState *dynamo.RestartState, checkpointInfos map[string]*checkpoint.CheckpointInfo) (ReconcileResult, error) {
	logger := log.FromContext(ctx)

	renderDeployment, existingPodCliqueSet, err := r.prepareGroveRenderDeployment(ctx, dynamoDeployment)
	if err != nil {
		return ReconcileResult{}, err
	}

	grovePodCliqueSetAsResource, err := r.reconcileGrovePodCliqueSet(ctx, dynamoDeployment, renderDeployment, existingPodCliqueSet, restartState, checkpointInfos)
	if err != nil {
		logger.Error(err, "failed to reconcile the Grove PodClique Set")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile the Grove PodClique Set: %w", err)
	}

	// Handle Grove scaling operations after structural changes
	if err := r.reconcileGroveScaling(ctx, dynamoDeployment, checkpointInfos); err != nil {
		logger.Error(err, "failed to reconcile Grove scaling")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile Grove scaling: %w", err)
	}

	// Reconcile headless services for model endpoint discovery
	if err := dynamo.ReconcileModelServicesForComponents(
		ctx,
		r,
		dynamoDeployment,
		dynamo.ComponentsByName(dynamoDeployment),
		dynamoDeployment.Namespace,
	); err != nil {
		logger.Error(err, "failed to reconcile model services")
		return ReconcileResult{}, fmt.Errorf("failed to reconcile model services: %w", err)
	}

	resources := []Resource{grovePodCliqueSetAsResource}
	for i := range renderDeployment.Spec.Components {
		component := &renderDeployment.Spec.Components[i]
		componentName := component.ComponentName

		// if k8s discovery is enabled, create a service for each component
		// else, only create for the frontend component
		isK8sDiscoveryEnabled := commoncontroller.IsK8sDiscoveryEnabled(r.Config.Discovery.Backend, dynamoDeployment.Annotations)
		if isK8sDiscoveryEnabled || string(component.ComponentType) == consts.ComponentTypeFrontend {
			dynamoNamespace := renderDeployment.GetDynamoNamespaceForComponent(component)
			mainComponentService, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
				ServiceName:     dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""),
				Namespace:       dynamoDeployment.Namespace,
				ComponentType:   string(component.ComponentType),
				DynamoNamespace: dynamoNamespace,
				ComponentName:   componentName,
				Labels:          dynamo.GetDGDComponentResourceLabels(renderDeployment, componentName, component),
				Annotations:     dynamo.GetDGDComponentResourceAnnotations(renderDeployment, componentName, component),
				IsK8sDiscovery:  isK8sDiscoveryEnabled,
			})
			if err != nil {
				logger.Error(err, "failed to generate the main component service")
				return ReconcileResult{}, fmt.Errorf("failed to generate the main component service: %w", err)
			}
			_, syncedMainComponentService, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*corev1.Service, bool, error) {
				return mainComponentService, false, nil
			})
			if err != nil {
				logger.Error(err, "failed to sync the main component service")
				return ReconcileResult{}, fmt.Errorf("failed to sync the main component service: %w", err)
			}
			if syncedMainComponentService != nil {
				if syncedMainComponentService.Annotations == nil {
					syncedMainComponentService.Annotations = make(map[string]string)
				}
				desiredAnnotations := dynamo.GetDGDComponentResourceAnnotations(renderDeployment, componentName, component)
				var updateAnnotations bool
				for key, value := range desiredAnnotations {
					if val, ok := syncedMainComponentService.Annotations[key]; !ok || val != value {
						syncedMainComponentService.Annotations[key] = value
						updateAnnotations = true
					}
				}
				if updateAnnotations {
					err = r.Update(ctx, syncedMainComponentService)
					if err != nil {
						logger.Error(err, fmt.Sprintf("Failed to update main component service %s.", componentName))
						r.GetRecorder().Eventf(dynamoDeployment, corev1.EventTypeWarning, "UpdateService", "Failed to update Service %s: %s", componentName, err)
						return ReconcileResult{}, fmt.Errorf("failed to update main component service %s: %w", componentName, err)
					}
				}
				mainComponentServiceAsResource, err := commoncontroller.NewResource(syncedMainComponentService,
					func() (bool, string) {
						return true, ""
					})
				if err != nil {
					return ReconcileResult{}, fmt.Errorf("failed to sync the main component service: %w", err)
				}
				resources = append(resources, mainComponentServiceAsResource)
			}
		}

		if string(component.ComponentType) == consts.ComponentTypeFrontend {
			// generate the main component ingress
			ingressSpec := dynamo.GenerateDefaultIngressSpec(dynamoDeployment, r.Config.Ingress)
			if preservedIngressSpec, ok := dynamo.GetDGDComponentPreservedIngressSpec(dynamoDeployment, componentName); ok {
				ingressSpec = preservedIngressSpec
			}
			mainComponentIngress := dynamo.GenerateComponentIngress(ctx, dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""), dynamoDeployment.Namespace, ingressSpec)
			_, syncedMainComponentIngress, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*networkingv1.Ingress, bool, error) {
				if !ingressSpec.Enabled || ingressSpec.IngressControllerClassName == nil {
					logger.Info("Ingress is not enabled")
					return mainComponentIngress, true, nil
				}
				return mainComponentIngress, false, nil
			})
			if err != nil {
				logger.Error(err, "failed to sync the main component ingress")
				return ReconcileResult{}, fmt.Errorf("failed to sync the main component ingress: %w", err)
			}
			if syncedMainComponentIngress != nil {
				mainComponentIngressAsResource, err := commoncontroller.NewResource(syncedMainComponentIngress,
					func() (bool, string) {
						return true, ""
					})
				if err != nil {
					return ReconcileResult{}, fmt.Errorf("failed to create the main component ingress resource: %w", err)
				}
				resources = append(resources, mainComponentIngressAsResource)
			}
			// generate the main component virtual service
			if r.Config.Ingress.UseVirtualService() {
				mainComponentVirtualService := dynamo.GenerateComponentVirtualService(ctx, dynamo.GetDCDResourceName(dynamoDeployment, componentName, ""), dynamoDeployment.Namespace, ingressSpec)
				_, syncedMainComponentVirtualService, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*networkingv1beta1.VirtualService, bool, error) {
					if !ingressSpec.IsVirtualServiceEnabled() {
						logger.Info("VirtualService is not enabled")
						return mainComponentVirtualService, true, nil
					}
					return mainComponentVirtualService, false, nil
				})
				if err != nil {
					logger.Error(err, "failed to sync the main component virtual service")
					return ReconcileResult{}, fmt.Errorf("failed to sync the main component virtual service: %w", err)
				}
				if syncedMainComponentVirtualService != nil {
					mainComponentVirtualServiceAsResource, err := commoncontroller.NewResource(syncedMainComponentVirtualService,
						func() (bool, string) {
							return true, ""
						})
					if err != nil {
						return ReconcileResult{}, fmt.Errorf("failed to create the main component virtual service resource: %w", err)
					}
					resources = append(resources, mainComponentVirtualServiceAsResource)
				}
			}
		}
	}

	// Check resource readiness and overlay the Grove-specific Ready reason.
	// Extracted to keep this function's cyclomatic complexity within the
	// gocyclo limit.
	return r.checkGroveResourcesReadiness(ctx, dynamoDeployment, resources)
}

// checkGroveResourcesReadiness computes the readiness result for the synced
// Grove resources and overlays the Grove-specific Ready reason
// classification on a not-ready result. A transient Grove read error is
// returned (not folded into the result) so the reconcile retries and does not
// advance ObservedGeneration on a blip.
func (r *DynamoGraphDeploymentReconciler) checkGroveResourcesReadiness(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment, resources []Resource) (ReconcileResult, error) {
	result := r.checkResourcesReadiness(resources)
	if err := r.applyGroveReadyClassification(ctx, dynamoDeployment, &result); err != nil {
		return ReconcileResult{}, err
	}
	return result, nil
}

// applyGroveReadyClassification replaces the generic not-ready reason on result
// with a Grove-specific classification (insufficient_capacity / pods_not_ready /
// updating / mixed_not_ready_reasons / some_resources_are_not_ready) computed
// from Grove PodClique / PodCliqueScalingGroup status (REQ 1). It only overrides
// when the result is not successful; the ready path keeps checkResourcesReadiness's
// success result.
//
// A non-nil error is a transient (non-NotFound) Grove read failure: the caller
// should return it so the reconcile retries rather than publishing a possibly
// wrong not-ready diagnosis and advancing ObservedGeneration. NotFound is not an
// error and is classified as a legitimate not-ready state.
func (r *DynamoGraphDeploymentReconciler) applyGroveReadyClassification(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment, result *ReconcileResult) error {
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		return nil
	}
	classification, err := dynamo.ClassifyGroveReadiness(ctx, r.Client, dynamoDeployment)
	if err != nil {
		return err
	}
	if classification != "" {
		result.Reason = Reason(classification)
	}
	return nil
}

// isNewRestartRequest checks if the current spec.restart.id represents a new restart request
func isNewRestartRequest(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if dgd.Status.Restart == nil || dgd.Status.Restart.ObservedID == "" || dgd.Spec.Restart.ID == "" {
		return true
	}
	return dgd.Spec.Restart.ID != dgd.Status.Restart.ObservedID
}

// computeParallelRestartStatus handles parallel restart where all components restart together.
func (r *DynamoGraphDeploymentReconciler) computeParallelRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *nvidiacomv1beta1.RestartStatus {
	logger := log.FromContext(ctx)

	specID := dgd.Spec.Restart.ID

	var componentsToCheck []string
	if isNewRestartRequest(dgd) {
		logger.Info("New restart request detected, resetting to all components", "specID", specID)
		componentsToCheck = make([]string, 0, len(dgd.Spec.Components))
		for i := range dgd.Spec.Components {
			componentsToCheck = append(componentsToCheck, dgd.Spec.Components[i].ComponentName)
		}
		// Sort for deterministic output
		sort.Strings(componentsToCheck)

		// For a new restart request with components, immediately return Restarting phase without checking readiness.
		if len(componentsToCheck) > 0 {
			return &nvidiacomv1beta1.RestartStatus{
				ObservedID: specID,
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: componentsToCheck,
			}
		}
		// If no components, fall through to the empty check below.
	} else if dgd.Status.Restart != nil && len(dgd.Status.Restart.InProgress) > 0 {
		// Continuing existing restart: use current InProgress list
		componentsToCheck = dgd.Status.Restart.InProgress
	} else {
		// No in-progress list but same ID - use all components.
		componentsToCheck = make([]string, 0, len(dgd.Spec.Components))
		for i := range dgd.Spec.Components {
			componentsToCheck = append(componentsToCheck, dgd.Spec.Components[i].ComponentName)
		}
		// Sort for deterministic output
		sort.Strings(componentsToCheck)
	}

	if len(componentsToCheck) == 0 {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	updatedInProgress := r.getUpdatedInProgress(ctx, dgd, componentsToCheck)

	if len(updatedInProgress) == 0 {
		logger.Info("Restart completed for all components")
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	return &nvidiacomv1beta1.RestartStatus{
		ObservedID: specID,
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: updatedInProgress,
	}
}

// computeSequentialRestartStatus handles sequential restart where components restart one at a time.
func (r *DynamoGraphDeploymentReconciler) computeSequentialRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	order []string,
) *nvidiacomv1beta1.RestartStatus {
	logger := log.FromContext(ctx)

	specID := dgd.Spec.Restart.ID
	if len(order) == 0 {
		logger.Info("Sequential restart completed with no components", "specID", specID)
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	// Get the current component being restarted from previous status.
	var currentComponent string
	if isNewRestartRequest(dgd) {
		// New restart request: start fresh from the first component.
		logger.Info("New restart request detected, starting from first component", "specID", specID, "firstComponent", order[0])
		currentComponent = order[0]
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{currentComponent},
		}
	}

	if dgd.Status.Restart != nil && len(dgd.Status.Restart.InProgress) > 0 {
		currentComponent = dgd.Status.Restart.InProgress[0] // For sequential, there's only one.
	}

	// If no current component, we're starting fresh - use the first component.
	if currentComponent == "" {
		currentComponent = order[0]
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{currentComponent},
		}
	}

	// Check if the current component is fully updated.
	updatedInProgress := r.getUpdatedInProgress(ctx, dgd, []string{currentComponent})

	if len(updatedInProgress) > 0 {
		// Still restarting
		logger.Info("Component restart not completed", "component", currentComponent, "updatedInProgress", updatedInProgress)
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{currentComponent},
		}
	}

	// Current component is fully updated - it's done.
	logger.Info("Component restart completed", "component", currentComponent)

	// Find the next component.
	nextComponent, currentFound := getNextComponentInOrder(order, currentComponent)
	if !currentFound {
		logger.Info("Current restart component is no longer in order, restarting sequence from first component", "component", currentComponent, "firstComponent", order[0])
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}

	if nextComponent == "" {
		// No more components, restart is complete.
		logger.Info("Restart completed for all components")
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	// Move to the next component.
	logger.Info("Starting next component restart", "component", nextComponent)
	return &nvidiacomv1beta1.RestartStatus{
		ObservedID: specID,
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: []string{nextComponent},
	}
}

// getNextComponentInOrder returns the component after the current component.
// The boolean reports whether currentComponent was found in order.
func getNextComponentInOrder(order []string, currentComponent string) (string, bool) {
	for i, componentName := range order {
		if componentName != currentComponent {
			continue
		}
		if i+1 < len(order) {
			return order[i+1], true
		}
		return "", true
	}
	return "", false
}

func (r *DynamoGraphDeploymentReconciler) computeRestartStatus(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) *nvidiacomv1beta1.RestartStatus {
	// No restart requested
	if dgd.Spec.Restart == nil || dgd.Spec.Restart.ID == "" {
		// Preserve existing terminal status
		if dgd.Status.Restart != nil && (dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseCompleted || dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseFailed || dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseSuperseded) {
			return dgd.Status.Restart
		}
		return nil
	}

	// If restart was already processed (completed, failed, or superseded), return existing status
	if isRestartAlreadyProcessed(dgd) {
		return dgd.Status.Restart
	}

	// Supersede restart if a rolling update is in progress
	if r.isRollingUpdateInProgress(dgd) {
		r.Recorder.Eventf(dgd, corev1.EventTypeWarning, "RestartSuperseded",
			"Restart %s superseded by rolling update", dgd.Spec.Restart.ID)
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: dgd.Spec.Restart.ID,
			Phase:      nvidiacomv1beta1.RestartPhaseSuperseded,
		}
	}

	order := dynamo.GetRestartOrder(dgd)

	if dynamo.IsParallelRestart(dgd) {
		return r.computeParallelRestartStatus(ctx, dgd)
	}

	return r.computeSequentialRestartStatus(ctx, dgd, order)
}

// checkComponentFullyUpdated checks if a DynamoComponentDeployment is fully updated.
func (r *DynamoGraphDeploymentReconciler) checkComponentFullyUpdated(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, componentName string) (bool, string) {
	if r.currentWorkerHashes(dgd).empty() {
		resourceName := dynamo.GetDCDResourceName(dgd, componentName, "")
		return checkDCDReady(ctx, r.Client, resourceName, dgd.Namespace)
	}

	hashes, err := r.desiredWorkerHashes(dgd)
	if err != nil {
		return false, err.Error()
	}

	var lastReason string
	for _, hash := range r.activeWorkerHashCandidates(dgd, hashes) {
		resourceName := dynamo.GetDCDResourceName(dgd, componentName, hash)
		ready, reason := checkDCDReady(ctx, r.Client, resourceName, dgd.Namespace)
		if ready || reason != "resource not found" {
			return ready, reason
		}
		lastReason = reason
	}
	return false, lastReason
}

// checkDCDReady checks if a DynamoComponentDeployment has completed its restart.
// A DCD is considered fully updated when:
// 1. The DCD controller has processed the latest spec (observedGeneration >= generation)
// 2. The Available condition is set to True
func checkDCDReady(ctx context.Context, client client.Client, resourceName, namespace string) (bool, string) {
	logger := log.FromContext(ctx)
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, dcd)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("DynamoComponentDeployment not found", "resourceName", resourceName)
			return false, "resource not found"
		}
		logger.V(1).Info("Failed to get DynamoComponentDeployment", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err)
	}

	// Log the DCD status for debugging
	logger.Info("CheckDCDFullyUpdated",
		"resourceName", resourceName,
		"generation", dcd.Generation,
		"observedGeneration", dcd.Status.ObservedGeneration,
		"conditionCount", len(dcd.Status.Conditions))

	if dcd.Status.ObservedGeneration < dcd.Generation {
		logger.V(1).Info("DynamoComponentDeployment spec not yet processed",
			"resourceName", resourceName,
			"generation", dcd.Generation,
			"observedGeneration", dcd.Status.ObservedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", dcd.Generation, dcd.Status.ObservedGeneration)
	}

	// Check if the Available condition is True
	for _, condition := range dcd.Status.Conditions {
		if condition.Type == nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable {
			if condition.Status == metav1.ConditionTrue {
				return true, ""
			}
			logger.V(1).Info("DynamoComponentDeployment not available",
				"resourceName", resourceName,
				"status", condition.Status,
				"reason", condition.Reason,
				"message", condition.Message)
			return false, fmt.Sprintf("not available: %s", condition.Message)
		}
	}

	logger.V(1).Info("DynamoComponentDeployment missing Available condition", "resourceName", resourceName)
	return false, "Available condition not found"
}

// getUpdatedInProgressForComponent checks which components are still in progress for DCD pathway.
func (r *DynamoGraphDeploymentReconciler) getUpdatedInProgressForComponent(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
	logger := log.FromContext(ctx)

	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		isFullyUpdated, reason := r.checkComponentFullyUpdated(ctx, dgd, componentName)
		if !isFullyUpdated {
			logger.V(1).Info("component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}
	return updatedInProgress
}

func (r *DynamoGraphDeploymentReconciler) checkResourcesReadiness(resources []Resource) ReconcileResult {
	// Sort resources by name to ensure deterministic ordering
	sort.Slice(resources, func(i, j int) bool {
		return resources[i].GetName() < resources[j].GetName()
	})

	var notReadyReasons []string
	notReadyResources := []string{}
	componentStatuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus)
	for _, resource := range resources {
		ready, reason := resource.IsReady()

		resourceComponentStatuses := resource.GetComponentStatuses()
		for componentName, componentStatus := range resourceComponentStatuses {
			componentStatuses[componentName] = componentStatus
		}

		if !ready {
			notReadyResources = append(notReadyResources, resource.GetName())
			notReadyReasons = append(notReadyReasons, fmt.Sprintf("%s: %s", resource.GetName(), reason))
		}
	}

	if len(notReadyResources) == 0 {
		return ReconcileResult{
			State:           nvidiacomv1beta1.DGDStateSuccessful,
			Reason:          "all_resources_are_ready",
			Message:         Message("All resources are ready"),
			ComponentStatus: componentStatuses,
		}
	}
	return ReconcileResult{
		State:           nvidiacomv1beta1.DGDStatePending,
		Reason:          "some_resources_are_not_ready",
		Message:         Message(fmt.Sprintf("Resources not ready: %s", strings.Join(notReadyReasons, "; "))),
		ComponentStatus: componentStatuses,
	}
}

func applyCheckpointStartupReadiness(
	result ReconcileResult,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) ReconcileResult {
	if result.State == nvidiacomv1beta1.DGDStateFailed {
		return result
	}
	var waiting []string
	for componentName, info := range checkpointInfos {
		if info == nil ||
			!info.Enabled ||
			info.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint ||
			info.Ready {
			continue
		}
		if info.CheckpointName != "" {
			waiting = append(waiting, fmt.Sprintf("%s (%s)", componentName, info.CheckpointName))
		} else {
			waiting = append(waiting, componentName)
		}
	}
	if len(waiting) == 0 {
		return result
	}
	sort.Strings(waiting)
	result.State = nvidiacomv1beta1.DGDStatePending
	result.Reason = "waiting_for_checkpoint"
	result.Message = Message(fmt.Sprintf("Waiting for checkpoints: %s", strings.Join(waiting, ", ")))
	return result
}

func (r *DynamoGraphDeploymentReconciler) reconcileDynamoComponentsDeployments(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment, restartState *dynamo.RestartState, checkpointInfos map[string]*checkpoint.CheckpointInfo) (ReconcileResult, error) {
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dynamoDeployment)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	if rollingUpdateCtx.InProgress() {
		logger.Info("Rolling update in progress",
			"newWorkerHash", rollingUpdateCtx.NewWorkerHash,
			"oldWorkerComponentReplicas", rollingUpdateCtx.OldWorkerReplicaTargetsByComponent)
	}

	// Generate all DCDs (handles both normal and rolling update cases)
	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(
		dynamoDeployment, restartState, existingRestartAnnotations, rollingUpdateCtx,
	)
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		return ReconcileResult{}, fmt.Errorf("failed to generate the DynamoComponentsDeployments: %w", err)
	}

	// Sync all generated DCDs
	for key, dcd := range dynamoComponentsDeployments {
		if err := applyDCDCheckpointStartupPolicy(dcd, checkpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		logger.Info("Reconciling DynamoComponentDeployment", "key", key, "name", dcd.Name)
		if err := r.preserveExistingDCDBackendFramework(ctx, dcd); err != nil {
			logger.Error(err, "failed to preserve existing DynamoComponentDeployment backendFramework", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to sync the DynamoComponentDeployment: %w", err)
		}
		resources = append(resources, syncedDCD)
	}

	// During rolling update, scale old worker DCDs via direct patching.
	// This is done separately from DCD generation to avoid overwriting the old spec
	// with the new spec (which would trigger an unwanted rolling update on old workers).
	if rollingUpdateCtx.InProgress() {
		if err := r.scaleOldWorkerDCDs(ctx, dynamoDeployment, rollingUpdateCtx); err != nil {
			logger.Error(err, "failed to scale old worker DCDs")
			return ReconcileResult{}, fmt.Errorf("failed to scale old worker DCDs: %w", err)
		}
	}

	// Check resource readiness
	result := r.checkResourcesReadiness(resources)

	// During rolling updates, aggregate old worker component statuses into the result
	// so that Replicas, ReadyReplicas, etc. reflect the total across old and new DCDs.
	if rollingUpdateCtx.InProgress() {
		oldWorkerStatuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dynamoDeployment, rollingUpdateCtx)
		if err != nil {
			logger.Error(err, "failed to aggregate old worker component statuses")
			// Non-fatal: continue with partial status
		} else if len(oldWorkerStatuses) > 0 {
			mergeWorkerComponentStatuses(result.ComponentStatus, oldWorkerStatuses)
		}
	}

	return result, nil
}

func applyDCDCheckpointStartupPolicy(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	if dcd == nil || checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	// DGD-managed automatic checkpoints deliberately bypass identity lookup and
	// are resolved by the exact DynamoCheckpoint CR created for this
	// DGD/component generation. Propagate that resolved reference into the child
	// DCD so the DCD controller does not independently fall back to legacy
	// identity-based reuse logic.
	if checkpointInfo.Exists && checkpointInfo.CheckpointName != "" {
		if dcd.Spec.Experimental == nil {
			dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
		}
		if dcd.Spec.Experimental.Checkpoint == nil {
			dcd.Spec.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{}
		}
		checkpointName := checkpointInfo.CheckpointName
		dcd.Spec.Experimental.Checkpoint.Enabled = true
		dcd.Spec.Experimental.Checkpoint.CheckpointRef = &checkpointName
		dcd.Spec.Experimental.Checkpoint.Identity = nil
		dcd.Spec.Experimental.Checkpoint.Job = nil
		startupPolicy := checkpointInfo.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}
		dcd.Spec.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicy(startupPolicy)
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		dcd.Spec.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy == "" ||
		checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		labels := dynamo.GetPodTemplateLabels(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
		if labels == nil {
			if dcd.Spec.PodTemplate == nil {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
			}
			if dcd.Spec.PodTemplate.Labels == nil {
				dcd.Spec.PodTemplate.Labels = map[string]string{}
			}
			labels = dcd.Spec.PodTemplate.Labels
		}
		annotations := dynamo.GetPodTemplateAnnotations(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
		if annotations == nil {
			if dcd.Spec.PodTemplate == nil {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
			}
			if dcd.Spec.PodTemplate.Annotations == nil {
				dcd.Spec.PodTemplate.Annotations = map[string]string{}
			}
			annotations = dcd.Spec.PodTemplate.Annotations
		}
		return checkpoint.ApplyRestoreCandidateMetadata(labels, annotations, checkpointInfo)
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) preserveExistingDCDBackendFramework(ctx context.Context, desired *nvidiacomv1beta1.DynamoComponentDeployment) error {
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get existing DynamoComponentDeployment %s/%s: %w", desired.Namespace, desired.Name, err)
	}

	// backendFramework is immutable on DCDs. Older generated children may have
	// an empty value, so preserve the stored value on update while allowing new
	// children to be created with the inferred backend.
	desired.Spec.BackendFramework = existing.Spec.BackendFramework
	return nil
}

func (r *DynamoGraphDeploymentReconciler) getExistingRestartAnnotationsDCD(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) (map[string]string, error) {
	logger := log.FromContext(ctx)

	hashes, err := r.desiredWorkerHashes(dgd)
	if err != nil {
		return nil, err
	}
	workerHashes := r.activeWorkerHashCandidates(dgd, hashes)

	restartAnnotations := make(map[string]string)
	for i := range dgd.Spec.Components {
		componentName := dgd.Spec.Components[i].ComponentName
		existingDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
		for _, workerHash := range workerHashes {
			dcdName := dynamo.GetDCDResourceName(dgd, componentName, workerHash)
			err := r.Get(ctx, types.NamespacedName{Name: dcdName, Namespace: dgd.Namespace}, existingDCD)

			if err == nil {
				break
			}
			if !errors.IsNotFound(err) {
				return nil, fmt.Errorf("failed to get DynamoComponentDeployment: %w", err)
			}
			logger.Info("DynamoComponentDeployment not found", "dcdName", dcdName)
		}
		if existingDCD.Name == "" {
			continue
		}
		if restartAt := dynamo.GetPodTemplateAnnotations(&existingDCD.Spec.DynamoComponentDeploymentSharedSpec)[consts.RestartAnnotation]; restartAt != "" {
			restartAnnotations[componentName] = restartAt
		}
	}
	return restartAnnotations, nil
}

func (r *DynamoGraphDeploymentReconciler) reconcileK8sDiscoveryResources(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)

	if !commoncontroller.IsK8sDiscoveryEnabled(r.Config.Discovery.Backend, dynamoDeployment.Annotations) {
		logger.Info("K8s discovery is not enabled")
		return nil
	}
	logger.Info("K8s discovery is enabled")

	serviceAccount := discovery.GetK8sDiscoveryServiceAccount(dynamoDeployment.Name, dynamoDeployment.Namespace)
	_, _, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*corev1.ServiceAccount, bool, error) {
		return serviceAccount, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the k8s discovery service account")
		return fmt.Errorf("failed to sync the k8s discovery service account: %w", err)
	}

	role := discovery.GetK8sDiscoveryRole(dynamoDeployment.Name, dynamoDeployment.Namespace)
	_, _, err = commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*rbacv1.Role, bool, error) {
		return role, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the k8s discovery role")
		return fmt.Errorf("failed to sync the k8s discovery role: %w", err)
	}

	roleBinding := discovery.GetK8sDiscoveryRoleBinding(dynamoDeployment.Name, dynamoDeployment.Namespace)
	_, _, err = commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*rbacv1.RoleBinding, bool, error) {
		return roleBinding, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the k8s discovery role binding")
		return fmt.Errorf("failed to sync the k8s discovery role binding: %w", err)
	}

	return nil

}

// reconcilePVC reconciles a single top-level PVC preserved from a converted v1alpha1 DGD.
func (r *DynamoGraphDeploymentReconciler) reconcilePVC(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment, pvcName string, pvcConfig nvidiacomv1alpha1.PVC) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)

	pvc := &corev1.PersistentVolumeClaim{}
	pvcNamespacedName := types.NamespacedName{Name: pvcName, Namespace: dynamoDeployment.Namespace}
	err := r.Get(ctx, pvcNamespacedName, pvc)
	if err != nil {
		if !errors.IsNotFound(err) {
			return nil, fmt.Errorf("unable to retrieve legacy top-level PVC %q: %w", pvcName, err)
		}
		if pvcConfig.Create == nil || !*pvcConfig.Create {
			return nil, fmt.Errorf("legacy top-level PVC %q does not exist and create is not enabled: %w", pvcName, err)
		}

		pvc = constructPVC(dynamoDeployment, pvcConfig)
		if err := controllerutil.SetControllerReference(dynamoDeployment, pvc, r.Client.Scheme()); err != nil {
			return nil, fmt.Errorf("failed to set controller reference for legacy top-level PVC %q: %w", pvcName, err)
		}

		if err := r.Create(ctx, pvc); err != nil {
			return nil, fmt.Errorf("failed to create legacy top-level PVC %q: %w", pvcName, err)
		}
		logger.Info("Legacy top-level PVC created", "pvcName", pvcName, "namespace", dynamoDeployment.Namespace)
	}

	return pvc, nil
}

// reconcilePVCs keeps v1alpha1 DGDs compatible after conversion. Native v1beta1
// DGDs have no top-level PVC declarations, so this is a no-op unless the
// conversion webhook preserved legacy spec.pvcs in the alpha annotation payload.
func (r *DynamoGraphDeploymentReconciler) reconcilePVCs(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)
	pvcs := dynamo.GetDGDPreservedAlphaPVCs(dynamoDeployment)
	if len(pvcs) == 0 {
		return nil
	}

	for _, pvcConfig := range pvcs {
		if pvcConfig.Name == nil || *pvcConfig.Name == "" {
			logger.Error(nil, "Legacy top-level PVC not reconcilable: name is required", "pvcConfig", pvcConfig)
			continue
		}

		pvcName := *pvcConfig.Name
		logger.Info("Reconciling legacy top-level PVC", "pvcName", pvcName, "namespace", dynamoDeployment.Namespace)

		if _, err := r.reconcilePVC(ctx, dynamoDeployment, pvcName, pvcConfig); err != nil {
			return err
		}
	}

	return nil
}

// reconcileCheckpoints reconciles checkpoint state for checkpoint-enabled components.
// Without checkpointRef, it creates DGD-managed DynamoCheckpoint CRs when they
// do not exist. With checkpointRef, it resolves the referenced checkpoint only.
// Returns per-component checkpoint status and resolved checkpoint info.
func (r *DynamoGraphDeploymentReconciler) reconcileCheckpoints(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[string]nvidiacomv1beta1.ComponentCheckpointStatus, map[string]*checkpoint.CheckpointInfo, error) {
	logger := log.FromContext(ctx)
	checkpointStatuses := make(map[string]nvidiacomv1beta1.ComponentCheckpointStatus)
	checkpointInfos := make(map[string]*checkpoint.CheckpointInfo)
	storageEnsured := false

	for i := range dynamoDeployment.Spec.Components {
		component := &dynamoDeployment.Spec.Components[i]
		componentName := component.ComponentName
		checkpointConfig := dynamo.GetCheckpoint(component)
		if checkpointConfig == nil {
			continue
		}
		if !r.RuntimeConfig.Gate.Enabled(features.Checkpoint) {
			return nil, nil, fmt.Errorf("component %s: checkpoint functionality is disabled in the operator configuration", componentName)
		}

		logger.Info("Reconciling checkpoint for component", "component", componentName)

		if !storageEnsured {
			if err := checkpoint.EnsureStoragePVC(ctx, r.Client, dynamoDeployment.Namespace, r.Config.Checkpoint.Storage); err != nil {
				logger.Error(err, "Failed to ensure checkpoint storage PVC", "component", componentName)
				return nil, nil, fmt.Errorf("failed to ensure checkpoint storage PVC for component %s: %w", componentName, err)
			}
			storageEnsured = true
		}

		alphaCheckpointConfig := dynamo.ToAlphaCheckpointConfig(checkpointConfig)
		startupPolicy := alphaCheckpointConfig.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}

		var info *checkpoint.CheckpointInfo
		var err error
		hasCheckpointRef := checkpointConfig.CheckpointRef != nil && *checkpointConfig.CheckpointRef != ""
		if !hasCheckpointRef {
			workerHash, err := r.checkpointWorkerHashForComponent(dynamoDeployment, componentName)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to compute checkpoint worker hash for component %s: %w", componentName, err)
			}
			checkpointID := checkpoint.DGDCheckpointID(
				dynamoDeployment.Namespace,
				dynamoDeployment.Name,
				string(dynamoDeployment.UID),
				componentName,
				workerHash,
			)
			checkpointName := fmt.Sprintf("checkpoint-%s", checkpointID)
			refConfig := *alphaCheckpointConfig.DeepCopy()
			refConfig.CheckpointRef = &checkpointName
			info, err = checkpoint.ResolveCheckpointForService(ctx, r.Client, dynamoDeployment.Namespace, &refConfig, r.RuntimeConfig.Gate)
			if errors.IsNotFound(err) {
				info = nil
				err = nil
			}
			if err == nil && info == nil {
				info = &checkpoint.CheckpointInfo{
					Enabled:       true,
					StartupPolicy: startupPolicy,
				}
			}
		} else {
			// Resolve checkpoint for this component.
			info, err = checkpoint.ResolveCheckpointForService(ctx, r.Client, dynamoDeployment.Namespace, alphaCheckpointConfig, r.RuntimeConfig.Gate)
		}
		if err != nil {
			logger.Error(err, "Failed to resolve checkpoint for component", "component", componentName)
			return nil, nil, fmt.Errorf("failed to resolve checkpoint for component %s: %w", componentName, err)
		}
		info.StartupPolicy = startupPolicy
		if len(info.RestoreTargetContainers) == 0 && alphaCheckpointConfig.TargetContainerName != "" {
			info.RestoreTargetContainers = []string{alphaCheckpointConfig.TargetContainerName}
		}
		if dynamo.IsIntraPodFailoverEnabled(component) {
			info.RestoreTargetContainers = dynamo.IntraPodFailoverEngineContainerNames()
		}
		if err := gms.OverlayClients(&info.GPUMemoryService, info.CheckpointName, info.Exists, dynamo.GetGPUMemoryService(component)); err != nil {
			return nil, nil, fmt.Errorf("failed to apply checkpoint gpuMemoryService config for component %s: %w", componentName, err)
		}

		// Store checkpoint info for later use in pod spec generation
		checkpointInfos[componentName] = info

		// Without checkpointRef, the DGD creates a scoped automatic checkpoint for
		// this component generation without looking for reusable checkpoints.
		if !hasCheckpointRef {
			if !info.Exists {
				logger.Info("Creating DGD-managed DynamoCheckpoint CR", "component", componentName)
			}

			ckpt, err := r.createCheckpointCR(ctx, dynamoDeployment, componentName, component)
			if err != nil {
				logger.Error(err, "Failed to create DynamoCheckpoint CR", "component", componentName)
				return nil, nil, fmt.Errorf("failed to create checkpoint for component %s: %w", componentName, err)
			}
			info.Exists = true
			info.CheckpointName = ckpt.Name
			info.Hash, err = checkpoint.CheckpointID(ckpt)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to resolve checkpoint ID for component %s: %w", componentName, err)
			}
			if info.GPUMemoryService == nil {
				info.GPUMemoryService = ckpt.Spec.GPUMemoryService
			}
			info.Ready = ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady
		}

		checkpointStatuses[componentName] = nvidiacomv1beta1.ComponentCheckpointStatus{
			CheckpointName: info.CheckpointName,
			CheckpointID:   info.Hash,
			IdentityHash:   info.Hash,
			Ready:          info.Ready,
		}
	}

	return checkpointStatuses, checkpointInfos, nil
}

// createCheckpointCR creates a DGD-managed DynamoCheckpoint CR for a component.
//
//nolint:gocyclo
func (r *DynamoGraphDeploymentReconciler) createCheckpointCR(
	ctx context.Context,
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	checkpointConfig := dynamo.GetCheckpoint(component)
	if checkpointConfig == nil {
		return nil, fmt.Errorf("checkpoint config is required")
	}

	workerHash, err := r.checkpointWorkerHashForComponent(dynamoDeployment, componentName)
	if err != nil {
		return nil, fmt.Errorf("failed to compute checkpoint worker hash for component %s: %w", componentName, err)
	}
	checkpointID := checkpoint.DGDCheckpointID(
		dynamoDeployment.Namespace,
		dynamoDeployment.Name,
		string(dynamoDeployment.UID),
		componentName,
		workerHash,
	)

	backendFramework, err := dynamo.BackendFrameworkForComponent(component, dynamoDeployment)
	if err != nil {
		return nil, fmt.Errorf("failed to determine backend framework for component %s: %w", componentName, err)
	}
	if (backendFramework == "" || backendFramework == dynamo.BackendFrameworkNoop) &&
		checkpointConfig.Identity != nil &&
		checkpointConfig.Identity.BackendFramework != "" {
		backendFramework, err = dynamo.ParseBackendFramework(checkpointConfig.Identity.BackendFramework)
		if err != nil {
			return nil, fmt.Errorf("invalid legacy checkpoint identity backend framework for component %s: %w", componentName, err)
		}
	}
	if backendFramework == "" || backendFramework == dynamo.BackendFrameworkNoop {
		return nil, fmt.Errorf("checkpoint backend framework for component %s could not be determined; set spec.backendFramework or use a recognizable worker command", componentName)
	}

	podTemplate, err := r.buildCheckpointJobPodTemplate(
		dynamoDeployment,
		component,
		componentName,
		backendFramework,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to build checkpoint job pod template: %w", err)
	}
	if commoncontroller.IsK8sDiscoveryEnabled(r.Config.Discovery.Backend, dynamoDeployment.Annotations) &&
		podTemplate.Spec.ServiceAccountName == "" {
		podTemplate.Spec.ServiceAccountName = discovery.GetK8sDiscoveryServiceAccountName(dynamoDeployment.Name)
	}
	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	podTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName] = dynamoDeployment.Name
	podTemplate.Labels[consts.KubeLabelDynamoComponent] = componentName
	if workerHash != "" {
		podTemplate.Labels[consts.KubeLabelDynamoWorkerHash] = workerHash
	}

	targetContainerName := consts.MainContainerName
	if checkpointConfig.TargetContainerName != "" {
		targetContainerName = checkpointConfig.TargetContainerName
	}
	targetContainer, err := findPodTemplateContainer(&podTemplate, targetContainerName)
	if err != nil {
		return nil, err
	}
	var gmsSpec *nvidiacomv1alpha1.GPUMemoryServiceSpec
	if converted := gms.ToAlphaSpec(dynamo.GetGPUMemoryService(component)); converted != nil {
		gmsSpec = converted.DeepCopy()
		gmsSpec.ExtraClientContainers = nil
		if checkpointConfig.Job != nil {
			gmsSpec.ExtraClientContainers = append([]string(nil), checkpointConfig.Job.GMSClientContainers...)
		}
	}
	var checkpointGMSClaimTemplateName string
	if gmsSpec != nil && gmsSpec.Enabled {
		if err := checkpoint.ValidateGMSSnapshotGate("spec.gpuMemoryService", true, gmsSpec, r.RuntimeConfig.Gate); err != nil {
			return nil, err
		}
		checkpointGMSClaimTemplateName = checkpointGMSResourceClaimTemplateName(checkpointID)
		checkpointGMSGPUCount, err := dra.ExtractGPUCountFromResourceRequirements(targetContainer.Resources)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU resource requirements for GMS checkpoint %s/%s: %w", dynamoDeployment.Name, componentName, err)
		}
		checkpointGMSDeviceClassName := gmsSpec.DeviceClassName
		if checkpointGMSDeviceClassName == "" {
			checkpointGMSDeviceClassName = dra.DefaultDeviceClassName
		}
		if err := r.syncCheckpointGMSResourceClaimTemplate(
			ctx,
			dynamoDeployment.Namespace,
			checkpointGMSClaimTemplateName,
			checkpointGMSGPUCount,
			checkpointGMSDeviceClassName,
		); err != nil {
			return nil, err
		}
		if err := prepareCheckpointGMSPodTemplate(&podTemplate, targetContainerName, checkpointID, gmsSpec); err != nil {
			return nil, err
		}
	}
	deletionPolicy := nvidiacomv1alpha1.CheckpointDeletionPolicy(checkpointConfig.DeletionPolicy)
	if deletionPolicy == "" {
		deletionPolicy = nvidiacomv1alpha1.CheckpointDeletionPolicyDelete
	}

	ckpt, err := checkpoint.CreateOrGetAutoCheckpoint(
		ctx,
		r.Client,
		dynamoDeployment.Namespace,
		checkpointID,
		nvidiacomv1alpha1.DynamoCheckpointIdentity{
			Model:            fmt.Sprintf("%s/%s", dynamoDeployment.Namespace, dynamoDeployment.Name),
			BackendFramework: string(backendFramework),
			ExtraParameters: map[string]string{
				"dgdUID":       string(dynamoDeployment.UID),
				"component":    componentName,
				"checkpointID": checkpointID,
			},
		},
		podTemplate,
		targetContainerName,
		deletionPolicy,
		gmsSpec,
		dynamoDeployment,
		r.RuntimeConfig.Gate,
	)
	if err != nil {
		return nil, err
	}
	if gmsSpec != nil && gmsSpec.Enabled {
		if err := r.adoptCheckpointGMSResourceClaimTemplate(ctx, ckpt, checkpointGMSClaimTemplateName); err != nil {
			return nil, err
		}
	}
	return ckpt, nil
}

func checkpointGMSResourceClaimTemplateName(checkpointID string) string {
	return dra.ResourceClaimTemplateName("checkpoint-"+checkpointID, "worker")
}

func (r *DynamoGraphDeploymentReconciler) syncCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	namespace string,
	claimTemplateName string,
	gpuCount int,
	deviceClassName string,
) error {
	_, _, err := commoncontroller.SyncResource(ctx, r, nil, func(ctx context.Context) (*resourcev1.ResourceClaimTemplate, bool, error) {
		return dra.GenerateResourceClaimTemplate(ctx, r.Client, claimTemplateName, namespace, gpuCount, deviceClassName)
	})
	if err != nil {
		return fmt.Errorf("failed to sync checkpoint GMS ResourceClaimTemplate %s/%s: %w", namespace, claimTemplateName, err)
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) adoptCheckpointGMSResourceClaimTemplate(
	ctx context.Context,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	claimTemplateName string,
) error {
	template := &resourcev1.ResourceClaimTemplate{}
	key := types.NamespacedName{Name: claimTemplateName, Namespace: ckpt.Namespace}
	if err := r.Get(ctx, key, template); err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return fmt.Errorf("failed to get checkpoint GMS ResourceClaimTemplate %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	if metav1.IsControlledBy(template, ckpt) {
		return nil
	}

	ownerReferences := template.GetOwnerReferences()
	filtered := make([]metav1.OwnerReference, 0, len(ownerReferences))
	for _, ref := range ownerReferences {
		if ref.Controller != nil && *ref.Controller {
			continue
		}
		filtered = append(filtered, ref)
	}
	template.SetOwnerReferences(filtered)
	if err := controllerutil.SetControllerReference(ckpt, template, r.Scheme()); err != nil {
		return fmt.Errorf("failed to set DynamoCheckpoint owner on GMS ResourceClaimTemplate %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	if err := r.Update(ctx, template); err != nil {
		return fmt.Errorf("failed to update checkpoint GMS ResourceClaimTemplate owner %s/%s: %w", ckpt.Namespace, claimTemplateName, err)
	}
	return nil
}

func prepareCheckpointGMSPodTemplate(
	podTemplate *corev1.PodTemplateSpec,
	targetContainerName string,
	checkpointID string,
	gmsSpec *nvidiacomv1alpha1.GPUMemoryServiceSpec,
) error {
	switch gmsSpec.Mode {
	case "", nvidiacomv1alpha1.GMSModeIntraPod:
	case nvidiacomv1alpha1.GMSModeInterPod:
		return fmt.Errorf("gpuMemoryService checkpoint jobs for mode %q are not implemented", gmsSpec.Mode)
	default:
		return fmt.Errorf("gpuMemoryService checkpoint job has unsupported mode %q", gmsSpec.Mode)
	}

	targetContainer, err := findPodTemplateContainer(podTemplate, targetContainerName)
	if err != nil {
		return err
	}
	ensureCheckpointGMSPodClaim(&podTemplate.Spec, checkpointGMSResourceClaimTemplateName(checkpointID))
	checkpoint.EnsureIntraPodGPUMemoryService(
		&podTemplate.Spec,
		[]*corev1.Container{targetContainer},
		gmsSpec.ExtraClientContainers,
	)
	return nil
}

func ensureCheckpointGMSPodClaim(podSpec *corev1.PodSpec, claimTemplateName string) {
	foundToleration := false
	for i := range podSpec.Tolerations {
		toleration := podSpec.Tolerations[i]
		if toleration.Key == consts.KubeResourceGPUNvidia && toleration.Effect == corev1.TaintEffectNoSchedule {
			foundToleration = true
			break
		}
	}
	if !foundToleration {
		podSpec.Tolerations = append(podSpec.Tolerations, corev1.Toleration{
			Key:      consts.KubeResourceGPUNvidia,
			Operator: corev1.TolerationOpExists,
			Effect:   corev1.TaintEffectNoSchedule,
		})
	}

	podClaim := corev1.PodResourceClaim{
		Name:                      dra.ClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	}
	for i := range podSpec.ResourceClaims {
		if podSpec.ResourceClaims[i].Name == dra.ClaimName {
			podSpec.ResourceClaims[i] = podClaim
			return
		}
	}
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, podClaim)
}

func (r *DynamoGraphDeploymentReconciler) deleteAutoCheckpointsForDGD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := r.List(
		ctx,
		checkpoints,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		return err
	}
	for i := range checkpoints.Items {
		ckpt := &checkpoints.Items[i]
		if ckpt.Annotations == nil || ckpt.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
			continue
		}
		if ckpt.Annotations[consts.CheckpointDeletionPolicyAnnotation] == string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain) {
			if err := r.detachRetainedAutoCheckpoint(ctx, ckpt); err != nil {
				return err
			}
			continue
		}
		if err := r.Delete(ctx, ckpt); err != nil && !errors.IsNotFound(err) {
			return fmt.Errorf("delete auto checkpoint %s/%s: %w", ckpt.Namespace, ckpt.Name, err)
		}
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) detachRetainedAutoCheckpoint(
	ctx context.Context,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
) error {
	updated := ckpt.DeepCopy()
	updated.OwnerReferences = nil
	if updated.Labels != nil {
		delete(updated.Labels, consts.KubeLabelDynamoGraphDeploymentName)
	}
	if equality.Semantic.DeepEqual(ckpt.OwnerReferences, updated.OwnerReferences) &&
		equality.Semantic.DeepEqual(ckpt.Labels, updated.Labels) {
		return nil
	}
	if err := r.Patch(ctx, updated, client.MergeFrom(ckpt)); err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("detach retained auto checkpoint %s/%s: %w", ckpt.Namespace, ckpt.Name, err)
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) checkpointWorkerHashForComponent(dgd *nvidiacomv1beta1.DynamoGraphDeployment, componentName string) (string, error) {
	if dgd == nil {
		return "", nil
	}
	component := dgd.GetComponentByName(componentName)
	if component == nil || !dynamo.IsWorkerComponent(string(component.ComponentType)) {
		return "", nil
	}
	if r == nil {
		return "", nil
	}
	desired, err := r.desiredWorkerHashes(dgd)
	if err != nil {
		return "", err
	}
	return r.activeWorkerHashForDCDGeneration(dgd, desired), nil
}

// buildCheckpointJobPodTemplate builds a checkpoint job template from the same
// component defaults used for regular DGD pods, then keeps only the target
// container plus any checkpoint-job sidecars supplied by the user.
//
//nolint:gocyclo
func (r *DynamoGraphDeploymentReconciler) buildCheckpointJobPodTemplate(
	dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	componentName string,
	backendFramework dynamo.BackendFramework,
) (corev1.PodTemplateSpec, error) {
	targetContainerName := consts.MainContainerName
	if checkpointConfig := dynamo.GetCheckpoint(component); checkpointConfig != nil && checkpointConfig.TargetContainerName != "" {
		targetContainerName = checkpointConfig.TargetContainerName
	}

	// Create a copy of the component spec stripped of features that buildCheckpointJob
	// or the checkpoint controller handle independently. GenerateBasePodSpec would
	// otherwise apply DGD-specific transforms (DRA claims, GMS server sidecar,
	// frontend sidecar, failover transforms) that conflict with the checkpoint path's
	// own setup.
	componentForJob := component.DeepCopy()
	if componentForJob.Experimental != nil {
		componentForJob.Experimental.Checkpoint = nil
		componentForJob.Experimental.GPUMemoryService = nil
		componentForJob.Experimental.Failover = nil
		if componentForJob.Experimental.GPUMemoryService == nil &&
			componentForJob.Experimental.Failover == nil &&
			componentForJob.Experimental.Checkpoint == nil {
			componentForJob.Experimental = nil
		}
	}
	componentForJob.FrontendSidecar = nil

	// Use the normal DGD path so graph-level defaults such as spec.env,
	// annotations, labels, and pod-template metadata are applied consistently.
	podSpec, err := dynamo.GeneratePodSpecForComponent(
		componentForJob,
		backendFramework,
		r.DockerSecretRetriever,
		dynamoDeployment,
		dynamo.RoleCheckpoint, // Use checkpoint role
		1,                     // Single node for checkpoint job
		r.Config,
		consts.MultinodeDeploymentTypeGrove, // Use Grove (single-node backends return early)
		componentName,
		nil, // No checkpoint info for checkpoint creation jobs
		nil, // Use default deployer
	)
	if err != nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("failed to generate base pod spec: %w", err)
	}

	if podSpec == nil {
		return corev1.PodTemplateSpec{}, fmt.Errorf("checkpoint job pod spec is nil")
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == targetContainerName {
			podSpec.Containers = []corev1.Container{*podSpec.Containers[i].DeepCopy()}
			break
		}
	}
	if len(podSpec.Containers) != 1 || podSpec.Containers[0].Name != targetContainerName {
		return corev1.PodTemplateSpec{}, fmt.Errorf("checkpoint target container %q not found", targetContainerName)
	}

	// Override RestartPolicy for job (must be Never or OnFailure)
	podSpec.RestartPolicy = corev1.RestartPolicyNever

	// Seed the checkpoint job pod-template metadata from the component's own
	// PodTemplate.ObjectMeta so workload-level labels/annotations (e.g. Istio
	// sidecar opt-out or policy annotations) are not silently dropped on the
	// auto-created checkpoint job. GeneratePodSpecForComponent only returns the
	// PodSpec, so the template metadata must be carried over explicitly here.
	// Precedence: component pod-template metadata < controller-managed labels <
	// explicit checkpoint.job.podTemplate overrides (applied below).
	podLabels := map[string]string{}
	podAnnotations := map[string]string{}
	if component.PodTemplate != nil {
		for k, v := range component.PodTemplate.Labels {
			podLabels[k] = v
		}
		for k, v := range component.PodTemplate.Annotations {
			podAnnotations[k] = v
		}
	}
	podLabels[consts.KubeLabelDynamoComponent] = componentName

	podTemplate := corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels:      podLabels,
			Annotations: podAnnotations,
		},
		Spec: *podSpec,
	}
	if checkpointConfig := dynamo.GetCheckpoint(component); checkpointConfig != nil && checkpointConfig.Job != nil {
		if overrides := checkpointConfig.Job.PodTemplate; overrides != nil {
			if len(overrides.Labels) > 0 {
				if podTemplate.Labels == nil {
					podTemplate.Labels = make(map[string]string, len(overrides.Labels))
				}
				for k, v := range overrides.Labels {
					podTemplate.Labels[k] = v
				}
			}
			if len(overrides.Annotations) > 0 {
				if podTemplate.Annotations == nil {
					podTemplate.Annotations = make(map[string]string, len(overrides.Annotations))
				}
				for k, v := range overrides.Annotations {
					podTemplate.Annotations[k] = v
				}
			}

			overlay := overrides.Spec.DeepCopy()
			containers := overlay.Containers
			initContainers := overlay.InitContainers
			volumes := overlay.Volumes
			overlay.Containers = nil
			overlay.InitContainers = nil
			overlay.Volumes = nil
			if err := mergo.Merge(&podTemplate.Spec, *overlay, mergo.WithOverride); err != nil {
				return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge checkpoint job pod spec: %w", err)
			}

			podTemplate.Spec.Volumes = mergeNamedSlice(podTemplate.Spec.Volumes, volumes, func(v corev1.Volume) string { return v.Name })
			podTemplate.Spec.InitContainers = mergeNamedSlice(podTemplate.Spec.InitContainers, initContainers, func(c corev1.Container) string { return c.Name })
			for _, override := range containers {
				if override.Name == "" {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, override)
					continue
				}
				var existing *corev1.Container
				for i := range podTemplate.Spec.Containers {
					if podTemplate.Spec.Containers[i].Name == override.Name {
						existing = &podTemplate.Spec.Containers[i]
						break
					}
				}
				if existing == nil {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, override)
					continue
				}

				baseEnv := existing.Env
				user := override.DeepCopy()
				if err := mergo.Merge(existing, *user, mergo.WithOverride); err != nil {
					return corev1.PodTemplateSpec{}, fmt.Errorf("failed to merge checkpoint job container %q: %w", override.Name, err)
				}
				existing.Env = dynamo.MergeEnvs(baseEnv, user.Env)
				if user.LivenessProbe != nil {
					existing.LivenessProbe = user.LivenessProbe.DeepCopy()
				}
				if user.ReadinessProbe != nil {
					existing.ReadinessProbe = user.ReadinessProbe.DeepCopy()
				}
				if user.StartupProbe != nil {
					existing.StartupProbe = user.StartupProbe.DeepCopy()
				}
			}
		}
	}
	return podTemplate, nil
}

// reconcileScalingAdapters ensures a DynamoGraphDeploymentScalingAdapter exists for each component in the DGD
// that has scaling adapter explicitly enabled. Components without scalingAdapter.enabled=true will not have a DGDSA.
// This enables pluggable autoscaling via HPA, KEDA, or Planner.
func (r *DynamoGraphDeploymentReconciler) reconcileScalingAdapters(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)

	// Process each component - SyncResource handles create, update, and delete via toDelete flag.
	for i := range dynamoDeployment.Spec.Components {
		component := &dynamoDeployment.Spec.Components[i]
		componentName := component.ComponentName
		// Check if scaling adapter is enabled for this component (disabled by default).
		scalingAdapterEnabled := component.ScalingAdapter != nil

		// Get current replicas (default to 1 if not set)
		currentReplicas := int32(1)
		if component.Replicas != nil {
			currentReplicas = *component.Replicas
		}

		// Use SyncResource to handle creation/updates/deletion
		// When toDelete=true, SyncResource will delete the existing resource if it exists
		_, _, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter, bool, error) {
			adapterName := generateAdapterName(dynamoDeployment.Name, componentName)
			adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      adapterName,
					Namespace: dynamoDeployment.Namespace,
					Labels: map[string]string{
						consts.KubeLabelDynamoGraphDeploymentName: dynamoDeployment.Name,
						consts.KubeLabelDynamoComponent:           componentName,
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
					Replicas: currentReplicas,
					DGDRef: nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
						Name:        dynamoDeployment.Name,
						ServiceName: componentName,
					},
				},
			}
			// Return toDelete=true if scaling adapter is not enabled
			return adapter, !scalingAdapterEnabled, nil
		})

		if err != nil {
			logger.Error(err, "Failed to sync DynamoGraphDeploymentScalingAdapter", "component", componentName)
			return err
		}
	}

	// Clean up adapters for components that were removed from DGD entirely.
	adapterList := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	if err := r.List(ctx, adapterList,
		client.InNamespace(dynamoDeployment.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dynamoDeployment.Name},
	); err != nil {
		logger.Error(err, "Failed to list DynamoGraphDeploymentScalingAdapters")
		return err
	}

	for i := range adapterList.Items {
		adapter := &adapterList.Items[i]
		componentName := adapter.Spec.DGDRef.ServiceName

		// Delete adapter if component no longer exists in DGD.
		if dynamoDeployment.GetComponentByName(componentName) == nil {
			logger.Info("Deleting orphaned DynamoGraphDeploymentScalingAdapter", "adapter", adapter.Name, "component", componentName)
			if err := r.Delete(ctx, adapter); err != nil && !errors.IsNotFound(err) {
				logger.Error(err, "Failed to delete orphaned adapter", "adapter", adapter.Name)
				return err
			}
			r.Recorder.Eventf(dynamoDeployment, corev1.EventTypeNormal, "AdapterDeleted",
				"Deleted orphaned scaling adapter %s for removed component %s", adapter.Name, componentName)
		}
	}

	return nil
}

// generateAdapterName creates a consistent name for a DynamoGraphDeploymentScalingAdapter.
// Component names are lowercased to comply with Kubernetes DNS subdomain naming requirements.
func generateAdapterName(dgdName, componentName string) string {
	return fmt.Sprintf("%s-%s", dgdName, strings.ToLower(componentName))
}

// hasEPPService checks if the DGD has an EPP service defined
// reconcileEPPResources reconciles all EPP-related resources (ConfigMaps, Services, InferencePools)
func (r *DynamoGraphDeploymentReconciler) reconcileEPPResources(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	logger := log.FromContext(ctx)

	componentName, eppService, hasEPP := dgd.GetEPPComponent()
	if !hasEPP {
		logger.V(1).Info("No EPP service defined, skipping EPP resource reconciliation")
		return nil
	}

	logger.Info("Reconciling EPP resources", "componentName", componentName)

	// 1. Reconcile EPP ConfigMap (if needed - not needed when ConfigMapRef is used)
	if eppService.EPPConfig == nil || eppService.EPPConfig.ConfigMapRef == nil {
		configMap, err := epp.GenerateConfigMap(ctx, dgd, componentName, eppService.EPPConfig)
		if err != nil {
			logger.Error(err, "Failed to generate EPP ConfigMap")
			return fmt.Errorf("failed to generate EPP ConfigMap: %w", err)
		}

		if configMap != nil {
			_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.ConfigMap, bool, error) {
				return configMap, false, nil
			})
			if err != nil {
				logger.Error(err, "Failed to sync EPP ConfigMap")
				return fmt.Errorf("failed to sync EPP ConfigMap: %w", err)
			}
		}
	}

	// 2. Reconcile InferencePool
	// Note: EPP Service is created automatically by the standard component reconciliation
	// via GenerateComponentService() in graph.go (see ComponentTypeEPP case)
	eppServiceName := dynamo.GetDCDResourceName(dgd, componentName, "")
	inferencePool, err := epp.GenerateInferencePool(dgd, componentName, eppServiceName, eppService.EPPConfig)
	if err != nil {
		logger.Error(err, "Failed to generate EPP InferencePool")
		return fmt.Errorf("failed to generate EPP InferencePool: %w", err)
	}

	_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*gaiev1.InferencePool, bool, error) {
		return inferencePool, false, nil
	})
	if err != nil {
		logger.Error(err, "Failed to sync EPP InferencePool")
		return fmt.Errorf("failed to sync EPP InferencePool: %w", err)
	}

	// 3. Reconcile service mesh resources (e.g., Istio DestinationRule).
	// Only attempt DestinationRule reconciliation when the Istio CRDs are
	// present on the cluster; otherwise the API call would fail on every
	// reconcile for Istio-less clusters.
	// When disabled, still best-effort clean up previously owned DestinationRules if the API exists.
	meshEnabled := r.RuntimeConfig.Gate.Enabled(features.Istio)
	istioAvailable := meshEnabled
	if !meshEnabled {
		istioAvailable = features.DetectIstioDestinationRuleAvailability(ctx, r.RestConfig)
	}
	if istioAvailable {
		destinationRule := dynamo.GenerateEPPDestinationRule(eppServiceName, dgd.Namespace, r.Config.ServiceMesh)
		_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*networkingv1beta1.DestinationRule, bool, error) {
			return destinationRule, !meshEnabled, nil
		})
		if err != nil {
			logger.Error(err, "Failed to sync EPP DestinationRule")
			return fmt.Errorf("failed to sync EPP DestinationRule: %w", err)
		}
		if meshEnabled {
			logger.Info("Synced EPP DestinationRule", "name", eppServiceName)
		} else {
			logger.Info("Cleaned up EPP DestinationRule", "name", eppServiceName)
		}
	}

	logger.Info("Successfully reconciled EPP resources", "poolName", inferencePool.GetName())
	return nil
}

// reconcileWaitLeaderConfigMap ensures the wait-for-leader Python script
// ConfigMap exists for multinode DGDs. The ConfigMap is only mounted by
// vLLM mp worker pods (via UpdatePodSpec); for other backends it is inert.
func (r *DynamoGraphDeploymentReconciler) reconcileWaitLeaderConfigMap(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	if !dgd.HasAnyMultinodeComponent() {
		return nil
	}

	cm := dynamo.GenerateWaitLeaderConfigMap(dgd.Name, dgd.Namespace)
	_, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.ConfigMap, bool, error) {
		return cm, false, nil
	})
	return err
}

func (r *DynamoGraphDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	return r.deleteAutoCheckpointsForDGD(ctx, dynamoDeployment)
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoGraphDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&corev1.Pod{},
		dgdComponentPodIndex,
		dgdComponentPodIndexValues,
	); err != nil {
		return fmt.Errorf("register DGD component Pod index: %w", err)
	}

	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeployment{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named(consts.ResourceTypeDynamoGraphDeployment).
		Watches(
			&nvidiacomv1alpha1.DynamoCheckpoint{},
			handler.EnqueueRequestsFromMapFunc(r.mapAutoCheckpointToDGDRequests),
			builder.WithPredicates(predicate.Funcs{
				CreateFunc:  func(ce event.CreateEvent) bool { return false },
				DeleteFunc:  func(de event.DeleteEvent) bool { return true },
				UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
				GenericFunc: func(ge event.GenericEvent) bool { return true },
			}),
		).
		Watches(
			&corev1.Pod{},
			handler.EnqueueRequestsFromMapFunc(mapDGDWorkerPodToRequests),
			builder.WithPredicates(dgdWorkerPodEventPredicate()),
		).
		Owns(&nvidiacomv1beta1.DynamoComponentDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the adapter
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return false }, // Adapter updates are handled by adapter controller
			GenericFunc: func(ge event.GenericEvent) bool { return false },
		})).
		Owns(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the PVC
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig))
	if r.RuntimeConfig.Gate.Enabled(features.Istio) {
		ctrlBuilder = ctrlBuilder.Owns(&networkingv1beta1.DestinationRule{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return false },
		}))
	}
	if r.RuntimeConfig.Gate.Enabled(features.Grove) {
		ctrlBuilder = ctrlBuilder.Owns(&grovev1alpha1.PodCliqueSet{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the pod gang set
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
			// Watch PodClique resources - only on status changes
			Watches(
				&grovev1alpha1.PodClique{},
				handler.EnqueueRequestsFromMapFunc(r.mapPodCliqueToRequests),
				builder.WithPredicates(predicate.Funcs{
					CreateFunc: func(ce event.CreateEvent) bool { return false },
					DeleteFunc: func(de event.DeleteEvent) bool { return false },
					UpdateFunc: func(ue event.UpdateEvent) bool {
						oldPC, okOld := ue.ObjectOld.(*grovev1alpha1.PodClique)
						newPC, okNew := ue.ObjectNew.(*grovev1alpha1.PodClique)
						if !okOld || !okNew {
							return false
						}
						return podCliqueStatusChangeIsSignificant(oldPC, newPC)
					},
					GenericFunc: func(ge event.GenericEvent) bool { return false },
				}),
			).
			// Watch PodCliqueScalingGroup resources on status-replica changes.
			// PCSG.Status.AvailableReplicas is independently recomputed by the PCSG
			// controller and can land after the last PodClique event the DGD
			// controller sees. Without this watch, the DGD aggregate
			// (CheckPCSGReady reads pcsg.Status.AvailableReplicas) can stay stale
			// indefinitely even though the underlying PCSG is already ready.
			Watches(
				&grovev1alpha1.PodCliqueScalingGroup{},
				handler.EnqueueRequestsFromMapFunc(r.mapPodCliqueScalingGroupToRequests),
				builder.WithPredicates(predicate.Funcs{
					CreateFunc: func(ce event.CreateEvent) bool { return false },
					DeleteFunc: func(de event.DeleteEvent) bool { return false },
					UpdateFunc: func(ue event.UpdateEvent) bool {
						oldPCSG, okOld := ue.ObjectOld.(*grovev1alpha1.PodCliqueScalingGroup)
						newPCSG, okNew := ue.ObjectNew.(*grovev1alpha1.PodCliqueScalingGroup)
						if !okOld || !okNew {
							return false
						}
						return pcsgStatusChangeIsSignificant(oldPCSG, newPCSG)
					},
					GenericFunc: func(ge event.GenericEvent) bool { return false },
				}),
			)

	}
	// Wrap with metrics collection
	observedReconciler := observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeployment)
	return ctrlBuilder.Complete(observedReconciler)
}

func (r *DynamoGraphDeploymentReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}

func isDGDManagedWorkerPod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	labels := pod.GetLabels()
	return labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		labels[consts.KubeLabelDynamoComponent] != "" &&
		labels[consts.KubeLabelDynamoSelector] != "" &&
		dynamo.IsWorkerComponent(labels[consts.KubeLabelDynamoComponentType])
}

func dgdComponentPodIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	labels := pod.GetLabels()
	dgdName := labels[consts.KubeLabelDynamoGraphDeploymentName]
	componentName := labels[consts.KubeLabelDynamoComponent]
	if dgdName == "" || componentName == "" {
		return nil
	}
	return []string{dgdComponentPodIndexValue(dgdName, componentName)}
}

func dgdComponentPodIndexValue(dgdName, componentName string) string {
	// Both inputs are label values, which cannot contain '/', so this encoding
	// is collision-free.
	return dgdName + "/" + componentName
}

// dgdWorkerPodEventPredicate admits only events that can change whether a
// Recreate rollout is blocked by an old pod. Creation and deletion change pod
// membership; updates matter only when membership or terminality changes. A
// deletion timestamp alone leaves the pod non-terminal and does not enqueue.
func dgdWorkerPodEventPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return isDGDManagedWorkerPod(e.Object)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return isDGDManagedWorkerPod(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldManaged := isDGDManagedWorkerPod(e.ObjectOld)
			newManaged := isDGDManagedWorkerPod(e.ObjectNew)
			if oldManaged != newManaged {
				return true
			}
			if !newManaged {
				return false
			}
			oldPod := e.ObjectOld.(*corev1.Pod)
			newPod := e.ObjectNew.(*corev1.Pod)
			if oldPod.Labels[consts.KubeLabelDynamoGraphDeploymentName] != newPod.Labels[consts.KubeLabelDynamoGraphDeploymentName] ||
				oldPod.Labels[consts.KubeLabelDynamoComponent] != newPod.Labels[consts.KubeLabelDynamoComponent] ||
				oldPod.Labels[consts.KubeLabelDynamoSelector] != newPod.Labels[consts.KubeLabelDynamoSelector] {
				return true
			}
			return isTerminalPhase(oldPod.Status.Phase) != isTerminalPhase(newPod.Status.Phase)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

func mapDGDWorkerPodToRequests(_ context.Context, obj client.Object) []ctrl.Request {
	if !isDGDManagedWorkerPod(obj) {
		return nil
	}
	dgdName := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: dgdName}}}
}

func (r *DynamoGraphDeploymentReconciler) mapAutoCheckpointToDGDRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	ckpt, ok := obj.(*nvidiacomv1alpha1.DynamoCheckpoint)
	if !ok || ckpt == nil {
		return nil
	}
	if ckpt.Annotations == nil || ckpt.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
		return nil
	}
	if ckpt.Labels == nil {
		return nil
	}
	dgdName := ckpt.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	if dgdName == "" {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: ckpt.Namespace, Name: dgdName}}}
}

// mapPodCliqueToRequests maps a PodClique to reconcile requests for its owning DGD
// Uses the nvidia.com/dynamo-graph-deployment-name label for direct lookup - no API calls needed!
func (r *DynamoGraphDeploymentReconciler) mapPodCliqueToRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	podClique, ok := obj.(*grovev1alpha1.PodClique)
	if !ok {
		return nil
	}

	// PodCliques are labeled with the DGD name and live in the same namespace
	dgdName, hasLabel := podClique.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	if !hasLabel || dgdName == "" {
		log.FromContext(ctx).V(1).Info("PodClique missing DGD label",
			"podClique", podClique.Name,
			"namespace", podClique.Namespace)
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Name:      dgdName,
			Namespace: podClique.Namespace,
		},
	}}
}

// mapPodCliqueScalingGroupToRequests maps a PodCliqueScalingGroup to reconcile
// requests for its owning DGD.
//
// The PCSG is owned by a PodCliqueSet (controller ownerRef), which is in turn
// owned by the DynamoGraphDeployment. The PCS name may differ from the DGD name
// when auto-truncation is applied (see PCSNameForDGD), so we walk the ownerRef
// chain (PCSG -> PCS -> DGD) to find the actual DGD name.
func (r *DynamoGraphDeploymentReconciler) mapPodCliqueScalingGroupToRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	pcsg, ok := obj.(*grovev1alpha1.PodCliqueScalingGroup)
	if !ok {
		return nil
	}

	controllerRef := metav1.GetControllerOf(pcsg)
	if controllerRef == nil ||
		controllerRef.Kind != "PodCliqueSet" ||
		controllerRef.APIVersion != grovev1alpha1.SchemeGroupVersion.String() {
		log.FromContext(ctx).V(1).Info("PodCliqueScalingGroup missing PodCliqueSet controller ownerReference",
			"podCliqueScalingGroup", pcsg.Name,
			"namespace", pcsg.Namespace)
		return nil
	}

	// Look up the PCS to walk the ownerRef chain to the DGD, since PCS name
	// may be truncated and no longer match the DGD name.
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.Client.Get(ctx, types.NamespacedName{
		Name:      controllerRef.Name,
		Namespace: pcsg.Namespace,
	}, pcs); err != nil {
		log.FromContext(ctx).V(1).Info("failed to look up PodCliqueSet for PCSG",
			"podCliqueScalingGroup", pcsg.Name,
			"pcsName", controllerRef.Name,
			"error", err)
		return nil
	}

	pcsOwnerRef := metav1.GetControllerOf(pcs)
	if pcsOwnerRef == nil ||
		pcsOwnerRef.Kind != consts.ResourceTypeDynamoGraphDeployment {
		log.FromContext(ctx).V(1).Info("PodCliqueSet missing DynamoGraphDeployment controller ownerReference",
			"pcsName", pcs.Name,
			"namespace", pcs.Namespace)
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Name:      pcsOwnerRef.Name,
			Namespace: pcsg.Namespace,
		},
	}}
}

// ptrInt64Equal returns true when two *int64 values are equivalent, treating
// nil and a pointer to the same value as equal. Used to compare optional
// status fields like ObservedGeneration without tripping on pointer identity.
func ptrInt64Equal(a, b *int64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

// groveScheduledConditionChanged reports whether either of the Grove scheduling
// conditions consumed by the Ready-reason classification (dynamo/grove.go)
// changed between two condition slices. It watches PodCliqueScheduled (set on
// PodCliques) and MinAvailableBreached (set on PodCliqueScalingGroups); a single
// helper serves both since each object only carries the condition relevant to
// it. A change in Status or Reason on either condition is treated as a change,
// so a scheduling shortfall clearing (or appearing) re-wakes the DGD even when
// the replica counters are otherwise flat.
func groveScheduledConditionChanged(oldConds, newConds []metav1.Condition) bool {
	for _, condType := range []string{
		groveconstants.ConditionTypePodCliqueScheduled,
		groveconstants.ConditionTypeMinAvailableBreached,
	} {
		oldCond := meta.FindStatusCondition(oldConds, condType)
		newCond := meta.FindStatusCondition(newConds, condType)
		if (oldCond == nil) != (newCond == nil) {
			return true
		}
		if oldCond != nil && newCond != nil &&
			(oldCond.Status != newCond.Status || oldCond.Reason != newCond.Reason) {
			return true
		}
	}
	return false
}

// podCliqueStatusChangeIsSignificant reports whether a PodClique status update
// affects the DGD's Grove readiness/classification and therefore must not be
// filtered out by the watch predicate.
//
// It mirrors every field CheckPodCliqueReady (dynamo/grove.go) reads: the
// replica counters gate readiness and rolling-update detection; ScheduledReplicas
// and ScheduleGatedReplicas feed the InsufficientCapacity classification and the
// exposed status.components[*].scheduledReplicas; the PodCliqueScheduled
// condition is another capacity signal; ObservedGeneration gates "spec not yet
// processed". In particular, scheduling can advance (e.g. 1/N -> N/N) while the
// ready/updated/replica counters and the condition stay flat, so ScheduledReplicas
// must be compared or that event would be dropped and the DGD left stale.
func podCliqueStatusChangeIsSignificant(oldPC, newPC *grovev1alpha1.PodClique) bool {
	return oldPC.Status.ReadyReplicas != newPC.Status.ReadyReplicas ||
		oldPC.Status.UpdatedReplicas != newPC.Status.UpdatedReplicas ||
		oldPC.Status.Replicas != newPC.Status.Replicas ||
		oldPC.Status.ScheduledReplicas != newPC.Status.ScheduledReplicas ||
		oldPC.Status.ScheduleGatedReplicas != newPC.Status.ScheduleGatedReplicas ||
		oldPC.Spec.Replicas != newPC.Spec.Replicas ||
		!ptrInt64Equal(oldPC.Status.ObservedGeneration, newPC.Status.ObservedGeneration) ||
		groveScheduledConditionChanged(oldPC.Status.Conditions, newPC.Status.Conditions)
}

// pcsgStatusChangeIsSignificant is the PodCliqueScalingGroup analogue of
// podCliqueStatusChangeIsSignificant. It mirrors every field CheckPCSGReady
// reads, including ScheduledReplicas (capacity classification + exposed counter)
// and the MinAvailableBreached condition, so a scheduling-only advance re-wakes
// the DGD.
func pcsgStatusChangeIsSignificant(oldPCSG, newPCSG *grovev1alpha1.PodCliqueScalingGroup) bool {
	return oldPCSG.Status.AvailableReplicas != newPCSG.Status.AvailableReplicas ||
		oldPCSG.Status.UpdatedReplicas != newPCSG.Status.UpdatedReplicas ||
		oldPCSG.Status.Replicas != newPCSG.Status.Replicas ||
		oldPCSG.Status.ScheduledReplicas != newPCSG.Status.ScheduledReplicas ||
		oldPCSG.Spec.Replicas != newPCSG.Spec.Replicas ||
		!ptrInt64Equal(oldPCSG.Status.ObservedGeneration, newPCSG.Status.ObservedGeneration) ||
		groveScheduledConditionChanged(oldPCSG.Status.Conditions, newPCSG.Status.Conditions)
}
