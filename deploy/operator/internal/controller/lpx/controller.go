// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// graphReconciler reconciles LPXGraphDeployments using their owning DGD's configuration.
type graphReconciler struct {
	enabled               bool
	recorder              events.EventRecorder
	runtimeConfig         *commoncontroller.RuntimeConfig
	modelRegistry         lpx.ModelRegistry
	config                *configv1alpha1.OperatorConfiguration
	dockerSecretRetriever dynamo.SecretsRetriever

	client.Client
}

// Setup registers the LPX controller and its dependencies.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig, secrets dynamo.SecretsRetriever) error {
	r := &graphReconciler{
		// Disabled integration still serves child status; deletion uses owner garbage collection.
		enabled:               integrationDisabledMessage(runtimeConfig) == "",
		recorder:              mgr.GetEventRecorder("lpxgraphdeployment"),
		runtimeConfig:         runtimeConfig,
		config:                config,
		dockerSecretRetriever: secrets,
		Client:                mgr.GetClient(),
	}

	if r.enabled {
		var err error
		r.modelRegistry, err = newLPXModelRegistry(config)
		if err != nil {
			return err
		}

		if err := setupLPUEviction(mgr); err != nil {
			return fmt.Errorf("unable to create LPU Eviction controller: %w", err)
		}
	}

	if err := r.setupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create LPXGraphDeployment controller: %w", err)
	}

	return nil
}

func (r *graphReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets/finalizers,verbs=update
// +kubebuilder:rbac:groups=scheduling.lpu.nvidia.com,resources=lpupipelinerequests,verbs=get;list;watch;create;delete

// Reconcile observes dependencies once, then persists the final child status.
// Cached misses and write conflicts are retried; no APIReader is used.
// These observations are not an atomic snapshot; watches converge later edits.
func (r *graphReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	deployment := &v1alpha1.LPXGraphDeployment{}
	if err := r.Get(ctx, req.NamespacedName, deployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !deployment.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// Persist the workload outcome before converting errors into deadline retries.
	defer func(previous *v1alpha1.LPXGraphDeploymentStatus) {
		if err != nil {
			setReadyCondition(deployment, v1beta1.DGDStateFailed, err.Error())
		} else {
			deployment.Status.ObservedGeneration = deployment.Generation
		}

		if statusErr := r.updateStatus(ctx, deployment, previous); statusErr != nil {
			err = errors.Join(err, statusErr)
			result = ctrl.Result{}
			return
		}

		// Workload errors retain a retry delay only while a scheduling deadline is active.
		if err != nil && result.RequeueAfter > 0 {
			log.FromContext(ctx).Error(err, "retrying reconciliation under the pipeline request scheduling deadline")
			err = nil
		}
	}(deployment.Status.DeepCopy())

	for name, component := range deployment.Status.Components {
		component.Ready = false
		deployment.Status.Components[name] = component
	}

	if !r.enabled {
		setReadyCondition(deployment, v1beta1.DGDStateFailed, integrationDisabledMessage(r.runtimeConfig))
		return ctrl.Result{}, nil
	}

	// DGD identity and input revision are checked before any workload writes.
	dgd, err := getDynamoGraphDeployment(ctx, r.Client, deployment)
	if err != nil {
		return ctrl.Result{}, err
	}

	// The parent owns deselection/deletion; a cache miss or delayed handoff is not cleanup authority.
	if dgd == nil {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the matching DynamoGraphDeployment")
		return ctrl.Result{}, nil
	}

	// Resolve the owned Grove hierarchy from one cache observation.
	pcs, err := getPodCliqueSet(ctx, r.Client, deployment)
	if err != nil {
		return ctrl.Result{}, err
	}

	if pcs != nil && !pcs.DeletionTimestamp.IsZero() {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for PodCliqueSet garbage collection")
		return ctrl.Result{}, nil
	}

	var (
		pcsgs    map[string]*grovev1alpha1.PodCliqueScalingGroup
		requests []lpxv1alpha1.LPUPipelineRequest
	)

	if pcs != nil {
		// Wait for every configured group before reconciling an existing PCS.
		pcsgs, err = getPodCliqueScalingGroups(ctx, r.Client, pcs)
		if err != nil {
			return ctrl.Result{}, err
		}

		if len(pcsgs) != len(pcs.Spec.Template.PodCliqueScalingGroupConfigs) {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for all LPX scaling groups")
			return ctrl.Result{}, nil
		}

		// Foreground PCS replacement waits for its blocking LPRs; discover them only by owner UID.
		requests, err = r.getPipelineRequests(ctx, pcs)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	if result, err := r.reconcileModelDownloads(ctx, deployment, dgd); err != nil || result.RequeueAfter > 0 {
		return result, err
	}

	return r.reconcileWorkloads(ctx, deployment, dgd, pcs, pcsgs, requests)
}

// reconcileWorkloads consumes owned observations; DGD and deployment are non-nil.
// A nil pcs means initial creation, with no pcsgs or requests. Otherwise pcsgs
// contains every configured, non-deleting group. On error, Reconcile persists
// status before applying deadline retries.
func (r *graphReconciler) reconcileWorkloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (result ctrl.Result, err error) {
	// Resolve and render every workload before deleting the running PCS.
	workloads, plans, err := r.resolveWorkloads(ctx, deployment, dgd)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Share the rendered identities with capacity management and request publication.
	desiredPCS, resources, err := r.renderPodCliqueSet(ctx, deployment, dgd, workloads, plans)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Immutable composition changes replace the PCS; replica-only changes use /scale.
	if pcs != nil && pcs.Annotations[lpx.WorkloadDigestAnnotation] != desiredPCS.Annotations[lpx.WorkloadDigestAnnotation] {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the previous PodCliqueSet and its requests to be deleted")
		return ctrl.Result{}, deletePodCliqueSet(ctx, r, pcs)
	}

	// Resolve capacity and requests for the complete graph before any group is changed.
	var (
		groupNames = slices.Sorted(maps.Keys(workloads))

		desiredRequests  = make(map[string]*lpxv1alpha1.LPUPipelineRequest)
		explicitReplicas = make(map[string]*int32, len(plans))
		missingRequests  []*lpxv1alpha1.LPUPipelineRequest
	)

	// Index the graph's observed requests once, retaining scheduler-owned receipts.
	requestsByName := make(map[string]*lpxv1alpha1.LPUPipelineRequest, len(requests))
	for i := range requests {
		request := &requests[i]
		requestsByName[request.Name] = request
	}

	// Resolve each workload in the same order used for request publication.
	for _, groupName := range groupNames {
		plan := plans[groupName]
		workload := workloads[groupName]
		pcsg := pcsgs[plan.LPXScalingGroup]
		explicitReplicas[plan.LPXScalingGroup] = dgd.GetComponentByName(groupName).Replicas

		// External scalers own live capacity once Grove has created the groups.
		if pcs != nil && explicitReplicas[plan.LPXScalingGroup] == nil {
			plan.Replicas = pcsg.Spec.Replicas
			if err := plan.ValidateReplicaCount(); err != nil {
				return ctrl.Result{}, err
			}
		}

		desired, missing, intentChanged := resolvePipelineRequests(deployment, requestsByName, workload, plan)

		if intentChanged {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the previous PodCliqueSet and its requests to be deleted")
			return ctrl.Result{}, deletePodCliqueSet(ctx, r, pcs)
		}

		maps.Copy(desiredRequests, desired)
		missingRequests = append(missingRequests, missing...)
	}

	// One graph-wide deadline observation preserves the earliest wakeup on every return.
	expiredRequests, deadlineAt := pipelineRequestDeadlines(desiredRequests, pipelineRequestDeadlineSeconds(dgd))
	defer func() {
		result = requeueForPipelineRequestDeadline(deadlineAt, result, err)
	}()

	// Scale only existing groups with explicitly managed capacity.
	for groupName, pcsg := range pcsgs {
		if replicas := explicitReplicas[groupName]; replicas != nil {
			if err := scaleDownPodCliqueScalingGroup(ctx, r, pcsg, *replicas); err != nil {
				return ctrl.Result{}, err
			}
		}
	}

	// Remove requests only after all scale-down writes have succeeded.
	removed := pipelineRequestsPendingDeletion(requests, desiredRequests)
	if err := r.deletePipelineRequests(ctx, removed); err != nil {
		return ctrl.Result{}, err
	}

	// Failure blocks publication while each workload independently cleans up expired suffixes.
	if len(expiredRequests) > 0 || isSchedulingFailedConditionCurrent(deployment) {
		retained := slices.DeleteFunc(slices.Clone(requests), func(request lpxv1alpha1.LPUPipelineRequest) bool {
			_, desired := desiredRequests[request.Name]
			return !desired
		})
		return r.reconcileSchedulingFailure(ctx, deployment, pcsgs, explicitReplicas, retained, expiredRequests)
	}

	acknowledgeSchedulingRetry(deployment)

	// Terminating requests must settle before synchronization or scale-out.
	if len(removed) > 0 {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for removed LPX requests to finish deletion")
		return ctrl.Result{}, nil
	}

	modified, _, err := commoncontroller.SyncObservedResource(ctx, r, deployment, pcs, desiredPCS, commoncontroller.WithPreservedListOrder())
	if err != nil {
		if apierrors.IsAlreadyExists(err) || apierrors.IsConflict(err) {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the PodCliqueSet cache observation")
			return ctrl.Result{}, nil
		}

		return ctrl.Result{}, err
	}

	if err := r.reconcileRuntimeResources(ctx, deployment, resources); err != nil {
		return ctrl.Result{}, err
	}

	// Observe the created or updated PCS before scaling or publishing requests.
	if pcs == nil || modified {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for Grove to observe the workload")
		return ctrl.Result{}, nil
	}

	// Scale-out proceeds after deleting old names, without waiting for Pods.
	for _, groupName := range groupNames {
		plan := plans[groupName]
		if replicas := explicitReplicas[plan.LPXScalingGroup]; replicas != nil {
			if err := scalePodCliqueScalingGroup(ctx, r, pcsgs[plan.LPXScalingGroup], *replicas); err != nil {
				return ctrl.Result{}, err
			}
		}
	}

	if len(missingRequests) > 0 {
		return ctrl.Result{}, r.reconcilePipelineRequests(ctx, deployment, pcs, missingRequests)
	}

	defer func() {
		// Old configmaps remain available until all workloads are Ready.
		if err == nil {
			err = r.deleteUnusedConfigMaps(ctx, deployment, resources)
		}
	}()

	return r.reconcileReadiness(ctx, deployment, dgd, pcs, pcsgs, plans, desiredRequests)
}

// reconcileRuntimeResources synchronizes additional resources for the LPX deployment.
func (r *graphReconciler) reconcileRuntimeResources(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, resources []client.Object) error {
	for _, resource := range resources {
		if _, _, err := commoncontroller.SyncResource(ctx, r, deployment, func(context.Context) (client.Object, bool, error) {
			return resource, false, nil
		}); err != nil {
			return err
		}
	}
	return nil
}

// reconcileReadiness combines scheduler receipts with Grove runtime readiness.
// All pointer inputs are non-nil and refer to the same observed workload.
func (r *graphReconciler) reconcileReadiness(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	plans map[string]*lpx.MaterializationPlan,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
) (ctrl.Result, error) {
	readiness := dynamo.GroveReadiness{Ready: true}
	deployment.Status.Components = make(map[string]v1beta1.ComponentReplicaStatus)

	componentGroups := lpx.ComponentGroups(dgd)

	for _, groupName := range slices.Sorted(maps.Keys(plans)) {
		plan := plans[groupName]

		observed, err := dynamo.EvaluateLPXGroveReadiness(ctx, r.Client, dgd, groupName, componentGroups[groupName], pcs, pcsgs[plan.LPXScalingGroup])
		if err != nil {
			return ctrl.Result{}, err
		}

		maps.Copy(deployment.Status.Components, observed.ComponentStatuses)

		if !observed.Ready && readiness.Ready {
			readiness = observed
		}
	}

	if !setPipelineRequestReadyCondition(deployment, requests) {
		return ctrl.Result{}, nil
	}

	if !readiness.Ready {
		setReadyCondition(deployment, v1beta1.DGDStatePending, readiness.Message)
		return ctrl.Result{}, nil
	}

	setReadyCondition(deployment, v1beta1.DGDStateSuccessful, readiness.Message)

	delay := modelDownloadRefreshInterval
	if download := deployment.Status.ModelDownload; download != nil && download.LastCheckedAt != nil {
		delay = max(modelDownloadRequeueAfter, time.Until(download.LastCheckedAt.Add(modelDownloadRefreshInterval)))
	}

	return ctrl.Result{RequeueAfter: delay}, nil
}

// deleteUnusedConfigMaps runs only after readiness so existing pods keep their
// immutable configuration during replacement. Maps belong to the non-nil LPXGD,
// not the PCS, and therefore survive PCS garbage collection.
func (r *graphReconciler) deleteUnusedConfigMaps(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, resources []client.Object) error {
	// Keep old runtime configuration until the replacement workload is Ready.
	if !meta.IsStatusConditionTrue(deployment.Status.Conditions, v1alpha1.LPXReadyCondition) {
		return nil
	}

	// Retain every ConfigMap rendered for the current workload.
	desiredNames := make(map[string]struct{}, len(resources))
	for _, resource := range resources {
		if _, ok := resource.(*corev1.ConfigMap); ok {
			desiredNames[resource.GetName()] = struct{}{}
		}
	}

	// Discover obsolete ConfigMaps rooted in this exact LPX child.
	configMaps := &corev1.ConfigMapList{}
	if err := r.List(ctx, configMaps,
		client.InNamespace(deployment.Namespace),
		client.MatchingLabels{deploymentUIDLabel: string(deployment.UID)},
	); err != nil {
		return err
	}

	// Preconditions prevent stale observations from deleting replacements.
	for index := range configMaps.Items {
		configMap := &configMaps.Items[index]
		if !metav1.IsControlledBy(configMap, deployment) || !configMap.DeletionTimestamp.IsZero() {
			continue
		}
		if _, desired := desiredNames[configMap.Name]; desired {
			continue
		}
		uid, resourceVersion := configMap.GetUID(), configMap.GetResourceVersion()
		if err := r.Delete(ctx, configMap, &client.DeleteOptions{Preconditions: &metav1.Preconditions{
			UID: &uid, ResourceVersion: &resourceVersion,
		}}); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// integrationDisabledMessage returns an empty string when integration is enabled.
// runtimeConfig is non-nil; child status remains available when disabled.
func integrationDisabledMessage(runtimeConfig *commoncontroller.RuntimeConfig) string {
	if !runtimeConfig.Gate.Enabled(features.Grove) {
		return "Grove is disabled"
	}
	if !runtimeConfig.Gate.Enabled(features.LPX) {
		return "LPX integration is disabled"
	}
	return ""
}
