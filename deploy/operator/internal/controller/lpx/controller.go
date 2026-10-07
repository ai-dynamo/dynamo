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
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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
	recorder              events.EventRecorder
	runtimeConfig         *commoncontroller.RuntimeConfig
	modelRegistry         lpx.ModelRegistry
	config                *configv1alpha1.OperatorConfiguration
	dockerSecretRetriever dynamo.SecretsRetriever

	client.Client
}

// Setup registers the LPX controller and its dependencies when LPX is enabled.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig, secrets dynamo.SecretsRetriever) error {
	if !runtimeConfig.Gate.Enabled(features.LPX) {
		return nil
	}

	r := &graphReconciler{
		recorder:              mgr.GetEventRecorder("lpxgraphdeployment"),
		runtimeConfig:         runtimeConfig,
		config:                config,
		dockerSecretRetriever: secrets,
		Client:                mgr.GetClient(),
	}

	if runtimeConfig.Gate.Enabled(features.Grove) {
		var err error
		r.modelRegistry, err = newLPXModelRegistry(config)
		if err != nil {
			return err
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
// SetControllerReference sets blockOwnerDeletion=true on PCS-owned LPUPipelineRequests,
// requiring update permission on the owner's finalizers subresource.
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets/finalizers,verbs=update
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale;podcliquescalinggroups/scale,verbs=update
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
		component.Conditions = nil
		deployment.Status.Components[name] = component
	}

	if !r.runtimeConfig.Gate.Enabled(features.Grove) {
		setReadyCondition(deployment, v1beta1.DGDStateFailed, "Grove is disabled")
		return ctrl.Result{}, nil
	}

	// DGD identity and input revision are checked before any workload writes.
	dgd, err := getDynamoGraphDeployment(ctx, r.Client, deployment)
	if err != nil {
		return ctrl.Result{}, err
	}

	// A cache miss or delayed handoff is not cleanup authority; owner GC handles deletion.
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
		pclqs    map[string]*grovev1alpha1.PodClique
		requests map[string]*lpxv1alpha1.LPUPipelineRequest
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

		pclqs, err = getPodCliques(ctx, r.Client, pcs, pcsgs)
		if err != nil {
			return ctrl.Result{}, err
		}

		requests, err = r.getPipelineRequests(ctx, pcs)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	if result, err := r.reconcileModelDownloads(ctx, deployment, dgd); err != nil || result.RequeueAfter > 0 {
		return result, err
	}

	return r.reconcileWorkloads(ctx, deployment, dgd, pcs, pcsgs, pclqs, requests)
}

// reconcileWorkloads consumes owned observations; DGD and deployment are non-nil.
// A nil pcs means initial creation, with no pcsgs, pclqs or requests. Otherwise
// pcsgs contains every configured group; pclqs contains owned, non-deleting cliques.
// On error, Reconcile persists status before applying deadline retries.
func (r *graphReconciler) reconcileWorkloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsgs map[string]*grovev1alpha1.PodCliqueScalingGroup,
	pclqs map[string]*grovev1alpha1.PodClique,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
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

	// Grove rolls compatible layouts; immutable structure or OnDelete builds replace the PCS.
	if pcs != nil && !podCliqueSetLayoutMatches(pcs, desiredPCS) {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the previous PodCliqueSet and its requests to be deleted")
		return ctrl.Result{}, deletePodCliqueSet(ctx, r, pcs)
	}

	// Resolve capacity and requests for the complete graph before any group is changed.
	var (
		groupNames = slices.Sorted(maps.Keys(workloads))

		desiredRequests  = make(map[string]*lpxv1alpha1.LPUPipelineRequest)
		explicitReplicas = make(map[string]*int32, len(plans))
		missingRequests  []*lpxv1alpha1.LPUPipelineRequest
		expiredRequests  []*lpxv1alpha1.LPUPipelineRequest
		deadlineAt       time.Time
	)

	// Prepare requests only after observing the PCS, in publication order.
	if pcs != nil {
		for _, groupName := range groupNames {
			plan := plans[groupName]
			workload := workloads[groupName]
			pcsg := pcsgs[plan.LPXScalingGroup]
			explicitReplicas[plan.LPXScalingGroup] = dgd.GetComponentByName(groupName).Replicas

			// External scalers own live capacity once Grove has created the groups.
			if explicitReplicas[plan.LPXScalingGroup] == nil {
				plan.Replicas = pcsg.Spec.Replicas
				if err := plan.ValidateReplicaCount(); err != nil {
					return ctrl.Result{}, err
				}
			}

			desired, missing := resolvePipelineRequests(deployment, requests, workload, plan)

			maps.Copy(desiredRequests, desired)
			missingRequests = append(missingRequests, missing...)

			// Expanded models use their source component's policy, not the conductor's.
			secondsByModel := make(map[string]*int64)
			for _, projection := range workload.ModelProjections() {
				if scheduling := dgd.GetComponentByName(projection.ComponentName()).LPX.Scheduling; scheduling != nil {
					secondsByModel[projection.Model()] = scheduling.AttemptDeadlineSeconds
				}
			}
			expired, next := pipelineRequestDeadlines(desired, secondsByModel, deadlineAt)
			expiredRequests = append(expiredRequests, expired...)
			deadlineAt = next
		}
	}

	// Preserve the earliest deadline across workloads on every subsequent return.
	defer func() {
		result = requeueForPipelineRequestDeadline(deadlineAt, result, err)
	}()

	// Scale only existing groups with explicitly managed capacity.
	var capacityChanged bool
	for groupName, pcsg := range pcsgs {
		if replicas := explicitReplicas[groupName]; replicas != nil {
			changed, err := scaleDownPodCliqueScalingGroup(ctx, r, pcsg, *replicas)
			if err != nil {
				return ctrl.Result{}, err
			}

			capacityChanged = capacityChanged || changed
		}
	}

	// Retire released requests after scale-down, keeping all removed names pending.
	removed := pipelineRequestsPendingDeletion(requests, desiredRequests)
	if err := r.deletePipelineRequests(ctx, slices.DeleteFunc(slices.Clone(removed), pipelineRequestCommitted)); err != nil {
		return ctrl.Result{}, err
	}

	if capacityChanged {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting to observe LPX capacity changes")
		return ctrl.Result{}, nil
	}

	// Failure blocks publication while each workload independently cleans up expired suffixes.
	if len(expiredRequests) > 0 {
		return r.reconcileSchedulingFailure(ctx, deployment, pcsgs, explicitReplicas, requests, desiredRequests, expiredRequests)
	}

	// Keep publication blocked after the failed requests have been removed.
	if isSchedulingFailedConditionCurrent(deployment) {
		setSchedulingFailedCondition(deployment, false)
		return ctrl.Result{}, nil
	}

	acknowledgeSchedulingRetry(deployment)

	if err := r.reconcileRuntimeResources(ctx, deployment, resources); err != nil {
		return ctrl.Result{}, err
	}

	modified, _, err := commoncontroller.SyncObservedResource(ctx, r, deployment, pcs, desiredPCS, commoncontroller.WithPreservedListOrder())
	if err != nil {
		if apierrors.IsAlreadyExists(err) || apierrors.IsConflict(err) {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the PodCliqueSet cache observation")
			return ctrl.Result{}, nil
		}

		return ctrl.Result{}, err
	}

	// Observe the created or updated PCS before scaling or publishing requests.
	if modified {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for Grove to observe the workload")
		return ctrl.Result{}, nil
	}

	// Scale-out proceeds after deleting old names, without waiting for Pods.
	for _, groupName := range groupNames {
		if len(removed) > 0 {
			break
		}
		workload := workloads[groupName]
		plan := plans[groupName]
		pcsg := pcsgs[plan.LPXScalingGroup]
		component := dgd.GetComponentByName(groupName)

		changed, err := r.reconcileWorkloadCapacity(ctx, component, pcsg, pclqs, workload, plan)
		if err != nil {
			return ctrl.Result{}, err
		}

		if changed {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting to observe LPX capacity changes")
			return ctrl.Result{}, nil
		}
	}

	if len(missingRequests) > 0 {
		return ctrl.Result{}, r.reconcilePipelineRequests(ctx, deployment, pcs, pcsgs, pclqs, requests, missingRequests)
	}

	if len(removed) > 0 {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for removed LPX requests to finish deletion")
		return ctrl.Result{}, nil
	}

	result = r.reconcileReadiness(ctx, deployment, dgd, pcs, pcsgs, pclqs, plans, desiredRequests)
	// Old configmaps remain available until all workloads are Ready.
	return result, r.deleteUnusedConfigMaps(ctx, deployment, resources)
}

// reconcileWorkloadCapacity applies explicit capacity and validates external Cyborg counts.
// Component, pcsg, workload and plan are non-nil. Omitted replica counts are never written.
// The returned bool reports successful scale writes that require a fresh observation.
func (r *graphReconciler) reconcileWorkloadCapacity(
	ctx context.Context,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	pclqs map[string]*grovev1alpha1.PodClique,
	workload *lpx.Workload,
	plan *lpx.MaterializationPlan,
) (bool, error) {
	if replicas := component.Replicas; replicas != nil {
		changed, err := scalePodCliqueScalingGroup(ctx, r, pcsg, *replicas)
		if changed || err != nil {
			return changed, err
		}
	}

	if plan.CyborgTemplate == "" {
		return false, nil
	}

	if replicas := component.ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas; replicas != nil {
		return scalePodCliques(ctx, r, pcsg, pclqs, plan.CyborgTemplate, *replicas)
	}

	for index := range pcsg.Spec.Replicas {
		name := grovecommon.GeneratePodCliqueName(grovecommon.ResourceNameReplica{Name: pcsg.Name, Replica: int(index)}, plan.CyborgTemplate)
		pclq := pclqs[name]
		if pclq == nil {
			continue
		}

		if err := workload.ValidateCyborgReplicas(pclq.Spec.Replicas); err != nil {
			return false, err
		}
	}

	return false, nil
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
	pclqs map[string]*grovev1alpha1.PodClique,
	plans map[string]*lpx.MaterializationPlan,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
) ctrl.Result {
	readiness := dynamo.GroveReadiness{Ready: true}
	deployment.Status.Components = make(map[string]v1alpha1.LPXComponentStatus)

	componentGroups := lpx.ComponentGroups(dgd)

	for _, groupName := range slices.Sorted(maps.Keys(plans)) {
		plan := plans[groupName]

		observed := dynamo.EvaluateLPXGroveReadiness(ctx, dgd, groupName, componentGroups[groupName], pcs, pcsgs[plan.LPXScalingGroup], pclqs)

		state := v1beta1.DGDStatePending
		if observed.Ready {
			state = v1beta1.DGDStateSuccessful
		}
		conditions := []metav1.Condition{readyCondition(deployment.Generation, state, observed.Message)}
		groupRequests := make(map[string]*lpxv1alpha1.LPUPipelineRequest)
		for name, request := range requests {
			if request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.Name == plan.LPXScalingGroup {
				groupRequests[name] = request
			}
		}
		setPipelineRequestReadyCondition(&conditions, deployment.Generation, groupRequests)
		for name, status := range observed.ComponentStatuses {
			deployment.Status.Components[name] = v1alpha1.LPXComponentStatus{ComponentReplicaStatus: status, Conditions: conditions}
		}

		if !observed.Ready && readiness.Ready {
			readiness = observed
		}
	}

	if !setPipelineRequestReadyCondition(&deployment.Status.Conditions, deployment.Generation, requests) {
		return ctrl.Result{}
	}

	if !readiness.Ready {
		setReadyCondition(deployment, v1beta1.DGDStatePending, readiness.Message)
		return ctrl.Result{}
	}

	setReadyCondition(deployment, v1beta1.DGDStateSuccessful, readiness.Message)

	if download := deployment.Status.ModelDownload; download != nil && download.LastCheckedAt != nil {
		return ctrl.Result{RequeueAfter: max(modelDownloadRequeueAfter, time.Until(download.LastCheckedAt.Add(modelDownloadRefreshInterval)))}
	}

	return ctrl.Result{}
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

// podCliqueSetLayoutMatches compares the variable immutable fields emitted by LPX.
// OnDelete also fixes the workload digests; the desired strategy governs transitions.
func podCliqueSetLayoutMatches(observed, desired *grovev1alpha1.PodCliqueSet) bool {
	onDelete := desired.Spec.UpdateStrategy != nil && desired.Spec.UpdateStrategy.Type == grovev1alpha1.OnDeleteStrategy
	layout := func(pcs *grovev1alpha1.PodCliqueSet) grovev1alpha1.PodCliqueSetTemplateSpec {
		result := grovev1alpha1.PodCliqueSetTemplateSpec{TopologyConstraint: pcs.Spec.Template.TopologyConstraint}
		for _, clique := range pcs.Spec.Template.Cliques {
			result.Cliques = append(result.Cliques, &grovev1alpha1.PodCliqueTemplateSpec{
				Name: clique.Name, TopologyConstraint: clique.TopologyConstraint,
				Spec: grovev1alpha1.PodCliqueSpec{MinAvailable: clique.Spec.MinAvailable},
			})
		}
		for _, group := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
			// Grove defaults this mutable rollout policy; it does not change the layout.
			group.RollingUpdate = nil
			if !onDelete {
				group.Annotations = nil
			}
			result.PodCliqueScalingGroupConfigs = append(result.PodCliqueScalingGroupConfigs, group)
		}
		return result
	}
	return apiequality.Semantic.DeepEqual(layout(observed), layout(desired))
}
