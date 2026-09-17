// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
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
		pcsg     *grovev1alpha1.PodCliqueScalingGroup
		requests []lpxv1alpha1.LPUPipelineRequest
	)

	if pcs != nil {
		// Foreground PCS replacement waits for its blocking LPRs; discover them only by owner UID.
		requests, err = r.getPipelineRequests(ctx, pcs)
		if err != nil {
			return ctrl.Result{}, err
		}

		pcsg, err = getPodCliqueScalingGroup(ctx, r.Client, pcs)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	return r.reconcileWorkload(ctx, deployment, dgd, pcs, pcsg, requests)
}

// reconcileWorkload consumes owned observations; DGD and deployment are non-nil.
// A nil pcs means initial creation (pcsg is nil and requests is empty). A nil pcsg
// means no current, non-deleting group is observed, not zero capacity. A non-nil
// pcsg is owned by pcs; only its live Spec.Replicas controls existing engine capacity.
// On error, a nonzero RequeueAfter requests a bounded deadline retry; Reconcile
// records the error and persists status before returning that retry to controller-runtime.
func (r *graphReconciler) reconcileWorkload(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (result ctrl.Result, err error) {
	if result, err := r.reconcileModelDownloads(ctx, deployment, dgd); err != nil || result.RequeueAfter > 0 {
		return result, err
	}

	workload, err := lpx.ResolveSelectedWorkload(ctx, dgd, r.modelRegistry)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Validate the complete replacement before either immutable change can delete the running PCS.
	plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment))
	if err != nil {
		return ctrl.Result{}, err
	}
	desiredPCS, resources, err := r.renderPodCliqueSet(ctx, deployment, dgd, workload, plan)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Model composition changes require replacement: Grove makes clique membership immutable.
	// Replica-only changes, including top-level PCS replica drift, use normal mutable sync.
	if pcs != nil && pcs.Annotations[lpx.WorkloadDigestAnnotation] != workload.Digest().String() {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the previous PodCliqueSet and its requests to be deleted")
		return ctrl.Result{}, r.deletePodCliqueSet(ctx, pcs)
	}

	// Explicit DGD replicas authorize scale writes; omission leaves the live count externally managed.
	explicitReplicas := lpx.ServingComponent(dgd).Replicas
	if explicitReplicas == nil && pcs != nil {
		// Do not derive an externally managed request set from an unknown replica count.
		if pcsg == nil {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the LPX scaling group")
			return ctrl.Result{}, nil
		}

		plan.Replicas = pcsg.Spec.Replicas
		if err := plan.ValidateReplicaCount(); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Preserve matching scheduler receipts and collect new LPRs in publication order.
	desiredRequests, missingRequests, intentChanged := resolvePipelineRequests(deployment, requests, workload, plan)
	if intentChanged {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the previous PodCliqueSet and its requests to be deleted")
		return ctrl.Result{}, r.deletePodCliqueSet(ctx, pcs)
	}

	// Schedule deadline wakes without consuming errors needed by the status boundary.
	expiredRequests, deadlineAt := pipelineRequestDeadlines(desiredRequests, pipelineRequestDeadlineSeconds(dgd))
	defer func() {
		result = requeueForPipelineRequestDeadline(deadlineAt, result, err)
	}()

	// Apply explicit scale-in before deadline failures can block publication.
	if err := r.scaleDownPodCliqueScalingGroup(ctx, pcsg, explicitReplicas); err != nil {
		return ctrl.Result{}, err
	}

	// Deadline failures stop publication even when no timer or cleanup is needed.
	if len(expiredRequests) > 0 || isSchedulingFailedConditionCurrent(deployment) {
		// Explicit or external scale-in has already lowered capacity; failure must not prevent its LPR cleanup.
		if pcsg != nil {
			if err := r.deletePipelineRequests(ctx, pipelineRequestsPendingDeletion(requests, desiredRequests)); err != nil {
				return ctrl.Result{}, err
			}

			// Do not delete those names again during deadline cleanup or mutate the expired requests' backing slice.
			requests = slices.DeleteFunc(slices.Clone(requests), func(request lpxv1alpha1.LPUPipelineRequest) bool {
				_, desired := desiredRequests[request.Name]
				return !desired
			})
		}
		return r.reconcilePipelineRequestDeadline(ctx, deployment, pcsg, requests, expiredRequests, explicitReplicas != nil)
	}

	clearPreviousSchedulingFailure(deployment)

	// Existing LPRs require an observed group so scale-down precedes their deletion.
	// Without LPRs, initial PCS synchronization can proceed before Grove creates the group.
	if pcsg == nil && len(requests) > 0 {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the LPX scaling group")
		return ctrl.Result{}, nil
	}

	// Deletion is terminal for this pass; LPR events resume publication after cleanup.
	if removed := pipelineRequestsPendingDeletion(requests, desiredRequests); len(removed) > 0 {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for removed LPX requests to finish deletion")
		return ctrl.Result{}, r.deletePipelineRequests(ctx, removed)
	}

	modified, _, err := commoncontroller.SyncObservedResource(ctx, r, deployment, pcs, desiredPCS, commoncontroller.WithPreservedListOrder(), commoncontroller.WithMetadataSync())
	if err != nil {
		if apierrors.IsAlreadyExists(err) || apierrors.IsConflict(err) {
			setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for the PodCliqueSet cache observation")
			return ctrl.Result{}, nil
		}

		return ctrl.Result{}, err
	}

	if err := r.reconcileRuntimeResources(ctx, deployment, dgd, resources); err != nil {
		return ctrl.Result{}, err
	}

	// Publication requires an observed PCS spec and its current, non-deleting group.
	if modified || pcsg == nil {
		setReadyCondition(deployment, v1beta1.DGDStatePending, "Waiting for Grove to observe the workload")
		return ctrl.Result{}, nil
	}

	// Scale-out proceeds after deleting old names, without waiting for Pods.
	if err := r.scalePodCliqueScalingGroup(ctx, pcsg, explicitReplicas); err != nil {
		return ctrl.Result{}, err
	}

	// Publication is terminal for this pass; readiness needs watched scheduler receipts.
	if len(missingRequests) > 0 {
		return ctrl.Result{}, r.reconcilePipelineRequests(ctx, deployment, pcs, missingRequests)
	}

	defer func() {
		if err == nil {
			err = r.deleteUnusedConfigMaps(ctx, deployment, resources)
		}
	}()

	return r.reconcileReadiness(ctx, deployment, dgd, pcs, pcsg, desiredRequests)
}

// scaleDownPodCliqueScalingGroup lowers capacity before asynchronous request cleanup.
// A nil pcsg defers scaling until the group is observed; a non-nil pcsg is owned
// and not deleting. Nil replicas leaves capacity externally managed. Scale-out
// must wait until terminating request names disappear.
func (r *graphReconciler) scaleDownPodCliqueScalingGroup(ctx context.Context, pcsg *grovev1alpha1.PodCliqueScalingGroup, replicas *int32) error {
	if pcsg == nil || replicas == nil || *replicas >= pcsg.Spec.Replicas {
		return nil
	}
	return r.scalePodCliqueScalingGroup(ctx, pcsg, replicas)
}

// scalePodCliqueScalingGroup updates the non-nil, already-owned PCSG using its
// observed resource version. Nil replicas leaves capacity to external autoscaling.
// A successful write updates the supplied observation for the rest of this pass.
func (r *graphReconciler) scalePodCliqueScalingGroup(ctx context.Context, pcsg *grovev1alpha1.PodCliqueScalingGroup, replicas *int32) error {
	if replicas == nil || pcsg.Spec.Replicas == *replicas {
		return nil
	}
	scale := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{ResourceVersion: pcsg.ResourceVersion},
		Spec:       autoscalingv1.ScaleSpec{Replicas: *replicas},
	}
	if err := r.SubResource("scale").Update(ctx, pcsg, client.WithSubResourceBody(scale)); err != nil {
		return fmt.Errorf("scale LPX PodCliqueScalingGroup %q: %w", pcsg.Name, err)
	}
	pcsg.Spec.Replicas = *replicas
	return nil
}

// reconcileRuntimeResources synchronizes native rendered resources and the
// serving endpoint. deployment and DGD are non-nil.
func (r *graphReconciler) reconcileRuntimeResources(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment, resources []client.Object) error {
	for _, resource := range resources {
		if _, _, err := commoncontroller.SyncResource(ctx, r, deployment, func(context.Context) (client.Object, bool, error) {
			return resource, false, nil
		}); err != nil {
			return err
		}
	}
	return r.reconcileEndpoint(ctx, deployment, dgd)
}

func (r *graphReconciler) reconcileEndpoint(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment) error {
	// One serving endpoint belongs to the engine, independently of mutable component names.
	pcsName := dynamo.PCSNameForLPX(deployment)
	serviceName := pcsName + "-serve"
	backend := commoncontroller.GetDiscoveryBackend(r.config.Discovery.Backend, dgd.Annotations)

	observed := &corev1.Service{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: serviceName}, observed); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		observed = nil
	}

	// Switching discovery backends removes only this deployment's endpoint.
	if backend != configv1alpha1.DiscoveryBackendKubernetes {
		if observed == nil {
			return nil
		}
		if err := commoncontroller.CheckControllerOwnership(observed, deployment, r.Scheme()); err != nil {
			return err
		}
		return client.IgnoreNotFound(r.Delete(ctx, observed, client.Preconditions{UID: &observed.UID, ResourceVersion: &observed.ResourceVersion}))
	}

	component := lpx.ServingComponent(dgd)
	service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName: serviceName, Namespace: deployment.Namespace,
		ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
		DynamoNamespace: dgd.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
		Labels:      dynamo.GetDGDComponentResourceLabels(dgd, component.ComponentName, component),
		Annotations: dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component),
	})
	if err != nil {
		return err
	}
	// Keep cleanup and pod selection scoped to this materialization.
	service.Labels[deploymentUIDLabel] = string(deployment.UID)
	service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
	service.Spec.Selector[grovecommon.LabelPartOfKey] = pcsName

	// Preserve API-allocated addresses and IP families across selector updates.
	if observed != nil {
		service.Spec.ClusterIP = observed.Spec.ClusterIP
		service.Spec.ClusterIPs = observed.Spec.ClusterIPs
		service.Spec.IPFamilies = observed.Spec.IPFamilies
		service.Spec.IPFamilyPolicy = observed.Spec.IPFamilyPolicy
	}
	_, _, err = commoncontroller.SyncObservedResource(ctx, r, deployment, observed, service, commoncontroller.WithMetadataSync())
	return err
}

// reconcileReadiness combines scheduler receipts with Grove runtime readiness.
// All pointer inputs are non-nil and refer to the same observed workload.
func (r *graphReconciler) reconcileReadiness(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
	requests map[string]*lpxv1alpha1.LPUPipelineRequest,
) (ctrl.Result, error) {
	readiness, err := dynamo.EvaluateLPXGroveReadiness(ctx, r.Client, dgd, pcs, pcsg)
	if err != nil {
		return ctrl.Result{}, err
	}

	deployment.Status.Components = readiness.ComponentStatuses

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

// deletePodCliqueSet deletes the already-validated, non-nil PCS.
func (r *graphReconciler) deletePodCliqueSet(ctx context.Context, pcs *grovev1alpha1.PodCliqueSet) error {
	return client.IgnoreNotFound(
		r.Delete(
			ctx,
			pcs,
			// Foreground GC so that blocking LPRs and Grove dependents can be removed.
			client.PropagationPolicy(metav1.DeletePropagationForeground),
			client.Preconditions{UID: &pcs.UID, ResourceVersion: &pcs.ResourceVersion},
		),
	)
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
