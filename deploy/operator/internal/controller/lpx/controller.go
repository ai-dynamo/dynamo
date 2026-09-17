// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const legacyLPXGraphDeploymentFinalizer = "nvidia.com/lpx-graph-deployment"

func lpxSourceKey(deployment *v1alpha1.LPXGraphDeployment) (client.ObjectKey, error) {
	owner := metav1.GetControllerOf(deployment)
	if owner == nil || owner.APIVersion != v1beta1.GroupVersion.String() || owner.Kind != "DynamoGraphDeployment" || owner.Name == "" {
		return client.ObjectKey{}, fmt.Errorf("LPXGraphDeployment requires a DynamoGraphDeployment controller owner")
	}
	return client.ObjectKey{Namespace: deployment.Namespace, Name: owner.Name}, nil
}

func (r *graphReconciler) resolveLPXSource(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment) (*v1beta1.DynamoGraphDeployment, *lpxRejected, error) {
	sourceKey, err := lpxSourceKey(deployment)
	if err != nil {
		return nil, &lpxRejected{reason: err.Error()}, nil
	}
	source := &v1beta1.DynamoGraphDeployment{}
	if err := r.Get(ctx, sourceKey, source); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, &lpxRejected{reason: "The exact source DGD no longer exists"}, nil
		}
		return nil, nil, err
	}
	if err := dynamo.ValidateLPXSource(deployment, source); err != nil {
		return source, nil, err
	}
	return source, nil, nil
}

// graphReconciler owns the complete compiled engine lifecycle.
// The source DGD supplies intent, never an alternate status or owner identity.
type graphReconciler struct {
	client.Client
	recorder              events.EventRecorder
	apiReader             client.Reader
	runtimeConfig         *commoncontroller.RuntimeConfig
	modelRegistry         lpx.ModelRegistry
	Config                *configv1alpha1.OperatorConfiguration
	DockerSecretRetriever dynamo.SecretsRetriever
}

func (r *graphReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

// Setup registers the LPX controller and its dependencies.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig, secrets dynamo.SecretsRetriever) error {
	// Disabled integration still serves child status; deletion uses owner garbage collection.
	r := &graphReconciler{
		Client:                mgr.GetClient(),
		recorder:              mgr.GetEventRecorder("lpxgraphdeployment"),
		apiReader:             mgr.GetAPIReader(),
		runtimeConfig:         runtimeConfig,
		Config:                config,
		DockerSecretRetriever: secrets,
	}

	// Resolve availability before activating any runtime dependencies.
	enabled := r.unavailableReason() == ""
	if enabled {
		var err error
		r.modelRegistry, err = newLPXModelRegistry(config)
		if err != nil {
			return err
		}
	}

	if err := r.setupWithManager(mgr); err != nil {
		return fmt.Errorf("unable to create LPXGraphDeployment controller: %w", err)
	}
	if enabled {
		if err := setupLPUEviction(mgr); err != nil {
			return fmt.Errorf("unable to create LPU Eviction controller: %w", err)
		}
	}
	return nil
}

// unavailableReason owns the LPX activation policy; an empty reason means enabled.
// Status remains available regardless of this policy.
func (r *graphReconciler) unavailableReason() string {
	if !r.runtimeConfig.Gate.Enabled(features.Grove) {
		return "Grove is disabled"
	}
	if !r.runtimeConfig.Gate.Enabled(features.LPX) {
		return "LPX integration is disabled"
	}
	return ""
}

// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=lpxgraphdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=scheduling.lpu.nvidia.com,resources=lpupipelinerequests,verbs=get;list;watch;create;delete

func (r *graphReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	deployment := &v1alpha1.LPXGraphDeployment{}
	if err := r.Get(ctx, req.NamespacedName, deployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !deployment.DeletionTimestamp.IsZero() {
		if controllerutil.RemoveFinalizer(deployment, legacyLPXGraphDeploymentFinalizer) {
			return ctrl.Result{}, r.Update(ctx, deployment)
		}
		return ctrl.Result{}, nil
	}

	unavailableReason := r.unavailableReason()
	previous := deployment.Status.DeepCopy()
	if deployment.Status.ObservedGeneration != deployment.Generation {
		deployment.Status.Components = nil
		deployment.Status.ModelDownload = nil
	}
	state := reconcileOutcome{State: v1beta1.DGDStatePending, Reason: "LPXPending", Message: "Waiting for the current LPX engine"}
	var deadlineAt time.Time
	var source *v1beta1.DynamoGraphDeployment
	defer func() {
		// Reconcile intended removals before deadline cleanup, including on failed renders.
		if unavailableReason == "" {
			classification, wake, _, deadlineErr := r.reconcileLPXAttemptDeadline(ctx, deployment, source)
			deadlineAt, err = wake, errors.Join(err, deadlineErr)
			if classification != nil {
				state, result = lpxResult(classification), projectLPXLifecycleStatus(deployment, classification)
			}
		}
		if err != nil {
			state.State, state.Reason, state.Message = v1beta1.DGDStateFailed, "LPXReconciliationFailed", truncateLPXMessage(err.Error())
		} else if unavailableReason == "" {
			deployment.Status.ObservedGeneration = deployment.Generation
		}
		ready, failed := metav1.ConditionFalse, metav1.ConditionFalse
		if state.State == v1beta1.DGDStateSuccessful {
			ready = metav1.ConditionTrue
		}
		if state.State == v1beta1.DGDStateFailed {
			failed = metav1.ConditionTrue
		}
		// Every authored component observes the complete shared runtime gate.
		for name, component := range deployment.Status.Components {
			component.Ready = ready == metav1.ConditionTrue
			deployment.Status.Components[name] = component
		}
		for condition, value := range map[string]metav1.ConditionStatus{"Ready": ready, "Failed": failed} {
			meta.SetStatusCondition(&deployment.Status.Conditions, metav1.Condition{Type: condition, Status: value, ObservedGeneration: deployment.Generation, Reason: state.Reason, Message: state.Message})
		}
		if !apiequality.Semantic.DeepEqual(previous, &deployment.Status) {
			if statusErr := r.Status().Update(ctx, deployment); statusErr != nil {
				err = errors.Join(err, fmt.Errorf("persist LPX lifecycle status: %w", statusErr))
			}
		}
		if ready == metav1.ConditionTrue && deployment.Status.ModelDownload != nil && deployment.Status.ModelDownload.LastCheckedAt != nil && result.RequeueAfter == 0 {
			result.RequeueAfter = max(modelDownloadRequeueAfter, time.Until(deployment.Status.ModelDownload.LastCheckedAt.Add(modelDownloadRefreshInterval)))
		}
		result, err = completeLPXDeadline(ctx, deadlineAt, result, err)
	}()

	// Like Grove, disabling integration reports unavailability without touching workloads.
	if unavailableReason != "" {
		state.State, state.Reason, state.Message = v1beta1.DGDStateFailed, "LPXUnavailable", unavailableReason
		return ctrl.Result{}, nil
	}

	var rejected *lpxRejected
	var sourceErr error
	source, rejected, sourceErr = r.resolveLPXSource(ctx, deployment)
	if sourceErr != nil {
		return ctrl.Result{}, sourceErr
	}
	if rejected != nil {
		state = lpxResult(rejected)
		return r.retireInvalidWorkload(ctx, deployment, rejected.reason)
	}
	selected, classification, err := r.reconcileLPXSafetyPreflight(ctx, deployment, source, nil)
	if err != nil {
		return ctrl.Result{}, err
	}
	if classification != nil {
		state = lpxResult(classification)
		return projectLPXLifecycleStatus(deployment, classification), nil
	}
	ready, err := r.reconcileModelDownloads(ctx, deployment, source, selected == nil)
	if err != nil {
		return ctrl.Result{}, err
	}
	if !ready {
		state.Reason, state.Message = modelDownloadPendingReason, modelDownloadPendingMessage
		return ctrl.Result{RequeueAfter: modelDownloadRequeueAfter}, nil
	}
	if selected == nil {
		selected, classification, err = r.reconcileSelectedLPXSafetyPreflight(ctx, deployment, source)
		if err != nil {
			return ctrl.Result{}, err
		}
		if classification != nil {
			state = lpxResult(classification)
			return projectLPXLifecycleStatus(deployment, classification), nil
		}
	}
	state, result, err = r.reconcileWorkload(ctx, deployment, source, selected)
	return result, err
}

// reconcileWorkload publishes and observes the selected engine after source,
// deadline, placement, and model-download gates have passed. deployment, source
// and selected are non-nil and supplied by that preflight.
func (r *graphReconciler) reconcileWorkload(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, source *v1beta1.DynamoGraphDeployment, selected *lpxMaterializing) (reconcileOutcome, ctrl.Result, error) {
	state := reconcileOutcome{State: v1beta1.DGDStatePending, Reason: "LPXPending", Message: "Waiting for the current LPX engine"}

	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: selected.plan.PodCliqueSetName}, pcs); err != nil {
		if !apierrors.IsNotFound(err) {
			return state, ctrl.Result{}, err
		}
		pcs = nil
	}
	desired, resources, err := renderPodCliqueSet(ctx, source, r.Config, r.runtimeConfig, r.Client, r.DockerSecretRetriever, selected.workload, selected.plan, deployment)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return state, ctrl.Result{}, err
	}
	// Scale the live Grove group before stale request retirement. LPX scheduler
	// finalizers may wait for Grove to remove the corresponding pods, so the
	// publication fence must not block this write.
	if err := r.reconcileLPXScalingGroupScale(ctx, source, pcs, selected.plan.LPXScalingGroup, true); err != nil {
		return state, ctrl.Result{}, err
	}
	currents, classification, err := r.reconcileLPXPublicationFence(ctx, deployment, selected, pcs == nil)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	if classification != nil {
		state = lpxResult(classification)
		return state, projectLPXLifecycleStatus(deployment, classification), nil
	}
	for _, resource := range resources {
		if err := r.syncLPXResource(ctx, deployment, resource); err != nil {
			return state, ctrl.Result{}, err
		}
	}
	// Grove seeds native scale only at creation; preserve that seed when replicas are omitted.
	if pcs != nil && lpx.ServingComponent(source).Replicas == nil {
		group := &desired.Spec.Template.PodCliqueScalingGroupConfigs[0]
		for _, existing := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
			if existing.Name == group.Name {
				group.Replicas = existing.Replicas
				break
			}
		}
	}

	changed, synced, err := commoncontroller.SyncObservedResource(
		ctx,
		r,
		deployment,
		pcs,
		desired,
		commoncontroller.WithPreservedListOrder(),
	)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	if err := r.reconcileEndpoint(ctx, deployment, source); err != nil {
		return state, ctrl.Result{}, err
	}
	if err := r.reconcileLPXScalingGroupScale(ctx, source, synced, selected.plan.LPXScalingGroup, false); err != nil {
		return state, ctrl.Result{}, err
	}
	readiness, err := dynamo.EvaluateLPXGroveReadiness(ctx, r.Client, source, deployment, synced)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	deployment.Status.Components = readiness.ComponentStatuses
	shape, err := dynamo.ResolveLPXGPUShape(ctx, r.Client, synced)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	// The serving component owns shared GPU costs exactly once.
	name := selected.workload.LPXComponentName()
	component := deployment.Status.Components[name]
	component.GPUsPerEngine, component.GPUsPerReplica = ptr.To(shape.GPUsPerEngine), ptr.To(shape.GPUsPerReplica)
	deployment.Status.Components[name] = component
	state.Reason, state.Message = readiness.Classification, readiness.Message
	if readiness.Ready && !changed {
		if err := r.deleteStaleLPXConfigMaps(ctx, deployment, resources); err != nil {
			return state, ctrl.Result{}, err
		}
		state.State = v1beta1.DGDStateSuccessful
	}
	classification, err = r.reconcileSelectedLPXFromCurrentRequests(ctx, deployment, selected, currents)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	state = overlayLPXResult(state, classification)
	return state, projectLPXLifecycleStatus(deployment, classification), nil
}

// reconcileLPXScalingGroupScale applies explicit component replicas to the
// Grove group. Grove seeds this group once, so later replica changes use its
// scale subresource without rewriting the PCS template.
func (r *graphReconciler) reconcileLPXScalingGroupScale(
	ctx context.Context,
	source *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	groupName string,
	scaleDownOnly bool,
) error {
	replicas := lpx.ServingComponent(source).Replicas
	if replicas == nil || pcs == nil {
		return nil
	}
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: pcs.Namespace, Name: groupName}, group); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !metav1.IsControlledBy(group, pcs) || !group.DeletionTimestamp.IsZero() {
		return fmt.Errorf("LPX scaling group lacks the current PCS owner")
	}
	if group.Spec.Replicas == *replicas || scaleDownOnly && group.Spec.Replicas < *replicas {
		return nil
	}
	scale := &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{ResourceVersion: group.ResourceVersion},
		Spec:       autoscalingv1.ScaleSpec{Replicas: *replicas},
	}
	return r.SubResource("scale").Update(ctx, group, client.WithSubResourceBody(scale))
}

// projectLPXLifecycleStatus transfers reconcile-owned results to the same child before returning.
func projectLPXLifecycleStatus(deployment *v1alpha1.LPXGraphDeployment, classification lpxClassification) ctrl.Result {
	if transition, ok := classification.(*lpxDeadlineTransition); ok {
		if deployment.Status.Placement == nil {
			deployment.Status.Placement = &v1beta1.PlacementStatus{}
		}
		deployment.Status.Placement.LPXAttempt = transition.attempt
		return ctrl.Result{RequeueAfter: transition.requeueAfter}
	}
	if _, retiring := classification.(*lpxRetiring); retiring {
		return ctrl.Result{RequeueAfter: lpxRetirementRequeueAfter}
	}
	return ctrl.Result{}
}

func (r *graphReconciler) retireInvalidWorkload(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, reason string) (ctrl.Result, error) {
	retiring, err := r.retireInvalidLPXWorkload(ctx, deployment, reason)
	if err != nil {
		return ctrl.Result{}, err
	}
	if retiring != nil {
		return ctrl.Result{RequeueAfter: lpxRetirementRequeueAfter}, nil
	}
	return ctrl.Result{}, nil
}

// Publication is an identity-sensitive write boundary: an uncached read fences
// a source/child replacement or input edit, not an attempt to repair informer lag.
func (r *graphReconciler) validateLPXPublicationSource(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment) error {
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return err
	}
	source := &v1beta1.DynamoGraphDeployment{}
	sourceKey, err := lpxSourceKey(deployment)
	if err != nil {
		return err
	}
	if err := r.apiReader.Get(ctx, sourceKey, source); err != nil {
		return err
	}
	return dynamo.ValidateLPXSource(deployment, source)
}

// validateLPXDeploymentAuthority fences writes against a replaced or newer child,
// independently of whether its source remains valid for publication.
func (r *graphReconciler) validateLPXDeploymentAuthority(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment) error {
	current := &v1alpha1.LPXGraphDeployment{}
	if err := r.apiReader.Get(ctx, client.ObjectKeyFromObject(deployment), current); err != nil {
		return err
	}
	if current.UID != deployment.UID || current.ResourceVersion != deployment.ResourceVersion || !current.DeletionTimestamp.IsZero() {
		return fmt.Errorf("LPXGraphDeployment authority changed before LPX resource mutation")
	}
	return nil
}

func (r *graphReconciler) syncLPXResource(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, desired client.Object) error {
	// Observe the resource once and reject foreign ownership before preparing a write.
	live := desired.DeepCopyObject().(client.Object)
	if err := r.Get(ctx, client.ObjectKeyFromObject(desired), live); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		live = nil
	} else if !metav1.IsControlledBy(live, deployment) {
		return fmt.Errorf("refusing to adopt LPX resource %s/%s without its exact owner", live.GetNamespace(), live.GetName())
	}

	// Prepare LPX-owned metadata while preserving shared synchronization's bookkeeping.
	metadataChanged := false
	if live != nil {
		labels, annotations := desired.GetLabels(), maps.Clone(desired.GetAnnotations())
		if annotations == nil {
			annotations = make(map[string]string)
		}
		for _, key := range []string{commoncontroller.NvidiaAnnotationHashKey, commoncontroller.NvidiaAnnotationGenerationKey} {
			annotations[key] = live.GetAnnotations()[key]
		}
		metadataChanged = !maps.Equal(labels, live.GetLabels()) || !maps.Equal(annotations, live.GetAnnotations())
		if metadataChanged {
			live.SetLabels(maps.Clone(labels))
			live.SetAnnotations(annotations)
		}
	}

	// Content updates carry prepared metadata; only metadata-only drift needs another path.
	modified, synced, err := commoncontroller.SyncObservedResource(ctx, r, deployment, live, desired)
	if err != nil || modified || !metadataChanged {
		return err
	}
	return r.Update(ctx, synced)
}

func (r *graphReconciler) deleteStaleLPXConfigMaps(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, resources []client.Object) error {
	// Retain every ConfigMap rendered for the current workload.
	desiredNames := make(map[string]struct{}, len(resources))
	for _, resource := range resources {
		if _, ok := resource.(*corev1.ConfigMap); ok {
			desiredNames[resource.GetName()] = struct{}{}
		}
	}

	// Discover obsolete ConfigMaps rooted in this exact LPX child.
	stale := make([]client.Object, 0)
	configMaps := &corev1.ConfigMapList{}
	if err := r.apiReader.List(ctx, configMaps,
		client.InNamespace(deployment.Namespace),
		client.MatchingLabels{lpxOwnerUIDLabel: string(deployment.UID)},
	); err != nil {
		return err
	}
	for index := range configMaps.Items {
		configMap := &configMaps.Items[index]
		if metav1.IsControlledBy(configMap, deployment) {
			if _, desired := desiredNames[configMap.GetName()]; !desired {
				stale = append(stale, configMap)
			}
		}
	}
	if len(stale) == 0 {
		return nil
	}

	// Revalidate publication authority before deleting the discovered objects.
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return err
	}

	// Preconditions prevent stale observations from deleting replacements.
	for _, configMap := range stale {
		if !configMap.GetDeletionTimestamp().IsZero() {
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

func (r *graphReconciler) reconcileEndpoint(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, source *v1beta1.DynamoGraphDeployment) error {
	// One serving endpoint belongs to the engine, independently of mutable component names.
	pcsName := dynamo.PCSNameForLPX(deployment)
	serviceName := pcsName + "-serve"

	if !commoncontroller.IsK8sDiscoveryEnabled(r.Config.Discovery.Backend, source.Annotations) {
		service := &corev1.Service{}
		key := client.ObjectKey{Namespace: deployment.Namespace, Name: serviceName}
		if err := r.Get(ctx, key, service); err != nil {
			return client.IgnoreNotFound(err)
		}
		if !metav1.IsControlledBy(service, deployment) {
			return fmt.Errorf("refusing to delete a foreign LPX endpoint %q", service.Name)
		}
		return client.IgnoreNotFound(r.Delete(ctx, service, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &service.UID, ResourceVersion: &service.ResourceVersion}}))
	}

	// Source component metadata is needed only when publishing the serving endpoint.
	component := lpx.ServingComponent(source)
	service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName: serviceName, Namespace: deployment.Namespace,
		ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
		DynamoNamespace: source.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
		Labels:      dynamo.GetDGDComponentResourceLabels(source, component.ComponentName, component),
		Annotations: dynamo.GetDGDComponentResourceAnnotations(source, component.ComponentName, component),
	})
	if err != nil {
		return err
	}
	// Keep cleanup and pod selection scoped to this materialization.
	service.Labels[lpxOwnerUIDLabel] = string(deployment.UID)
	service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
	service.Spec.Selector[grovecommon.LabelPartOfKey] = pcsName
	return r.syncLPXResource(ctx, deployment, service)
}
