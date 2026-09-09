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
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
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

const lpxGraphDeploymentFinalizer = "nvidia.com/lpx-graph-deployment"

// graphReconciler owns the complete compiled engine lifecycle.
// The source DGD supplies intent, never an alternate status or owner identity.
type graphReconciler struct {
	client.Client
	recorder              events.EventRecorder
	apiReader             client.Reader
	runtimeConfig         *commoncontroller.RuntimeConfig
	modelRegistry         lpxModelRegistry
	Config                *configv1alpha1.OperatorConfiguration
	DockerSecretRetriever dynamo.SecretsRetriever
}

func (r *graphReconciler) GetRecorder() events.EventRecorder {
	return r.recorder
}

// Setup registers the LPX controller and its dependencies.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig, secrets dynamo.SecretsRetriever) error {
	// Disabled integration still serves child status and explicit deletion.
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
// Status and explicit deletion remain available regardless of this policy.
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
// +kubebuilder:rbac:groups=scheduler.grove.io,resources=podgangs,verbs=get;list;watch
// +kubebuilder:rbac:groups=scheduling.lpu.nvidia.com,resources=lpupipelinerequests,verbs=get;list;watch;create;update;delete

func (r *graphReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	deployment := &v1alpha1.LPXGraphDeployment{}
	if err := r.Get(ctx, req.NamespacedName, deployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !deployment.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, r.finalize(ctx, deployment)
	}

	unavailableReason := r.unavailableReason()
	previous := deployment.Status.DeepCopy()
	if deployment.Status.ObservedGeneration != deployment.Generation {
		deployment.Status.Components = nil
		deployment.Status.ModelDownload = nil
	}
	state := reconcileOutcome{State: v1beta1.DGDStatePending, Reason: "LPXPending", Message: "Waiting for the current LPX engine"}
	var deadlineAt time.Time
	defer func() {
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
			result.RequeueAfter = max(time.Second, time.Until(deployment.Status.ModelDownload.LastCheckedAt.Add(modelDownloadRefreshInterval)))
		}
		result, err = completeLPXDeadline(ctx, deadlineAt, result, err)
	}()

	// Like Grove, disabling integration reports unavailability without touching workloads.
	if unavailableReason != "" {
		state.State, state.Reason, state.Message = v1beta1.DGDStateFailed, "LPXUnavailable", unavailableReason
		return ctrl.Result{}, nil
	}

	source := &v1beta1.DynamoGraphDeployment{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: deployment.Name}, source); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		state = lpxResult(&lpxRejected{reason: "The exact source DGD no longer exists"})
		return r.retireInvalidWorkload(ctx, deployment, state.Message)
	}
	if sourceErr := dynamo.ValidateLPXSource(deployment, source); sourceErr != nil {
		state = lpxResult(&lpxRejected{reason: sourceErr.Error()})
		return r.retireInvalidWorkload(ctx, deployment, sourceErr.Error())
	}
	if !controllerutil.ContainsFinalizer(deployment, lpxGraphDeploymentFinalizer) {
		controllerutil.AddFinalizer(deployment, lpxGraphDeploymentFinalizer)
		return ctrl.Result{RequeueAfter: time.Nanosecond}, r.Update(ctx, deployment)
	}

	classification, wake, requests, err := r.reconcileLPXAttemptDeadline(ctx, deployment, source)
	deadlineAt = wake
	var selected *lpxMaterializing
	if classification == nil && err == nil {
		selected, classification, err = r.reconcileLPXSafetyPreflight(ctx, deployment, source, requests)
	}
	if err != nil {
		return ctrl.Result{}, err
	}
	if classification != nil {
		state, result, err = r.finishLPXPreflight(ctx, deployment, classification)
		return result, err
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
			state, result, err = r.finishLPXPreflight(ctx, deployment, classification)
			return result, err
		}
	}
	state, result, err = r.reconcileWorkload(ctx, deployment, source, selected)
	return result, err
}

// finishLPXPreflight projects a non-nil classification and retires rejected workloads.
func (r *graphReconciler) finishLPXPreflight(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, classification lpxClassification) (reconcileOutcome, ctrl.Result, error) {
	state := lpxResult(classification)
	if _, rejected := classification.(*lpxRejected); rejected {
		result, err := r.retireInvalidWorkload(ctx, deployment, state.Message)
		return state, result, err
	}
	return state, projectLPXLifecycleStatus(deployment, classification), nil
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
		retiring, fenceErr := r.retireInvalidLPXWorkload(ctx, deployment, fmt.Sprintf("The current LPX workload can no longer be rendered safely: %v", err))
		if fenceErr != nil {
			return state, ctrl.Result{}, errors.Join(err, fenceErr)
		}
		if retiring != nil {
			state = lpxResult(retiring)
			return state, projectLPXLifecycleStatus(deployment, retiring), nil
		}
		return state, ctrl.Result{}, err
	}
	if err := r.validateLPXPublicationSource(ctx, deployment); err != nil {
		return state, ctrl.Result{}, err
	}
	for _, resource := range resources {
		if err := r.syncLPXResource(ctx, deployment, resource); err != nil {
			return state, ctrl.Result{}, err
		}
	}
	synced, changed, err := r.reconcileGrovePodCliqueSetForLPX(ctx, deployment, pcs, desired, selected)
	if err != nil {
		var retiring *lpxRetiring
		if errors.As(err, &retiring) {
			state = lpxResult(retiring)
			return state, projectLPXLifecycleStatus(deployment, retiring), nil
		}
		if errors.Is(err, errLPXGrovePodCliqueSetRecreating) {
			return state, ctrl.Result{RequeueAfter: lpxRetirementRequeueAfter}, nil
		}
		return state, ctrl.Result{}, err
	}
	if err := r.reconcileEndpoint(ctx, deployment, source); err != nil {
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
		state.State = v1beta1.DGDStateSuccessful
	}
	classification, err := r.reconcileSelectedLPX(ctx, deployment, selected)
	if err != nil {
		return state, ctrl.Result{}, err
	}
	state = overlayLPXResult(state, classification)
	return state, projectLPXLifecycleStatus(deployment, classification), nil
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

func (r *graphReconciler) finalize(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment) error {
	if !controllerutil.ContainsFinalizer(deployment, lpxGraphDeploymentFinalizer) {
		return nil
	}
	if err := r.finalizeLPXRequests(ctx, deployment); err != nil {
		return err
	}
	controllerutil.RemoveFinalizer(deployment, lpxGraphDeploymentFinalizer)
	return r.Update(ctx, deployment)
}

// Publication is an identity-sensitive write boundary: an uncached read fences
// a source/child replacement or input edit, not an attempt to repair informer lag.
func (r *graphReconciler) validateLPXPublicationSource(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment) error {
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return err
	}
	source := &v1beta1.DynamoGraphDeployment{}
	if err := r.apiReader.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: deployment.Name}, source); err != nil {
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
	if current.UID != deployment.UID || current.Generation != deployment.Generation || !apiequality.Semantic.DeepEqual(current.Spec, deployment.Spec) || !apiequality.Semantic.DeepEqual(current.OwnerReferences, deployment.OwnerReferences) || !current.DeletionTimestamp.IsZero() {
		return fmt.Errorf("LPXGraphDeployment authority changed before LPX resource mutation")
	}
	for _, key := range []string{dynamo.LPXRestartAnnotation, lpx.DGDGenerationAnnotation} {
		if current.Annotations[key] != deployment.Annotations[key] {
			return fmt.Errorf("LPXGraphDeployment publication metadata changed before LPX resource mutation")
		}
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

func (r *graphReconciler) reconcileEndpoint(ctx context.Context, deployment *v1alpha1.LPXGraphDeployment, source *v1beta1.DynamoGraphDeployment) error {
	component := lpx.ServingComponent(source)
	if !commoncontroller.IsK8sDiscoveryEnabled(r.Config.Discovery.Backend, source.Annotations) {
		service := &corev1.Service{}
		key := client.ObjectKey{Namespace: source.Namespace, Name: dynamo.NormalizeKubeResourceName(dynamo.GetDCDResourceName(source, component.ComponentName, ""))}
		if err := r.Get(ctx, key, service); err != nil {
			return client.IgnoreNotFound(err)
		}
		if !metav1.IsControlledBy(service, deployment) {
			return fmt.Errorf("refusing to delete a foreign LPX endpoint %q", service.Name)
		}
		return client.IgnoreNotFound(r.Delete(ctx, service, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &service.UID, ResourceVersion: &service.ResourceVersion}}))
	}
	service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName: dynamo.GetDCDResourceName(source, component.ComponentName, ""), Namespace: source.Namespace,
		ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
		DynamoNamespace: source.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
		Labels:      dynamo.GetDGDComponentResourceLabels(source, component.ComponentName, component),
		Annotations: dynamo.GetDGDComponentResourceAnnotations(source, component.ComponentName, component),
	})
	if err != nil {
		return err
	}
	service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
	return r.syncLPXResource(ctx, deployment, service)
}
