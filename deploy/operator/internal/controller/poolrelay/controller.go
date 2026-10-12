/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// relayReconciler owns relay infrastructure independently of inference DGD lifecycles.
type relayReconciler struct {
	client.Client
	config        *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
}

// Setup registers the Pool Relay controller and its dependency watches.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig) error {
	// Bind the controller to the manager's cache and shared operator configuration.
	r := &relayReconciler{
		Client:        mgr.GetClient(),
		config:        config,
		runtimeConfig: runtimeConfig,
	}

	return r.setupWithManager(mgr)
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=replicasets,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services;serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamoworkermetadatas,verbs=get;list;watch;create;patch;delete
// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch

// Reconcile converges owned resources and records transport availability for this generation.
func (r *relayReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Owner-reference garbage collection handles deletion without touching inference workers.
	relay := &v1alpha1.DynamoPoolRelay{}
	if err := r.Get(ctx, req.NamespacedName, relay); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !relay.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	baseline := relay.DeepCopy()
	names := make([]string, 0, len(relay.Spec.Sources.DGDs.Refs))
	for _, ref := range relay.Spec.Sources.DGDs.Refs {
		names = append(names, ref.Name)
	}
	selected, missing, err := selectRelayDGDs(ctx, r.Client, relay.Namespace, relay.Spec.Sources.DGDs.LabelSelector, names)
	if err != nil {
		return ctrl.Result{}, err
	}
	relay.Status.SelectedDGDs = int32(len(selected))
	selection := metav1.Condition{Type: "SelectionResolved", Status: metav1.ConditionTrue, Reason: "Selected", Message: "DGD selection resolved", ObservedGeneration: relay.Generation}
	if len(missing) > 0 {
		selection.Status = metav1.ConditionFalse
		selection.Reason = "MissingDGD"
		selection.Message = "Missing DGD references: " + strings.Join(missing, ", ")
	} else if len(selected) == 0 {
		selection.Reason = "WaitingForDGD"
		selection.Message = "No DGDs currently match the selection"
	}
	meta.SetStatusCondition(&relay.Status.Conditions, selection)
	connection, sourceEntries, resolveErr := resolveRelayConnections(ctx, r.Client, r.config, selected)
	connectionRevision := ""
	if resolveErr == nil && len(selected) == 0 {
		connection, connectionRevision, resolveErr = retainRelayConnection(ctx, r.Client, relay, connection)
	}
	if resolveErr == nil {
		var preview []client.Object
		preview, resolveErr = renderRelayResources(relay, connection, newRelaySources(connectionRevision, sourceEntries))
		if resolveErr == nil {
			resolveErr = r.validateRelayWorkload(ctx, preview)
		}
	}
	configuration := metav1.Condition{Type: "ConfigurationResolved", Status: metav1.ConditionTrue, Reason: "Compatible", Message: "Selected workers use a compatible connection", ObservedGeneration: relay.Generation}
	if resolveErr != nil || len(missing) > 0 {
		configuration.Status = metav1.ConditionFalse
		configuration.Reason = "ConfigurationUnresolved"
		configuration.Message = "Waiting for all explicit DGD references"
		if resolveErr != nil {
			configuration.Message = resolveErr.Error()
		}
		meta.SetStatusCondition(&relay.Status.Conditions, configuration)
		if err := r.withdrawDeselectedRelaySources(ctx, relay, selected); err != nil {
			return ctrl.Result{}, err
		}
		condition := metav1.Condition{Type: "Ready", Status: metav1.ConditionFalse, Reason: "ConfigurationUnresolved", Message: "Desired source configuration has not been applied", ObservedGeneration: relay.Generation}
		return ctrl.Result{RequeueAfter: 15 * time.Second}, r.updateRelayStatus(ctx, relay, baseline, condition, false)
	}
	meta.SetStatusCondition(&relay.Status.Conditions, configuration)
	if connectionRevision == "" {
		connectionRevision, err = r.relayConnectionRevision(ctx, relay, connection)
		if err != nil {
			return ctrl.Result{}, err
		}
	}
	sources := newRelaySources(connectionRevision, sourceEntries)

	// Apply dependencies before the workload; retain the write result rather than rereading the cache.
	resources, err := renderRelayResources(relay, connection, sources)
	if err != nil {
		return ctrl.Result{}, err
	}
	for _, desired := range resources {
		if err := r.reconcileRelayResource(ctx, relay, desired); err != nil {
			condition := metav1.Condition{Type: "Ready", Status: metav1.ConditionFalse, Reason: "ReconcileFailed", Message: err.Error(), ObservedGeneration: relay.Generation}
			return ctrl.Result{}, errors.Join(err, r.updateRelayStatus(ctx, relay, baseline, condition, false))
		}
	}
	relay.Status.DesiredSourcesRevision = sources.Revision
	deployment := resources[len(resources)-1].(*appsv1.Deployment)

	sourcesCondition, err := r.observeRelaySources(ctx, relay, deployment, sources)
	if err != nil {
		return ctrl.Result{}, err
	}
	meta.SetStatusCondition(&relay.Status.Conditions, sourcesCondition)

	// A ready previous ReplicaSet must not make an incomplete active-active rollout Ready.
	relay.Status.Replicas = deployment.Status.Replicas
	relay.Status.UpdatedReplicas = deployment.Status.UpdatedReplicas
	relay.Status.AvailableReplicas = deployment.Status.AvailableReplicas
	desiredReplicas := *deployment.Spec.Replicas
	ready := deployment.Status.ObservedGeneration >= deployment.Generation &&
		deployment.Status.UpdatedReplicas == desiredReplicas && deployment.Status.Replicas == desiredReplicas &&
		deployment.Status.AvailableReplicas == desiredReplicas
	condition := metav1.Condition{Type: "Ready", Status: metav1.ConditionFalse, Reason: "DeploymentProgressing",
		Message: "Waiting for all requested current Relay replicas and gRPC listeners; pool and model readiness are not evaluated", ObservedGeneration: relay.Generation}
	if ready {
		condition.Status = metav1.ConditionTrue
		condition.Reason = "ListenerAvailable"
		condition.Message = "All requested current Relay replicas and gRPC listeners are available; pool and model readiness are not evaluated"
	}
	if err := r.updateRelayStatus(ctx, relay, baseline, condition, true); err != nil {
		return ctrl.Result{}, err
	}
	if !ready {
		return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
	}
	return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
}

// reconcileRelayResource updates only owned fields, preserving API defaults and service allocation.
// relay and desired must be non-nil. desired receives the persisted object and its status.
func (r *relayReconciler) reconcileRelayResource(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, desired client.Object) error {
	if deployment, ok := desired.(*appsv1.Deployment); ok {
		return r.reconcileRelayDeployment(ctx, relay, deployment)
	}

	// Create from a cached miss; an AlreadyExists race is retried without adoption.
	target := desired.DeepCopyObject().(client.Object)
	if err := r.Get(ctx, client.ObjectKeyFromObject(desired), desired); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		if err := controllerutil.SetControllerReference(relay, desired, r.Scheme()); err != nil {
			return err
		}
		return r.Create(ctx, desired)
	}

	// Reuse the operator ownership contract before modifying the observed object.
	if err := commoncontroller.CheckControllerOwnership(desired, relay, r.Scheme()); err != nil {
		return fmt.Errorf("resource %T %s/%s is not owned by this DynamoPoolRelay: %w", desired, desired.GetNamespace(), desired.GetName(), err)
	}
	if !desired.GetDeletionTimestamp().IsZero() {
		return fmt.Errorf("resource %T %s/%s is being deleted", desired, desired.GetNamespace(), desired.GetName())
	}
	before := desired.DeepCopyObject().(client.Object)
	if err := controllerutil.SetControllerReference(relay, desired, r.Scheme()); err != nil {
		return err
	}

	// Preserve unrelated metadata while correcting labels required for discovery.
	if len(target.GetLabels()) > 0 {
		labels := desired.GetLabels()
		if labels == nil {
			labels = map[string]string{}
		}
		maps.Copy(labels, target.GetLabels())
		desired.SetLabels(labels)
	}
	switch current := desired.(type) {
	case *corev1.ConfigMap:
		current.Data = target.(*corev1.ConfigMap).Data
	case *corev1.ServiceAccount:
		// Runtime credentials are supplied by Kubernetes using this account.
	case *rbacv1.Role:
		current.Rules = target.(*rbacv1.Role).Rules
	case *rbacv1.RoleBinding:
		spec := target.(*rbacv1.RoleBinding)
		current.RoleRef, current.Subjects = spec.RoleRef, spec.Subjects
	case *corev1.Service:
		spec := target.(*corev1.Service).Spec
		current.Spec.Type, current.Spec.Selector, current.Spec.Ports = spec.Type, spec.Selector, spec.Ports
	default:
		return fmt.Errorf("unsupported relay resource %T", desired)
	}

	// Fence the write by resource version so replacement objects remain untouched.
	if equality.Semantic.DeepEqual(before, desired) {
		return nil
	}
	patch := client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{})
	if err := r.Patch(ctx, desired, patch); err != nil {
		return fmt.Errorf("reconcile %T %s: %w", desired, desired.GetName(), err)
	}
	return nil
}

// updateRelayStatus avoids writes on no-op reconciliations. relay must be non-nil.
func (r *relayReconciler) updateRelayStatus(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, before *v1alpha1.DynamoPoolRelay, condition metav1.Condition, reconciled bool) error {
	// Preserve the last successfully applied generation on resource reconciliation failures.
	meta.SetStatusCondition(&relay.Status.Conditions, condition)
	if reconciled {
		relay.Status.ObservedGeneration = relay.Generation
		relay.Status.DeploymentName = relayResourceName(relay.Name)
		port := int32(5561)
		if relay.Spec.Service != nil {
			for _, candidate := range relay.Spec.Service.Ports {
				if candidate.Name == "grpc" {
					port = candidate.Port
				}
			}
		}
		relay.Status.Endpoint = fmt.Sprintf("%s.%s.svc:%d", relay.Status.DeploymentName, relay.Namespace, port)
	}
	// Bound diagnostics independently of the number or lengths of selected objects.
	for i := range relay.Status.Conditions {
		message := []rune(relay.Status.Conditions[i].Message)
		if len(message) > 1024 {
			relay.Status.Conditions[i].Message = string(message[:1021]) + "..."
		}
	}
	if equality.Semantic.DeepEqual(before.Status, relay.Status) {
		return nil
	}
	return r.Status().Patch(ctx, relay, client.MergeFrom(before))
}

// setupWithManager registers all owned-resource watches and the operator namespace filter.
func (r *relayReconciler) setupWithManager(mgr ctrl.Manager) error {
	if err := mgr.GetFieldIndexer().IndexField(context.Background(), &v1beta1.DynamoGraphDeployment{}, relayDependencyIndex, relayDGDDependencies); err != nil {
		return err
	}
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DynamoPoolRelay{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Owns(&appsv1.Deployment{}).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(r.relaysForPod)).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Watches(&v1beta1.DynamoGraphDeployment{}, handler.EnqueueRequestsFromMapFunc(r.relaysForDGD)).
		Watches(&corev1.Secret{}, handler.EnqueueRequestsFromMapFunc(r.relaysForDependency)).
		Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(r.relaysForDependency)).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.config, r.runtimeConfig)).
		Complete(r)
}

// withdrawDeselectedRelaySources preserves the last valid connection but never
// keeps explicitly deselected DGD incarnations while unrelated additions are blocked.
func (r *relayReconciler) withdrawDeselectedRelaySources(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, selected []v1beta1.DynamoGraphDeployment) error {
	// Never leave a successful acknowledgement attached to a newly withdrawn sources.
	meta.SetStatusCondition(&relay.Status.Conditions, metav1.Condition{Type: "SourcesApplied", Status: metav1.ConditionFalse, Reason: "WaitingForAcknowledgement", Message: "Waiting for acknowledgement of the retained source set", ObservedGeneration: relay.Generation})
	relay.Status.AppliedSourcesRevision = ""
	relay.Status.SourcesAppliedReplicas = 0
	var current corev1.ConfigMap
	if err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: relayResourceName(relay.Name)}, &current); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(&current, relay) {
		return fmt.Errorf("sources ConfigMap is not owned by this DynamoPoolRelay")
	}
	var sources relaySources
	if err := json.Unmarshal([]byte(current.Data["sources.json"]), &sources); err != nil {
		return fmt.Errorf("existing Relay sources is malformed")
	}
	allowed := map[relaySource]bool{}
	for i := range selected {
		for _, source := range relayDGDSources(&selected[i]) {
			allowed[source] = true
		}
	}
	kept := make([]relaySource, 0, len(sources.Sources))
	for _, source := range sources.Sources {
		if allowed[source] {
			kept = append(kept, source)
		}
	}
	desired := newRelaySources(sources.ConnectionRevision, kept)
	data, _ := json.Marshal(desired)
	current.Data = map[string]string{"sources.json": string(data)}
	if err := r.reconcileRelayResource(ctx, relay, &current); err != nil {
		return err
	}
	relay.Status.DesiredSourcesRevision = desired.Revision

	// Observe the withdrawal even while an incompatible addition blocks configuration.
	var deployment appsv1.Deployment
	if err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: relayResourceName(relay.Name)}, &deployment); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(&deployment, relay) {
		return fmt.Errorf("existing Deployment is not owned by this DynamoPoolRelay")
	}
	condition, err := r.observeRelaySources(ctx, relay, &deployment, desired)
	if err != nil {
		return err
	}
	meta.SetStatusCondition(&relay.Status.Conditions, condition)
	return nil
}
