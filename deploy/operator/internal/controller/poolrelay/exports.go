/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"unicode/utf8"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/multicluster"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// The pool's exports: when the multiClusterExport gate is on, the Relay
// controller exports its DGD's PoolRelay and frontend Services to the other
// clusters of the ClusterSet through the configured Multi-Cluster Services (MCS)
// implementation. The Relay owns the exports, which follow its own decision: they
// are reconciled while it publishes its Service for the DGD, and withdrawn
// whenever it withdraws that Service.

// +kubebuilder:rbac:groups=multicluster.x-k8s.io,resources=serviceexports,verbs=get;list;watch;create;patch;delete
// +kubebuilder:rbac:groups=about.k8s.io,resources=clusterproperties,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocomponentdeployments,verbs=get;list;watch

// exportFieldManager is the field manager of the controller's writes to
// ServiceExports, as of its other writes.
const exportFieldManager = "dynamo-pool-relay"

// The Relay's Exported condition reports on the pool's exports: True once the
// Services that the pool can export are exported, False with reason Withdrawn
// while the Relay is withdrawn, and otherwise False with the reason of the pool's
// first problem. Apart from a failure, the problems are the ones that only the
// operator can see; the MCS implementation reports on each export in the export's
// own status.
//
// The controller records a Warning event for each problem when the condition
// changes, not on every reconcile. A problem's event names the DGD or
// ClusterProperty that it is about, at the version read, so that a problem with
// new details is not merged into the old one: the events API counts an event
// with the same object, reason, action and related object as a repeat, whatever
// its note. Normal events record each export that the controller creates,
// repairs or deletes, and name that export.
const (
	conditionExported = "Exported"

	reasonExported           = "Exported"
	reasonWithdrawn          = "Withdrawn"
	reasonServiceNotOwned    = "ServiceNotOwned"
	reasonInvalidServiceName = "InvalidServiceName"
	reasonServiceNameTooLong = "ServiceNameTooLong"
	reasonNoFrontend         = "NoFrontend"
	reasonClusterIDMismatch  = "ClusterIDMismatch"
	reasonExportFailed       = "ExportFailed"
)

// Limits of the API server: an event's note, and a condition's message.
const (
	maxEventNote        = 1024
	maxConditionMessage = 32768
)

// relayGroupKind is the kind that controls a pool's exports.
var relayGroupKind = v1alpha1.GroupVersion.WithKind("DynamoPoolRelay").GroupKind()

// dcdGroupKind is the kind that controls a frontend's Service on the DCD path; a
// DGD controls each of its DCDs.
var dcdGroupKind = v1beta1.GroupVersion.WithKind("DynamoComponentDeployment").GroupKind()

// exportSettings configures the pool's exports, which the controller maintains
// when the multiClusterExport gate is on.
type exportSettings struct {
	// cache reads unstructured objects, ServiceExports and ClusterProperties, from
	// the informers that the controller's watches start. The manager's client
	// reads unstructured objects from the API server.
	cache         client.Reader
	recorder      events.EventRecorder
	config        *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
	// clusterPropertyWatched is true when the controller watches the About API's
	// ClusterProperties: the MCS implementation labels imports with the cluster's
	// ID in its ClusterSet, and the cluster serves the API. Otherwise the check of
	// a pool's cluster ID is skipped.
	clusterPropertyWatched bool
}

// eventFilter passes the events of the namespaces that the operator manages, and
// those of the cluster-scoped ClusterProperty, whose mapper selects the Relays of
// those namespaces.
func eventFilter(
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
) predicate.Predicate {
	return predicate.Or(
		commoncontroller.EphemeralDeploymentEventFilter(config, runtimeConfig),
		predicate.NewPredicateFuncs(func(object client.Object) bool {
			return object.GetObjectKind().GroupVersionKind().GroupKind() == multicluster.ClusterPropertyGVK.GroupKind()
		}),
	)
}

// watchExports turns on the pool's exports, and adds the watches that they need:
// the ServiceExports, the Services that a pool may export and, with a standard
// implementation in a cluster that serves the About API, the ClusterProperty that
// holds the cluster's ID in its ClusterSet. The Relay's DGD is watched already.
func (r *reconciler) watchExports(
	mgr ctrl.Manager,
	controllerBuilder *builder.Builder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
) (*builder.Builder, error) {
	r.exports = &exportSettings{
		cache:         mgr.GetCache(),
		recorder:      mgr.GetEventRecorder("dynamo-pool-relay"),
		config:        config,
		runtimeConfig: runtimeConfig,
	}
	controllerBuilder = controllerBuilder.Watches(
		multicluster.NewServiceExport(runtimeConfig.Gate.ServiceExportVersion),
		handler.EnqueueRequestsFromMapFunc(r.mapServiceExportToRelays),
		builder.WithPredicates(serviceExportChanged),
	).Watches(
		&corev1.Service{},
		handler.EnqueueRequestsFromMapFunc(r.mapServiceToRelays),
		builder.WithPredicates(serviceControllerChanged),
	)
	if !multicluster.UsesClusterSetID(config.MultiCluster.MCSImplementation) {
		return controllerBuilder, nil
	}
	served, err := clusterPropertyServed(mgr.GetRESTMapper())
	if err != nil {
		return controllerBuilder, err
	}
	if !served {
		mgr.GetLogger().Info("The cluster does not serve the About API's ClusterProperty, " +
			"so the clusterId of DynamoPoolRelays is not checked; to check it, install the About API and restart the operator")
		return controllerBuilder, nil
	}
	r.exports.clusterPropertyWatched = true
	return controllerBuilder.Watches(
		multicluster.NewClusterProperty(),
		handler.EnqueueRequestsFromMapFunc(r.mapClusterPropertyToRelays),
		builder.WithPredicates(clusterIDPropertyEvents),
	), nil
}

// exportProblem is a problem of the pool's exports that only the operator can
// see, or a failure to reconcile them. related is the object that it is about,
// if any.
type exportProblem struct {
	reason  string
	message string
	related runtime.Object
}

// reconcileExports exports the Services of the pool that the Relay publishes for
// dgd, withdraws the Relay's other exports, and reports the outcome in the
// Relay's Exported condition. It decides from what exists now, not from the
// events that led there, so a DGD that is deleted and created again under the
// Relay's name, quickly or not, ends up with the new DGD's exports.
func (r *reconciler) reconcileExports(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
) error {
	if r.exports == nil {
		return nil
	}
	exported, missing, problems, err := r.convergeExports(ctx, relay, dgd)
	message := fmt.Sprintf("Exported %s through MCS implementation %s",
		serviceList(exported), r.exports.config.MultiCluster.MCSImplementation)
	if len(missing) > 0 {
		message += fmt.Sprintf("; %s does not exist yet", serviceList(missing))
	}
	exportedCondition := metav1.Condition{
		Type: conditionExported, Status: metav1.ConditionTrue, Reason: reasonExported, Message: message,
	}
	return errors.Join(err, r.setExported(ctx, relay, exportedCondition, problems, err))
}

// withdrawExports withdraws every export that the Relay controls, as it withdraws
// the Relay's Service, and reports that in the Relay's Exported condition.
func (r *reconciler) withdrawExports(ctx context.Context, relay *v1alpha1.DynamoPoolRelay) error {
	if r.exports == nil {
		return nil
	}
	err := r.withdrawAllExports(ctx, relay)
	withdrawnCondition := metav1.Condition{
		Type: conditionExported, Status: metav1.ConditionFalse, Reason: reasonWithdrawn,
		Message: "The Relay withdrew its Service, and with it the pool's exports",
	}
	return errors.Join(err, r.setExported(ctx, relay, withdrawnCondition, nil, err))
}

// setExported sets the Relay's Exported condition to condition or, with problems
// or err, to False with the first one's reason and every one's message. When the
// condition changes, it writes the Relay's status and then records a Warning
// event for each problem, so the events follow the condition, not each reconcile.
func (r *reconciler) setExported(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	condition metav1.Condition,
	problems []exportProblem,
	err error,
) error {
	if err != nil {
		problems = append([]exportProblem{{
			reason:  reasonExportFailed,
			message: fmt.Sprintf("Failed to reconcile the pool's ServiceExports: %v", err),
		}}, problems...)
	}
	if len(problems) > 0 {
		messages := make([]string, 0, len(problems))
		for _, problem := range problems {
			messages = append(messages, problem.message)
		}
		condition.Status = metav1.ConditionFalse
		condition.Reason = problems[0].reason
		condition.Message = strings.Join(messages, "; ")
	}
	condition.Message = truncate(condition.Message, maxConditionMessage)
	condition.ObservedGeneration = relay.Generation

	previous := meta.FindStatusCondition(relay.Status.Conditions, conditionExported)
	changed := previous == nil || previous.Status != condition.Status ||
		previous.Reason != condition.Reason || previous.Message != condition.Message
	if !changed && previous.ObservedGeneration == condition.ObservedGeneration {
		return nil
	}
	meta.SetStatusCondition(&relay.Status.Conditions, condition)
	if err := r.Status().Update(ctx, relay); err != nil {
		return fmt.Errorf("failed to update the Exported condition: %w", err)
	}
	if changed {
		for _, problem := range problems {
			r.exports.recorder.Eventf(relay, problem.related, corev1.EventTypeWarning, problem.reason, "Export",
				"%s", truncate(problem.message, maxEventNote))
		}
	}
	return nil
}

// unexportedService is a pool Service that the implementation cannot export.
type unexportedService struct {
	name    string
	problem multicluster.NameProblem
	detail  string
}

// convergeExports exports each Service of the pool that the implementation can
// export and that the pool controls, and withdraws the Relay's other exports. It
// returns the names exported, those whose Service does not exist yet, the pool's
// problems, and every failed read and write: one failure does not hold back the
// others.
func (r *reconciler) convergeExports(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
) (exported, missing []string, problems []exportProblem, err error) {
	implementation := r.exports.config.MultiCluster.MCSImplementation
	existing, err := r.listExports(ctx, relay)
	if err != nil {
		return nil, nil, nil, err
	}

	// The pool exports each of its Services that the implementation can export and
	// that the pool controls: a missing export is created, and one that the Relay
	// controls is repaired. A user's export, or one being deleted, stays as it is.
	// A Service that does not exist, or that the pool does not control, is not
	// exported, so a Service of the same name that someone else created is never
	// exported in the pool's name.
	var errs []error
	var unexported []unexportedService
	var notControlled []string
	keep := map[string]bool{}
	for _, name := range multicluster.ServiceNames(dgd) {
		if problem, detail := multicluster.CheckServiceName(implementation, name); problem != multicluster.NameExportable {
			unexported = append(unexported, unexportedService{name: name, problem: problem, detail: detail})
			continue
		}
		found, controlled, err := r.poolControlsService(ctx, relay, dgd, name)
		switch {
		case err != nil:
			// Keep the export until the Service can be read again.
			keep[name] = true
			errs = append(errs, err)
			continue
		case !found:
			missing = append(missing, name)
			continue
		case !controlled:
			notControlled = append(notControlled, name)
			continue
		}
		keep[name] = true
		current, found := existing[name]
		switch {
		case !found:
			err = r.createExport(ctx, relay, dgd, name)
		case current.ControlledBy(relay.UID) && !current.Deleting:
			err = r.repairExport(ctx, relay, dgd, current)
		}
		if err != nil {
			errs = append(errs, err)
			continue
		}
		exported = append(exported, name)
	}

	// Withdraw the Relay's exports of Services that the pool does not export, such
	// as a renamed or removed frontend's, or one that someone else's Service took
	// over.
	for _, name := range slices.Sorted(maps.Keys(existing)) {
		if keep[name] {
			continue
		}
		if err := r.withdrawExport(ctx, relay, existing[name]); err != nil {
			errs = append(errs, err)
		}
	}
	problems, err = r.exportProblems(ctx, relay, dgd, unexported, notControlled)
	return exported, missing, problems, errors.Join(append(errs, err)...)
}

// poolControlsService reads the Service named name in the Relay's namespace, and
// reports whether it exists and whether the pool controls it: the Relay controls
// its PoolRelay Service, and the DGD, or on the DCD path one of its DCDs, a
// frontend's.
func (r *reconciler) poolControlsService(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	name string,
) (found, controlled bool, err error) {
	service := &corev1.Service{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: name}, service); err != nil {
		if apierrors.IsNotFound(err) {
			return false, false, nil
		}
		return false, false, fmt.Errorf("failed to read Service %s: %w", name, err)
	}
	controller := metav1.GetControllerOf(service)
	switch {
	case controller == nil:
		return true, false, nil
	case controller.UID == relay.UID || controller.UID == dgd.UID:
		return true, true, nil
	case schema.FromAPIVersionAndKind(controller.APIVersion, controller.Kind).GroupKind() != dcdGroupKind:
		return true, false, nil
	}
	dcd := &v1beta1.DynamoComponentDeployment{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: controller.Name}, dcd); err != nil {
		if apierrors.IsNotFound(err) {
			return true, false, nil
		}
		return true, false, fmt.Errorf("failed to read DynamoComponentDeployment %s: %w", controller.Name, err)
	}
	return true, dcd.UID == controller.UID && metav1.IsControlledBy(dcd, dgd), nil
}

func (r *reconciler) withdrawAllExports(ctx context.Context, relay *v1alpha1.DynamoPoolRelay) error {
	existing, err := r.listExports(ctx, relay)
	if err != nil {
		return err
	}
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(existing)) {
		if err := r.withdrawExport(ctx, relay, existing[name]); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// listExports returns the ServiceExports in the Relay's namespace by name.
func (r *reconciler) listExports(ctx context.Context, relay *v1alpha1.DynamoPoolRelay) (map[string]multicluster.Export, error) {
	return multicluster.ListExports(ctx, r.exports.cache, r.exports.runtimeConfig.Gate.ServiceExportVersion, relay.Namespace)
}

// desiredExport returns the Relay's ServiceExport of the Service named name, with
// only the fields that the controller writes: the DGD label, and the Relay as its
// controller, so that the export is deleted with the Relay.
func (r *reconciler) desiredExport(
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	name string,
) (*unstructured.Unstructured, error) {
	serviceExport := multicluster.RenderServiceExport(
		r.exports.runtimeConfig.Gate.ServiceExportVersion, relay.Namespace, name, dgd.Name)
	if err := ctrl.SetControllerReference(relay, serviceExport, r.scheme); err != nil {
		return nil, fmt.Errorf("failed to set the owner of ServiceExport %s: %w", name, err)
	}
	return serviceExport, nil
}

// createExport creates the Relay's ServiceExport of the Service named name. A
// create fails on any existing export, so the Relay never takes over a user's
// export of the same name that the cache has not seen yet.
func (r *reconciler) createExport(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	name string,
) error {
	serviceExport, err := r.desiredExport(relay, dgd, name)
	if err != nil {
		return err
	}
	if err := r.Create(ctx, serviceExport, client.FieldOwner(exportFieldManager)); err != nil {
		// The cache has not seen an export created moments ago. Its watch event
		// requeues the Relay.
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("failed to create ServiceExport %s: %w", name, err)
	}
	r.exports.recorder.Eventf(relay, serviceExport, corev1.EventTypeNormal, "CreateServiceExport", "Create",
		"Exported Service %s through MCS implementation %s", name, r.exports.config.MultiCluster.MCSImplementation)
	return nil
}

// repairExport restores the fields that the controller writes on an export the
// Relay controls, when they differ from desiredExport's, for example after
// someone removed the DGD label. Server-side apply writes only those fields, so
// the export keeps its UID, the MCS implementation's status, its finalizers, and
// other labels and annotations.
func (r *reconciler) repairExport(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	current multicluster.Export,
) error {
	desired, err := r.desiredExport(relay, dgd, current.Name)
	if err != nil {
		return err
	}
	if hasDesiredFields(current, desired) {
		return nil
	}

	// The UID and resourceVersion tie the apply to the export whose controller was
	// just checked. Once it has changed, been replaced or been deleted, the API
	// server refuses the apply with a conflict instead of writing to an export
	// that the Relay may no longer control, and the change's watch event requeues
	// the Relay. Forcing ownership takes back a field that someone else changed:
	// without it, that field would also be refused as a conflict.
	desired.SetUID(current.UID)
	desired.SetResourceVersion(current.ResourceVersion)
	err = r.Apply(ctx, client.ApplyConfigurationFromUnstructured(desired),
		client.FieldOwner(exportFieldManager), client.ForceOwnership)
	if apierrors.IsConflict(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to repair ServiceExport %s: %w", current.Name, err)
	}
	r.exports.recorder.Eventf(relay, desired, corev1.EventTypeNormal, "RepairServiceExport", "Apply",
		"Restored the DGD label and controller reference of the export of Service %s", current.Name)
	return nil
}

// hasDesiredFields reports whether export has the labels and the controller
// reference of desired. Other labels do not matter.
func hasDesiredFields(export multicluster.Export, desired *unstructured.Unstructured) bool {
	for key, value := range desired.GetLabels() {
		if export.Labels[key] != value {
			return false
		}
	}
	return equality.Semantic.DeepEqual(export.Controller, metav1.GetControllerOf(desired))
}

// withdrawExport deletes export if the Relay controls it and it is not already
// being deleted.
func (r *reconciler) withdrawExport(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, export multicluster.Export) error {
	if !export.ControlledBy(relay.UID) || export.Deleting {
		return nil
	}

	// The UID and resourceVersion preconditions delete exactly the export just
	// read, at the version whose controller was checked. A change to the export
	// since, such as someone removing or transferring the controller reference,
	// which keeps the UID, makes the API server refuse the delete with a conflict.
	// That is retried from the change's watch event, which requeues the Relay to
	// check the controller again.
	serviceExport := multicluster.NewServiceExport(r.exports.runtimeConfig.Gate.ServiceExportVersion)
	serviceExport.SetNamespace(relay.Namespace)
	serviceExport.SetName(export.Name)
	if err := r.Delete(ctx, serviceExport, client.Preconditions{
		UID:             &export.UID,
		ResourceVersion: &export.ResourceVersion,
	}); err != nil {
		if apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
			return nil
		}
		return fmt.Errorf("failed to delete ServiceExport %s: %w", export.Name, err)
	}
	// The event names the export deleted.
	serviceExport.SetUID(export.UID)
	serviceExport.SetResourceVersion(export.ResourceVersion)
	r.exports.recorder.Eventf(relay, serviceExport, corev1.EventTypeNormal, "DeleteServiceExport", "Delete",
		"Withdrew the export of Service %s", export.Name)
	return nil
}

// exportProblems returns the problems of the pool that only the operator can
// see: a Service that the pool does not control, a Service that the
// implementation cannot export, a DGD without a frontend, and a cluster ID that
// differs from the cluster's ID in its ClusterSet. A failed read of that ID is an
// error.
func (r *reconciler) exportProblems(
	ctx context.Context,
	relay *v1alpha1.DynamoPoolRelay,
	dgd *v1beta1.DynamoGraphDeployment,
	unexported []unexportedService,
	notControlled []string,
) ([]exportProblem, error) {
	var problems []exportProblem
	if len(notControlled) > 0 {
		problems = append(problems, exportProblem{reason: reasonServiceNotOwned, related: dgd, message: fmt.Sprintf(
			"Neither the Relay nor DGD %s controls %s, so it is not exported", dgd.Name, serviceList(notControlled))})
	}
	if names, detail := unexportedNames(unexported, multicluster.NameInvalid); len(names) > 0 {
		problems = append(problems, exportProblem{reason: reasonInvalidServiceName, related: dgd, message: fmt.Sprintf(
			"No Service can be named %s, so it is not exported: %s", strings.Join(names, ", "), detail)})
	}
	if names, detail := unexportedNames(unexported, multicluster.NameTooLong); len(names) > 0 {
		problems = append(problems, exportProblem{reason: reasonServiceNameTooLong, related: dgd, message: fmt.Sprintf(
			"%s, so %s is not exported; shorten the DGD or component name", detail, strings.Join(names, ", "))})
	}
	if len(multicluster.FrontendServiceNames(dgd)) == 0 {
		problems = append(problems, exportProblem{reason: reasonNoFrontend, related: dgd, message: fmt.Sprintf(
			"DGD %s has no frontend component, so the Global Router has nowhere to send the pool's requests", dgd.Name)})
	}

	// A standard implementation labels the pool's imports with the cluster's ID in
	// its ClusterSet, and the Global Discovery Service joins the pool's
	// registration to them only under the same cluster ID.
	if !r.exports.clusterPropertyWatched {
		return problems, nil
	}
	property, err := multicluster.ReadClusterIDProperty(ctx, r.exports.cache)
	if err != nil || property == nil {
		return problems, err
	}
	if clusterID := multicluster.ClusterPropertyValue(property); clusterID != "" && clusterID != relay.Spec.ClusterID {
		problems = append(problems, exportProblem{reason: reasonClusterIDMismatch, related: property, message: fmt.Sprintf(
			"clusterId %q differs from the ID %q that ClusterProperty %s gives this cluster in its ClusterSet; "+
				"the Global Discovery Service lists the pool only when they match",
			relay.Spec.ClusterID, clusterID, multicluster.ClusterIDProperty)})
	}
	return problems, nil
}

// serviceList names the Services named names in a message.
func serviceList(names []string) string {
	switch len(names) {
	case 0:
		return "no Service"
	case 1:
		return "Service " + names[0]
	}
	return "Services " + strings.Join(names, ", ")
}

// truncate shortens s to at most limit bytes without splitting a character.
func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	end := limit
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end]
}

// unexportedNames returns the names of the services with problem, and the first
// one's detail.
func unexportedNames(services []unexportedService, problem multicluster.NameProblem) ([]string, string) {
	var names []string
	detail := ""
	for _, service := range services {
		if service.problem != problem {
			continue
		}
		if len(names) == 0 {
			detail = service.detail
		}
		names = append(names, service.name)
	}
	return names, detail
}

// mapServiceExportToRelays requeues the Relay that controls the export, and each
// Relay whose DGD exports a Service of the export's name: such a Relay takes over
// when a user's export of that Service is deleted.
func (r *reconciler) mapServiceExportToRelays(ctx context.Context, export client.Object) []ctrl.Request {
	var requests []ctrl.Request
	if controller := metav1.GetControllerOf(export); controller != nil &&
		schema.FromAPIVersionAndKind(controller.APIVersion, controller.Kind).GroupKind() == relayGroupKind {
		requests = append(requests, ctrl.Request{Namespace: export.GetNamespace(), Name: controller.Name})
	}
	for _, request := range r.relaysExportingService(ctx, export.GetNamespace(), export.GetName()) {
		if !slices.Contains(requests, request) {
			requests = append(requests, request)
		}
	}
	return requests
}

// mapServiceToRelays requeues each Relay whose DGD exports a Service of the
// Service's name: the pool exports a Service only while it exists and the pool
// controls it.
func (r *reconciler) mapServiceToRelays(ctx context.Context, service client.Object) []ctrl.Request {
	return r.relaysExportingService(ctx, service.GetNamespace(), service.GetName())
}

// relaysExportingService returns a request for each Relay in namespace whose DGD
// exports a Service named name.
func (r *reconciler) relaysExportingService(ctx context.Context, namespace, name string) []ctrl.Request {
	relays := &v1alpha1.DynamoPoolRelayList{}
	if err := r.List(ctx, relays, client.InNamespace(namespace)); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoPoolRelays for a Service or ServiceExport event")
		return nil
	}
	var requests []ctrl.Request
	for i := range relays.Items {
		dgd := &v1beta1.DynamoGraphDeployment{}
		if err := r.Get(ctx, client.ObjectKeyFromObject(&relays.Items[i]), dgd); err != nil {
			continue
		}
		if slices.Contains(multicluster.ServiceNames(dgd), name) {
			requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(&relays.Items[i])})
		}
	}
	return requests
}

// mapClusterPropertyToRelays requeues every Relay in a namespace that the
// operator manages: each pool's cluster ID is compared with the cluster's ID in
// its ClusterSet.
func (r *reconciler) mapClusterPropertyToRelays(ctx context.Context, _ client.Object) []ctrl.Request {
	relays := &v1alpha1.DynamoPoolRelayList{}
	if err := r.List(ctx, relays); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoPoolRelays for ClusterProperty event")
		return nil
	}
	requests := make([]ctrl.Request, 0, len(relays.Items))
	for i := range relays.Items {
		relay := &relays.Items[i]
		if commoncontroller.NamespaceAllowed(r.exports.config, r.exports.runtimeConfig, relay, relay.Namespace) {
			requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(relay)})
		}
	}
	return requests
}

// serviceExportChanged passes any update of a ServiceExport with a controller,
// before or after the update: the Relay that controls the export repairs it, and
// one that stopped controlling it leaves it alone. This also covers a change that
// made a repair or a withdrawal conflict. The creation and deletion of any export
// pass too.
var serviceExportChanged = predicate.Funcs{UpdateFunc: func(update event.UpdateEvent) bool {
	return metav1.GetControllerOf(update.ObjectOld) != nil || metav1.GetControllerOf(update.ObjectNew) != nil
}}

// serviceControllerChanged passes the creation and deletion of a Service, and an
// update that changes its controller: whether the pool exports a Service depends
// only on whether it exists and who controls it.
var serviceControllerChanged = predicate.Funcs{UpdateFunc: func(update event.UpdateEvent) bool {
	return !equality.Semantic.DeepEqual(metav1.GetControllerOf(update.ObjectOld), metav1.GetControllerOf(update.ObjectNew))
}}

// clusterIDPropertyEvents passes the events of the ClusterProperty that holds the
// cluster's ID in its ClusterSet: its creation, its deletion and a change of its
// value.
var clusterIDPropertyEvents = predicate.Funcs{
	CreateFunc:  func(e event.CreateEvent) bool { return e.Object.GetName() == multicluster.ClusterIDProperty },
	DeleteFunc:  func(e event.DeleteEvent) bool { return e.Object.GetName() == multicluster.ClusterIDProperty },
	UpdateFunc:  clusterIDPropertyChanged,
	GenericFunc: func(event.GenericEvent) bool { return false },
}

// clusterIDPropertyChanged passes the updates of the ClusterProperty that holds
// the cluster's ID in its ClusterSet that change its value.
func clusterIDPropertyChanged(update event.UpdateEvent) bool {
	oldProperty, oldOK := update.ObjectOld.(*unstructured.Unstructured)
	newProperty, newOK := update.ObjectNew.(*unstructured.Unstructured)
	return oldOK && newOK && newProperty.GetName() == multicluster.ClusterIDProperty &&
		multicluster.ClusterPropertyValue(oldProperty) != multicluster.ClusterPropertyValue(newProperty)
}

// clusterPropertyServed reports whether the cluster serves the About API's
// ClusterProperty, which the controller then watches.
func clusterPropertyServed(mapper meta.RESTMapper) (bool, error) {
	_, err := mapper.RESTMapping(multicluster.ClusterPropertyGVK.GroupKind(), multicluster.ClusterPropertyGVK.Version)
	if meta.IsNoMatchError(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("discover the ClusterProperty API for the pool's exports: %w", err)
	}
	return true, nil
}
