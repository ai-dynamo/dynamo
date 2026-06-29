/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"hash/fnv"
	"maps"
	"math"
	"net"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/selectionservice"
)

const (
	selectionEndpointSlicePodIndex = ".metadata.selectionTargetPod"
	selectorServiceIndex           = ".metadata.selectorService"
	selectionTopologyRequeueAfter  = 30 * time.Second
	selectionTopologyResyncAfter   = 2 * time.Minute
	selectionServiceRequestTimeout = 5 * time.Second
	selectionSGLangRequestTimeout  = 5 * time.Second
)

type selectionClient interface {
	UpsertWorker(context.Context, selectionservice.WorkerRequest) (selectionservice.WorkerRecord, error)
	ListWorkers(context.Context, string, string) ([]selectionservice.WorkerRecord, error)
	DeleteWorker(context.Context, uint64, bool) (selectionservice.WorkerRecord, error)
}

type selectionClientFactory func(baseURL string) (selectionClient, error)

func defaultSelectionClientFactory(baseURL string) (selectionClient, error) {
	return selectionservice.NewClient(baseURL, selectionServiceRequestTimeout)
}

type knownWorkerTargets struct {
	SelectorTargetURLs sets.Set[string]
}

type selectionTopologyOwner struct {
	Kind      string
	Namespace string
	Name      string
	UID       string
}

type selectionReconcileScope struct {
	Owner       selectionTopologyOwner
	AdapterName string
	TenantID    string
}

type selectionWorkerRegistration struct {
	WorkerID        uint64
	WorkerEndpoint  string
	TenantID        string
	RequireKVEvents bool
	StableRoutingID string
	TopologyDomains map[string]string
}

type workerTarget struct {
	selectionWorkerRegistration
	AdapterName string
	Owner       selectionTopologyOwner
	PodName     string
}

// selectorTarget separates the HTTP endpoint used for a selector
// replica from the stable catalog ownership key written into worker metadata.
type selectorTarget struct {
	TargetURL string
	ScopeKey  string
}

type parsedSelectorURL struct {
	normalized  string
	parsed      url.URL
	serviceName string
	namespace   string
	serviceDNS  bool
}

// selectorCatalogPlan is the desired catalog and deactivation scope for one
// concrete selector target URL.
type selectorCatalogPlan struct {
	Desired          map[uint64]selectionservice.WorkerRequest
	Scopes           sets.Set[selectionReconcileScope]
	SelectorScopeKey string
	TenantID         string
}

const (
	selectionMetadataManagedBy    = "managed-by"
	selectionMetadataManagedByVal = "dynamo-operator"
	selectionMetadataOwnerKind    = "owner-kind"
	selectionMetadataOwnerNS      = "owner-namespace"
	selectionMetadataOwnerName    = "owner-name"
	selectionMetadataOwnerUID     = "owner-uid"
	selectionMetadataAdapter      = "adapter"
	selectionMetadataSelectorURL  = "selector-url"
)

// SelectionTopologyReconciler watches platform topology and reconciles raw
// engine workers into the runtime-free selection service.
type SelectionTopologyReconciler struct {
	client.Client
	Recorder      record.EventRecorder
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig

	SelectionClientFactory      selectionClientFactory
	SGLangMetadataClientFactory sglangMetadataClientFactory

	knownMu        sync.Mutex
	knownByService map[types.NamespacedName]map[uint64]knownWorkerTargets
}

// +kubebuilder:rbac:groups=discovery.k8s.io,resources=endpointslices,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch

func (r *SelectionTopologyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Treat the worker Service as the reconcile unit: Service, Pod, EndpointSlice,
	// and selector changes all converge the Service's desired worker catalog.
	ownerService := &corev1.Service{}
	if err := r.Get(ctx, req.NamespacedName, ownerService); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, r.deactivateAllKnownWorkersForService(ctx, req.NamespacedName)
		}
		return ctrl.Result{}, fmt.Errorf("get worker Service %s: %w", req.NamespacedName, err)
	}
	logger = logger.WithValues("workerService", ownerService.Name, "namespace", ownerService.Namespace)

	scope, rawSelectorURL, ok, err := selectionReconcileScopeForWorkerService(ownerService)
	if err != nil {
		logger.Info("Skipping invalid worker Service selection topology config", "error", err.Error())
		return ctrl.Result{}, r.deactivateAllKnownWorkersForService(ctx, req.NamespacedName)
	}
	if !ok {
		return ctrl.Result{}, r.deactivateAllKnownWorkersForService(ctx, req.NamespacedName)
	}

	slices, err := r.endpointSlicesForService(ctx, ownerService.Namespace, ownerService.Name)
	if err != nil {
		return ctrl.Result{}, err
	}

	workerTargets, err := r.workerTargetsForEndpointSlices(ctx, ownerService, slices)
	if err != nil {
		return ctrl.Result{}, err
	}
	previousKnownWorkers := r.knownWorkers(req.NamespacedName)
	clientsByTargetURL := map[string]selectionClient{}

	// All ready workers owned by this Service fan out to the same selector
	// replicas, so resolve the selector topology once per Service reconcile.
	selectorTargets, err := r.selectorTargetsForURL(ctx, rawSelectorURL, ownerService.Namespace)
	if err != nil {
		return ctrl.Result{}, err
	}
	if len(selectorTargets) == 0 {
		logger.Info("Selector URL has no ready targets; requeueing",
			"selectorURL", rawSelectorURL)
		return ctrl.Result{RequeueAfter: selectionTopologyRequeueAfter}, nil
	}
	activeSelectorTargetURLs := selectorActiveTargetURLs(selectorTargets)

	catalogPlans, nextKnownWorkers, needsRequeue := r.buildSelectorCatalogPlans(
		ctx,
		logger,
		scope,
		workerTargets,
		selectorTargets,
		previousKnownWorkers,
	)

	// Apply desired state after all probes finish. The catalog reconciler reuses
	// ListWorkers to skip unchanged upserts and deactivate stale owned entries.
	if err := r.reconcileSelectorCatalogWorkers(ctx, logger, catalogPlans, clientsByTargetURL); err != nil {
		return ctrl.Result{}, err
	}
	if err := r.deactivateKnownWorkersNotInDesired(ctx, previousKnownWorkers, nextKnownWorkers, clientsByTargetURL, activeSelectorTargetURLs, true); err != nil {
		return ctrl.Result{}, err
	}
	r.setKnownWorkers(req.NamespacedName, nextKnownWorkers)

	if needsRequeue {
		return ctrl.Result{RequeueAfter: selectionTopologyRequeueAfter}, nil
	}
	return ctrl.Result{RequeueAfter: selectionTopologyResyncAfter}, nil
}

func (r *SelectionTopologyReconciler) buildSelectorCatalogPlans(
	ctx context.Context,
	logger logr.Logger,
	serviceScope selectionReconcileScope,
	workerTargets []workerTarget,
	selectorTargets []selectorTarget,
	previousKnownWorkers map[uint64]knownWorkerTargets,
) (map[string]*selectorCatalogPlan, map[uint64]knownWorkerTargets, bool) {
	catalogPlans := map[string]*selectorCatalogPlan{}
	nextKnownWorkers := make(map[uint64]knownWorkerTargets)
	needsRequeue := false

	// Keep an empty desired plan when no workers are ready so owned catalog
	// entries from previous reconciles are still removed from selector replicas.
	if len(workerTargets) == 0 {
		for _, selectorTarget := range selectorTargets {
			selectorCatalogPlanFor(catalogPlans, selectorTarget).addScope(serviceScope)
		}
		return catalogPlans, nextKnownWorkers, false
	}

	for _, target := range workerTargets {
		if target.AdapterName != selectionAdapterExternalSGLang {
			logger.Info("Unsupported selection topology adapter; deactivating worker registration",
				"adapter", target.AdapterName,
				"workerID", target.WorkerID,
				"pod", target.PodName,
				"reason", "adapter is not registered")
			addSelectionReconcileScopes(catalogPlans, selectorTargets, selectionReconcileScope{
				Owner:    target.Owner,
				TenantID: normalizeSelectionTenantID(target.TenantID),
			})
			removeKnownWorkerSelectorTargets(previousKnownWorkers, target.WorkerID, selectorTargets)
			continue
		}

		// Fail closed for unusable runtime metadata: deactivate owned registrations
		// and requeue so SGLang startup races can converge later.
		worker, err := externalSGLangSelectionAdapter{
			MetadataClientFactory: r.SGLangMetadataClientFactory,
		}.BuildWorker(ctx, target.selectionWorkerRegistration)
		if err != nil {
			logger.Info("Selection topology adapter failed; deactivating worker registration",
				"adapter", target.AdapterName,
				"workerID", target.WorkerID,
				"pod", target.PodName,
				"error", err)
			addSelectionReconcileScopes(catalogPlans, selectorTargets, selectionReconcileScope{
				Owner:       target.Owner,
				AdapterName: target.AdapterName,
				TenantID:    normalizeSelectionTenantID(target.TenantID),
			})
			removeKnownWorkerSelectorTargets(previousKnownWorkers, target.WorkerID, selectorTargets)
			needsRequeue = true
			continue
		}

		selectorTargetURLs := sets.New[string]()
		scope := selectionReconcileScope{
			Owner:       target.Owner,
			AdapterName: target.AdapterName,
			TenantID:    worker.TenantID,
		}
		for _, selectorTarget := range selectorTargets {
			workerForSelector := worker
			workerForSelector.Metadata = workerCatalogMetadata(target, selectorTarget.ScopeKey)
			plan := selectorCatalogPlanFor(catalogPlans, selectorTarget)
			plan.Desired[target.WorkerID] = workerForSelector
			plan.addScope(scope)
			selectorTargetURLs.Insert(selectorTarget.TargetURL)
		}
		nextKnownWorkers[target.WorkerID] = knownWorkerTargets{SelectorTargetURLs: selectorTargetURLs}
	}

	return catalogPlans, nextKnownWorkers, needsRequeue
}

func (r *SelectionTopologyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.SelectionClientFactory == nil {
		r.SelectionClientFactory = defaultSelectionClientFactory
	}
	// Index EndpointSlices by target Pod name so Pod readiness or topology
	// label changes can requeue the owning worker Service.
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&discoveryv1.EndpointSlice{},
		selectionEndpointSlicePodIndex,
		func(obj client.Object) []string {
			return endpointSliceTargetPodNames(obj.(*discoveryv1.EndpointSlice))
		},
	); err != nil {
		return err
	}
	// Index worker Services by referenced selector Service so selector
	// EndpointSlice changes can converge the worker catalog to every replica.
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&corev1.Service{},
		selectorServiceIndex,
		func(obj client.Object) []string {
			return selectorServiceIndexKeys(obj.(*corev1.Service))
		},
	); err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Service{}, builder.WithPredicates(selectionTopologyServicePredicate())).
		Named(consts.ResourceTypeSelectionTopology).
		Watches(
			&discoveryv1.EndpointSlice{},
			handler.EnqueueRequestsFromMapFunc(r.findWorkerServicesForEndpointSlice),
			builder.WithPredicates(predicate.Funcs{
				GenericFunc: func(e event.GenericEvent) bool { return false },
			}),
		).
		Watches(
			&corev1.Pod{},
			handler.EnqueueRequestsFromMapFunc(r.findWorkerServicesForPod),
			builder.WithPredicates(predicate.Funcs{
				GenericFunc: func(e event.GenericEvent) bool { return false },
			}),
		).
		Watches(
			&corev1.Service{},
			handler.EnqueueRequestsFromMapFunc(r.findWorkerServicesForSelectorService),
			builder.WithPredicates(predicate.Funcs{
				GenericFunc: func(e event.GenericEvent) bool { return false },
			}),
		).
		// This global filter applies to worker and selector objects. Cross-namespace
		// selector Services require cluster-wide mode or otherwise cache-visible namespaces.
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(observability.NewObservedReconciler(r, consts.ResourceTypeSelectionTopology))
}

func (r *SelectionTopologyReconciler) findWorkerServicesForPod(ctx context.Context, obj client.Object) []reconcile.Request {
	slices := &discoveryv1.EndpointSliceList{}
	if err := r.List(ctx, slices,
		client.InNamespace(obj.GetNamespace()),
		client.MatchingFields{selectionEndpointSlicePodIndex: obj.GetName()},
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list EndpointSlices for selection topology Pod", "pod", obj.GetName())
		return nil
	}

	requests := make([]reconcile.Request, 0, len(slices.Items))
	for _, slice := range slices.Items {
		if req, ok := workerServiceRequestForEndpointSlice(&slice); ok {
			requests = appendAnnotatedWorkerServiceRequest(ctx, r.Client, requests, req)
			requests = append(requests, r.findWorkerServicesForSelectorKey(ctx, req.NamespacedName)...)
		}
	}
	return dedupeRequests(requests)
}

func (r *SelectionTopologyReconciler) findWorkerServicesForEndpointSlice(ctx context.Context, obj client.Object) []reconcile.Request {
	slice := obj.(*discoveryv1.EndpointSlice)
	var requests []reconcile.Request
	if req, ok := workerServiceRequestForEndpointSlice(slice); ok {
		requests = appendAnnotatedWorkerServiceRequest(ctx, r.Client, requests, req)
		requests = append(requests, r.findWorkerServicesForSelectorKey(ctx, req.NamespacedName)...)
	}
	return dedupeRequests(requests)
}

func workerServiceRequestForEndpointSlice(slice *discoveryv1.EndpointSlice) (reconcile.Request, bool) {
	serviceName := strings.TrimSpace(slice.Labels[discoveryv1.LabelServiceName])
	if serviceName == "" {
		return reconcile.Request{}, false
	}
	return reconcile.Request{
		NamespacedName: types.NamespacedName{
			Namespace: slice.Namespace,
			Name:      serviceName,
		},
	}, true
}

func (r *SelectionTopologyReconciler) findWorkerServicesForSelectorService(ctx context.Context, obj client.Object) []reconcile.Request {
	return r.findWorkerServicesForSelectorKey(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: obj.GetName()})
}

func (r *SelectionTopologyReconciler) findWorkerServicesForSelectorKey(ctx context.Context, selectorKey types.NamespacedName) []reconcile.Request {
	services := &corev1.ServiceList{}
	if err := r.List(ctx, services,
		client.MatchingFields{selectorServiceIndex: selectorServiceKey(selectorKey.Namespace, selectorKey.Name)},
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list worker Services for selector Service", "selectorService", selectorKey.String())
		return nil
	}

	requests := make([]reconcile.Request, 0, len(services.Items))
	for _, service := range services.Items {
		requests = append(requests, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(&service)})
	}
	return dedupeRequests(requests)
}

func selectionTopologyServicePredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return serviceHasSelectionTopologyTrigger(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			return serviceHasSelectionTopologyTrigger(e.ObjectOld) ||
				serviceHasSelectionTopologyTrigger(e.ObjectNew)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return serviceHasSelectionTopologyTrigger(e.Object)
		},
		GenericFunc: func(e event.GenericEvent) bool { return false },
	}
}

func appendAnnotatedWorkerServiceRequest(ctx context.Context, kubeClient client.Client, requests []reconcile.Request, req reconcile.Request) []reconcile.Request {
	service := &corev1.Service{}
	if err := kubeClient.Get(ctx, req.NamespacedName, service); err != nil {
		if !apierrors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "Failed to get worker Service",
				"workerService", req.NamespacedName.Name,
				"namespace", req.NamespacedName.Namespace)
		}
		return requests
	}
	if !serviceHasSelectionTopologyTrigger(service) {
		return requests
	}
	return append(requests, req)
}

func serviceHasSelectionTopologyTrigger(obj client.Object) bool {
	if obj == nil {
		return false
	}
	annotations := obj.GetAnnotations()
	return strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionAdapter]) != "" ||
		strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionServiceURL]) != ""
}

func endpointSliceTargetPodNames(slice *discoveryv1.EndpointSlice) []string {
	names := sets.New[string]()
	for _, endpoint := range slice.Endpoints {
		if podKey, ok := endpointPodKey(endpoint, slice.Namespace); ok {
			names.Insert(podKey.Name)
		}
	}
	return names.UnsortedList()
}

func selectorServiceIndexKeys(service *corev1.Service) []string {
	annotations := service.GetAnnotations()
	if strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionAdapter]) == "" {
		return nil
	}
	key, ok := selectorServiceIndexKey(annotations[consts.KubeAnnotationDynamoSelectionServiceURL], service.Namespace)
	if !ok {
		return nil
	}
	return []string{key}
}

func selectorServiceIndexKey(rawURL string, defaultNamespace string) (string, bool) {
	parsedSelectorURL, err := parseSelectorURL(rawURL, defaultNamespace)
	if err != nil || !parsedSelectorURL.serviceDNS {
		return "", false
	}
	return selectorServiceKey(parsedSelectorURL.namespace, parsedSelectorURL.serviceName), true
}

func selectorServiceKey(namespace string, name string) string {
	return namespace + "/" + name
}

func dedupeRequests(requests []reconcile.Request) []reconcile.Request {
	seen := sets.New[types.NamespacedName]()
	result := make([]reconcile.Request, 0, len(requests))
	for _, request := range requests {
		if seen.Has(request.NamespacedName) {
			continue
		}
		seen.Insert(request.NamespacedName)
		result = append(result, request)
	}
	return result
}

func (r *SelectionTopologyReconciler) workerTargetsForEndpointSlices(ctx context.Context, ownerService *corev1.Service, slices []*discoveryv1.EndpointSlice) ([]workerTarget, error) {
	var workerTargets []workerTarget

	for _, slice := range slices {
		logger := log.FromContext(ctx).WithValues("endpointSlice", slice.Name, "namespace", slice.Namespace)
		for _, endpoint := range slice.Endpoints {
			if !endpointReady(endpoint) || len(endpoint.Addresses) == 0 {
				continue
			}
			podKey, ok := endpointPodKey(endpoint, slice.Namespace)
			if !ok {
				continue
			}

			pod := &corev1.Pod{}
			if err := r.Get(ctx, podKey, pod); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				return nil, err
			}
			target, ok, err := workerTargetForPodEndpoint(slice, ownerService, pod, endpoint.Addresses[0])
			if err != nil {
				logger.Info("Skipping invalid selection topology worker target",
					"pod", podKey.Name,
					"error", err.Error())
				continue
			}
			if !ok {
				continue
			}
			// Worker IDs are Pod-scoped, so only one annotated Service may own
			// a ready Pod endpoint.
			conflictingService, conflict, err := r.overlappingWorkerServiceForPod(ctx, ownerService, pod)
			if err != nil {
				return nil, err
			}
			if conflict {
				logger.Info("Skipping overlapping selection topology worker target",
					"pod", podKey.Name,
					"workerID", target.WorkerID,
					"conflictingWorkerService", client.ObjectKeyFromObject(conflictingService).String(),
					"reason", "a Pod endpoint can be owned by only one annotated worker Service")
				continue
			}
			workerTargets = append(workerTargets, target)
		}
	}

	return workerTargets, nil
}

func (r *SelectionTopologyReconciler) overlappingWorkerServiceForPod(ctx context.Context, ownerService *corev1.Service, pod *corev1.Pod) (*corev1.Service, bool, error) {
	slices := &discoveryv1.EndpointSliceList{}
	if err := r.List(ctx, slices,
		client.InNamespace(pod.Namespace),
		client.MatchingFields{selectionEndpointSlicePodIndex: pod.Name},
	); err != nil {
		return nil, false, fmt.Errorf("list EndpointSlices for Pod %s/%s: %w", pod.Namespace, pod.Name, err)
	}

	for i := range slices.Items {
		slice := &slices.Items[i]
		if !endpointSliceHasReadyPodEndpoint(slice, pod) {
			continue
		}
		serviceName := strings.TrimSpace(slice.Labels[discoveryv1.LabelServiceName])
		if serviceName == "" || serviceName == ownerService.Name {
			continue
		}

		service := &corev1.Service{}
		serviceKey := types.NamespacedName{Namespace: slice.Namespace, Name: serviceName}
		if err := r.Get(ctx, serviceKey, service); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			return nil, false, fmt.Errorf("get overlapping worker Service %s: %w", serviceKey, err)
		}
		if _, _, ok, err := selectionTopologyConfigFromAnnotations(service); err != nil || !ok {
			continue
		}
		if workerServicePrecedes(service, ownerService) {
			return service, true, nil
		}
	}

	return nil, false, nil
}

func workerServicePrecedes(candidate *corev1.Service, owner *corev1.Service) bool {
	if !candidate.CreationTimestamp.IsZero() || !owner.CreationTimestamp.IsZero() {
		if !candidate.CreationTimestamp.Equal(&owner.CreationTimestamp) {
			return candidate.CreationTimestamp.Before(&owner.CreationTimestamp)
		}
	}

	candidateKey := client.ObjectKeyFromObject(candidate).String()
	ownerKey := client.ObjectKeyFromObject(owner).String()
	if candidateKey != ownerKey {
		return candidateKey < ownerKey
	}
	return string(candidate.UID) < string(owner.UID)
}

func endpointSliceHasReadyPodEndpoint(slice *discoveryv1.EndpointSlice, pod *corev1.Pod) bool {
	for _, endpoint := range slice.Endpoints {
		if !endpointReady(endpoint) {
			continue
		}
		podKey, ok := endpointPodKey(endpoint, slice.Namespace)
		if ok && podKey.Name == pod.Name && podKey.Namespace == pod.Namespace {
			return true
		}
	}
	return false
}

func endpointPodKey(endpoint discoveryv1.Endpoint, defaultNamespace string) (types.NamespacedName, bool) {
	if endpoint.TargetRef == nil || endpoint.TargetRef.Kind != "Pod" || endpoint.TargetRef.Name == "" {
		return types.NamespacedName{}, false
	}
	return types.NamespacedName{
		Namespace: endpointTargetNamespace(endpoint, defaultNamespace),
		Name:      endpoint.TargetRef.Name,
	}, true
}

func endpointTargetNamespace(endpoint discoveryv1.Endpoint, defaultNamespace string) string {
	if endpoint.TargetRef.Namespace != "" {
		return endpoint.TargetRef.Namespace
	}
	return defaultNamespace
}

func workerTargetForPodEndpoint(slice *discoveryv1.EndpointSlice, service *corev1.Service, pod *corev1.Pod, address string) (workerTarget, bool, error) {
	annotations := service.GetAnnotations()
	adapterName, _, ok, err := selectionTopologyConfigFromAnnotations(service)
	if err != nil || !ok {
		return workerTarget{}, ok, err
	}
	port, ok := workerEndpointSlicePort(slice)
	if !ok {
		return workerTarget{}, false, fmt.Errorf("endpoint slice %s/%s has no port named %q",
			slice.Namespace,
			slice.Name,
			consts.DynamoSystemPortName)
	}
	requireKVEvents, err := parseOptionalBool(annotations[consts.KubeAnnotationDynamoSelectionRequireKVEvents], true)
	if err != nil {
		return workerTarget{}, false, fmt.Errorf("parse %s on service %s/%s: %w",
			consts.KubeAnnotationDynamoSelectionRequireKVEvents,
			service.Namespace,
			service.Name,
			err)
	}

	endpointURL := (&url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort(address, strconv.Itoa(int(port))),
	}).String()

	return workerTarget{
		selectionWorkerRegistration: selectionWorkerRegistration{
			WorkerID:        workerIDForPod(pod),
			WorkerEndpoint:  endpointURL,
			TenantID:        strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionTenantID]),
			RequireKVEvents: requireKVEvents,
			TopologyDomains: topologyDomainsFromLabels(pod.Labels),
		},
		AdapterName: adapterName,
		Owner:       workerServiceOwner(service),
		PodName:     pod.Namespace + "/" + pod.Name,
	}, true, nil
}

func selectionTopologyConfigFromAnnotations(service *corev1.Service) (string, string, bool, error) {
	annotations := service.GetAnnotations()
	adapterName := strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionAdapter])
	if adapterName == "" {
		return "", "", false, nil
	}
	rawSelectorURL := strings.TrimSpace(annotations[consts.KubeAnnotationDynamoSelectionServiceURL])
	if rawSelectorURL == "" {
		return "", "", false, fmt.Errorf("service %s/%s has %s=%q but no %s annotation",
			service.Namespace,
			service.Name,
			consts.KubeAnnotationDynamoSelectionAdapter,
			adapterName,
			consts.KubeAnnotationDynamoSelectionServiceURL)
	}
	return adapterName, rawSelectorURL, true, nil
}

func (r *SelectionTopologyReconciler) endpointSlicesForService(ctx context.Context, namespace string, serviceName string) ([]*discoveryv1.EndpointSlice, error) {
	slices := &discoveryv1.EndpointSliceList{}
	if err := r.List(ctx, slices,
		client.InNamespace(namespace),
		client.MatchingLabels{discoveryv1.LabelServiceName: serviceName},
	); err != nil {
		return nil, fmt.Errorf("list EndpointSlices for Service %s/%s: %w", namespace, serviceName, err)
	}
	result := make([]*discoveryv1.EndpointSlice, 0, len(slices.Items))
	for i := range slices.Items {
		result = append(result, &slices.Items[i])
	}
	return result, nil
}

func workerServiceOwner(service *corev1.Service) selectionTopologyOwner {
	return selectionTopologyOwner{
		Kind:      "Service",
		Namespace: service.Namespace,
		Name:      service.Name,
		UID:       string(service.UID),
	}
}

func selectionReconcileScopeForWorkerService(service *corev1.Service) (selectionReconcileScope, string, bool, error) {
	adapterName, rawSelectorURL, ok, err := selectionTopologyConfigFromAnnotations(service)
	if err != nil || !ok {
		return selectionReconcileScope{}, "", ok, err
	}
	return selectionReconcileScope{
		Owner:       workerServiceOwner(service),
		AdapterName: adapterName,
		TenantID:    normalizeSelectionTenantID(strings.TrimSpace(service.GetAnnotations()[consts.KubeAnnotationDynamoSelectionTenantID])),
	}, rawSelectorURL, true, nil
}

func normalizeSelectionTenantID(tenantID string) string {
	return selectionservice.NormalizeTenantID(tenantID)
}

func addSelectionReconcileScopes(catalogPlans map[string]*selectorCatalogPlan, selectorTargets []selectorTarget, scope selectionReconcileScope) {
	for _, selectorTarget := range selectorTargets {
		selectorCatalogPlanFor(catalogPlans, selectorTarget).addScope(scope)
	}
}

func (p *selectorCatalogPlan) addScope(scope selectionReconcileScope) {
	p.Scopes.Insert(scope)
	if p.TenantID == "" {
		p.TenantID = scope.TenantID
	}
}

func removeKnownWorkerSelectorTargets(knownWorkers map[uint64]knownWorkerTargets, workerID uint64, selectorTargets []selectorTarget) {
	known, ok := knownWorkers[workerID]
	if !ok {
		return
	}
	for _, selectorTarget := range selectorTargets {
		known.SelectorTargetURLs.Delete(selectorTarget.TargetURL)
	}
	if known.SelectorTargetURLs.Len() == 0 {
		delete(knownWorkers, workerID)
		return
	}
	knownWorkers[workerID] = known
}

func selectorCatalogPlanFor(catalogPlans map[string]*selectorCatalogPlan, selectorTarget selectorTarget) *selectorCatalogPlan {
	plan := catalogPlans[selectorTarget.TargetURL]
	if plan == nil {
		plan = &selectorCatalogPlan{
			Desired:          map[uint64]selectionservice.WorkerRequest{},
			Scopes:           sets.New[selectionReconcileScope](),
			SelectorScopeKey: selectorTarget.ScopeKey,
		}
		catalogPlans[selectorTarget.TargetURL] = plan
	}
	return plan
}

func workerCatalogMetadata(target workerTarget, selectorScopeKey string) map[string]string {
	return map[string]string{
		selectionMetadataManagedBy:   selectionMetadataManagedByVal,
		selectionMetadataOwnerKind:   target.Owner.Kind,
		selectionMetadataOwnerNS:     target.Owner.Namespace,
		selectionMetadataOwnerName:   target.Owner.Name,
		selectionMetadataOwnerUID:    target.Owner.UID,
		selectionMetadataAdapter:     target.AdapterName,
		selectionMetadataSelectorURL: selectorScopeKey,
	}
}

func selectorActiveTargetURLs(selectorTargets []selectorTarget) sets.Set[string] {
	urls := sets.New[string]()
	for _, selectorTarget := range selectorTargets {
		urls.Insert(selectorTarget.TargetURL)
	}
	return urls
}

func (r *SelectionTopologyReconciler) selectorTargetsForURL(ctx context.Context, rawURL string, defaultNamespace string) ([]selectorTarget, error) {
	parsedSelectorURL, err := parseSelectorURL(rawURL, defaultNamespace)
	if err != nil {
		return nil, err
	}
	if !parsedSelectorURL.serviceDNS {
		log.FromContext(ctx).V(1).Info("Using raw selector URL; operator-managed selector replica fan-out is disabled",
			"selectorURL", parsedSelectorURL.normalized)
		return []selectorTarget{{TargetURL: parsedSelectorURL.normalized, ScopeKey: parsedSelectorURL.scopeKey()}}, nil
	}
	scopeKey := parsedSelectorURL.scopeKey()

	service := &corev1.Service{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: parsedSelectorURL.namespace, Name: parsedSelectorURL.serviceName}, service); err != nil {
		if apierrors.IsNotFound(err) {
			return []selectorTarget{{TargetURL: parsedSelectorURL.normalized, ScopeKey: scopeKey}}, nil
		}
		return nil, fmt.Errorf("get selector Service %s/%s: %w", parsedSelectorURL.namespace, parsedSelectorURL.serviceName, err)
	}

	slices, err := r.endpointSlicesForService(ctx, service.Namespace, service.Name)
	if err != nil {
		return nil, err
	}
	servicePort := parsedSelectorURL.parsed.Port()
	selectorTargetURLSet := sets.New[string]()
	for _, slice := range slices {
		targetPort, ok := selectorEndpointSlicePort(service, slice, servicePort)
		if !ok {
			continue
		}
		for _, endpoint := range slice.Endpoints {
			if !endpointReady(endpoint) || len(endpoint.Addresses) == 0 {
				continue
			}
			targetURL := parsedSelectorURL.parsed
			targetURL.Host = net.JoinHostPort(endpoint.Addresses[0], strconv.Itoa(int(targetPort)))
			selectorTargetURLSet.Insert(targetURL.String())
		}
	}
	if selectorTargetURLSet.Len() == 0 {
		return nil, nil
	}
	targetURLs := sets.List(selectorTargetURLSet)
	selectorTargets := make([]selectorTarget, 0, len(targetURLs))
	for _, targetURL := range targetURLs {
		selectorTargets = append(selectorTargets, selectorTarget{TargetURL: targetURL, ScopeKey: scopeKey})
	}
	return selectorTargets, nil
}

func parseSelectorURL(rawURL string, defaultNamespace string) (parsedSelectorURL, error) {
	normalized := strings.TrimRight(strings.TrimSpace(rawURL), "/")
	parsed, err := url.Parse(normalized)
	if err != nil {
		return parsedSelectorURL{}, fmt.Errorf("parse selector URL %q: %w", rawURL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return parsedSelectorURL{}, fmt.Errorf("selector URL %q must use http or https", rawURL)
	}
	if parsed.Host == "" {
		return parsedSelectorURL{}, fmt.Errorf("selector URL %q must include a host", rawURL)
	}
	serviceName, namespace, serviceDNS := kubernetesServiceDNS(parsed.Hostname(), defaultNamespace)
	return parsedSelectorURL{
		normalized:  normalized,
		parsed:      *parsed,
		serviceName: serviceName,
		namespace:   namespace,
		serviceDNS:  serviceDNS,
	}, nil
}

func (u parsedSelectorURL) scopeKey() string {
	if !u.serviceDNS {
		return u.normalized
	}
	return selectorServiceScopeKey(u.namespace, u.serviceName, u.parsed.Port())
}

func selectorServiceScopeKey(namespace string, name string, port string) string {
	// Scope by Kubernetes Service identity so selector replica IP churn and
	// EndpointSlice address ordering do not create duplicate ownership domains.
	key := "service:" + selectorServiceKey(namespace, name)
	if port != "" {
		key += ":" + port
	}
	return key
}

func kubernetesServiceDNS(host string, defaultNamespace string) (string, string, bool) {
	if host == "" || net.ParseIP(host) != nil || strings.EqualFold(host, "localhost") {
		return "", "", false
	}
	trimmed := strings.TrimSuffix(host, ".")
	parts := strings.Split(trimmed, ".")
	switch {
	case len(parts) == 1:
		return parts[0], defaultNamespace, defaultNamespace != ""
	case len(parts) == 2:
		return parts[0], parts[1], parts[0] != "" && parts[1] != ""
	case len(parts) >= 3 && parts[2] == "svc":
		return parts[0], parts[1], parts[0] != "" && parts[1] != ""
	default:
		return "", "", false
	}
}

func firstEndpointSlicePort(slice *discoveryv1.EndpointSlice) (int32, bool) {
	for _, port := range slice.Ports {
		if port.Port != nil {
			return *port.Port, true
		}
	}
	return 0, false
}

func endpointSlicePortByName(slice *discoveryv1.EndpointSlice, name string) (int32, bool) {
	for _, port := range slice.Ports {
		if port.Port != nil && port.Name != nil && *port.Name == name {
			return *port.Port, true
		}
	}
	return 0, false
}

func selectorEndpointSlicePort(service *corev1.Service, slice *discoveryv1.EndpointSlice, servicePort string) (int32, bool) {
	if servicePort != "" {
		servicePortNumber, err := strconv.ParseInt(servicePort, 10, 32)
		if err != nil {
			return 0, false
		}
		for _, port := range service.Spec.Ports {
			if port.Port != int32(servicePortNumber) {
				continue
			}
			if port.Name != "" {
				if targetPort, ok := endpointSlicePortByName(slice, port.Name); ok {
					return targetPort, true
				}
			}
			if port.TargetPort.IntVal > 0 {
				return port.TargetPort.IntVal, true
			}
			return 0, false
		}
		return 0, false
	}
	return firstEndpointSlicePort(slice)
}

func (r *SelectionTopologyReconciler) selectionClientForTargetURL(selectorTargetURL string, clientsByTargetURL map[string]selectionClient) (selectionClient, error) {
	selectionClient := clientsByTargetURL[selectorTargetURL]
	if selectionClient != nil {
		return selectionClient, nil
	}
	selectionClient, err := r.SelectionClientFactory(selectorTargetURL)
	if err != nil {
		return nil, err
	}
	clientsByTargetURL[selectorTargetURL] = selectionClient
	return selectionClient, nil
}

func (r *SelectionTopologyReconciler) reconcileSelectorCatalogWorkers(
	ctx context.Context,
	logger logr.Logger,
	catalogPlans map[string]*selectorCatalogPlan,
	clientsByTargetURL map[string]selectionClient,
) error {
	for selectorTargetURL, plan := range catalogPlans {
		selectionClient, err := r.selectionClientForTargetURL(selectorTargetURL, clientsByTargetURL)
		if err != nil {
			return fmt.Errorf("create selection service client for %q: %w", selectorTargetURL, err)
		}
		if err := r.deactivateOrphanedCatalogWorkers(ctx, logger, selectionClient, selectorTargetURL, plan.SelectorScopeKey); err != nil {
			return err
		}
		// Tenant is safe to push down. Model stays broad so stale records from
		// older model names in this owner scope can still be removed.
		actualWorkers, err := selectionClient.ListWorkers(ctx, "", plan.TenantID)
		if err != nil {
			return fmt.Errorf("list selection workers from %q: %w", selectorTargetURL, err)
		}
		seenDesired := sets.New[uint64]()
		for _, actual := range actualWorkers {
			if !catalogWorkerInScopes(actual, plan.Scopes, plan.SelectorScopeKey) {
				continue
			}
			expected, ok := plan.Desired[actual.WorkerID]
			if !ok {
				// DELETE /workers leaves an unschedulable tombstone in the
				// selection service catalog. Treat that as already deactivated
				// so steady-state reconciles do not repeatedly delete it.
				if actual.Lifecycle == selectionservice.WorkerLifecycleUnschedulable {
					continue
				}
				if _, err := selectionClient.DeleteWorker(ctx, actual.WorkerID, true); err != nil {
					return fmt.Errorf("deactivate selection worker %d from %q: %w", actual.WorkerID, selectorTargetURL, err)
				}
				continue
			}
			seenDesired.Insert(actual.WorkerID)
			if catalogWorkerMatchesDesired(actual, expected) {
				continue
			}
			if actual.Lifecycle != selectionservice.WorkerLifecycleSchedulable &&
				catalogWorkerDesiredFieldsMatch(actual, expected) {
				logger.Info("Selection worker is not schedulable after desired registration; re-upserting",
					"selectorURL", selectorTargetURL,
					"workerID", actual.WorkerID,
					"lifecycle", actual.Lifecycle,
					"notSchedulableReasons", actual.NotSchedulableReasons)
			}
			if _, err := selectionClient.UpsertWorker(ctx, expected); err != nil {
				return fmt.Errorf("upsert selection worker %d to %q: %w", expected.WorkerID, selectorTargetURL, err)
			}
		}
		for workerID, expected := range plan.Desired {
			if seenDesired.Has(workerID) {
				continue
			}
			if _, err := selectionClient.UpsertWorker(ctx, expected); err != nil {
				return fmt.Errorf("upsert selection worker %d to %q: %w", expected.WorkerID, selectorTargetURL, err)
			}
		}
	}
	return nil
}

func (r *SelectionTopologyReconciler) deactivateOrphanedCatalogWorkers(
	ctx context.Context,
	logger logr.Logger,
	selectionClient selectionClient,
	selectorTargetURL string,
	selectorScopeKey string,
) error {
	actualWorkers, err := selectionClient.ListWorkers(ctx, "", "")
	if err != nil {
		return fmt.Errorf("list selection workers from %q for orphan deactivation: %w", selectorTargetURL, err)
	}
	for _, worker := range actualWorkers {
		if worker.Lifecycle == selectionservice.WorkerLifecycleUnschedulable ||
			worker.Metadata[selectionMetadataManagedBy] != selectionMetadataManagedByVal ||
			worker.Metadata[selectionMetadataOwnerKind] != "Service" ||
			worker.Metadata[selectionMetadataSelectorURL] != selectorScopeKey {
			continue
		}
		ownerNamespace := worker.Metadata[selectionMetadataOwnerNS]
		ownerName := worker.Metadata[selectionMetadataOwnerName]
		ownerUID := worker.Metadata[selectionMetadataOwnerUID]
		if ownerNamespace == "" || ownerName == "" || ownerUID == "" {
			continue
		}
		service := &corev1.Service{}
		serviceKey := types.NamespacedName{Namespace: ownerNamespace, Name: ownerName}
		reason := ""
		if err := r.Get(ctx, serviceKey, service); err != nil {
			if !apierrors.IsNotFound(err) {
				return fmt.Errorf("get selection worker owner Service %s: %w", serviceKey, err)
			}
			reason = "owner Service no longer exists"
		} else if string(service.UID) != ownerUID {
			reason = "owner Service UID changed"
		} else if currentScope, ownerSelectorURL, ok, err := selectionReconcileScopeForWorkerService(service); err != nil || !ok {
			reason = "owner Service no longer has valid selection topology annotations"
		} else if worker.Metadata[selectionMetadataAdapter] != currentScope.AdapterName {
			reason = "owner Service adapter changed"
		} else if normalizeSelectionTenantID(worker.TenantID) != currentScope.TenantID {
			reason = "owner Service tenant changed"
		} else if currentScopeKey, ok := selectorScopeKeyForURL(ownerSelectorURL, service.Namespace); !ok || currentScopeKey != selectorScopeKey {
			reason = "owner Service selector scope changed"
		}
		if reason == "" {
			continue
		}
		logger.Info("Deactivating orphaned selection worker",
			"selectorURL", selectorTargetURL,
			"workerID", worker.WorkerID,
			"ownerService", serviceKey.String(),
			"reason", reason)
		if _, err := selectionClient.DeleteWorker(ctx, worker.WorkerID, true); err != nil {
			return fmt.Errorf("deactivate orphaned selection worker %d from %q: %w", worker.WorkerID, selectorTargetURL, err)
		}
	}
	return nil
}

func selectorScopeKeyForURL(rawURL string, defaultNamespace string) (string, bool) {
	parsedSelectorURL, err := parseSelectorURL(rawURL, defaultNamespace)
	if err != nil {
		return "", false
	}
	return parsedSelectorURL.scopeKey(), true
}

func catalogWorkerMatchesDesired(actual selectionservice.WorkerRecord, expected selectionservice.WorkerRequest) bool {
	// Compare the routing fields produced by the current external SGLang adapter.
	return actual.Lifecycle == selectionservice.WorkerLifecycleSchedulable &&
		catalogWorkerDesiredFieldsMatch(actual, expected)
}

func catalogWorkerDesiredFieldsMatch(actual selectionservice.WorkerRecord, expected selectionservice.WorkerRequest) bool {
	return actual.WorkerID == expected.WorkerID &&
		selectionservice.NormalizeModelName(actual.ModelName) == selectionservice.NormalizeModelName(expected.ModelName) &&
		selectionservice.NormalizeTenantID(actual.TenantID) == selectionservice.NormalizeTenantID(expected.TenantID) &&
		actual.Endpoint == expected.Endpoint &&
		maps.Equal(actual.KVEventsEndpoints, expected.KVEventsEndpoints) &&
		uintRecordFieldMatches(actual.BlockSize, expected.BlockSize) &&
		uintRecordFieldMatches(actual.DataParallelStartRank, expected.DataParallelStartRank) &&
		uintRecordFieldMatches(actual.DataParallelSize, expected.DataParallelSize) &&
		uintRecordFieldMatches(actual.MaxNumBatchedTokens, expected.MaxNumBatchedTokens) &&
		uintRecordFieldMatches(actual.TotalKVBlocks, expected.TotalKVBlocks) &&
		actual.StableRoutingID == expected.StableRoutingID &&
		ptr.Equal(actual.IsEagle, expected.IsEagle) &&
		maps.Equal(actual.TopologyDomains, expected.TopologyDomains) &&
		maps.Equal(actual.Metadata, expected.Metadata)
}

func uintRecordFieldMatches[T ~uint32 | ~uint64](actual *T, expected T) bool {
	if expected == 0 {
		return actual == nil || *actual == 0
	}
	return actual != nil && *actual == expected
}

func catalogWorkerInScopes(worker selectionservice.WorkerRecord, scopes sets.Set[selectionReconcileScope], selectorScopeKey string) bool {
	for scope := range scopes {
		if catalogWorkerInScope(worker, scope, selectorScopeKey) {
			return true
		}
	}
	return false
}

func catalogWorkerInScope(worker selectionservice.WorkerRecord, scope selectionReconcileScope, selectorScopeKey string) bool {
	metadata := worker.Metadata
	if metadata[selectionMetadataManagedBy] != selectionMetadataManagedByVal ||
		metadata[selectionMetadataOwnerKind] != scope.Owner.Kind ||
		metadata[selectionMetadataOwnerNS] != scope.Owner.Namespace ||
		metadata[selectionMetadataOwnerName] != scope.Owner.Name ||
		metadata[selectionMetadataOwnerUID] != scope.Owner.UID ||
		metadata[selectionMetadataSelectorURL] != selectorScopeKey {
		return false
	}
	if scope.AdapterName != "" && metadata[selectionMetadataAdapter] != scope.AdapterName {
		return false
	}
	if scope.TenantID != "" && selectionservice.NormalizeTenantID(worker.TenantID) != scope.TenantID {
		return false
	}
	return true
}

func workerEndpointSlicePort(slice *discoveryv1.EndpointSlice) (int32, bool) {
	for _, port := range slice.Ports {
		if port.Port == nil {
			continue
		}
		if port.Name != nil && *port.Name == consts.DynamoSystemPortName {
			return *port.Port, true
		}
	}
	return 0, false
}

func endpointReady(endpoint discoveryv1.Endpoint) bool {
	return ptr.Deref(endpoint.Conditions.Ready, true)
}

func parseOptionalBool(value string, defaultValue bool) (bool, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return defaultValue, nil
	}
	return strconv.ParseBool(value)
}

func topologyDomainsFromLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	result := map[string]string{}
	for key, value := range labels {
		if strings.HasPrefix(key, consts.KubeLabelDynamoTopologyPrefix) && value != "" {
			domain := strings.TrimPrefix(key, consts.KubeLabelDynamoTopologyPrefix)
			if domain != "" {
				result[domain] = value
			}
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// workerIDForPod is globally unique and stable across selector replicas for one
// Pod lifetime. Pod restarts intentionally get a new ID; restart-stable routing
// identity should be added through an explicit future API.
func workerIDForPod(pod *corev1.Pod) uint64 {
	hash := fnv.New64a()
	_, _ = hash.Write([]byte(pod.Namespace))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(pod.Name))
	_, _ = hash.Write([]byte{0})
	_, _ = hash.Write([]byte(pod.UID))
	workerID := hash.Sum64() & math.MaxInt64
	if workerID == 0 {
		return 1
	}
	return workerID
}

func (r *SelectionTopologyReconciler) deactivateAllKnownWorkersForService(ctx context.Context, serviceKey types.NamespacedName) error {
	knownWorkers := r.knownWorkers(serviceKey)
	if err := r.deactivateKnownWorkersNotInDesired(ctx, knownWorkers, map[uint64]knownWorkerTargets{}, map[string]selectionClient{}, nil, false); err != nil {
		return err
	}
	r.setKnownWorkers(serviceKey, nil)
	return nil
}

func (r *SelectionTopologyReconciler) deactivateKnownWorkersNotInDesired(
	ctx context.Context,
	knownWorkers map[uint64]knownWorkerTargets,
	desiredWorkers map[uint64]knownWorkerTargets,
	clientsByTargetURL map[string]selectionClient,
	activeSelectorTargetURLs sets.Set[string],
	bestEffortInactiveSelectorCleanup bool,
) error {
	logger := log.FromContext(ctx)
	skipInactiveSelectorCleanup := func(workerID uint64, selectorTargetURL string, err error) bool {
		if !bestEffortInactiveSelectorCleanup || activeSelectorTargetURLs.Has(selectorTargetURL) {
			return false
		}
		logger.Info("Skipping deactivation for inactive selector URL",
			"workerID", workerID,
			"selectorURL", selectorTargetURL,
			"reason", "selector URL is no longer active",
			"error", err.Error())
		return true
	}
	for workerID, known := range knownWorkers {
		desiredTargetURLs := desiredWorkers[workerID].SelectorTargetURLs
		for selectorTargetURL := range known.SelectorTargetURLs {
			if desiredTargetURLs.Has(selectorTargetURL) {
				continue
			}
			selectionClient, err := r.selectionClientForTargetURL(selectorTargetURL, clientsByTargetURL)
			if err != nil {
				if skipInactiveSelectorCleanup(workerID, selectorTargetURL, err) {
					continue
				}
				return fmt.Errorf("create selection service client for %q: %w", selectorTargetURL, err)
			}
			if _, err := selectionClient.DeleteWorker(ctx, workerID, true); err != nil {
				if skipInactiveSelectorCleanup(workerID, selectorTargetURL, err) {
					continue
				}
				return fmt.Errorf("deactivate selection worker %d from %q: %w", workerID, selectorTargetURL, err)
			}
		}
	}
	return nil
}

func (r *SelectionTopologyReconciler) knownWorkers(serviceKey types.NamespacedName) map[uint64]knownWorkerTargets {
	r.knownMu.Lock()
	defer r.knownMu.Unlock()

	stored := r.knownByService[serviceKey]
	result := make(map[uint64]knownWorkerTargets, len(stored))
	for workerID, known := range stored {
		result[workerID] = knownWorkerTargets{
			SelectorTargetURLs: known.SelectorTargetURLs.Clone(),
		}
	}
	return result
}

func (r *SelectionTopologyReconciler) setKnownWorkers(serviceKey types.NamespacedName, workers map[uint64]knownWorkerTargets) {
	r.knownMu.Lock()
	defer r.knownMu.Unlock()

	if len(workers) == 0 {
		delete(r.knownByService, serviceKey)
		return
	}
	copied := make(map[uint64]knownWorkerTargets, len(workers))
	for workerID, known := range workers {
		copied[workerID] = knownWorkerTargets{
			SelectorTargetURLs: known.SelectorTargetURLs.Clone(),
		}
	}
	if r.knownByService == nil {
		r.knownByService = map[types.NamespacedName]map[uint64]knownWorkerTargets{}
	}
	r.knownByService[serviceKey] = copied
}
