/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/workermetadata"
)

// frontendAdmissionView is what the ready frontends of a DGD publish about the
// workers they route requests to. A rollout step reads it once.
type frontendAdmissionView struct {
	// gated reports that worker rollouts wait for frontends to serve
	// replacement workers before retiring old ones.
	gated bool
	// pending reports that gating applies but what some ready frontend serves
	// is unknown, because its record is missing or unsynced, because no
	// frontend is ready, or because the frontends are rolling out.
	pending bool
	// frontendsRolling reports that a frontend component runs pods of more
	// than one pod template.
	frontendsRolling bool
	// admissions holds what each ready frontend publishes.
	admissions []workermetadata.FrontendAdmission
	// unsynced names the ready frontends without a synced admission record.
	unsynced []string
	// metadataByPodUID indexes the discovery records in the namespace by the
	// UID of the pod that published them.
	metadataByPodUID map[types.UID][]*unstructured.Unstructured
	// templates lists, sorted, the pod-template hashes of frontends known to
	// publish admission records, as the DGD should record them.
	templates []string
}

// resolveFrontendAdmission reads what the ready frontends of dgd publish about
// the workers they serve.
//
// Pod readiness does not mean a frontend can route to a worker: the frontend
// first discovers the worker, builds a pipeline for its model card, needs the
// worker's generation to form a complete serving topology, such as both
// prefill and decode, and activates the routing hops between them. Frontends
// publish the workers they can route to in their DynamoWorkerMetadata.
//
// Gating applies while a ready frontend publishes these records or comes from
// a pod template the DGD records as publishing them, and, when no frontend is
// ready, while any template is recorded. A frontend of a recorded template
// that restarts or loses its records therefore keeps the rollout waiting. A
// template is forgotten once none of its pods remain, so frontends that never
// publish records, from earlier releases or after a downgrade, roll on pod
// availability once no frontend of a recorded template is ready. While a
// frontend component runs pods of more than one template, as when one edit
// changes frontends and workers together, which frontends will serve, and
// whether they publish records, is not known yet, so gating applies and the
// rollout keeps every available old worker until the frontends converge. A DGD
// without a frontend component, or with zero frontend replicas, is not gated.
func (r *dgdWorkerRolloutReconciler) resolveFrontendAdmission(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (frontendAdmissionView, error) {
	recorded := recordedFrontendAdmissionTemplates(dgd)
	hasFrontend, frontendReplicas := frontendComponentReplicas(dgd)
	if !hasFrontend {
		return frontendAdmissionView{}, nil
	}
	if frontendReplicas == 0 {
		return frontendAdmissionView{templates: sortedKeys(recorded)}, nil
	}

	// Find the frontends, including those not yet or no longer ready. This
	// code only reads what it lists, so it shares the cache's objects rather
	// than copying them; that halves the cost of a rollout step.
	frontendPods := &corev1.PodList{}
	if err := r.List(ctx, frontendPods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponentType:       consts.ComponentTypeFrontend,
		},
		client.UnsafeDisableDeepCopy,
	); err != nil {
		return frontendAdmissionView{}, fmt.Errorf("list frontend pods: %w", err)
	}
	metadataByPodUID, err := r.workerMetadataByPodUID(ctx, dgd.Namespace)
	if err != nil {
		return frontendAdmissionView{}, err
	}

	// Read what each ready frontend publishes, which templates publish, and
	// which templates each frontend component runs.
	view := frontendAdmissionView{metadataByPodUID: metadataByPodUID}
	existingTemplates := make(map[string]struct{})
	publishingTemplates := make(map[string]struct{})
	liveTemplatesByComponent := make(map[string]map[string]struct{})
	for i := range frontendPods.Items {
		pod := &frontendPods.Items[i]
		template := pod.Labels[appsv1.DefaultDeploymentUniqueLabelKey]
		if template != "" {
			existingTemplates[template] = struct{}{}
		}
		if template != "" && pod.DeletionTimestamp == nil {
			component := pod.Labels[consts.KubeLabelDynamoComponent]
			if liveTemplatesByComponent[component] == nil {
				liveTemplatesByComponent[component] = make(map[string]struct{})
			}
			liveTemplatesByComponent[component][template] = struct{}{}
		}
		if !podServesTraffic(pod) {
			continue
		}
		admission := workermetadata.FrontendAdmission{Members: make(map[string]struct{})}
		for _, metadata := range metadataByPodUID[pod.UID] {
			workermetadata.AddFrontendAdmission(metadata, &admission)
		}
		_, recordedTemplate := recorded[template]
		if admission.Capable && template != "" {
			publishingTemplates[template] = struct{}{}
		}
		view.gated = view.gated || admission.Capable || (template != "" && recordedTemplate)
		if !admission.Synced {
			view.unsynced = append(view.unsynced, pod.Name)
		}
		view.admissions = append(view.admissions, admission)
	}
	if len(view.admissions) == 0 && len(recorded) > 0 {
		view.gated = true
	}

	// Wait while a frontend component rolls out, or what a frontend serves is unknown.
	for _, templates := range liveTemplatesByComponent {
		view.frontendsRolling = view.frontendsRolling || len(templates) > 1
	}
	view.gated = view.gated || view.frontendsRolling
	view.pending = view.gated && (view.frontendsRolling || len(view.admissions) == 0 || len(view.unsynced) > 0)

	// Keep recorded templates while their pods exist, and record publishing ones.
	templates := make(map[string]struct{}, len(recorded)+len(publishingTemplates))
	for template := range recorded {
		if _, exists := existingTemplates[template]; exists || len(frontendPods.Items) == 0 {
			templates[template] = struct{}{}
		}
	}
	for template := range publishingTemplates {
		templates[template] = struct{}{}
	}
	view.templates = sortedKeys(templates)
	return view, nil
}

// workerMetadataByPodUID indexes the projected DynamoWorkerMetadata resources
// in namespace by the UID of the pod that published them, reading them from the
// shared informer cache. The returned objects are the cache's own, so callers
// must not modify them. It returns nil when the API is not served.
func (r *dgdWorkerRolloutReconciler) workerMetadataByPodUID(
	ctx context.Context,
	namespace string,
) (map[types.UID][]*unstructured.Unstructured, error) {
	metadataList := workermetadata.NewList()
	if err := r.cachedReader.List(ctx, metadataList, client.InNamespace(namespace), client.UnsafeDisableDeepCopy); err != nil {
		if apierrors.IsNotFound(err) || apimeta.IsNoMatchError(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("list DynamoWorkerMetadata: %w", err)
	}
	metadataByPodUID := make(map[types.UID][]*unstructured.Unstructured)
	for i := range metadataList.Items {
		if _, podUID, ok := workermetadata.OwningPod(&metadataList.Items[i]); ok {
			metadataByPodUID[podUID] = append(metadataByPodUID[podUID], &metadataList.Items[i])
		}
	}
	return metadataByPodUID, nil
}

// recordFrontendAdmissionTemplates persists the frontend templates known to
// publish admission records when they changed, keeping dgd at the persisted
// resource version.
func (r *dgdWorkerRolloutReconciler) recordFrontendAdmissionTemplates(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	templates []string,
) error {
	value := strings.Join(templates, ",")
	if dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates] == value {
		return nil
	}
	patch := client.MergeFrom(dgd.DeepCopy())
	if value == "" {
		delete(dgd.Annotations, consts.AnnotationFrontendAdmissionTemplates)
	} else {
		if dgd.Annotations == nil {
			dgd.Annotations = make(map[string]string)
		}
		dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates] = value
	}
	if err := r.Patch(ctx, dgd, patch); err != nil {
		return fmt.Errorf("record frontend admission templates: %w", err)
	}
	return nil
}

func recordedFrontendAdmissionTemplates(dgd *nvidiacomv1beta1.DynamoGraphDeployment) map[string]struct{} {
	recorded := make(map[string]struct{})
	for _, template := range strings.Split(dgd.Annotations[consts.AnnotationFrontendAdmissionTemplates], ",") {
		if template = strings.TrimSpace(template); template != "" {
			recorded[template] = struct{}{}
		}
	}
	return recorded
}

// frontendComponentReplicas reports whether dgd declares a frontend component
// and how many frontend replicas it declares.
func frontendComponentReplicas(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (bool, int32) {
	found := false
	var replicas int32
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if string(component.ComponentType) != consts.ComponentTypeFrontend {
			continue
		}
		found = true
		replicas += ptr.Deref(component.Replicas, 1)
	}
	return found, replicas
}

func sortedKeys(set map[string]struct{}) []string {
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// Condition ReplacementsServed reports, during a rolling update that waits for
// frontends, whether every ready frontend serves the available replacement
// workers.
const (
	conditionTypeReplacementsServed = "ReplacementsServed"
	reasonReplacementsServed        = "replacements_served"
	reasonWaitingForFrontends       = "waiting_for_frontends"
	reasonFrontendRecordsMissing    = "frontend_records_missing"
	reasonFrontendsRolling          = "frontends_rolling"
)

// replacementsServedCondition reports whether the ready frontends serve the
// available replacement replicas of steps, or nil when no rolling component
// waits for frontends.
func replacementsServedCondition(
	generation int64,
	admission frontendAdmissionView,
	steps []workerRolloutStep,
) *metav1.Condition {
	condition := &metav1.Condition{
		Type:               conditionTypeReplacementsServed,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: generation,
		Reason:             reasonReplacementsServed,
		Message:            "Every ready frontend serves the available replacement workers",
	}
	gated := 0
	waiting := make([]string, 0, len(steps))
	for i := range steps {
		step := &steps[i]
		if !step.gated {
			continue
		}
		gated++
		if step.newServed < step.newAvailable {
			waiting = append(waiting, fmt.Sprintf("%s %d/%d", step.componentName, step.newServed, step.newAvailable))
		}
	}
	switch {
	case gated == 0:
		return nil
	case admission.frontendsRolling:
		condition.Status = metav1.ConditionFalse
		condition.Reason = reasonFrontendsRolling
		condition.Message = "Frontends are rolling out; keeping every available previous worker until they finish"
	case admission.pending:
		condition.Status = metav1.ConditionFalse
		condition.Reason = reasonFrontendRecordsMissing
		condition.Message = "No frontend is ready; keeping every available previous worker"
		if len(admission.admissions) > 0 {
			condition.Message = fmt.Sprintf(
				"Frontends without current serving records: %s; keeping every available previous worker",
				strings.Join(admission.unsynced, ", "))
		}
	case len(waiting) > 0:
		condition.Status = metav1.ConditionFalse
		condition.Reason = reasonWaitingForFrontends
		condition.Message = "Available replacement workers that every ready frontend serves: " +
			strings.Join(waiting, ", ")
	}
	return condition
}

// componentServing counts, by owning DCD, the ready pods of one worker
// component that every ready frontend, or at least one, serves.
type componentServing struct {
	byEveryFrontend map[string]int32
	byAnyFrontend   map[string]int32
	// registersModels reports that the component's workers publish model
	// cards, which frontends serve. Ready pods of other components count as
	// served by every frontend.
	registersModels bool
}

// frontendServing reports which ready pods of a worker component the ready
// frontends in view serve. A frontend whose view is not synced serves none.
// It reads the component's pods without copying them out of the cache.
func (r *dgdWorkerRolloutReconciler) frontendServing(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	view frontendAdmissionView,
	componentName string,
) (componentServing, error) {
	componentPods, err := r.listDGDComponentPods(ctx, dgd, componentName, client.UnsafeDisableDeepCopy)
	if err != nil {
		return componentServing{}, fmt.Errorf("list pods for component %s: %w", componentName, err)
	}

	// Collect each pod's model cards. A worker can become Ready before it
	// publishes its cards, so once any pod of the component publishes cards, a
	// pod without them is not served yet; otherwise it has nothing to serve.
	serving := componentServing{
		byEveryFrontend: make(map[string]int32),
		byAnyFrontend:   make(map[string]int32),
	}
	cardsByPod := make(map[string]map[string]struct{}, len(componentPods))
	for i := range componentPods {
		pod := &componentPods[i]
		cards := make(map[string]struct{})
		for _, metadata := range view.metadataByPodUID[pod.UID] {
			workermetadata.AddBaseModelCards(metadata, cards)
		}
		cardsByPod[pod.Name] = cards
		serving.registersModels = serving.registersModels || len(cards) > 0
	}

	// Count each DCD's ready pods by how many frontends serve them.
	for i := range componentPods {
		pod := &componentPods[i]
		if !podServesTraffic(pod) {
			continue
		}
		dcdName := pod.Labels[consts.KubeLabelDynamoSelector]
		cards := cardsByPod[pod.Name]
		if len(cards) == 0 {
			if !serving.registersModels {
				serving.byEveryFrontend[dcdName]++
				serving.byAnyFrontend[dcdName]++
			}
			continue
		}
		servingFrontends := 0
		for _, admission := range view.admissions {
			if admission.Synced && servesAll(admission, cards) {
				servingFrontends++
			}
		}
		if servingFrontends > 0 && servingFrontends == len(view.admissions) {
			serving.byEveryFrontend[dcdName]++
		}
		if servingFrontends > 0 {
			serving.byAnyFrontend[dcdName]++
		}
	}
	return serving, nil
}

// servesAll reports whether admission lists every card in cards.
func servesAll(admission workermetadata.FrontendAdmission, cards map[string]struct{}) bool {
	for card := range cards {
		if _, ok := admission.Members[card]; !ok {
			return false
		}
	}
	return true
}

func isDGDFrontendPod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	return pod.Labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		pod.Labels[consts.KubeLabelDynamoComponentType] == consts.ComponentTypeFrontend
}

// dgdFrontendPodEventPredicate admits frontend events that change which
// frontends must serve a replacement worker before a rollout retires an old one.
func dgdFrontendPodEventPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return isDGDFrontendPod(e.Object)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return isDGDFrontendPod(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldFrontend := isDGDFrontendPod(e.ObjectOld)
			newFrontend := isDGDFrontendPod(e.ObjectNew)
			if oldFrontend != newFrontend {
				return true
			}
			if !newFrontend {
				return false
			}
			return podServesTraffic(e.ObjectOld.(*corev1.Pod)) != podServesTraffic(e.ObjectNew.(*corev1.Pod))
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

// mapDGDFrontendPodToRequests enqueues the DGD of a frontend pod while the
// DGD's worker rollout can wait for its frontends.
func (r *DynamoGraphDeploymentReconciler) mapDGDFrontendPodToRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	if !isDGDFrontendPod(obj) {
		return nil
	}
	return r.requestsAwaitingFrontendAdmission(ctx, obj.GetNamespace(), obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName])
}

// requestsAwaitingFrontendAdmission enqueues the named DGD while a managed
// worker rollout is in progress or about to start, the only time the DGD
// reads what its frontends serve. Other DGDs, including idle ones whose
// frontends churn records with their workers, are not requeued.
func (r *DynamoGraphDeploymentReconciler) requestsAwaitingFrontendAdmission(
	ctx context.Context,
	namespace string,
	dgdName string,
) []ctrl.Request {
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if dgdName == "" || r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: dgdName}, dgd) != nil {
		return nil
	}
	if !isRollingUpdateInProgress(&dgd.Status) {
		desired, err := desiredWorkerHashes(dgd)
		if err == nil && currentWorkerHashesMatchDesired(currentWorkerHashes(dgd), desired) {
			return nil
		}
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: namespace, Name: dgdName}}}
}

// workerMetadataChangedPredicate admits DynamoWorkerMetadata events whose
// projected discovery records changed. The cache holds projections, so changes
// confined to dropped fields, such as a model card's contents, are ignored.
func workerMetadataChangedPredicate() predicate.Predicate {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldMetadata, oldOK := e.ObjectOld.(*unstructured.Unstructured)
			newMetadata, newOK := e.ObjectNew.(*unstructured.Unstructured)
			if !oldOK || !newOK {
				return true
			}
			return !equality.Semantic.DeepEqual(oldMetadata.Object["spec"], newMetadata.Object["spec"])
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

// mapWorkerMetadataToDGDRequests enqueues the DGD whose pod published obj
// while the DGD's worker rollout can wait for its frontends.
func (r *DynamoGraphDeploymentReconciler) mapWorkerMetadataToDGDRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	metadata, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil
	}
	podName, podUID, ok := workermetadata.OwningPod(metadata)
	if !ok {
		return nil
	}
	pod := &corev1.Pod{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: podName}, pod); err != nil || pod.UID != podUID {
		return nil
	}
	return r.requestsAwaitingFrontendAdmission(ctx, obj.GetNamespace(), pod.Labels[consts.KubeLabelDynamoGraphDeploymentName])
}

// podServesTraffic reports whether pod is Ready and neither terminating nor
// finished.
func podServesTraffic(pod *corev1.Pod) bool {
	if pod.DeletionTimestamp != nil || isTerminalPhase(pod.Status.Phase) {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
