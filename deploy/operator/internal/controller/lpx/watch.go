/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"maps"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
)

const (
	dgdControllerOwnerIndex = "lpx.dgdController"
	podCliqueSetKind        = "PodCliqueSet"
	// pipelineRequestPCSOwnerUIDIndex indexes controlling PCS UIDs, never reusable PCS names.
	pipelineRequestPCSOwnerUIDIndex = ".metadata.controller"
)

func (r *graphReconciler) setupWithManager(mgr ctrl.Manager) error {
	// DGD events address exact controller owners, not DGD-named materializations.
	if err := mgr.GetFieldIndexer().IndexField(context.Background(), &v1alpha1.LPXGraphDeployment{}, dgdControllerOwnerIndex, dgdControllerOwnerKey); err != nil {
		return fmt.Errorf("register LPX DGD owner index: %w", err)
	}

	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.LPXGraphDeployment{}, builder.WithPredicates(lpxGraphDeploymentPredicate())).
		Named("lpxgraphdeployment").
		Watches(&v1beta1.DynamoGraphDeployment{}, handler.EnqueueRequestsFromMapFunc(r.mapDGDToLPXGraphDeployments), builder.WithPredicates(dgdPredicate())).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.config, r.runtimeConfig))

	// Primary and DGD events are sufficient while disabled.
	if !r.enabled {
		return ctrlBuilder.Complete(r)
	}

	// Active integration observes its workload and scheduling dependencies.
	if err := mgr.GetFieldIndexer().IndexField(context.Background(), &lpxv1alpha1.LPUPipelineRequest{}, pipelineRequestPCSOwnerUIDIndex, pipelineRequestOwnerUID); err != nil {
		return fmt.Errorf("register LPR owner UID index: %w", err)
	}

	return ctrlBuilder.Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Service{}).
		Owns(&grovev1alpha1.PodCliqueSet{}).
		Watches(&grovev1alpha1.PodClique{}, handler.EnqueueRequestsFromMapFunc(mapChildToLPXGraphDeployment), builder.WithPredicates(podCliquePredicate())).
		Watches(&grovev1alpha1.PodCliqueScalingGroup{}, handler.EnqueueRequestsFromMapFunc(mapChildToLPXGraphDeployment), builder.WithPredicates(podCliqueScalingGroupPredicate())).
		Watches(&lpxv1alpha1.LPUPipelineRequest{}, handler.EnqueueRequestsFromMapFunc(mapChildToLPXGraphDeployment)).
		Complete(r)
}

func pipelineRequestOwnerUID(obj client.Object) []string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil || owner.APIVersion != grovev1alpha1.SchemeGroupVersion.String() || owner.Kind != podCliqueSetKind {
		return nil
	}
	return []string{string(owner.UID)}
}

func dgdControllerOwnerKey(obj client.Object) []string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil || owner.APIVersion != v1beta1.GroupVersion.String() || owner.Kind != v1beta1.DynamoGraphDeploymentGVK.Kind {
		return nil
	}
	return []string{owner.Name + "/" + string(owner.UID)}
}

func (r *graphReconciler) mapDGDToLPXGraphDeployments(ctx context.Context, obj client.Object) []ctrl.Request {
	deployments := &v1alpha1.LPXGraphDeploymentList{}
	ownerKey := obj.GetName() + "/" + string(obj.GetUID())
	if err := r.List(ctx, deployments, client.InNamespace(obj.GetNamespace()), client.MatchingFields{dgdControllerOwnerIndex: ownerKey}); err != nil {
		ctrl.LoggerFrom(ctx).Error(err, "Unable to list LPX deployments for DGD", "dgd", client.ObjectKeyFromObject(obj), "dgdUID", obj.GetUID())
		return nil
	}
	requests := make([]ctrl.Request, 0, len(deployments.Items))
	for index := range deployments.Items {
		requests = append(requests, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(&deployments.Items[index])})
	}
	return requests
}

// mapChildToLPXGraphDeployment receives non-nil, namespaced Grove informer objects.
func mapChildToLPXGraphDeployment(_ context.Context, obj client.Object) []ctrl.Request {
	deploymentName := obj.GetAnnotations()[lpx.DeploymentNameAnnotation]
	if deploymentName == "" {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: deploymentName}}}
}

// lpxGraphDeploymentPredicate observes every field that grants publication authority.
func lpxGraphDeploymentPredicate() predicate.Predicate {
	return predicate.Or(
		commoncontroller.GenerationOrDeletionChangedPredicate(),
		predicate.AnnotationChangedPredicate{},
		predicate.Funcs{UpdateFunc: func(e event.UpdateEvent) bool {
			return !apiequality.Semantic.DeepEqual(e.ObjectOld.GetOwnerReferences(), e.ObjectNew.GetOwnerReferences())
		}},
	)
}

// dgdPredicate ignores ordinary scaling/status traffic but wakes the child
// for relevant intent, DGD identity/deletion, and persisted restart selection.
// Its registered informer supplies non-nil DGDs.
func dgdPredicate() predicate.Predicate {
	filter := predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return obj.(*v1beta1.DynamoGraphDeployment).HasLPXComponent()
	})
	filter.UpdateFunc = func(e event.UpdateEvent) bool {
		oldDGD := e.ObjectOld.(*v1beta1.DynamoGraphDeployment)
		newDGD := e.ObjectNew.(*v1beta1.DynamoGraphDeployment)
		if !oldDGD.HasLPXComponent() && !newDGD.HasLPXComponent() {
			return false
		}
		if oldDGD.UID != newDGD.UID || !apiequality.Semantic.DeepEqual(oldDGD.DeletionTimestamp, newDGD.DeletionTimestamp) {
			return true
		}
		// Spec edits advance generation; only the selected restart depends on status.
		// Avoid conversion and hashing for ordinary status-only events.
		oldRestart, newRestart := dynamo.LPXRestartToken(oldDGD, ""), dynamo.LPXRestartToken(newDGD, "")
		if oldDGD.Generation == newDGD.Generation && oldRestart == newRestart &&
			maps.Equal(oldDGD.Labels, newDGD.Labels) && maps.Equal(oldDGD.Annotations, newDGD.Annotations) {
			return false
		}
		oldRevision, oldErr := dynamo.LPXInputRevision(oldDGD, oldRestart)
		newRevision, newErr := dynamo.LPXInputRevision(newDGD, newRestart)
		return oldErr != nil || newErr != nil || oldRevision != newRevision
	}
	return filter
}

func isLPXPodClique(obj client.Object) bool {
	clique := obj.(*grovev1alpha1.PodClique)
	role := clique.Annotations[lpxv1alpha1.PodRoleAnnotation]
	return clique.Annotations[lpx.WorkloadDigestAnnotation] != "" &&
		(role == lpxv1alpha1.PodRoleAgent || role == lpxv1alpha1.PodRoleConductor || role == lpxv1alpha1.PodRoleCyborgWorker)
}

func isLPXPodCliqueScalingGroup(obj client.Object) bool {
	pcsg := obj.(*grovev1alpha1.PodCliqueScalingGroup)
	return pcsg.Annotations[lpx.WorkloadDigestAnnotation] != ""
}

func podCliquePredicate() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return isLPXPodClique(e.Object) },
		DeleteFunc: func(e event.DeleteEvent) bool { return isLPXPodClique(e.Object) },
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldClique := e.ObjectOld.(*grovev1alpha1.PodClique)
			newClique := e.ObjectNew.(*grovev1alpha1.PodClique)
			if !isLPXPodClique(oldClique) && !isLPXPodClique(newClique) {
				return false
			}
			return commoncontroller.PodCliqueStatusChangeIsSignificant(oldClique, newClique) ||
				!ptr.Equal(oldClique.Status.CurrentPodTemplateHash, newClique.Status.CurrentPodTemplateHash) ||
				oldClique.Generation != newClique.Generation ||
				!maps.Equal(oldClique.Annotations, newClique.Annotations) ||
				!maps.Equal(oldClique.Labels, newClique.Labels) ||
				!apiequality.Semantic.DeepEqual(oldClique.OwnerReferences, newClique.OwnerReferences) ||
				!apiequality.Semantic.DeepEqual(oldClique.DeletionTimestamp, newClique.DeletionTimestamp)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

func podCliqueScalingGroupPredicate() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return isLPXPodCliqueScalingGroup(e.Object) },
		DeleteFunc: func(e event.DeleteEvent) bool { return isLPXPodCliqueScalingGroup(e.Object) },
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldPCSG := e.ObjectOld.(*grovev1alpha1.PodCliqueScalingGroup)
			newPCSG := e.ObjectNew.(*grovev1alpha1.PodCliqueScalingGroup)
			if !isLPXPodCliqueScalingGroup(oldPCSG) && !isLPXPodCliqueScalingGroup(newPCSG) {
				return false
			}
			return commoncontroller.PodCliqueScalingGroupStatusChangeIsSignificant(oldPCSG, newPCSG) ||
				oldPCSG.Generation != newPCSG.Generation ||
				!maps.Equal(oldPCSG.Annotations, newPCSG.Annotations) ||
				!maps.Equal(oldPCSG.Labels, newPCSG.Labels) ||
				!apiequality.Semantic.DeepEqual(oldPCSG.OwnerReferences, newPCSG.OwnerReferences) ||
				!apiequality.Semantic.DeepEqual(oldPCSG.DeletionTimestamp, newPCSG.DeletionTimestamp)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}
