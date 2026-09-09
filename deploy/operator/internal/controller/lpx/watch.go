/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"maps"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
)

func (r *graphReconciler) setupWithManager(mgr ctrl.Manager) error {
	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.LPXGraphDeployment{}, builder.WithPredicates(lpxDeploymentPrimaryPredicate())).
		Named("lpxgraphdeployment").
		Watches(&nvidiacomv1beta1.DynamoGraphDeployment{}, handler.EnqueueRequestsFromMapFunc(func(_ context.Context, obj client.Object) []ctrl.Request {
			return []ctrl.Request{{NamespacedName: client.ObjectKeyFromObject(obj)}}
		}), builder.WithPredicates(lpxSourcePredicate())).
		WithEventFilter(r.lpxControllerEventFilter())
	// Primary/source events and finalizer retries are sufficient while disabled.
	if r.unavailableReason() != "" {
		return ctrlBuilder.Complete(r)
	}

	// Active integration observes its workload and scheduling dependencies.
	ctrlBuilder = ctrlBuilder.Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Service{}).
		Owns(&grovev1alpha1.PodCliqueSet{}).
		Watches(&grovev1alpha1.PodClique{}, handler.EnqueueRequestsFromMapFunc(mapLPXChildToRequests), builder.WithPredicates(lpxPodCliqueEventPredicates())).
		Watches(&grovev1alpha1.PodCliqueScalingGroup{}, handler.EnqueueRequestsFromMapFunc(mapLPXChildToRequests), builder.WithPredicates(lpxScalingGroupEventPredicates()))
	if err := mgr.GetFieldIndexer().IndexField(context.Background(), &nvidiacomv1beta1.DynamoGraphDeployment{}, lpxTopologyBindingRefIndex, lpxTopologyBindingReferences); err != nil {
		return fmt.Errorf("register LPX topology reference index: %w", err)
	}
	ctrlBuilder = r.addLPXWatches(ctrlBuilder)
	if r.runtimeConfig.Gate.Enabled(features.DRA) {
		// Index authored references at each hop without reading other cached objects.
		for _, index := range []struct {
			object client.Object
			name   string
			values client.IndexerFunc
		}{
			{&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxResourceClaimRefIndex, lpxDRAClaimReferences(false)},
			{&nvidiacomv1beta1.DynamoGraphDeployment{}, lpxResourceClaimTemplateRefIndex, lpxDRAClaimReferences(true)},
			{&resourcev1.ResourceClaim{}, lpxDeviceClassRefIndex, lpxDeviceClassReferences},
			{&resourcev1.ResourceClaimTemplate{}, lpxDeviceClassRefIndex, lpxDeviceClassReferences},
		} {
			if err := mgr.GetFieldIndexer().IndexField(context.Background(), index.object, index.name, index.values); err != nil {
				return fmt.Errorf("register LPX %T index %s: %w", index.object, index.name, err)
			}
		}
		for _, dependency := range []client.Object{&resourcev1.ResourceClaim{}, &resourcev1.ResourceClaimTemplate{}, &resourcev1.DeviceClass{}} {
			ctrlBuilder = ctrlBuilder.Watches(dependency, handler.EnqueueRequestsFromMapFunc(r.mapLPXDRADependencyToRequests), builder.WithPredicates(predicate.GenerationChangedPredicate{}))
		}
	}
	return ctrlBuilder.Complete(r)
}

// lpxDeploymentPrimaryPredicate observes every field that grants publication authority.
func lpxDeploymentPrimaryPredicate() predicate.Predicate {
	ownerChanged := predicate.Funcs{UpdateFunc: func(e event.UpdateEvent) bool {
		oldDeployment := e.ObjectOld.(*nvidiacomv1alpha1.LPXGraphDeployment)
		newDeployment := e.ObjectNew.(*nvidiacomv1alpha1.LPXGraphDeployment)
		return !apiequality.Semantic.DeepEqual(oldDeployment.OwnerReferences, newDeployment.OwnerReferences)
	}}
	return predicate.Or(commoncontroller.GenerationOrDeletionChangedPredicate(), predicate.AnnotationChangedPredicate{}, ownerChanged)
}

func (r *graphReconciler) addLPXWatches(ctrlBuilder *builder.Builder) *builder.Builder {
	ctrlBuilder = ctrlBuilder.
		Owns(&lpxv1alpha1.LPUPipelineRequest{}).
		Watches(
			&grovev1alpha1.ClusterTopologyBinding{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				return r.indexedLPXDependencyRequests(ctx, obj, lpxTopologyBindingRefIndex)
			}),
			builder.WithPredicates(predicate.GenerationChangedPredicate{}),
		).
		Watches(
			&groveschedulerv1alpha1.PodGang{},
			handler.EnqueueRequestsFromMapFunc(mapLPXChildToRequests),
			builder.WithPredicates(lpxPodGangPredicate()),
		)
	return ctrlBuilder
}

func (r *graphReconciler) lpxControllerEventFilter() predicate.Predicate {
	// The topology mapper applies namespace filtering to each consuming DGD.
	return predicate.NewPredicateFuncs(func(obj client.Object) bool {
		if _, deviceClass := obj.(*resourcev1.DeviceClass); deviceClass {
			return true
		}
		if _, topology := obj.(*grovev1alpha1.ClusterTopologyBinding); topology && obj.GetNamespace() == "" {
			return true
		}
		return commoncontroller.NamespaceAllowed(r.Config, r.runtimeConfig, obj, obj.GetNamespace())
	})
}

// mapLPXDRADependencyToRequests receives non-nil claim, template and DeviceClass
// events from their registered informers, keeping LPX watches out of DGD.
func (r *graphReconciler) mapLPXDRADependencyToRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	switch dependency := obj.(type) {
	case *resourcev1.ResourceClaim:
		return r.indexedLPXDependencyRequests(ctx, dependency, lpxResourceClaimRefIndex)
	case *resourcev1.ResourceClaimTemplate:
		return r.indexedLPXDependencyRequests(ctx, dependency, lpxResourceClaimTemplateRefIndex)
	case *resourcev1.DeviceClass:
		return r.mapLPXDeviceClassToRequests(ctx, dependency)
	default:
		return nil
	}
}

// mapLPXChildToRequests receives non-nil, namespaced Grove informer objects.
func mapLPXChildToRequests(_ context.Context, obj client.Object) []ctrl.Request {
	dgdName := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	if dgdName == "" {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: dgdName}}}
}

// lpxPodGangPredicate receives non-nil PodGangs from the registered informer.
func lpxPodGangPredicate() predicate.Predicate {
	isSchedulerWitness := func(obj client.Object) bool {
		if obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName] == "" {
			return false
		}
		schedulerName := obj.GetLabels()[grovecommon.LabelSchedulerName]
		return schedulerName == dynamolpx.SchedulerName || schedulerName == corev1.DefaultSchedulerName
	}
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return isSchedulerWitness(e.Object) },
		DeleteFunc: func(e event.DeleteEvent) bool { return isSchedulerWitness(e.Object) },
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldGang := e.ObjectOld.(*groveschedulerv1alpha1.PodGang)
			newGang := e.ObjectNew.(*groveschedulerv1alpha1.PodGang)
			if !isSchedulerWitness(oldGang) && !isSchedulerWitness(newGang) {
				return false
			}
			return oldGang.UID != newGang.UID ||
				oldGang.Generation != newGang.Generation ||
				!maps.Equal(oldGang.Labels, newGang.Labels) ||
				!maps.Equal(oldGang.Annotations, newGang.Annotations) ||
				!apiequality.Semantic.DeepEqual(oldGang.OwnerReferences, newGang.OwnerReferences) ||
				!apiequality.Semantic.DeepEqual(oldGang.DeletionTimestamp, newGang.DeletionTimestamp)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

func lpxPodCliqueMaterializationEvent(obj client.Object) bool {
	clique := obj.(*grovev1alpha1.PodClique)
	role := clique.Annotations[lpxv1alpha1.PodRoleAnnotation]
	return clique.Annotations[dynamolpx.WorkloadDigestAnnotation] != "" &&
		(role == lpxv1alpha1.PodRoleAgent || role == lpxv1alpha1.PodRoleConductor || role == lpxv1alpha1.PodRoleCyborgWorker)
}

func lpxPodCliqueScalingGroupMaterializationEvent(obj client.Object) bool {
	group := obj.(*grovev1alpha1.PodCliqueScalingGroup)
	return group.Annotations[dynamolpx.WorkloadDigestAnnotation] != ""
}

func lpxPodCliqueEventPredicates() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return lpxPodCliqueMaterializationEvent(e.Object) },
		DeleteFunc: func(e event.DeleteEvent) bool { return lpxPodCliqueMaterializationEvent(e.Object) },
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldClique := e.ObjectOld.(*grovev1alpha1.PodClique)
			newClique := e.ObjectNew.(*grovev1alpha1.PodClique)
			if !lpxPodCliqueMaterializationEvent(oldClique) && !lpxPodCliqueMaterializationEvent(newClique) {
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

func lpxScalingGroupEventPredicates() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return lpxPodCliqueScalingGroupMaterializationEvent(e.Object) },
		DeleteFunc: func(e event.DeleteEvent) bool { return lpxPodCliqueScalingGroupMaterializationEvent(e.Object) },
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldGroup := e.ObjectOld.(*grovev1alpha1.PodCliqueScalingGroup)
			newGroup := e.ObjectNew.(*grovev1alpha1.PodCliqueScalingGroup)
			if !lpxPodCliqueScalingGroupMaterializationEvent(oldGroup) && !lpxPodCliqueScalingGroupMaterializationEvent(newGroup) {
				return false
			}
			return commoncontroller.PodCliqueScalingGroupStatusChangeIsSignificant(oldGroup, newGroup) ||
				oldGroup.Generation != newGroup.Generation ||
				!maps.Equal(oldGroup.Annotations, newGroup.Annotations) ||
				!maps.Equal(oldGroup.Labels, newGroup.Labels) ||
				!apiequality.Semantic.DeepEqual(oldGroup.OwnerReferences, newGroup.OwnerReferences) ||
				!apiequality.Semantic.DeepEqual(oldGroup.DeletionTimestamp, newGroup.DeletionTimestamp)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

// lpxSourcePredicate ignores ordinary scaling/status traffic but wakes the child
// for relevant intent, source identity/deletion, and persisted restart selection.
// Its registered informer supplies non-nil DGDs.
func lpxSourcePredicate() predicate.Predicate {
	filter := predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return obj.(*nvidiacomv1beta1.DynamoGraphDeployment).HasLPXComponent()
	})
	filter.UpdateFunc = func(e event.UpdateEvent) bool {
		oldSource := e.ObjectOld.(*nvidiacomv1beta1.DynamoGraphDeployment)
		newSource := e.ObjectNew.(*nvidiacomv1beta1.DynamoGraphDeployment)
		if !oldSource.HasLPXComponent() && !newSource.HasLPXComponent() {
			return false
		}
		if oldSource.UID != newSource.UID || !apiequality.Semantic.DeepEqual(oldSource.DeletionTimestamp, newSource.DeletionTimestamp) {
			return true
		}
		// Spec edits advance generation; only the selected restart depends on status.
		// Avoid conversion and hashing for ordinary status-only events.
		oldRestart, newRestart := dynamo.LPXRestartToken(oldSource, ""), dynamo.LPXRestartToken(newSource, "")
		if oldSource.Generation == newSource.Generation && oldRestart == newRestart &&
			maps.Equal(oldSource.Labels, newSource.Labels) && maps.Equal(oldSource.Annotations, newSource.Annotations) {
			return false
		}
		oldRevision, oldErr := dynamo.LPXInputRevision(oldSource, oldRestart)
		newRevision, newErr := dynamo.LPXInputRevision(newSource, newRestart)
		return oldErr != nil || newErr != nil || oldRevision != newRevision
	}
	return filter
}
