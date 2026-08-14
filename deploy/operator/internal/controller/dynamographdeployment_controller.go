/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"

	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
)

const (
	reasonFailedToInitializeWorkerHash Reason = "failed_to_initialize_worker_hash"
	reasonFailedToMigrateWorkerHash    Reason = "failed_to_migrate_worker_hash"
	reasonNoMultinodeOrchestrator      Reason = "no_multinode_orchestrator_available"
	reasonFailedToReconcileResources   Reason = "failed_to_reconcile_the_resources"
	reasonRollingUpdateFailed          Reason = "rolling_update_failed"
	reasonWaitingForCheckpoint         Reason = "waiting_for_checkpoint"

	dgdComponentPodIndex = ".metadata.dgdComponent"
)

// rbacManager interface for managing RBAC resources
type rbacManager interface {
	EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

// DynamoGraphDeploymentReconciler reconciles a DynamoGraphDeployment object
type DynamoGraphDeploymentReconciler struct {
	client.Client
	Config                *configv1alpha1.OperatorConfiguration
	RuntimeConfig         *commoncontroller.RuntimeConfig
	RestConfig            *rest.Config
	Recorder              events.EventRecorder
	DockerSecretRetriever DockerSecretRetriever
	ScaleClient           scale.ScalesGetter
	SSHKeyManager         *secret.SSHKeyManager
	RBACManager           rbacManager
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentscalingadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=grove.io,resources=podcliquesets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=grove.io,resources=podcliques,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliques/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquescalinggroups,verbs=get;list;watch
// +kubebuilder:rbac:groups=grove.io,resources=podcliquescalinggroups/scale,verbs=get;update;patch
// +kubebuilder:rbac:groups=grove.io,resources=clustertopologybindings,verbs=get;list;watch
// +kubebuilder:rbac:groups=scheduling.run.ai,resources=queues,verbs=get;list
// +kubebuilder:rbac:groups=inference.networking.k8s.io,resources=inferencepools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.istio.io,resources=destinationrules,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamocheckpoints,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaimtemplates,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=deviceclasses,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=rolebindings,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoGraphDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *DynamoGraphDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (result ctrl.Result, err error) {
	logger := log.FromContext(ctx)
	// retrieve the CRD
	dynamoDeployment := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err = r.Get(ctx, req.NamespacedName, dynamoDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Finalize deleting resources before validating their now-immutable live configuration.
	if !dynamoDeployment.GetDeletionTimestamp().IsZero() {
		_, err = commoncontroller.HandleFinalizer(ctx, dynamoDeployment, r.Client, r)
		if err != nil {
			logger.Error(err, "failed to handle the finalizer")
		}
		return ctrl.Result{}, err
	}

	// Reject unsupported stored configurations before any primary-resource mutation.
	var compatibilityErrs []error
	for i := range dynamoDeployment.Spec.Components {
		component := &dynamoDeployment.Spec.Components[i]
		for _, compatibilityErr := range checkpoint.ValidateCheckpointCompatibility(component.Experimental) {
			compatibilityErrs = append(
				compatibilityErrs,
				fmt.Errorf("component %q: %w", component.ComponentName, compatibilityErr),
			)
		}
	}
	if compatibilityErr := errors.Join(compatibilityErrs...); compatibilityErr != nil {
		programResult := newWorkloadProgramResult(dynamoDeployment)
		programResult.Fail(dynamoDeployment.Generation, reasonFailedToReconcileResources, compatibilityErr)
		if statusErr := r.persistWorkloadProgramResult(ctx, dynamoDeployment, programResult); statusErr != nil {
			logger.Error(statusErr, "unable to update status after compatibility failure")
			return ctrl.Result{}, statusErr
		}
		return ctrl.Result{}, nil
	}

	// Add the finalizer only after the live object passes compatibility preflight.
	if _, err = commoncontroller.HandleFinalizer(ctx, dynamoDeployment, r.Client, r); err != nil {
		logger.Error(err, "failed to handle the finalizer")
		programResult := newWorkloadProgramResult(dynamoDeployment)
		programResult.Fail(dynamoDeployment.Generation, "failed_to_handle_the_finalizer", err)
		if statusErr := r.persistWorkloadProgramResult(ctx, dynamoDeployment, programResult); statusErr != nil {
			logger.Error(statusErr, "unable to update status after finalizer failure")
		}
		return ctrl.Result{}, err
	}

	program := r.selectWorkloadProgram(dynamoDeployment)
	programResult, programErr := program.Reconcile(ctx, workloadProgramRequest{
		DGD: dynamoDeployment,
	})
	result = programResult.Result
	if statusErr := r.persistWorkloadProgramResult(ctx, dynamoDeployment, programResult); statusErr != nil {
		logger.Error(statusErr, "unable to persist workload program status")
		return result, statusErr
	}
	if programErr != nil {
		logger.Error(programErr, "failed to reconcile the workload program")
		return result, programErr
	}

	logger.Info("Reconciliation done")
	return result, nil
}

func (r *DynamoGraphDeploymentReconciler) persistWorkloadProgramResult(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	result workloadProgramResult,
) error {
	dgd.Status = result.Status
	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("update DynamoGraphDeployment status: %w", err)
	}
	if r.Recorder != nil {
		for _, event := range result.Events {
			r.Recorder.Eventf(dgd, nil, event.Type, event.Reason, "Update", "%s", event.Message)
		}
	}
	return nil
}

func (r *DynamoGraphDeploymentReconciler) isGrovePathway(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return r.RuntimeConfig.Gate.Enabled(features.Grove) && (dgd.Annotations == nil ||
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse)
}

func (r *DynamoGraphDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoDeployment *nvidiacomv1beta1.DynamoGraphDeployment) error {
	syncer := newDGDResourceSyncer(r.Client, r.Recorder)
	return newDGDCheckpointsReconciler(
		syncer,
		r.Config,
		r.RuntimeConfig,
		r.DockerSecretRetriever,
	).deleteAutoCheckpointsForDGD(ctx, dynamoDeployment)
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoGraphDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&corev1.Pod{},
		dgdComponentPodIndex,
		dgdComponentPodIndexValues,
	); err != nil {
		return fmt.Errorf("register DGD component Pod index: %w", err)
	}

	ctrlBuilder := ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeployment{}, builder.WithPredicates(
			generationOrDeletionChangedPredicate(),
		)).
		Named(consts.ResourceTypeDynamoGraphDeployment).
		Watches(
			&nvidiacomv1alpha1.DynamoCheckpoint{},
			handler.EnqueueRequestsFromMapFunc(r.mapAutoCheckpointToDGDRequests),
			builder.WithPredicates(predicate.Funcs{
				CreateFunc:  func(ce event.CreateEvent) bool { return false },
				DeleteFunc:  func(de event.DeleteEvent) bool { return true },
				UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
				GenericFunc: func(ge event.GenericEvent) bool { return true },
			}),
		).
		Watches(
			&corev1.Pod{},
			handler.EnqueueRequestsFromMapFunc(mapDGDWorkerPodToRequests),
			builder.WithPredicates(dgdWorkerPodEventPredicate()),
		).
		Owns(&nvidiacomv1beta1.DynamoComponentDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the adapter
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return false }, // Adapter updates are handled by adapter controller
			GenericFunc: func(ge event.GenericEvent) bool { return false },
		})).
		Owns(&corev1.PersistentVolumeClaim{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the PVC
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig))
	if r.RuntimeConfig.Gate.Enabled(features.Istio) {
		ctrlBuilder = ctrlBuilder.Owns(&networkingv1beta1.DestinationRule{}, builder.WithPredicates(predicate.Funcs{
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return false },
		}))
	}
	if r.RuntimeConfig.Gate.Enabled(features.Grove) {
		ctrlBuilder = newGroveWatchSetup(r.Client).addTo(ctrlBuilder)
	}
	// Wrap with metrics collection
	observedReconciler := observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeployment)
	return ctrlBuilder.Complete(observedReconciler)
}

func (r *DynamoGraphDeploymentReconciler) GetRecorder() events.EventRecorder {
	return r.Recorder
}

func isDGDManagedWorkerPod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	labels := pod.GetLabels()
	return labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		labels[consts.KubeLabelDynamoComponent] != "" &&
		labels[consts.KubeLabelDynamoSelector] != "" &&
		dynamo.IsWorkerComponent(labels[consts.KubeLabelDynamoComponentType])
}

func dgdComponentPodIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	labels := pod.GetLabels()
	dgdName := labels[consts.KubeLabelDynamoGraphDeploymentName]
	componentName := labels[consts.KubeLabelDynamoComponent]
	if dgdName == "" || componentName == "" {
		return nil
	}
	return []string{dgdComponentPodIndexValue(dgdName, componentName)}
}

func dgdComponentPodIndexValue(dgdName, componentName string) string {
	// Both inputs are label values, which cannot contain '/', so this encoding
	// is collision-free.
	return dgdName + "/" + componentName
}

// dgdWorkerPodEventPredicate admits only events that can change whether a
// Recreate rollout is blocked by an old pod. Creation and deletion change pod
// membership; updates matter only when membership or terminality changes. A
// deletion timestamp alone leaves the pod non-terminal and does not enqueue.
func dgdWorkerPodEventPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return isDGDManagedWorkerPod(e.Object)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return isDGDManagedWorkerPod(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldManaged := isDGDManagedWorkerPod(e.ObjectOld)
			newManaged := isDGDManagedWorkerPod(e.ObjectNew)
			if oldManaged != newManaged {
				return true
			}
			if !newManaged {
				return false
			}
			oldPod := e.ObjectOld.(*corev1.Pod)
			newPod := e.ObjectNew.(*corev1.Pod)
			if oldPod.Labels[consts.KubeLabelDynamoGraphDeploymentName] != newPod.Labels[consts.KubeLabelDynamoGraphDeploymentName] ||
				oldPod.Labels[consts.KubeLabelDynamoComponent] != newPod.Labels[consts.KubeLabelDynamoComponent] ||
				oldPod.Labels[consts.KubeLabelDynamoSelector] != newPod.Labels[consts.KubeLabelDynamoSelector] {
				return true
			}
			return isTerminalPhase(oldPod.Status.Phase) != isTerminalPhase(newPod.Status.Phase)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

func mapDGDWorkerPodToRequests(_ context.Context, obj client.Object) []ctrl.Request {
	if !isDGDManagedWorkerPod(obj) {
		return nil
	}
	dgdName := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: dgdName}}}
}

func (r *DynamoGraphDeploymentReconciler) mapAutoCheckpointToDGDRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	ckpt, ok := obj.(*nvidiacomv1alpha1.DynamoCheckpoint)
	if !ok || ckpt == nil {
		return nil
	}
	if ckpt.Annotations == nil || ckpt.Annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
		return nil
	}
	if ckpt.Labels == nil {
		return nil
	}
	dgdName := ckpt.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	if dgdName == "" {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: ckpt.Namespace, Name: dgdName}}}
}
