/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	"dario.cat/mergo"
	"emperror.dev/errors"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1alpha1 "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/v1alpha1"
	commonController "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/internal/controller_common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/internal/nim"
)

const (
	FailedState  = "failed"
	ReadyState   = "successful"
	PendingState = "pending"
)

// CompoundAIDeploymentReconciler reconciles a CompoundAIDeployment object
type CompoundAIDeploymentReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Config   commonController.Config
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=nvidia.com,resources=compoundaideployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundaideployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=compoundaideployments/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the CompoundAIDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *CompoundAIDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var err error
	reason := "undefined"
	readyStatus := metav1.ConditionFalse
	// retrieve the CRD
	compoundAIDeployment := &nvidiacomv1alpha1.CompoundAIDeployment{}
	if err = r.Get(ctx, req.NamespacedName, compoundAIDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if err != nil {
		// not found, nothing to do
		return ctrl.Result{}, nil
	}

	defer func() {
		message := ""
		if err != nil {
			compoundAIDeployment.SetState(FailedState)
			message = err.Error()
		}
		// update the CRD status condition
		compoundAIDeployment.Status.Conditions = []metav1.Condition{
			{
				Type:               "Ready",
				Status:             readyStatus,
				Reason:             reason,
				Message:            message,
				LastTransitionTime: metav1.Now(),
			},
		}
		err = r.Status().Update(ctx, compoundAIDeployment)
		if err != nil {
			logger.Error(err, "Unable to update the CRD status", "crd", req.NamespacedName)
		}
		logger.Info("Reconciliation done")
	}()

	// fetch the CompoundAINIMConfig
	compoundAINIMConfig, err := nim.GetCompoundAINIMConfig(ctx, compoundAIDeployment, r.getSecret, r.Recorder)
	if err != nil {
		reason = "failed_to_get_the_CompoundAINIMConfig"
		return ctrl.Result{}, err
	}

	// generate the CompoundAINimDeployments from the config
	compoundAINimDeployments, err := nim.GenerateCompoundAINIMDeployments(compoundAIDeployment, compoundAINIMConfig)
	if err != nil {
		reason = "failed_to_generate_the_CompoundAINimDeployments"
		return ctrl.Result{}, err
	}

	// merge the CompoundAINimDeployments with the CompoundAINimDeployments from the CRD
	for serviceName, deployment := range compoundAINimDeployments {
		if _, ok := compoundAIDeployment.Spec.Services[serviceName]; ok {
			err := mergo.Merge(deployment, compoundAIDeployment.Spec.Services[serviceName], mergo.WithOverride)
			if err != nil {
				reason = "failed_to_merge_the_CompoundAINimDeployments"
				return ctrl.Result{}, err
			}
		}
	}

	// reconcile the compoundAINimRequest
	compoundAINimRequest := &nvidiacomv1alpha1.CompoundAINimRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      generateCompoundAINimRequestName(compoundAIDeployment.Spec.CompoundAINim),
			Namespace: compoundAIDeployment.Namespace,
		},
		Spec: nvidiacomv1alpha1.CompoundAINimRequestSpec{
			BentoTag: compoundAIDeployment.Spec.CompoundAINim,
		},
	}
	if err := ctrl.SetControllerReference(compoundAIDeployment, compoundAINimRequest, r.Scheme); err != nil {
		reason = "failed_to_set_the_controller_reference_for_the_CompoundAINimRequest"
		return ctrl.Result{}, err
	}
	_, err = commonController.SyncResource(ctx, r.Client, compoundAINimRequest, types.NamespacedName{Name: compoundAINimRequest.Name, Namespace: compoundAINimRequest.Namespace}, true)
	if err != nil {
		reason = "failed_to_sync_the_CompoundAINimRequest"
		return ctrl.Result{}, err
	}

	allAreReady := true
	// reconcile the CompoundAINimDeployments
	for serviceName, compoundAINimDeployment := range compoundAINimDeployments {
		logger.Info("Reconciling the CompoundAINimDeployment", "serviceName", serviceName, "compoundAINimDeployment", compoundAINimDeployment)
		if err := ctrl.SetControllerReference(compoundAIDeployment, compoundAINimDeployment, r.Scheme); err != nil {
			reason = "failed_to_set_the_controller_reference_for_the_CompoundAINimDeployment"
			return ctrl.Result{}, err
		}
		compoundAINimDeployment, err = commonController.SyncResource(ctx, r.Client, compoundAINimDeployment, types.NamespacedName{Name: compoundAINimDeployment.Name, Namespace: compoundAINimDeployment.Namespace}, true)
		if err != nil {
			reason = "failed_to_sync_the_CompoundAINimDeployment"
			return ctrl.Result{}, err
		}
		if !compoundAINimDeployment.Status.IsReady() {
			allAreReady = false
		}
	}
	if allAreReady {
		compoundAIDeployment.SetState(ReadyState)
		readyStatus = metav1.ConditionTrue
	} else {
		compoundAIDeployment.SetState(PendingState)
	}

	return ctrl.Result{}, nil

}

func (r *CompoundAIDeploymentReconciler) getSecret(ctx context.Context, namespace, name string) (*corev1.Secret, error) {
	secret := &corev1.Secret{}
	err := r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: name}, secret)
	return secret, errors.Wrap(err, "get secret")
}

// SetupWithManager sets up the controller with the Manager.
func (r *CompoundAIDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.CompoundAIDeployment{}).
		Named("compoundaideployment").
		Owns(&nvidiacomv1alpha1.CompoundAINimDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
}
