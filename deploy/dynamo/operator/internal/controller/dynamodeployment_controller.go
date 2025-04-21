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
	"fmt"
	"strings"

	"dario.cat/mergo"
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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/nim"
)

const (
	FailedState  = "failed"
	ReadyState   = "successful"
	PendingState = "pending"
)

type etcdStorage interface {
	DeleteKeys(ctx context.Context, prefix string) error
}

// DynamoDeploymentReconciler reconciles a DynamoDeployment object
type DynamoDeploymentReconciler struct {
	client.Client
	Scheme                     *runtime.Scheme
	Config                     commonController.Config
	Recorder                   record.EventRecorder
	VirtualServiceGateway      string
	IngressControllerClassName string
	IngressControllerTLSSecret string
	IngressHostSuffix          string
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamodeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamodeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamodeployments/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *DynamoDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var err error
	reason := "undefined"
	message := ""
	readyStatus := metav1.ConditionFalse
	// retrieve the CRD
	dynamoDeployment := &nvidiacomv1alpha1.DynamoDeployment{}
	if err = r.Get(ctx, req.NamespacedName, dynamoDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if err != nil {
		// not found, nothing to do
		return ctrl.Result{}, nil
	}

	defer func() {
		if err != nil {
			dynamoDeployment.SetState(FailedState)
			message = err.Error()
		}
		// update the CRD status condition
		dynamoDeployment.AddStatusCondition(metav1.Condition{
			Type:               "Ready",
			Status:             readyStatus,
			Reason:             reason,
			Message:            message,
			LastTransitionTime: metav1.Now(),
		})
		err = r.Status().Update(ctx, dynamoDeployment)
		if err != nil {
			logger.Error(err, "Unable to update the CRD status", "crd", req.NamespacedName)
		}
		logger.Info("Reconciliation done")
	}()

	deleted, err := commonController.HandleFinalizer(ctx, dynamoDeployment, r.Client, r)
	if err != nil {
		reason = "failed_to_handle_the_finalizer"
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	// fetch the DynamoNIMConfig
	dynamoNIMConfig, err := nim.GetDynamoNIMConfig(ctx, dynamoDeployment, r.Recorder)
	if err != nil {
		reason = "failed_to_get_the_DynamoNIMConfig"
		return ctrl.Result{}, err
	}

	// generate the DynamoNimDeployments from the config
	dynamoNimDeployments, err := nim.GenerateDynamoNIMDeployments(ctx, dynamoDeployment, dynamoNIMConfig, r.generateDefaultIngressSpec(dynamoDeployment))
	if err != nil {
		reason = "failed_to_generate_the_DynamoNimDeployments"
		return ctrl.Result{}, err
	}

	// merge the DynamoNimDeployments with the DynamoNimDeployments from the CRD
	for serviceName, deployment := range dynamoNimDeployments {
		if _, ok := dynamoDeployment.Spec.Services[serviceName]; ok {
			err := mergo.Merge(&deployment.Spec.DynamoNimDeploymentSharedSpec, dynamoDeployment.Spec.Services[serviceName].DynamoNimDeploymentSharedSpec, mergo.WithOverride)
			if err != nil {
				reason = "failed_to_merge_the_DynamoNimDeployments"
				return ctrl.Result{}, err
			}
		}
		if deployment.Spec.Ingress.Enabled {
			dynamoDeployment.SetEndpointStatus((r.isEndpointSecured()), getIngressHost(deployment.Spec.Ingress))
		}
	}

	// Set common env vars on each of the dynamoNimDeployments
	for _, deployment := range dynamoNimDeployments {
		if len(dynamoDeployment.Spec.Envs) > 0 {
			deployment.Spec.Envs = mergeEnvs(dynamoDeployment.Spec.Envs, deployment.Spec.Envs)
		}
	}

	// reconcile the dynamoNimRequest
	dynamoNimRequest := &nvidiacomv1alpha1.DynamoNimRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      strings.ReplaceAll(dynamoDeployment.Spec.DynamoNim, ":", "--"),
			Namespace: dynamoDeployment.Namespace,
		},
		Spec: nvidiacomv1alpha1.DynamoNimRequestSpec{
			BentoTag: dynamoDeployment.Spec.DynamoNim,
		},
	}
	if err := ctrl.SetControllerReference(dynamoDeployment, dynamoNimRequest, r.Scheme); err != nil {
		reason = "failed_to_set_the_controller_reference_for_the_DynamoNimRequest"
		return ctrl.Result{}, err
	}
	_, err = commonController.SyncResource(ctx, r.Client, dynamoNimRequest, types.NamespacedName{Name: dynamoNimRequest.Name, Namespace: dynamoNimRequest.Namespace}, false)
	if err != nil {
		reason = "failed_to_sync_the_DynamoNimRequest"
		return ctrl.Result{}, err
	}

	notReadyDeployments := []string{}
	// reconcile the DynamoNimDeployments
	for serviceName, dynamoNimDeployment := range dynamoNimDeployments {
		logger.Info("Reconciling the DynamoNimDeployment", "serviceName", serviceName, "dynamoNimDeployment", dynamoNimDeployment)
		if err := ctrl.SetControllerReference(dynamoDeployment, dynamoNimDeployment, r.Scheme); err != nil {
			reason = "failed_to_set_the_controller_reference_for_the_DynamoNimDeployment"
			return ctrl.Result{}, err
		}
		dynamoNimDeployment, err = commonController.SyncResource(ctx, r.Client, dynamoNimDeployment, types.NamespacedName{Name: dynamoNimDeployment.Name, Namespace: dynamoNimDeployment.Namespace}, false)
		if err != nil {
			reason = "failed_to_sync_the_DynamoNimDeployment"
			return ctrl.Result{}, err
		}
		if !dynamoNimDeployment.Status.IsReady() {
			notReadyDeployments = append(notReadyDeployments, dynamoNimDeployment.Name)
		}
	}
	if len(notReadyDeployments) == 0 {
		dynamoDeployment.SetState(ReadyState)
		reason = "all_deployments_are_ready"
		message = "All deployments are ready"
		readyStatus = metav1.ConditionTrue
	} else {
		reason = "some_deployments_are_not_ready"
		message = fmt.Sprintf("The following deployments are not ready: %v", notReadyDeployments)
		dynamoDeployment.SetState(PendingState)
	}

	return ctrl.Result{}, nil

}

func (r *DynamoDeploymentReconciler) generateDefaultIngressSpec(dynamoDeployment *nvidiacomv1alpha1.DynamoDeployment) *nvidiacomv1alpha1.IngressSpec {
	res := &nvidiacomv1alpha1.IngressSpec{
		Enabled:           r.VirtualServiceGateway != "" || r.IngressControllerClassName != "",
		Host:              dynamoDeployment.Name,
		UseVirtualService: r.VirtualServiceGateway != "",
	}
	if r.IngressControllerClassName != "" {
		res.IngressControllerClassName = &r.IngressControllerClassName
	}
	if r.IngressControllerTLSSecret != "" {
		res.TLS = &nvidiacomv1alpha1.IngressTLSSpec{
			SecretName: r.IngressControllerTLSSecret,
		}
	}
	if r.IngressHostSuffix != "" {
		res.HostSuffix = &r.IngressHostSuffix
	}
	if r.VirtualServiceGateway != "" {
		res.VirtualServiceGateway = &r.VirtualServiceGateway
	}
	return res
}

func (r *DynamoDeploymentReconciler) isEndpointSecured() bool {
	return r.IngressControllerTLSSecret != ""
}

func mergeEnvs(common, specific []corev1.EnvVar) []corev1.EnvVar {
	envMap := make(map[string]corev1.EnvVar)

	// Add all common environment variables.
	for _, env := range common {
		envMap[env.Name] = env
	}

	// Override or add with service-specific environment variables.
	for _, env := range specific {
		envMap[env.Name] = env
	}

	// Convert the map back to a slice.
	merged := make([]corev1.EnvVar, 0, len(envMap))
	for _, env := range envMap {
		merged = append(merged, env)
	}
	return merged
}

func (r *DynamoDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoDeployment) error {
	// for now doing nothing
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoDeployment{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named("dynamodeployment").
		Owns(&nvidiacomv1alpha1.DynamoNimDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
}
