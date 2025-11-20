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
	"bytes"
	"context"
	"fmt"
	"io"
	"text/template"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	sigsyaml "sigs.k8s.io/yaml"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/webhook/validation"
)

const (
	// State constants
	StateEmpty             = ""
	StatePending           = "Pending"
	StateProfiling         = "Profiling"
	StateDeploying         = "Deploying"
	StateReady             = "Ready"
	StateDeploymentDeleted = "DeploymentDeleted"
	StateFailed            = "Failed"

	// Condition types
	ConditionTypeValidation      = "Validation"
	ConditionTypeProfiling       = "Profiling"
	ConditionTypeSpecGenerated   = "SpecGenerated"
	ConditionTypeDeploymentReady = "DeploymentReady"

	// Event reasons
	EventReasonInitialized          = "Initialized"
	EventReasonValidationFailed     = "ValidationFailed"
	EventReasonProfilingJobCreated  = "ProfilingJobCreated"
	EventReasonProfilingJobFailed   = "ProfilingJobFailed"
	EventReasonAIConfiguratorFailed = "AIConfiguratorFailed"
	EventReasonSpecGenerated        = "SpecGenerated"
	EventReasonSpecChangeRejected   = "SpecChangeRejected"
	EventReasonDeploymentCreated    = "DeploymentCreated"
	EventReasonDeploymentReady      = "DeploymentReady"
	EventReasonDeploymentDegraded   = "DeploymentDegraded"
	EventReasonDeploymentDeleted    = "DeploymentDeleted"

	// Label keys
	LabelApp           = "app"
	LabelDGDR          = "dgdr"
	LabelDGDRName      = "dgdr.nvidia.com/name"
	LabelDGDRNamespace = "dgdr.nvidia.com/namespace"
	LabelManagedBy     = "nvidia.com/managed-by"

	// Label values
	LabelValueDynamoProfiler = "dynamo-profiler"
	LabelValueAICProfiler    = "aic-profiler"
	LabelValueDynamoOperator = "dynamo-operator"

	// Job naming
	JobNamePrefixOnline = "profile-online-"
	JobNamePrefixAIC    = "profile-aic-"

	// Container names
	ContainerNameProfiler     = "profiler"
	ContainerNameOutputCopier = "output-copier"

	// ServiceAccount
	ServiceAccountProfilingJob = "dgdr-profiling-job"

	// ConfigMap naming
	ConfigMapOutputPrefix = "dgdr-output-"

	// Annotation keys
	AnnotationAdditionalResources = "dgdr.nvidia.com/additional-resources"

	// Size limits
	MaxAnnotationSize = 250000 // ~250KB, below K8s 256KB limit

	// Sidecar image
	SidecarImage = "bitnami/kubectl:latest"

	// Volume names
	VolumeNameProfilingConfig = "profiling-config"
	VolumeNameProfilingOutput = "profiling-output"

	// Volume paths
	ProfilingOutputPath = "/data"
	ProfilingOutputFile = "config_with_planner.yaml"
	ProfilingConfigPath = "/config"
	ProfilingConfigFile = "disagg.yaml"

	// Command line arguments
	ArgModel   = "--model"
	ArgBackend = "--backend"
	ArgTTFT    = "--ttft"
	ArgITL     = "--itl"
	ArgConfig  = "--config"

	// Messages
	MessageInitialized               = "DGDR initialized successfully"
	MessageProfilingJobCreated       = "Profiling job created"
	MessageAICProfilingJobCreated    = "AIC profiling job created"
	MessageProfilingInProgress       = "Profiling is in progress"
	MessageSpecGenerated             = "DynamoGraphDeployment spec generated successfully"
	MessageSpecAvailable             = "Generated spec is available in status.generatedDeployment"
	MessageDeploymentCreated         = "DynamoGraphDeployment %s created successfully"
	MessageDeploymentReady           = "DynamoGraphDeployment %s is ready"
	MessageDeploymentDegraded        = "DynamoGraphDeployment %s degraded from Ready to %s"
	MessageDeploymentDeleted         = "DGD %s was deleted. DGDR will not recreate it. Delete this DGDR and create a new one to redeploy."
	MessageInvalidState              = "Invalid state"
	MessageSpecChangeRejected        = "Cannot modify spec in state '%s'. DynamoGraphDeploymentRequest is immutable once profiling starts. Create a new resource with a different name instead."
	MessageJobCreationFailed         = "JobCreationFailed"
	MessageDeploymentCreationFailed  = "DeploymentCreationFailed"
	MessageResultsRetrievalFailed    = "ResultsRetrievalFailed"
	MessageGenerationFailed          = "GenerationFailed"
	MessageAIConfiguratorCheckFailed = "AIConfiguratorCheckFailed"
	MessageProfilingCheckFailed      = "ProfilingCheckFailed"
	MessageConfigMapNotFound         = "ConfigMap %s not found in namespace %s"
	MessageConfigMapKeyNotFound      = "key %s not found in ConfigMap %s"

	// Validation messages
	ValidationErrorModelRequired  = "model is required"
	ValidationErrorITLPositive    = "sla.itl must be positive"
	ValidationErrorTTFTPositive   = "sla.ttft must be positive"
	ValidationErrorInvalidBackend = "invalid backend: %s (must be vllm, sglang, or trtllm)"

	// Valid backend values
	BackendVLLM   = "vllm"
	BackendSGLang = "sglang"
	BackendTRTLLM = "trtllm"
)

// shell script template for the output copier sidecar
const sidecarScriptTemplate = `
set -e
set -o pipefail
# Wait for the profiler container to complete, not just for the file to exist
# This ensures we capture the final config, not intermediate results
echo "Waiting for profiler to complete..."
while true; do
  # Check if profiler container has finished (either Completed or Error state)
  # Use kubectl to check the pod's container status
  STATUS=$(kubectl get pod $HOSTNAME -n {{.Namespace}} -o jsonpath='{.status.containerStatuses[?(@.name=="profiler")].state}' 2>/dev/null || echo "")
  if echo "$STATUS" | grep -q "terminated"; then
    echo "Profiler container has terminated"
    break
  fi
  sleep 5
done

# Now wait for the output file to exist
echo "Waiting for output file {{.OutputPath}}/{{.OutputFile}}..."
while [ ! -f {{.OutputPath}}/{{.OutputFile}} ]; do sleep 2; done
echo "Output file found, creating ConfigMap..."

# Start building ConfigMap YAML with DGD spec
cat >/tmp/cm.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{.ConfigMapName}}
  namespace: {{.Namespace}}
  labels:
    dgdr.nvidia.com/name: {{.DGDRName}}
    nvidia.com/managed-by: dynamo-operator
data:
  {{.OutputFile}}: |
EOF
sed 's/^/    /' {{.OutputPath}}/{{.OutputFile}} >> /tmp/cm.yaml

# Note: Profiling data (raw_data.npz converted to JSON) is included in the
# generated DGD YAML as a separate ConfigMap by the profiler, no need to add it here

kubectl apply -f /tmp/cm.yaml
echo "Saved profiling output to ConfigMap {{.ConfigMapName}}"
`

// DynamoGraphDeploymentRequestReconciler reconciles a DynamoGraphDeploymentRequest object
type DynamoGraphDeploymentRequestReconciler struct {
	client.Client
	Recorder record.EventRecorder
	Config   commonController.Config

	// RBACMgr handles RBAC setup for profiling jobs
	RBACManager RBACManager
}

// RBACManager interface for managing RBAC resources
type RBACManager interface {
	EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

// GetRecorder implements commonController.Reconciler interface
func (r *DynamoGraphDeploymentRequestReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}

// FinalizeResource implements commonController.Finalizer interface
func (r *DynamoGraphDeploymentRequestReconciler) FinalizeResource(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	logger.Info("DGDR finalized successfully", "name", dgdr.Name)
	return nil
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch

// Reconcile handles the reconciliation loop for DynamoGraphDeploymentRequest
func (r *DynamoGraphDeploymentRequestReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Reconciling DynamoGraphDeploymentRequest", "name", req.Name, "namespace", req.Namespace)

	// Fetch the DGDR instance
	dgdr := &nvidiacomv1alpha1.DynamoGraphDeploymentRequest{}
	if err := r.Get(ctx, req.NamespacedName, dgdr); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("DGDR resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get DGDR")
		return ctrl.Result{}, err
	}

	// Handle finalizer using common function
	finalized, err := commonController.HandleFinalizer(ctx, dgdr, r.Client, r)
	if err != nil {
		return ctrl.Result{}, err
	}
	if finalized {
		// Resource was deleted and finalized
		return ctrl.Result{}, nil
	}

	// Check for spec changes (immutability enforcement)
	if dgdr.Status.ObservedGeneration > 0 && dgdr.Status.ObservedGeneration != dgdr.Generation {
		// Spec changed after initial processing
		if dgdr.Status.State == StateProfiling || dgdr.Status.State == StateDeploying ||
			dgdr.Status.State == StateReady || dgdr.Status.State == StateDeploymentDeleted {
			logger.Info("Spec change detected in immutable state",
				"state", dgdr.Status.State,
				"observedGeneration", dgdr.Status.ObservedGeneration,
				"currentGeneration", dgdr.Generation)

			r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonSpecChangeRejected,
				fmt.Sprintf(MessageSpecChangeRejected, dgdr.Status.State))

			// Keep the old observedGeneration to continue rejecting changes
			// No state transition - stay in current state with old spec
			return ctrl.Result{}, nil
		}
	}
	// State machine: handle different states
	switch dgdr.Status.State {
	case StateEmpty:
		return r.handleInitialState(ctx, dgdr)
	case StatePending:
		return r.handlePendingState(ctx, dgdr)
	case StateProfiling:
		return r.handleProfilingState(ctx, dgdr)
	case StateDeploying:
		return r.handleDeployingState(ctx, dgdr)
	case StateReady:
		return r.handleReadyState(ctx, dgdr)
	case StateDeploymentDeleted:
		return r.handleDeploymentDeletedState(ctx, dgdr)
	case StateFailed:
		return r.handleFailedState(ctx, dgdr)
	default:
		logger.Info("Unknown state", "state", dgdr.Status.State)
		return r.updateStateAndRequeue(ctx, dgdr, StateFailed, MessageInvalidState)
	}
}

// handleInitialState processes newly created DGDR resources
func (r *DynamoGraphDeploymentRequestReconciler) handleInitialState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling initial state", "name", dgdr.Name)

	// Validate the spec
	if err := r.validateSpec(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonValidationFailed, err.Error())
		return r.updateStateWithCondition(ctx, dgdr, StateFailed, ConditionTypeValidation, metav1.ConditionFalse, EventReasonValidationFailed, err.Error())
	}

	// Set observedGeneration to track the spec we're processing
	dgdr.Status.ObservedGeneration = dgdr.Generation

	// Populate backend in status from spec for display in kubectl output
	dgdr.Status.Backend = dgdr.Spec.Backend

	// Initialize status
	r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonInitialized, MessageInitialized)
	return r.updateStateAndRequeue(ctx, dgdr, StatePending, MessageInitialized)
}

// handlePendingState starts the profiling process
func (r *DynamoGraphDeploymentRequestReconciler) handlePendingState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling pending state", "name", dgdr.Name)

	// Create profiling job (online or AIC)
	if err := r.createProfilingJob(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonProfilingJobFailed, err.Error())
		return r.updateStateWithCondition(ctx, dgdr, StateFailed, ConditionTypeProfiling, metav1.ConditionFalse, MessageJobCreationFailed, err.Error())
	}

	// Record event with appropriate message
	if isOnlineProfiling(dgdr) {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonProfilingJobCreated, MessageProfilingJobCreated)
	} else {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonProfilingJobCreated, MessageAICProfilingJobCreated)
	}

	// Update to Profiling state with Running status
	return r.updateStateWithCondition(ctx, dgdr, StateProfiling, ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingRunning", MessageProfilingInProgress)
}

// handleProfilingState monitors profiling progress and generates spec when complete
func (r *DynamoGraphDeploymentRequestReconciler) handleProfilingState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling profiling state", "name", dgdr.Name)

	// Check profiling job status (both online and offline/AIC run as Jobs)
	// Note: We watch the Job via Owns(), so we'll be triggered automatically on Job changes
	completed, err := r.checkProfilingJobStatus(ctx, dgdr)
	if err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageProfilingCheckFailed, err.Error())
		// Job failed - transition to Failed state
		return r.updateStateWithCondition(ctx, dgdr, StateFailed, ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingFailed", err.Error())
	}

	if !completed {
		logger.Info("Profiling job still running", "name", dgdr.Name)
		// Don't requeue - we'll be triggered when the Job completes/fails
		return ctrl.Result{}, nil
	}

	// Mark profiling as completed successfully
	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:               ConditionTypeProfiling,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: dgdr.Generation,
		Reason:             "ProfilingCompleted",
		Message:            "Profiling job completed successfully",
	})

	// Retrieve profiling results and generate spec
	if err := r.generateDGDSpec(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageGenerationFailed, err.Error())
		return r.updateStateWithCondition(ctx, dgdr, StateFailed, ConditionTypeSpecGenerated, metav1.ConditionFalse, MessageGenerationFailed, err.Error())
	}

	// Record spec generation event
	r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonSpecGenerated, MessageSpecGenerated)

	// Create additional resources (ConfigMaps) immediately after profiling
	// This ensures that the `planner-profile-data` ConfigMap is available for both auto and manual deployment
	targetNamespace := dgdr.Namespace
	if dgdr.Spec.DeploymentOverrides != nil && dgdr.Spec.DeploymentOverrides.Namespace != "" {
		targetNamespace = dgdr.Spec.DeploymentOverrides.Namespace
	}
	if err := r.createAdditionalResources(ctx, dgdr, targetNamespace); err != nil {
		logger.Error(err, "Failed to create additional resources after profiling")
		// Don't fail the DGDR, just log the error - ConfigMaps can be created manually
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, "ConfigMapCreationFailed",
			fmt.Sprintf("Failed to create ConfigMaps from profiling output: %v", err))
	}

	// If autoApply is enabled, transition to Deploying state
	if dgdr.Spec.AutoApply {
		logger.Info("AutoApply enabled, transitioning to Deploying state")
		return r.updateStateWithCondition(ctx, dgdr, StateDeploying, ConditionTypeSpecGenerated, metav1.ConditionTrue, EventReasonSpecGenerated, MessageSpecGenerated)
	}

	// Otherwise, transition to Ready state
	return r.updateStateWithCondition(ctx, dgdr, StateReady, ConditionTypeSpecGenerated, metav1.ConditionTrue, EventReasonSpecGenerated, MessageSpecAvailable)
}

// handleReadyState handles DGDR in Ready state
func (r *DynamoGraphDeploymentRequestReconciler) handleReadyState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is ready", "name", dgdr.Name)

	// If autoApply is not enabled, nothing to monitor
	if !dgdr.Spec.AutoApply {
		return ctrl.Result{}, nil
	}

	// Check if DGD still exists and monitor its status
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      dgdr.Status.Deployment.Name,
		Namespace: dgdr.Status.Deployment.Namespace,
	}, dgd)

	if apierrors.IsNotFound(err) {
		// DGD was deleted by user
		return r.handleDGDDeleted(ctx, dgdr)
	}

	if err != nil {
		return ctrl.Result{}, err
	}

	// Update deployment status
	dgdr.Status.Deployment.State = dgd.Status.State

	// Check if DGD degraded from Ready
	if dgd.Status.State != "Ready" {
		logger.Info("DGD degraded, transitioning back to Deploying",
			"dgdState", dgd.Status.State)

		dgdr.Status.State = StateDeploying

		r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonDeploymentDegraded,
			fmt.Sprintf(MessageDeploymentDegraded, dgd.Name, dgd.Status.State))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    ConditionTypeDeploymentReady,
			Status:  metav1.ConditionFalse,
			Reason:  EventReasonDeploymentDegraded,
			Message: fmt.Sprintf("Deployment degraded to %s", dgd.Status.State),
		})
	}

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// handleDeployingState handles DGD creation and monitors deployment
func (r *DynamoGraphDeploymentRequestReconciler) handleDeployingState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling deploying state", "name", dgdr.Name)

	if !dgdr.Spec.AutoApply {
		// Shouldn't be in this state without autoApply
		logger.Info("AutoApply not enabled, transitioning to Ready")
		dgdr.Status.State = StateReady
		return ctrl.Result{}, r.Status().Update(ctx, dgdr)
	}

	// Check if we need to create DGD
	if dgdr.Status.Deployment == nil || !dgdr.Status.Deployment.Created {
		return r.createDGD(ctx, dgdr)
	}

	// DGD was already created, check its status
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      dgdr.Status.Deployment.Name,
		Namespace: dgdr.Status.Deployment.Namespace,
	}, dgd)

	if apierrors.IsNotFound(err) {
		// DGD was deleted by user
		return r.handleDGDDeleted(ctx, dgdr)
	}

	if err != nil {
		return ctrl.Result{}, err
	}

	// Update deployment status
	dgdr.Status.Deployment.State = dgd.Status.State

	// Check if DGD is Ready
	if dgd.Status.State == "Ready" {
		logger.Info("DGD is Ready, transitioning to Ready state")
		dgdr.Status.State = StateReady

		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonDeploymentReady,
			fmt.Sprintf(MessageDeploymentReady, dgd.Name))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    ConditionTypeDeploymentReady,
			Status:  metav1.ConditionTrue,
			Reason:  EventReasonDeploymentReady,
			Message: fmt.Sprintf(MessageDeploymentReady, dgd.Name),
		})
	}

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// handleDeploymentDeletedState is a terminal state for when auto-created DGD is deleted
func (r *DynamoGraphDeploymentRequestReconciler) handleDeploymentDeletedState(_ context.Context, _ *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	// Terminal state - nothing to do
	// User must delete this DGDR and create a new one to redeploy
	return ctrl.Result{}, nil
}

// handleDGDDeleted handles the case when auto-created DGD is deleted by user
func (r *DynamoGraphDeploymentRequestReconciler) handleDGDDeleted(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGD was deleted by user, transitioning to DeploymentDeleted state")

	dgdr.Status.State = StateDeploymentDeleted
	dgdr.Status.Deployment.State = "Deleted"

	r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonDeploymentDeleted,
		fmt.Sprintf(MessageDeploymentDeleted, dgdr.Status.Deployment.Name))

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  EventReasonDeploymentDeleted,
		Message: "Deployment was deleted by user. Create a new DGDR to redeploy.",
	})

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createDGD creates a DynamoGraphDeployment with the generated spec
func (r *DynamoGraphDeploymentRequestReconciler) createDGD(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Extract DGD from RawExtension
	if dgdr.Status.GeneratedDeployment == nil {
		return ctrl.Result{}, fmt.Errorf("generatedDeployment is not set")
	}

	generatedDGD := &nvidiacomv1alpha1.DynamoGraphDeployment{}

	// RawExtension can have either Object (already decoded) or Raw (JSON bytes)
	if dgdr.Status.GeneratedDeployment.Object != nil {
		var ok bool
		generatedDGD, ok = dgdr.Status.GeneratedDeployment.Object.(*nvidiacomv1alpha1.DynamoGraphDeployment)
		if !ok {
			return ctrl.Result{}, fmt.Errorf("generatedDeployment.Object is not a DynamoGraphDeployment")
		}
	} else if dgdr.Status.GeneratedDeployment.Raw != nil {
		if err := yaml.Unmarshal(dgdr.Status.GeneratedDeployment.Raw, generatedDGD); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to unmarshal generated deployment: %w", err)
		}
	} else {
		return ctrl.Result{}, fmt.Errorf("generatedDeployment has neither Object nor Raw set")
	}

	// Determine DGD name and namespace
	dgdName := generatedDGD.Name
	dgdNamespace := dgdr.Namespace

	if dgdr.Spec.DeploymentOverrides != nil {
		if dgdr.Spec.DeploymentOverrides.Name != "" {
			dgdName = dgdr.Spec.DeploymentOverrides.Name
		}
		if dgdr.Spec.DeploymentOverrides.Namespace != "" {
			dgdNamespace = dgdr.Spec.DeploymentOverrides.Namespace
		}
	}

	// Build labels (start with generated DGD's labels)
	labels := make(map[string]string)
	if generatedDGD.Labels != nil {
		for k, v := range generatedDGD.Labels {
			labels[k] = v
		}
	}
	// Add/override with managed labels
	labels[LabelDGDRName] = dgdr.Name
	labels[LabelDGDRNamespace] = dgdr.Namespace
	labels[LabelManagedBy] = LabelValueDynamoOperator

	// Merge custom labels from overrides
	if dgdr.Spec.DeploymentOverrides != nil && dgdr.Spec.DeploymentOverrides.Labels != nil {
		for k, v := range dgdr.Spec.DeploymentOverrides.Labels {
			labels[k] = v
		}
	}

	// Build annotations (start with generated DGD's annotations)
	annotations := make(map[string]string)
	if generatedDGD.Annotations != nil {
		for k, v := range generatedDGD.Annotations {
			annotations[k] = v
		}
	}
	// Merge custom annotations from overrides
	if dgdr.Spec.DeploymentOverrides != nil && dgdr.Spec.DeploymentOverrides.Annotations != nil {
		for k, v := range dgdr.Spec.DeploymentOverrides.Annotations {
			annotations[k] = v
		}
	}

	// Create DGD from generated deployment
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        dgdName,
			Namespace:   dgdNamespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: generatedDGD.Spec,
	}

	// Note: We don't set owner reference on DGD
	// If a DGDR is deleted, the DGD may be serving traffic and should persist independently.
	// We use labels (LabelDGDRName) to track the relationship.

	logger.Info("Creating DynamoGraphDeployment", "name", dgdName, "namespace", dgdNamespace)

	if err := r.Create(ctx, dgd); err != nil {
		if apierrors.IsAlreadyExists(err) {
			// DGD already exists, just update status
			logger.Info("DGD already exists, updating status")
			dgdr.Status.Deployment = &nvidiacomv1alpha1.DeploymentStatus{
				Name:      dgdName,
				Namespace: dgdNamespace,
				State:     "Pending",
				Created:   true,
			}
			return ctrl.Result{}, r.Status().Update(ctx, dgdr)
		}
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageDeploymentCreationFailed, err.Error())
		return ctrl.Result{}, err
	}

	// Update status
	dgdr.Status.Deployment = &nvidiacomv1alpha1.DeploymentStatus{
		Name:      dgdName,
		Namespace: dgdNamespace,
		State:     "Pending",
		Created:   true,
	}

	r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonDeploymentCreated,
		fmt.Sprintf(MessageDeploymentCreated, dgdName))

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  EventReasonDeploymentCreated,
		Message: fmt.Sprintf("DGD %s created, waiting for Ready", dgdName),
	})

	logger.Info("DynamoGraphDeployment created successfully", "name", dgdName)

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createAdditionalResources creates ConfigMaps from the profiling output that should be deployed alongside the DGD
func (r *DynamoGraphDeploymentRequestReconciler) createAdditionalResources(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, targetNamespace string) error {
	logger := log.FromContext(ctx)

	// Check if there are additional resources stored in annotations
	if dgdr.Annotations == nil {
		return nil
	}

	resourcesYAML, exists := dgdr.Annotations[AnnotationAdditionalResources]
	if !exists || resourcesYAML == "" {
		return nil
	}

	// Parse using standard Kubernetes YAML decoder
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(resourcesYAML)), 4096)
	resourceCount := 0

	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if err == io.EOF {
				break
			}
			logger.Error(err, "Failed to decode resource, skipping")
			continue
		}

		if obj.GetKind() == "" {
			continue
		}

		resourceCount++

		// Only support ConfigMap for now (what profiler actually generates)
		if obj.GetKind() != "ConfigMap" {
			logger.Info("Skipping non-ConfigMap resource from profiling output", "kind", obj.GetKind(), "name", obj.GetName())
			continue
		}

		cm := &corev1.ConfigMap{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, cm); err != nil {
			logger.Error(err, "Failed to convert to ConfigMap", "name", obj.GetName())
			continue
		}

		// Override namespace and add tracking labels
		cm.Namespace = targetNamespace
		if cm.Labels == nil {
			cm.Labels = make(map[string]string)
		}
		cm.Labels[LabelDGDRName] = dgdr.Name
		cm.Labels[LabelDGDRNamespace] = dgdr.Namespace
		cm.Labels[LabelManagedBy] = LabelValueDynamoOperator

		// Create the ConfigMap
		if err := r.Create(ctx, cm); err != nil {
			if apierrors.IsAlreadyExists(err) {
				logger.Info("ConfigMap already exists, skipping", "name", cm.Name)
			} else {
				return fmt.Errorf("failed to create ConfigMap %s: %w", cm.Name, err)
			}
		} else {
			logger.Info("Created ConfigMap from profiling output", "name", cm.Name, "namespace", targetNamespace)
		}
	}

	if resourceCount > 0 {
		logger.Info("Deploying additional resources from profiling output", "count", resourceCount)
	}

	return nil
}

// handleFailedState handles DGDR in Failed state
func (r *DynamoGraphDeploymentRequestReconciler) handleFailedState(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is in failed state", "name", dgdr.Name)

	// Could implement retry logic here if desired
	return ctrl.Result{}, nil
}

// getProfilingJobName returns the job name for a DGDR
func getProfilingJobName(dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) string {
	// Use "profile-" prefix for all profiling jobs
	return fmt.Sprintf("profile-%s", dgdr.Name)
}

// getOutputConfigMapName returns the ConfigMap name for profiling output
func getOutputConfigMapName(dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) string {
	return fmt.Sprintf("%s%s", ConfigMapOutputPrefix, dgdr.Name)
}

// isOnlineProfiling determines whether online profiling or AI Configurator is being used
// based on the sweep.use_ai_configurator config value
func isOnlineProfiling(dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) bool {
	if dgdr.Spec.ProfilingConfig.Config == nil {
		return true
	}

	var config map[string]interface{}
	if err := yaml.Unmarshal(dgdr.Spec.ProfilingConfig.Config.Raw, &config); err != nil {
		return true // Default to online on parse error
	}

	if sweep, ok := config["sweep"].(map[string]interface{}); ok {
		if useAIC, exists := sweep["use_ai_configurator"].(bool); exists {
			return !useAIC
		}
	}
	// Default to online profiling if not specified
	return true
}

// validateSpec validates the DGDR spec
func (r *DynamoGraphDeploymentRequestReconciler) validateSpec(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) error {
	// Use the validator for simple validation (defense in depth - only when webhooks are disabled)
	if !r.Config.WebhooksEnabled {
		isClusterWide := r.Config.RestrictedNamespace == ""
		validator := webhookvalidation.NewDynamoGraphDeploymentRequestValidator(dgdr, isClusterWide)
		warnings, err := validator.Validate()
		if err != nil {
			return err
		}

		// Log warnings if any
		if len(warnings) > 0 {
			logger := log.FromContext(ctx)
			for _, warning := range warnings {
				logger.Info("Validation warning", "warning", warning)
			}
		}
	}

	// Validate ConfigMap if provided (for the DGD base config)
	// This requires cluster access and cannot be done in the stateless validator
	if dgdr.Spec.ProfilingConfig.ConfigMapRef != nil {
		cm := &corev1.ConfigMap{}
		err := r.Get(ctx, types.NamespacedName{
			Name:      dgdr.Spec.ProfilingConfig.ConfigMapRef.Name,
			Namespace: dgdr.Namespace,
		}, cm)

		if err != nil {
			if apierrors.IsNotFound(err) {
				return fmt.Errorf(MessageConfigMapNotFound,
					dgdr.Spec.ProfilingConfig.ConfigMapRef.Name, dgdr.Namespace)
			}
			return err
		}

		// Validate key exists
		key := dgdr.Spec.ProfilingConfig.ConfigMapRef.Key
		if key == "" {
			key = "disagg.yaml"
		}

		if _, exists := cm.Data[key]; !exists {
			return fmt.Errorf(MessageConfigMapKeyNotFound, key, cm.Name)
		}
	}

	// The profiler will validate the rest of the configuration
	return nil
}

// createProfilingJob creates a Kubernetes Job for profiling using SyncResource
func (r *DynamoGraphDeploymentRequestReconciler) createProfilingJob(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	// Delete any existing output ConfigMap to ensure fresh profiling results
	// This prevents using stale data from previous profiling runs
	outputConfigMapName := getOutputConfigMapName(dgdr)
	existingCM := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      outputConfigMapName,
		Namespace: dgdr.Namespace,
	}, existingCM)
	if err == nil {
		// ConfigMap exists, delete it
		logger.Info("Deleting existing output ConfigMap to ensure fresh profiling results", "configMap", outputConfigMapName)
		if err := r.Delete(ctx, existingCM); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "Failed to delete existing output ConfigMap", "configMap", outputConfigMapName)
			return fmt.Errorf("failed to delete existing output ConfigMap: %w", err)
		}
		logger.Info("Successfully deleted old output ConfigMap", "configMap", outputConfigMapName)
	} else if !apierrors.IsNotFound(err) {
		// Unexpected error checking for ConfigMap
		logger.Error(err, "Failed to check for existing output ConfigMap", "configMap", outputConfigMapName)
		return fmt.Errorf("failed to check for existing output ConfigMap: %w", err)
	}

	// Ensure profiling job RBAC exists (only for cluster-wide installation)
	if r.Config.RestrictedNamespace == "" {
		if err := r.RBACManager.EnsureServiceAccountWithRBAC(
			ctx,
			dgdr.Namespace,
			ServiceAccountProfilingJob,
			r.Config.RBAC.DGDRProfilingClusterRoleName,
		); err != nil {
			logger.Error(err, "Failed to ensure profiling job RBAC")
			return fmt.Errorf("failed to ensure profiling job RBAC: %w", err)
		}
	}

	// Use SyncResource to create/update the job
	modified, job, err := commonController.SyncResource(ctx, r, dgdr, func(ctx context.Context) (*batchv1.Job, bool, error) {
		jobName := getProfilingJobName(dgdr)
		outputConfigMapName := getOutputConfigMapName(dgdr)

		// Parse the profiling config from JSON
		var config map[string]interface{}
		if err := yaml.Unmarshal(dgdr.Spec.ProfilingConfig.Config.Raw, &config); err != nil {
			return nil, false, fmt.Errorf("failed to parse profiling config: %w", err)
		}

		// Set deployment.namespace if not already set
		deploymentVal, hasDeployment := config["deployment"]
		var deploymentConfig map[string]interface{}
		if !hasDeployment || deploymentVal == nil {
			deploymentConfig = make(map[string]interface{})
			config["deployment"] = deploymentConfig
		} else {
			var ok bool
			deploymentConfig, ok = deploymentVal.(map[string]interface{})
			if !ok {
				return nil, false, fmt.Errorf("profilingConfig.config.deployment must be an object, got %T", deploymentVal)
			}
		}
		if _, hasNamespace := deploymentConfig["namespace"]; !hasNamespace {
			deploymentConfig["namespace"] = dgdr.Namespace
		}

		// Set deployment.model from spec.model
		deploymentConfig["model"] = dgdr.Spec.Model

		// Set deployment.dgd_image from deploymentOverrides.workersImage if provided
		if dgdr.Spec.DeploymentOverrides != nil && dgdr.Spec.DeploymentOverrides.WorkersImage != "" {
			deploymentConfig["dgd_image"] = dgdr.Spec.DeploymentOverrides.WorkersImage
		}

		// Set output_dir if not already set
		if _, hasOutputDir := config["output_dir"]; !hasOutputDir {
			config["output_dir"] = ProfilingOutputPath
		}

		// Set engine.backend from spec.backend
		engineVal, hasEngine := config["engine"]
		var engineConfig map[string]interface{}
		if !hasEngine || engineVal == nil {
			engineConfig = make(map[string]interface{})
			config["engine"] = engineConfig
		} else {
			var ok bool
			engineConfig, ok = engineVal.(map[string]interface{})
			if !ok {
				return nil, false, fmt.Errorf("profilingConfig.config.engine must be an object, got %T", engineVal)
			}
		}
		engineConfig["backend"] = dgdr.Spec.Backend

		// If ConfigMapRef is provided, set engine.config path
		if dgdr.Spec.ProfilingConfig.ConfigMapRef != nil {
			engineConfig["config"] = fmt.Sprintf("%s/%s", ProfilingConfigPath, ProfilingConfigFile)
		}

		// Serialize config to YAML for passing to profiler
		configYAML, err := sigsyaml.Marshal(config)
		if err != nil {
			return nil, false, fmt.Errorf("failed to marshal profiling config to YAML: %w", err)
		}

		// Common environment variables
		profilerEnv := []corev1.EnvVar{
			{
				Name: "HUGGING_FACE_HUB_TOKEN",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: "hf-token-secret",
						},
						Key: "HF_TOKEN",
					},
				},
			},
			{
				Name:  "NATS_SERVER",
				Value: fmt.Sprintf("nats://%s-nats:4222", dgdr.Namespace),
			},
			{
				Name:  "ETCD_ENDPOINTS",
				Value: fmt.Sprintf("%s-etcd:2379", dgdr.Namespace),
			},
			// DGDR metadata for setting ownerReferences
			{
				Name:  "DGDR_NAME",
				Value: dgdr.Name,
			},
			{
				Name:  "DGDR_NAMESPACE",
				Value: dgdr.Namespace,
			},
			{
				Name:  "DGDR_UID",
				Value: string(dgdr.UID),
			},
		}

		// Build volume mounts
		volumeMounts := []corev1.VolumeMount{
			{
				Name:      VolumeNameProfilingOutput,
				MountPath: ProfilingOutputPath,
			},
		}

		// Add ConfigMap volume mount if provided
		if dgdr.Spec.ProfilingConfig.ConfigMapRef != nil {
			volumeMounts = append(volumeMounts, corev1.VolumeMount{
				Name:      VolumeNameProfilingConfig,
				MountPath: ProfilingConfigPath,
				ReadOnly:  true,
			})
		}

		// Profiler args: pass the config as an inline YAML string via --profile-config
		profilerArgs := []string{
			"--profile-config", string(configYAML),
		}

		// Add --enable-gpu-discovery flag based on DGDR spec
		// GPU discovery requires cluster-wide node access
		if dgdr.Spec.EnableGpuDiscovery {
			profilerArgs = append(profilerArgs, "--enable-gpu-discovery")
		}

		// Use profiler image from profilingConfig
		imageName := dgdr.Spec.ProfilingConfig.ProfilerImage
		logger.Info("Using profiler image", "image", imageName)

		profilerContainer := corev1.Container{
			Name:    ContainerNameProfiler,
			Image:   imageName,
			Command: []string{"python", "-m", "benchmarks.profiler.profile_sla"},
			Args:    profilerArgs,
			Resources: corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("16"),
					corev1.ResourceMemory: resource.MustParse("10Gi"),
				},
			},
			Env:          profilerEnv,
			VolumeMounts: volumeMounts,
		}

		// Generate sidecar script from template
		tmpl, err := template.New("sidecar").Parse(sidecarScriptTemplate)
		if err != nil {
			return nil, false, fmt.Errorf("failed to parse sidecar script template: %w", err)
		}

		var scriptBuf bytes.Buffer
		err = tmpl.Execute(&scriptBuf, map[string]string{
			"OutputPath":    ProfilingOutputPath,
			"OutputFile":    ProfilingOutputFile,
			"ConfigMapName": outputConfigMapName,
			"Namespace":     dgdr.Namespace,
			"DGDRName":      dgdr.Name,
		})
		if err != nil {
			return nil, false, fmt.Errorf("failed to execute sidecar script template: %w", err)
		}

		sidecarContainer := corev1.Container{
			Name:    ContainerNameOutputCopier,
			Image:   SidecarImage,
			Command: []string{"/bin/sh", "-c"},
			Args:    []string{scriptBuf.String()},
			VolumeMounts: []corev1.VolumeMount{{
				Name:      VolumeNameProfilingOutput,
				MountPath: ProfilingOutputPath,
				ReadOnly:  true,
			}},
		}

		// Build volumes - use emptyDir for profiling output
		// The sidecar saves all needed data to ConfigMaps, so persistence is not needed
		volumes := []corev1.Volume{{
			Name: VolumeNameProfilingOutput,
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		}}

		// Add ConfigMap volume if provided
		if dgdr.Spec.ProfilingConfig.ConfigMapRef != nil {
			key := dgdr.Spec.ProfilingConfig.ConfigMapRef.Key
			if key == "" {
				key = ProfilingConfigFile
			}

			volumes = append(volumes, corev1.Volume{
				Name: VolumeNameProfilingConfig,
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: dgdr.Spec.ProfilingConfig.ConfigMapRef.Name,
						},
						Items: []corev1.KeyToPath{{
							Key:  key,
							Path: ProfilingConfigFile,
						}},
					},
				},
			})
		}

		// Limit retries to prevent infinite loop
		backoffLimit := int32(3)

		// Determine label based on whether AI Configurator is used
		labelValue := LabelValueDynamoProfiler
		if !isOnlineProfiling(dgdr) {
			labelValue = LabelValueAICProfiler
		}

		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      jobName,
				Namespace: dgdr.Namespace,
				Labels: map[string]string{
					LabelApp:       labelValue,
					LabelDGDR:      dgdr.Name,
					LabelManagedBy: LabelValueDynamoOperator,
				},
			},
			Spec: batchv1.JobSpec{
				BackoffLimit: &backoffLimit,
				Template: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						ServiceAccountName: ServiceAccountProfilingJob,
						RestartPolicy:      corev1.RestartPolicyNever, SecurityContext: &corev1.PodSecurityContext{
							RunAsNonRoot: ptr.To(true),        // Enforces that container cannot run as root
							RunAsUser:    ptr.To[int64](1000), // Run as UID 1000 (non-privileged user)
							RunAsGroup:   ptr.To[int64](1000), // Run with GID 1000 (non-privileged group)
							FSGroup:      ptr.To[int64](1000), // Volume files owned by GID 1000
						},
						Containers: []corev1.Container{profilerContainer, sidecarContainer},
						Volumes:    volumes,
						ImagePullSecrets: []corev1.LocalObjectReference{
							{Name: "nvcr-imagepullsecret"},
						},
					},
				},
			},
		}

		return job, false, nil
	})

	if err != nil {
		return err
	}

	if modified {
		logger.Info("Profiling job created/updated", "job", job.Name)
	}

	return nil
}

// checkProfilingJobStatus checks if the profiling job has completed
func (r *DynamoGraphDeploymentRequestReconciler) checkProfilingJobStatus(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) (bool, error) {
	logger := log.FromContext(ctx)
	jobName := getProfilingJobName(dgdr)

	job := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: dgdr.Namespace}, job); err != nil {
		return false, err
	}

	// Check job conditions
	for _, condition := range job.Status.Conditions {
		if condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			logger.Info("Profiling job completed", "job", jobName)
			return true, nil
		}
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			// Get detailed error from pod logs
			detailedError := r.getProfilingJobErrorDetails(ctx, dgdr, job)
			if detailedError != "" {
				return false, fmt.Errorf("profiling job failed: %s. Details: %s", condition.Message, detailedError)
			}
			return false, fmt.Errorf("profiling job failed: %s", condition.Message)
		}
	}

	return false, nil
}

// getProfilingJobErrorDetails retrieves detailed error information from failed profiling job pods
func (r *DynamoGraphDeploymentRequestReconciler) getProfilingJobErrorDetails(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, job *batchv1.Job) string {
	logger := log.FromContext(ctx)

	// List pods owned by this job
	podList := &corev1.PodList{}
	labelSelector := client.MatchingLabels{
		"job-name": job.Name,
	}

	if err := r.List(ctx, podList, client.InNamespace(dgdr.Namespace), labelSelector); err != nil {
		logger.Error(err, "Failed to list pods for profiling job")
		return ""
	}

	// Look for failed pods and extract error details
	for _, pod := range podList.Items {
		// Check pod phase and container statuses
		if pod.Status.Phase == corev1.PodFailed {
			// Get profiler container status (first container)
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if containerStatus.Name == ContainerNameProfiler && containerStatus.State.Terminated != nil {
					terminated := containerStatus.State.Terminated
					// Construct detailed error message
					errorMsg := fmt.Sprintf("Pod: %s, Container: %s, ExitCode: %d, Reason: %s",
						pod.Name, containerStatus.Name, terminated.ExitCode, terminated.Reason)
					if terminated.Message != "" {
						errorMsg += fmt.Sprintf(", Message: %s", terminated.Message)
					}
					logger.Info("Retrieved profiling job error details", "error", errorMsg)
					return errorMsg
				}
			}

			// If no terminated state found, check waiting state
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if containerStatus.Name == ContainerNameProfiler && containerStatus.State.Waiting != nil {
					waiting := containerStatus.State.Waiting
					errorMsg := fmt.Sprintf("Pod: %s, Container: %s, Waiting - Reason: %s, Message: %s",
						pod.Name, containerStatus.Name, waiting.Reason, waiting.Message)
					logger.Info("Retrieved profiling job waiting details", "error", errorMsg)
					return errorMsg
				}
			}
		}
	}

	return ""
}

// generateDGDSpec generates DGD spec from profiling results (online or offline/AIC)
func (r *DynamoGraphDeploymentRequestReconciler) generateDGDSpec(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)
	logger.Info("Generating DGD spec from profiling results", "name", dgdr.Name)

	// Read the generated spec from ConfigMap (created by sidecar)
	outputConfigMapName := getOutputConfigMapName(dgdr)
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      outputConfigMapName,
		Namespace: dgdr.Namespace,
	}, cm)

	if err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("output ConfigMap %s not found - profiling may not have completed yet", outputConfigMapName)
		}
		return fmt.Errorf("failed to get output ConfigMap: %w", err)
	}

	// Get YAML content from ConfigMap
	yamlContent, exists := cm.Data[ProfilingOutputFile]
	if !exists {
		return fmt.Errorf("key %s not found in ConfigMap %s", ProfilingOutputFile, outputConfigMapName)
	}

	logger.Info("Found profiling output in ConfigMap", "configMap", outputConfigMapName, "size", len(yamlContent))

	// Extract DGD and any supporting resources from potentially multi-document YAML (ConfigMap + DGD)
	dgd, additionalResources, err := r.extractResourcesFromYAML([]byte(yamlContent))
	if err != nil {
		return fmt.Errorf("failed to extract DGD from %s: %w", ProfilingOutputFile, err)
	}

	logger.Info("Parsed profiling output", "dgdName", dgd.Name, "additionalResources", len(additionalResources))

	// Store additional resources (ConfigMaps) in annotations first
	if len(additionalResources) > 0 {
		if err := r.storeAdditionalResources(ctx, dgdr, additionalResources); err != nil {
			logger.Error(err, "Failed to store additional resources")
			return err
		}
		// Refetch the DGDR after updating annotations to get the latest resourceVersion
		if err := r.Get(ctx, types.NamespacedName{Name: dgdr.Name, Namespace: dgdr.Namespace}, dgdr); err != nil {
			return fmt.Errorf("failed to refetch DGDR after storing annotations: %w", err)
		}
	}

	// Store the generated DGD in status
	dgdr.Status.GeneratedDeployment = &runtime.RawExtension{
		Object: dgd,
	}
	dgdr.Status.ProfilingResults = fmt.Sprintf("configmap/%s", outputConfigMapName)

	return r.Status().Update(ctx, dgdr)
}

// storeAdditionalResources marshals additional resources to YAML and stores them in DGDR annotations.
// Validates annotation size and fails gracefully if too large.
func (r *DynamoGraphDeploymentRequestReconciler) storeAdditionalResources(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, resources []*unstructured.Unstructured) error {
	if len(resources) == 0 {
		return nil
	}

	var resourcesYAML []byte

	for i, res := range resources {
		resYAML, err := sigsyaml.Marshal(res.Object)
		if err != nil {
			return fmt.Errorf("failed to marshal resource %s/%s: %w", res.GetKind(), res.GetName(), err)
		}
		if i > 0 {
			resourcesYAML = append(resourcesYAML, []byte("\n---\n")...)
		}
		resourcesYAML = append(resourcesYAML, resYAML...)
	}

	// Validate size before storing
	if len(resourcesYAML) > MaxAnnotationSize {
		return fmt.Errorf("additional resources YAML size (%d bytes) exceeds maximum annotation size (%d bytes); "+
			"consider reducing the number of resources or storing them separately",
			len(resourcesYAML), MaxAnnotationSize)
	}

	if dgdr.Annotations == nil {
		dgdr.Annotations = make(map[string]string)
	}
	dgdr.Annotations[AnnotationAdditionalResources] = string(resourcesYAML)

	return r.Update(ctx, dgdr)
}

// extractResourcesFromYAML parses multi-document YAML from profiling output,
// extracting the DynamoGraphDeployment and any ConfigMaps that should be deployed with it.
func (r *DynamoGraphDeploymentRequestReconciler) extractResourcesFromYAML(yamlContent []byte) (*nvidiacomv1alpha1.DynamoGraphDeployment, []*unstructured.Unstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(yamlContent), 4096)

	var dgd *nvidiacomv1alpha1.DynamoGraphDeployment
	var additionalResources []*unstructured.Unstructured

	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if err == io.EOF {
				break
			}
			// Skip invalid documents and continue
			continue
		}

		// Skip empty objects
		if obj.GetKind() == "" {
			continue
		}

		if obj.GetKind() == "DynamoGraphDeployment" {
			dgd = &nvidiacomv1alpha1.DynamoGraphDeployment{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, dgd); err != nil {
				return nil, nil, fmt.Errorf("failed to convert to DynamoGraphDeployment: %w", err)
			}
		} else {
			// Store ConfigMaps or other resources for deployment
			additionalResources = append(additionalResources, obj)
		}
	}

	if dgd == nil {
		return nil, nil, fmt.Errorf("no DynamoGraphDeployment found in YAML content")
	}

	return dgd, additionalResources, nil
}

// extractDGDFromYAML is a convenience wrapper that extracts only the DGD (used by tests)
func (r *DynamoGraphDeploymentRequestReconciler) extractDGDFromYAML(yamlContent []byte) (*nvidiacomv1alpha1.DynamoGraphDeployment, error) {
	dgd, _, err := r.extractResourcesFromYAML(yamlContent)
	return dgd, err
}

// updateStateAndRequeue updates the DGDR state and requeues
func (r *DynamoGraphDeploymentRequestReconciler) updateStateAndRequeue(ctx context.Context, dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest, state, _ string) (ctrl.Result, error) {
	dgdr.Status.State = state
	if err := r.Status().Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

// updateStateWithCondition updates state and adds/updates a condition
func (r *DynamoGraphDeploymentRequestReconciler) updateStateWithCondition(
	ctx context.Context,
	dgdr *nvidiacomv1alpha1.DynamoGraphDeploymentRequest,
	state string,
	conditionType string,
	status metav1.ConditionStatus,
	reason string,
	message string,
) (ctrl.Result, error) {
	dgdr.Status.State = state

	condition := metav1.Condition{
		Type:               conditionType,
		Status:             status,
		ObservedGeneration: dgdr.Generation,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}

	dgdr.AddStatusCondition(condition)

	if err := r.Status().Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *DynamoGraphDeploymentRequestReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoGraphDeploymentRequest{}).
		Owns(&batchv1.Job{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the job
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})). // Watch Jobs created by this controller (via ownerReference)
		Watches(
			&nvidiacomv1alpha1.DynamoGraphDeployment{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				// Find DGDR by label instead of owner reference
				dgd := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
				dgdrName, hasName := dgd.Labels[LabelDGDRName]
				dgdrNamespace, hasNamespace := dgd.Labels[LabelDGDRNamespace]
				if !hasName || !hasNamespace {
					return nil
				}
				return []ctrl.Request{{
					NamespacedName: types.NamespacedName{
						Name:      dgdrName,
						Namespace: dgdrNamespace,
					},
				}}
			}),
			builder.WithPredicates(predicate.Funcs{
				// ignore creation cause we don't want to be called again after we create the DGD
				CreateFunc:  func(ce event.CreateEvent) bool { return false },
				DeleteFunc:  func(de event.DeleteEvent) bool { return true },
				UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
				GenericFunc: func(ge event.GenericEvent) bool { return true },
			}),
		). // Watch DGDs created by this controller (via label)
		Complete(r)
}
