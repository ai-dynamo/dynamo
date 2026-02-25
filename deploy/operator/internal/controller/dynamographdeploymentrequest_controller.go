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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"text/template"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	dgdv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
)

const (
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

	// Annotation keys for v1alpha1 round-trip compatibility.
	// The conversion layer stores v1alpha1 fields that have no v1beta1 spec equivalent
	// as annotations so the controller can still honour them for converted resources.
	AnnotationConfigMapRef = "nvidia.com/dgdr-config-map-ref"
	AnnotationOutputPVC    = "nvidia.com/dgdr-output-pvc"

	// Size limits
	MaxAnnotationSize = 250000 // ~250KB, below K8s 256KB limit

	// Sidecar image
	SidecarImage = "bitnami/kubectl:latest"

	// Volume names
	VolumeNameProfilingOutput = "profiling-output"
	VolumeNameProfilingConfig = "profiling-config"
	VolumeNameModelCache      = "model-cache"

	// Volume paths
	ProfilingOutputPath        = "/data"
	ProfilingOutputFile        = "config_with_planner.yaml"
	ProfilingOutputFileMocker  = "mocker_config_with_planner.yaml"
	ProfilingConfigMountPath   = "/config"
	ProfilingConfigDefaultKey  = "disagg.yaml"
	DefaultModelCacheMountPath = "/opt/model-cache"

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
	MessageSpecChangeRejected        = "Cannot modify spec in phase '%s'. DynamoGraphDeploymentRequest is immutable once profiling starts. Create a new resource with a different name instead."
	MessageJobCreationFailed         = "JobCreationFailed"
	MessageDeploymentCreationFailed  = "DeploymentCreationFailed"
	MessageResultsRetrievalFailed    = "ResultsRetrievalFailed"
	MessageGenerationFailed          = "GenerationFailed"
	MessageAIConfiguratorCheckFailed = "AIConfiguratorCheckFailed"
	MessageProfilingCheckFailed      = "ProfilingCheckFailed"
	MessageConfigMapNotFound         = "ConfigMap %s not found in namespace %s"
	MessageConfigMapKeyNotFound      = "key %s not found in ConfigMap %s"
	MessageModelCachePVCNotFound = "model cache PVC %s not found in namespace %s"
)

// shell script template for the output copier sidecar
const sidecarScriptTemplate = `
set -e
set -o pipefail

# Wait for profiler container to terminate (no timeout - profiling can take hours)
echo "Waiting for profiler to complete..."
START_TIME=$(date +%s)
LAST_PROGRESS_LOG=$START_TIME
PROGRESS_INTERVAL=300

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  # Log progress every 5 minutes
  if [ $((CURRENT_TIME - LAST_PROGRESS_LOG)) -ge $PROGRESS_INTERVAL ]; then
    echo "Still waiting... ($(($ELAPSED / 60)) minutes elapsed)"
    LAST_PROGRESS_LOG=$CURRENT_TIME
  fi

  # Check if profiler container terminated
  CONTAINER_STATUS=$(kubectl get pod $HOSTNAME -n {{.Namespace}} -o jsonpath='{.status.containerStatuses[?(@.name=="profiler")].state}' 2>/dev/null || echo "")
  if echo "$CONTAINER_STATUS" | grep -q "terminated"; then
    echo "Profiler terminated (ran for $(($ELAPSED / 60)) minutes)"
    break
  fi
  sleep 5
done

# Check profiler status file (2 minute timeout)
echo "Checking profiler status..."
STATUS_FILE="{{.OutputPath}}/profiler_status.yaml"
TIMEOUT=120
CHECK_START=$(date +%s)

# Wait for status file to exist
while [ ! -f "$STATUS_FILE" ]; do
  ELAPSED=$(($(date +%s) - CHECK_START))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: Status file not found after ${TIMEOUT}s"
    exit 1
  fi
  sleep 2
done

# Read and parse status from YAML file
STATUS=$(grep "^status:" "$STATUS_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")

if [ -z "$STATUS" ]; then
  echo "ERROR: Invalid status file format"
  exit 1
fi

# Check status value
case "$STATUS" in
  success)
    MESSAGE=$(grep "^message:" "$STATUS_FILE" | sed 's/^message: *//' | tr -d '"' | tr -d "'")
    echo "Profiler succeeded: $MESSAGE"
    ;;
  failed)
    ERROR=$(grep "^error:" "$STATUS_FILE" | sed 's/^error: *//' | tr -d '"' | tr -d "'")
    MESSAGE=$(grep "^message:" "$STATUS_FILE" | sed 's/^message: *//' | tr -d '"' | tr -d "'")
    echo "ERROR: Profiler failed: ${ERROR:-$MESSAGE}"
    exit 1
    ;;
  running)
    echo "ERROR: Profiler still running (unexpected)"
    exit 1
    ;;
  *)
    echo "ERROR: Unknown status: $STATUS"
    exit 1
    ;;
esac

echo "Creating ConfigMap..."

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

# Add mocker config (profiler always generates both real and mocker configs)
if [ -f {{.OutputPath}}/{{.MockerOutputFile}} ]; then
  echo "  {{.MockerOutputFile}}: |" >> /tmp/cm.yaml
  sed 's/^/    /' {{.OutputPath}}/{{.MockerOutputFile}} >> /tmp/cm.yaml
  echo "Added mocker config to ConfigMap"
fi

# Add profiler status file for debugging
if [ -f {{.OutputPath}}/profiler_status.yaml ]; then
  echo "  profiler_status.yaml: |" >> /tmp/cm.yaml
  sed 's/^/    /' {{.OutputPath}}/profiler_status.yaml >> /tmp/cm.yaml
fi

# Note: Profiling data (raw_data.npz converted to JSON) is included in the
# generated DGD YAML as a separate ConfigMap by the profiler, no need to add it here

kubectl apply -f /tmp/cm.yaml
echo "Saved profiling output to ConfigMap {{.ConfigMapName}}"
`

// DynamoGraphDeploymentRequestReconciler reconciles a DynamoGraphDeploymentRequest object
type DynamoGraphDeploymentRequestReconciler struct {
	client.Client
	Recorder      record.EventRecorder
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig

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
func (r *DynamoGraphDeploymentRequestReconciler) FinalizeResource(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
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
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
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
		if dgdr.Status.Phase == nvidiacomv1beta1.DGDRPhaseProfiling || dgdr.Status.Phase == nvidiacomv1beta1.DGDRPhaseDeploying ||
			dgdr.Status.Phase == nvidiacomv1beta1.DGDRPhaseReady || dgdr.Status.Phase == nvidiacomv1beta1.DGDRPhaseDeployed {
			logger.Info("Spec change detected in immutable phase",
				"phase", dgdr.Status.Phase,
				"observedGeneration", dgdr.Status.ObservedGeneration,
				"currentGeneration", dgdr.Generation)

			r.Recorder.Event(dgdr, corev1.EventTypeWarning, nvidiacomv1beta1.EventReasonSpecChangeRejected,
				fmt.Sprintf(MessageSpecChangeRejected, dgdr.Status.Phase))

			// Keep the old observedGeneration to continue rejecting changes
			// No phase transition - stay in current phase with old spec
			return ctrl.Result{}, nil
		}
	}
	// Phase machine: handle different phases
	switch dgdr.Status.Phase {
	case nvidiacomv1beta1.DGDRPhasePending, "":
		return r.handlePendingPhase(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseProfiling:
		return r.handleProfilingPhase(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseDeploying:
		return r.handleDeployingPhase(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseReady:
		return r.handleReadyPhase(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseDeployed:
		return r.handleDeployedPhase(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseFailed:
		return r.handleFailedPhase(ctx, dgdr)
	default:
		logger.Info("Unknown phase", "phase", dgdr.Status.Phase)
		return r.updatePhaseAndRequeue(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, MessageInvalidState)
	}
}

// handlePendingPhase processes newly created or pending DGDR resources.
// When ObservedGeneration == 0, performs initial validation (merged from v1alpha1 Initializing state).
// Otherwise, starts the profiling process.
func (r *DynamoGraphDeploymentRequestReconciler) handlePendingPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// First-time processing: validate spec (merged from handleInitialState)
	if dgdr.Status.ObservedGeneration == 0 {
		logger.Info("Handling initial validation", "name", dgdr.Name)

		// Validate the spec
		if err := r.validateSpec(ctx, dgdr); err != nil {
			r.Recorder.Event(dgdr, corev1.EventTypeWarning, nvidiacomv1beta1.EventReasonValidationFailed, err.Error())
			return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, nvidiacomv1beta1.ConditionTypeValidation, metav1.ConditionFalse, nvidiacomv1beta1.EventReasonValidationFailed, err.Error())
		}

		// Set observedGeneration to track the spec we're processing
		dgdr.Status.ObservedGeneration = dgdr.Generation

		// Initialize status
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonInitialized, MessageInitialized)
		return r.updatePhaseAndRequeue(ctx, dgdr, nvidiacomv1beta1.DGDRPhasePending, MessageInitialized)
	}

	logger.Info("Handling pending phase", "name", dgdr.Name)

	// Create profiling job (online or AIC)
	if err := r.createProfilingJob(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, nvidiacomv1beta1.EventReasonProfilingJobFailed, err.Error())
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, nvidiacomv1beta1.ConditionTypeProfiling, metav1.ConditionFalse, MessageJobCreationFailed, err.Error())
	}

	// Record event with appropriate message
	if isOnlineProfiling(dgdr) {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonProfilingJobCreated, MessageProfilingJobCreated)
	} else {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonProfilingJobCreated, MessageAICProfilingJobCreated)
	}

	// Update to Profiling phase with Running status
	dgdr.SetProfilingPhase(nvidiacomv1beta1.ProfilingPhaseInitializing)
	return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseProfiling, nvidiacomv1beta1.ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingRunning", MessageProfilingInProgress)
}

// handleProfilingPhase monitors profiling progress and generates spec when complete
func (r *DynamoGraphDeploymentRequestReconciler) handleProfilingPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling profiling phase", "name", dgdr.Name)

	// Check profiling job status (both online and offline/AIC run as Jobs)
	// Note: We watch the Job via Owns(), so we'll be triggered automatically on Job changes
	completed, err := r.checkProfilingJobStatus(ctx, dgdr)
	if err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageProfilingCheckFailed, err.Error())
		// Job failed - clear profiling sub-phase and transition to Failed
		dgdr.ClearProfilingPhase()
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, nvidiacomv1beta1.ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingFailed", err.Error())
	}

	if !completed {
		logger.Info("Profiling job still running", "name", dgdr.Name)
		// Don't requeue - we'll be triggered when the Job completes/fails
		return ctrl.Result{}, nil
	}

	// Profiling complete — clear the profiling sub-phase
	dgdr.ClearProfilingPhase()

	// Mark profiling as completed successfully
	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta1.ConditionTypeProfiling,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: dgdr.Generation,
		Reason:             "ProfilingCompleted",
		Message:            "Profiling job completed successfully",
	})

	// Retrieve profiling results and generate spec
	if err := r.generateDGDSpec(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageGenerationFailed, err.Error())
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, nvidiacomv1beta1.ConditionTypeSpecGenerated, metav1.ConditionFalse, MessageGenerationFailed, err.Error())
	}

	// Record spec generation event
	r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonSpecGenerated, MessageSpecGenerated)

	// Create additional resources (ConfigMaps) immediately after profiling
	// This ensures that the `planner-profile-data` ConfigMap is available for both auto and manual deployment
	// v1beta1 uses the DGDR namespace for additional resources.
	targetNamespace := dgdr.Namespace
	if err := r.createAdditionalResources(ctx, dgdr, targetNamespace); err != nil {
		logger.Error(err, "Failed to create additional resources after profiling")
		// Don't fail the DGDR, just log the error - ConfigMaps can be created manually
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, "ConfigMapCreationFailed",
			fmt.Sprintf("Failed to create ConfigMaps from profiling output: %v", err))
	}

	// If autoApply is enabled, transition to Deploying phase
	if dgdr.Spec.AutoApply {
		logger.Info("AutoApply enabled, transitioning to Deploying phase")
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseDeploying, nvidiacomv1beta1.ConditionTypeSpecGenerated, metav1.ConditionTrue, nvidiacomv1beta1.EventReasonSpecGenerated, MessageSpecGenerated)
	}

	// Otherwise, transition to Ready phase
	return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseReady, nvidiacomv1beta1.ConditionTypeSpecGenerated, metav1.ConditionTrue, nvidiacomv1beta1.EventReasonSpecGenerated, MessageSpecAvailable)
}

// handleReadyPhase handles DGDR in Ready phase (profiling complete, spec available)
func (r *DynamoGraphDeploymentRequestReconciler) handleReadyPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is ready", "name", dgdr.Name)

	// Nothing to monitor in Ready phase - spec is available for manual application
	return ctrl.Result{}, nil
}

// handleDeployingPhase handles DGD creation and monitors deployment
func (r *DynamoGraphDeploymentRequestReconciler) handleDeployingPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling deploying phase", "name", dgdr.Name)

	if !dgdr.Spec.AutoApply {
		// Shouldn't be in this phase without autoApply
		logger.Info("AutoApply not enabled, transitioning to Ready")
		dgdr.Status.Phase = nvidiacomv1beta1.DGDRPhaseReady
		setSucceededCondition(dgdr, nvidiacomv1beta1.DGDRPhaseReady)
		return ctrl.Result{}, r.Status().Update(ctx, dgdr)
	}

	// Check if we need to create DGD
	if dgdr.Status.DGDName == "" {
		return r.createDGD(ctx, dgdr)
	}

	// DGD was already created, check its status
	dgd := &dgdv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      dgdr.Status.DGDName,
		Namespace: dgdr.Namespace,
	}, dgd)

	if apierrors.IsNotFound(err) {
		// DGD was deleted by user
		return r.handleDGDDeleted(ctx, dgdr)
	}

	if err != nil {
		return ctrl.Result{}, err
	}

	// Check if DGD is Ready
	if dgd.Status.State == dgdv1alpha1.DGDStateSuccessful {
		logger.Info("DGD is Ready, transitioning to Deployed phase")
		dgdr.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeployed
		setSucceededCondition(dgdr, nvidiacomv1beta1.DGDRPhaseDeployed)

		r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonDeploymentReady,
			fmt.Sprintf(MessageDeploymentReady, dgd.Name))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeDeploymentReady,
			Status:  metav1.ConditionTrue,
			Reason:  nvidiacomv1beta1.EventReasonDeploymentReady,
			Message: fmt.Sprintf(MessageDeploymentReady, dgd.Name),
		})
	}

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// handleDeployedPhase monitors a healthy DGD and detects degradation or deletion
func (r *DynamoGraphDeploymentRequestReconciler) handleDeployedPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is deployed", "name", dgdr.Name)

	// Check if DGD still exists and monitor its status
	dgd := &dgdv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      dgdr.Status.DGDName,
		Namespace: dgdr.Namespace,
	}, dgd)

	if apierrors.IsNotFound(err) {
		// DGD was deleted by user
		return r.handleDGDDeleted(ctx, dgdr)
	}

	if err != nil {
		return ctrl.Result{}, err
	}

	// Check if DGD degraded from Ready
	if dgd.Status.State != dgdv1alpha1.DGDStateSuccessful {
		logger.Info("DGD degraded, transitioning back to Deploying",
			"dgdState", dgd.Status.State)

		dgdr.Status.Phase = nvidiacomv1beta1.DGDRPhaseDeploying
		setSucceededCondition(dgdr, nvidiacomv1beta1.DGDRPhaseDeploying)

		r.Recorder.Event(dgdr, corev1.EventTypeWarning, nvidiacomv1beta1.EventReasonDeploymentDegraded,
			fmt.Sprintf(MessageDeploymentDegraded, dgd.Name, string(dgd.Status.State)))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    nvidiacomv1beta1.ConditionTypeDeploymentReady,
			Status:  metav1.ConditionFalse,
			Reason:  nvidiacomv1beta1.EventReasonDeploymentDegraded,
			Message: fmt.Sprintf("Deployment degraded to %s", string(dgd.Status.State)),
		})
	}

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// handleDGDDeleted handles the case when auto-created DGD is deleted by user.
// In v1beta1, this transitions to Failed (DeploymentDeleted phase was removed).
func (r *DynamoGraphDeploymentRequestReconciler) handleDGDDeleted(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGD was deleted by user, transitioning to Failed phase")

	dgdr.Status.Phase = nvidiacomv1beta1.DGDRPhaseFailed
	setSucceededCondition(dgdr, nvidiacomv1beta1.DGDRPhaseFailed)

	r.Recorder.Event(dgdr, corev1.EventTypeWarning, nvidiacomv1beta1.EventReasonDeploymentDeleted,
		fmt.Sprintf(MessageDeploymentDeleted, dgdr.Status.DGDName))

	dgdr.Status.DGDName = ""
	dgdr.Status.DeploymentInfo = nil

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    nvidiacomv1beta1.ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  nvidiacomv1beta1.EventReasonDeploymentDeleted,
		Message: "Deployment was deleted by user. Create a new DGDR to redeploy.",
	})

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createDGD creates a DynamoGraphDeployment with the generated spec
func (r *DynamoGraphDeploymentRequestReconciler) createDGD(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Extract DGD spec from annotation (stored by generateDGDSpec)
	dgdSpecYAML, ok := dgdr.Annotations["nvidia.com/generated-dgd-spec"]
	if !ok || dgdSpecYAML == "" {
		return ctrl.Result{}, fmt.Errorf("generated DGD spec not found in annotation nvidia.com/generated-dgd-spec")
	}

	generatedDGD := &dgdv1alpha1.DynamoGraphDeployment{}
	if err := yaml.Unmarshal([]byte(dgdSpecYAML), generatedDGD); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to unmarshal generated deployment from annotation: %w", err)
	}

	// Determine DGD name and namespace from generated deployment
	dgdName := generatedDGD.Name
	dgdNamespace := dgdr.Namespace

	// Build labels (start with generated DGD's labels)
	labels := make(map[string]string)
	if generatedDGD.Labels != nil {
		for k, v := range generatedDGD.Labels {
			labels[k] = v
		}
	}
	// Add/override with managed labels
	labels[nvidiacomv1beta1.LabelDGDRName] = dgdr.Name
	labels[nvidiacomv1beta1.LabelDGDRNamespace] = dgdr.Namespace
	labels[nvidiacomv1beta1.LabelManagedBy] = nvidiacomv1beta1.LabelValueDynamoOperator

	// Build annotations (start with generated DGD's annotations)
	annotations := make(map[string]string)
	if generatedDGD.Annotations != nil {
		for k, v := range generatedDGD.Annotations {
			annotations[k] = v
		}
	}

	// Create DGD from generated deployment
	dgd := &dgdv1alpha1.DynamoGraphDeployment{
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
			dgdr.Status.DGDName = dgdName
			return ctrl.Result{}, r.Status().Update(ctx, dgdr)
		}
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageDeploymentCreationFailed, err.Error())
		return ctrl.Result{}, err
	}

	// Update status
	dgdr.Status.DGDName = dgdName

	r.Recorder.Event(dgdr, corev1.EventTypeNormal, nvidiacomv1beta1.EventReasonDeploymentCreated,
		fmt.Sprintf(MessageDeploymentCreated, dgdName))

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    nvidiacomv1beta1.ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  nvidiacomv1beta1.EventReasonDeploymentCreated,
		Message: fmt.Sprintf("DGD %s created, waiting for Ready", dgdName),
	})

	logger.Info("DynamoGraphDeployment created successfully", "name", dgdName)

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createAdditionalResources creates ConfigMaps from the profiling output that should be deployed alongside the DGD
func (r *DynamoGraphDeploymentRequestReconciler) createAdditionalResources(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, targetNamespace string) error {
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
		cm.Labels[nvidiacomv1beta1.LabelDGDRName] = dgdr.Name
		cm.Labels[nvidiacomv1beta1.LabelDGDRNamespace] = dgdr.Namespace
		cm.Labels[nvidiacomv1beta1.LabelManagedBy] = nvidiacomv1beta1.LabelValueDynamoOperator

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

// handleFailedPhase handles DGDR in Failed phase
func (r *DynamoGraphDeploymentRequestReconciler) handleFailedPhase(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is in failed phase", "name", dgdr.Name)

	// Could implement retry logic here if desired
	return ctrl.Result{}, nil
}

// getProfilingJobName returns the job name for a DGDR
func getProfilingJobName(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	// Use "profile-" prefix for all profiling jobs
	return fmt.Sprintf("profile-%s", dgdr.Name)
}

// getOutputConfigMapName returns the ConfigMap name for profiling output
func getOutputConfigMapName(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	return fmt.Sprintf("%s%s", ConfigMapOutputPrefix, dgdr.Name)
}

// isOnlineProfiling returns true. In v1beta1, the profiler decides online vs AIC
// mode internally based on its config. The controller always uses the same label.
func isOnlineProfiling(_ *nvidiacomv1beta1.DynamoGraphDeploymentRequest) bool {
	return true
}

// validateSpec validates the DGDR spec
func (r *DynamoGraphDeploymentRequestReconciler) validateSpec(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	// Validate model cache PVC if provided
	if dgdr.Spec.ModelCache != nil && dgdr.Spec.ModelCache.PVCName != "" {
		pvc := &corev1.PersistentVolumeClaim{}
		err := r.Get(ctx, types.NamespacedName{
			Name:      dgdr.Spec.ModelCache.PVCName,
			Namespace: dgdr.Namespace,
		}, pvc)

		if err != nil {
			if apierrors.IsNotFound(err) {
				return fmt.Errorf(MessageModelCachePVCNotFound, dgdr.Spec.ModelCache.PVCName, dgdr.Namespace)
			}
			return err
		}
	}

	if err := r.validateGPUHardwareInfo(ctx, dgdr); err != nil {
		return err
	}

	// The profiler will validate the rest of the configuration
	return nil
}

// validateGPUHardwareInfo ensures GPU hardware information is available when required for profiling
func (r *DynamoGraphDeploymentRequestReconciler) validateGPUHardwareInfo(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	// Check if user provided hardware info in the typed spec
	hasManualConfig := dgdr.Spec.Hardware != nil && (
		dgdr.Spec.Hardware.GPUSKU != "" ||
			dgdr.Spec.Hardware.VRAMMB != nil ||
			dgdr.Spec.Hardware.NumGPUsPerNode != nil)

	// If manual config is provided, validation passes
	if hasManualConfig {
		return nil
	}

	_, err := gpu.DiscoverGPUs(ctx, r.Client)
	if err == nil {
		// GPU discovery is available, validation passes
		return nil
	}

	logger.Info("GPU discovery not available", "reason", err.Error())

	isNamespaceScoped := r.Config.Namespace.Restricted != ""
	if isNamespaceScoped {
		return fmt.Errorf(
			"GPU hardware info required but cannot be auto-discovered." +
				"\n\nOptions to resolve:" +
				"\n\n1. Re-enable GPU discovery (if it was disabled during Helm install):" +
				"\n   helm upgrade ... --set dynamo-operator.gpuDiscovery.enabled=true" +
				"\n\n2. Add hardware config to spec.hardware:" +
				"\n   numGpusPerNode: 8" +
				"\n   gpuSku: \"H100-SXM5-80GB\"" +
				"\n   vramMb: 81920")
	}

	return fmt.Errorf("GPU hardware info required but auto-discovery failed. Add spec.hardware.gpuSku, spec.hardware.vramMb, spec.hardware.numGpusPerNode")
}

// createProfilingJob creates a Kubernetes Job for profiling using SyncResource
func (r *DynamoGraphDeploymentRequestReconciler) createProfilingJob(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
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
	if r.Config.Namespace.Restricted == "" {
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

	// Enrich hardware from GPU discovery before marshalling the spec.
	// This fills in gpuSku, vramMb, numGpusPerNode if the user didn't set them.
	if err := r.enrichHardwareFromDiscovery(ctx, dgdr); err != nil {
		logger.Info("GPU discovery not available, proceeding without enrichment", "reason", err.Error())
	}

	// Use SyncResource to create/update the job
	modified, job, err := commonController.SyncResource(ctx, r, dgdr, func(ctx context.Context) (*batchv1.Job, bool, error) {
		jobName := getProfilingJobName(dgdr)
		outputConfigMapName := getOutputConfigMapName(dgdr)

		// Marshal the DGDR spec to JSON — the profiler receives the spec verbatim
		specJSON, err := marshalDGDRSpec(dgdr)
		if err != nil {
			return nil, false, err
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

		// Add model cache PVC mount if configured
		modelCachePVC, modelCacheMountPath := extractModelCachePVCConfig(dgdr)
		if modelCachePVC != "" {
			logger.Info("Mounting model cache PVC to profiler pod", "pvc", modelCachePVC, "mountPath", modelCacheMountPath)
			volumeMounts = append(volumeMounts, corev1.VolumeMount{
				Name:      VolumeNameModelCache,
				MountPath: modelCacheMountPath,
				ReadOnly:  true,
			})
		}

		// v1alpha1 round-trip: mount ConfigMap if referenced via annotation
		cmRef := configMapRefFromAnnotation(dgdr)
		if cmRef != nil {
			volumeMounts = append(volumeMounts, corev1.VolumeMount{
				Name:      VolumeNameProfilingConfig,
				MountPath: ProfilingConfigMountPath,
				ReadOnly:  true,
			})
		}

		// Profiler args: pass the DGDR spec as JSON via --config
		profilerArgs := []string{"--config", specJSON}

		// Use image from spec
		imageName := dgdr.Spec.Image
		logger.Info("Using profiler image", "image", imageName)

		profilerContainer := corev1.Container{
			Name:         ContainerNameProfiler,
			Image:        imageName,
			Command:      []string{"python", "-m", "dynamo.profiler"},
			Args:         profilerArgs,
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
			"OutputPath":       ProfilingOutputPath,
			"OutputFile":       ProfilingOutputFile,
			"MockerOutputFile": ProfilingOutputFileMocker,
			"ConfigMapName":    outputConfigMapName,
			"Namespace":        dgdr.Namespace,
			"DGDRName":         dgdr.Name,
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

		// Use PVC for profiling output if round-tripped v1alpha1 annotation is present,
		// otherwise use emptyDir (v1beta1 default).
		var profilingOutputVolume corev1.Volume
		if outputPVC := outputPVCFromAnnotation(dgdr); outputPVC != "" {
			logger.Info("Using PVC for profiling output (from v1alpha1 annotation)", "pvc", outputPVC)
			profilingOutputVolume = corev1.Volume{
				Name: VolumeNameProfilingOutput,
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: outputPVC,
					},
				},
			}
		} else {
			profilingOutputVolume = corev1.Volume{
				Name: VolumeNameProfilingOutput,
				VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{},
				},
			}
		}
		volumes := []corev1.Volume{profilingOutputVolume}

		// Add model cache PVC volume if configured
		if modelCachePVC != "" {
			volumes = append(volumes, corev1.Volume{
				Name: VolumeNameModelCache,
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: modelCachePVC,
						ReadOnly:  true,
					},
				},
			})
		}

		// v1alpha1 round-trip: add ConfigMap volume if referenced via annotation
		if cmRef != nil {
			cmKey := cmRef.Key
			if cmKey == "" {
				cmKey = ProfilingConfigDefaultKey
			}
			volumes = append(volumes, corev1.Volume{
				Name: VolumeNameProfilingConfig,
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: cmRef.Name,
						},
						Items: []corev1.KeyToPath{{
							Key:  cmKey,
							Path: ProfilingConfigDefaultKey,
						}},
					},
				},
			})
		}

		// Limit retries to prevent infinite loop
		backoffLimit := int32(3)

		podSpec := corev1.PodSpec{
			ServiceAccountName: ServiceAccountProfilingJob,
			RestartPolicy:      corev1.RestartPolicyNever,
			SecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: ptr.To(true),
				RunAsUser:    ptr.To[int64](1000),
				RunAsGroup:   ptr.To[int64](1000),
				FSGroup:      ptr.To[int64](1000),
			},
			Containers: []corev1.Container{profilerContainer, sidecarContainer},
			Volumes:    volumes,
			ImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "nvcr-imagepullsecret"},
			},
		}

		// Apply overrides from spec.overrides.profilingJob if provided
		if dgdr.Spec.Overrides != nil && dgdr.Spec.Overrides.ProfilingJob != nil {
			overridePodSpec := dgdr.Spec.Overrides.ProfilingJob.Template.Spec
			if len(overridePodSpec.Containers) > 0 {
				profilerContainer.Resources = overridePodSpec.Containers[0].Resources
				podSpec.Containers[0] = profilerContainer
			}
			if len(overridePodSpec.Tolerations) > 0 {
				podSpec.Tolerations = overridePodSpec.Tolerations
			}
			if len(overridePodSpec.NodeSelector) > 0 {
				podSpec.NodeSelector = overridePodSpec.NodeSelector
			}
			if len(overridePodSpec.ImagePullSecrets) > 0 {
				podSpec.ImagePullSecrets = overridePodSpec.ImagePullSecrets
			}
			if overridePodSpec.ServiceAccountName != "" {
				podSpec.ServiceAccountName = overridePodSpec.ServiceAccountName
			}
		}

		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      jobName,
				Namespace: dgdr.Namespace,
				Labels: map[string]string{
					nvidiacomv1beta1.LabelApp:       nvidiacomv1beta1.LabelValueDynamoProfiler,
					nvidiacomv1beta1.LabelDGDR:      dgdr.Name,
					nvidiacomv1beta1.LabelManagedBy: nvidiacomv1beta1.LabelValueDynamoOperator,
				},
			},
			Spec: batchv1.JobSpec{
				BackoffLimit: &backoffLimit,
				Template: corev1.PodTemplateSpec{
					Spec: podSpec,
				},
			},
		}

		// Apply job-level overrides
		if dgdr.Spec.Overrides != nil && dgdr.Spec.Overrides.ProfilingJob != nil {
			if dgdr.Spec.Overrides.ProfilingJob.BackoffLimit != nil {
				job.Spec.BackoffLimit = dgdr.Spec.Overrides.ProfilingJob.BackoffLimit
			}
		}

		return job, false, nil
	})

	if err != nil {
		return err
	}

	if modified {
		logger.Info("Profiling job created/updated", "job", job.Name)
	}

	// Store the job name in status for observability
	dgdr.Status.ProfilingJobName = job.Name

	return nil
}

// marshalDGDRSpec produces the JSON string passed to the profiler via --config.
// The profiler receives the DGDR spec verbatim — no bespoke key mapping needed.
func marshalDGDRSpec(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (string, error) {
	specJSON, err := json.Marshal(dgdr.Spec)
	if err != nil {
		return "", fmt.Errorf("failed to marshal DGDR spec to JSON: %w", err)
	}
	return string(specJSON), nil
}

// enrichHardwareFromDiscovery fills in hardware fields that the user didn't set.
// Called before marshalDGDRSpec(). Mutates dgdr.Spec.Hardware in-place (memory only, not persisted).
func (r *DynamoGraphDeploymentRequestReconciler) enrichHardwareFromDiscovery(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	if dgdr.Spec.Hardware == nil {
		dgdr.Spec.Hardware = &nvidiacomv1beta1.HardwareSpec{}
	}
	hw := dgdr.Spec.Hardware
	if hw.GPUSKU != "" && hw.VRAMMB != nil && hw.NumGPUsPerNode != nil {
		return nil // all fields already set by user
	}

	gpuInfo, err := gpu.DiscoverGPUs(ctx, r.Client)
	if err != nil {
		return err
	}

	logger := log.FromContext(ctx)
	logger.Info("GPU discovery completed successfully",
		"gpusPerNode", gpuInfo.GPUsPerNode,
		"model", gpuInfo.Model,
		"vramMiB", gpuInfo.VRAMPerGPU)

	if hw.GPUSKU == "" {
		hw.GPUSKU = gpuInfo.Model
	}
	if hw.VRAMMB == nil {
		vram := float64(gpuInfo.VRAMPerGPU)
		hw.VRAMMB = &vram
	}
	if hw.NumGPUsPerNode == nil {
		n := int32(gpuInfo.GPUsPerNode)
		hw.NumGPUsPerNode = &n
	}
	return nil
}

// extractModelCachePVCConfig reads model cache PVC settings from the typed v1beta1 spec.
// Returns (pvcName, mountPath) — both empty if not configured.
func extractModelCachePVCConfig(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (string, string) {
	if dgdr.Spec.ModelCache == nil || dgdr.Spec.ModelCache.PVCName == "" {
		return "", ""
	}
	mountPath := dgdr.Spec.ModelCache.PVCMountPath
	if mountPath == "" {
		mountPath = DefaultModelCacheMountPath
	}
	return dgdr.Spec.ModelCache.PVCName, mountPath
}

// configMapKeySelector mirrors v1alpha1.ConfigMapKeySelector for annotation deserialization.
type configMapKeySelector struct {
	Name string `json:"name"`
	Key  string `json:"key,omitempty"`
}

// configMapRefFromAnnotation reads the ConfigMap reference from the round-trip annotation.
// Returns nil for native v1beta1 resources (no annotation present).
func configMapRefFromAnnotation(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) *configMapKeySelector {
	if dgdr.Annotations == nil {
		return nil
	}
	raw, ok := dgdr.Annotations[AnnotationConfigMapRef]
	if !ok || raw == "" {
		return nil
	}
	var ref configMapKeySelector
	if err := json.Unmarshal([]byte(raw), &ref); err != nil {
		return nil
	}
	return &ref
}

// outputPVCFromAnnotation reads the output PVC name from the round-trip annotation.
// Returns "" for native v1beta1 resources (always emptyDir).
func outputPVCFromAnnotation(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	if dgdr.Annotations == nil {
		return ""
	}
	return dgdr.Annotations[AnnotationOutputPVC]
}

// checkProfilingJobStatus checks if the profiling job has completed
func (r *DynamoGraphDeploymentRequestReconciler) checkProfilingJobStatus(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (bool, error) {
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
func (r *DynamoGraphDeploymentRequestReconciler) getProfilingJobErrorDetails(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, job *batchv1.Job) string {
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
func (r *DynamoGraphDeploymentRequestReconciler) generateDGDSpec(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)
	logger.Info("Generating DGD spec from profiling results", "name", dgdr.Name, "backend", dgdr.Spec.Backend)

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

	// Select the right config file based on mocker feature flag
	// Profiler always generates both real and mocker configs
	var outputFile string
	if dgdr.Spec.Features != nil && dgdr.Spec.Features.Mocker != nil && dgdr.Spec.Features.Mocker.Enabled {
		outputFile = ProfilingOutputFileMocker
		logger.Info("Using mocker deployment config")
	} else {
		outputFile = ProfilingOutputFile
	}

	// Get YAML content from ConfigMap
	yamlContent, exists := cm.Data[outputFile]
	if !exists {
		return fmt.Errorf("key %s not found in ConfigMap %s", outputFile, outputConfigMapName)
	}

	logger.Info("Found profiling output in ConfigMap", "configMap", outputConfigMapName, "outputFile", outputFile, "size", len(yamlContent))

	// Extract DGD and any supporting resources from potentially multi-document YAML (ConfigMap + DGD)
	dgd, additionalResources, err := r.extractResourcesFromYAML([]byte(yamlContent))
	if err != nil {
		return fmt.Errorf("failed to extract DGD from %s: %w", outputFile, err)
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

	// Store the generated DGD name in status and cache the spec in an annotation for createDGD
	dgdr.Status.DGDName = dgd.Name

	// Serialize the DGD spec to an annotation so createDGD can retrieve it
	dgdBytes, err := sigsyaml.Marshal(dgd)
	if err != nil {
		return fmt.Errorf("failed to marshal generated DGD: %w", err)
	}
	if dgdr.Annotations == nil {
		dgdr.Annotations = make(map[string]string)
	}
	dgdr.Annotations["nvidia.com/generated-dgd-spec"] = string(dgdBytes)

	// Update the object (annotations are on the object, not status)
	if err := r.Update(ctx, dgdr); err != nil {
		return fmt.Errorf("failed to update DGDR with generated DGD annotation: %w", err)
	}

	return r.Status().Update(ctx, dgdr)
}

// storeAdditionalResources marshals additional resources to YAML and stores them in DGDR annotations.
// Validates annotation size and fails gracefully if too large.
func (r *DynamoGraphDeploymentRequestReconciler) storeAdditionalResources(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, resources []*unstructured.Unstructured) error {
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
func (r *DynamoGraphDeploymentRequestReconciler) extractResourcesFromYAML(yamlContent []byte) (*dgdv1alpha1.DynamoGraphDeployment, []*unstructured.Unstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(yamlContent), 4096)

	var dgd *dgdv1alpha1.DynamoGraphDeployment
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
			dgd = &dgdv1alpha1.DynamoGraphDeployment{}
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
func (r *DynamoGraphDeploymentRequestReconciler) extractDGDFromYAML(yamlContent []byte) (*dgdv1alpha1.DynamoGraphDeployment, error) {
	dgd, _, err := r.extractResourcesFromYAML(yamlContent)
	return dgd, err
}

// setSucceededCondition sets the aggregate Succeeded condition based on the current phase.
func setSucceededCondition(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, phase nvidiacomv1beta1.DGDRPhase) {
	var status metav1.ConditionStatus
	var reason, message string

	switch phase {
	case nvidiacomv1beta1.DGDRPhasePending, "":
		status, reason, message = metav1.ConditionFalse, "Pending", "DGDR is pending"
	case nvidiacomv1beta1.DGDRPhaseProfiling:
		status, reason, message = metav1.ConditionFalse, "Profiling", "Profiling is in progress"
	case nvidiacomv1beta1.DGDRPhaseReady:
		status, reason, message = metav1.ConditionTrue, "SpecGenerated", "Profiling complete, spec available"
	case nvidiacomv1beta1.DGDRPhaseDeploying:
		status, reason, message = metav1.ConditionFalse, "Deploying", "Deployment is in progress"
	case nvidiacomv1beta1.DGDRPhaseDeployed:
		status, reason, message = metav1.ConditionTrue, "Deployed", "Deployment is healthy"
	case nvidiacomv1beta1.DGDRPhaseFailed:
		status, reason, message = metav1.ConditionFalse, "Failed", "DGDR has failed"
	default:
		status, reason, message = metav1.ConditionFalse, "Unknown", "Unknown phase"
	}

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta1.ConditionTypeSucceeded,
		Status:             status,
		ObservedGeneration: dgdr.Generation,
		Reason:             reason,
		Message:            message,
	})
}

// updatePhaseAndRequeue updates the DGDR phase and requeues
func (r *DynamoGraphDeploymentRequestReconciler) updatePhaseAndRequeue(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, phase nvidiacomv1beta1.DGDRPhase, _ string) (ctrl.Result, error) {
	dgdr.Status.Phase = phase
	setSucceededCondition(dgdr, phase)
	if err := r.Status().Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

// updatePhaseWithCondition updates phase and adds/updates a condition
func (r *DynamoGraphDeploymentRequestReconciler) updatePhaseWithCondition(
	ctx context.Context,
	dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	phase nvidiacomv1beta1.DGDRPhase,
	conditionType string,
	status metav1.ConditionStatus,
	reason string,
	message string,
) (ctrl.Result, error) {
	dgdr.Status.Phase = phase
	setSucceededCondition(dgdr, phase)

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
		For(&nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
		Named(consts.ResourceTypeDynamoGraphDeploymentRequest).
		Owns(&batchv1.Job{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the job
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})). // Watch Jobs created by this controller (via ownerReference)
		Watches(
			&dgdv1alpha1.DynamoGraphDeployment{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				// Find DGDR by label instead of owner reference
				dgd := obj.(*dgdv1alpha1.DynamoGraphDeployment)
				dgdrName, hasName := dgd.Labels[nvidiacomv1beta1.LabelDGDRName]
				dgdrNamespace, hasNamespace := dgd.Labels[nvidiacomv1beta1.LabelDGDRNamespace]
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
		).
		// Watch DGDs created by this controller (via label)
		// Set the event filter to ignore resources handled by other controllers in namespace-restricted mode
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeploymentRequest))
}
