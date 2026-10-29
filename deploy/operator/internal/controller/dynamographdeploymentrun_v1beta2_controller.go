/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/yaml"
)

const (
	dgdrRunJobControllerUIDIndex = ".metadata.dgdrRunControllerUID"
	dgdrRunMaxJobNameLength      = 63

	dgdrRunInputKey                = "input.yaml"
	dgdrRunPlannerImageKey         = "planner-image"
	dgdrRunPlannerPullPolicyKey    = "planner-image-pull-policy"
	dgdrRunRuntimeImageSourceKey   = "runtime-image-source"
	dgdrRunPublisherImageKey       = "publisher-image"
	dgdrRunPublisherPullPolicyKey  = "publisher-image-pull-policy"
	dgdrRunDynamoVersionKey        = "dynamo-version"
	dgdrRunGPUsPerNodeKey          = "gpus-per-node"
	dgdrRunSnapshotPathKey         = "snapshot-path"
	dgdrRunImagePullSecretsKey     = "image-pull-secrets.json"
	dgdrRunInputMountPath          = "/etc/aisimulate"
	dgdrRunInputPath               = dgdrRunInputMountPath + "/" + dgdrRunInputKey
	dgdrRunSnapshotPath            = "/var/run/aisimulate/output"
	dgdrRunServiceAccountMountPath = "/var/run/secrets/kubernetes.io/serviceaccount"
	dgdrRunPublisherBinary         = "/dgdr-run-publisher"
	dgdrRunSweeperContainer        = "sweeper"
	dgdrRunPublisherContainer      = "publisher"
	dgdrRunInputVolume             = "input"
	dgdrRunSnapshotVolume          = "snapshot"
	dgdrRunServiceAccountVolume    = "publisher-kube-api"
	dgdrRunGPUsPerNodeAnnotation   = "nvidia.com/dgdr-gpus-per-node"
	dgdrRunSnapshotPathAnnotation  = "nvidia.com/dgdr-snapshot-path"
	dgdrRunResourceComponent       = "aisimulate"
	dgdrRunServiceAccountTokenTTL  = int64(3600)
	dgdrRunReasonPending           = "JobPending"
	dgdrRunReasonRunning           = "JobRunning"
	dgdrRunReasonSucceeded         = "SearchSucceeded"
	dgdrRunReasonSearchFailed      = "SearchFailed"
	dgdrRunReasonProtocolViolation = "ProtocolViolation"
	dgdrRunReasonPublicationFailed = "PublicationFailed"
	dgdrRunReasonJobFailed         = "JobFailed"
	dgdrRunReasonSuperseded        = "Superseded"
	dgdrRunReasonInvalidExecution  = "InvalidExecution"
	dgdrRunReasonInvalidResource   = "InvalidResource"
	dgdrRunMessagePending          = "AISimulate Job is waiting to start"
	dgdrRunMessageRunning          = "AISimulate Job is running"
	dgdrRunMessageSucceeded        = "AISimulate search completed successfully"
	dgdrRunExecutionCacheRetry     = time.Second
)

var errDGDRRunExecutionPending = errors.New("immutable execution ConfigMap is not visible")

type dgdrRunPermanentExecutionError struct {
	cause error
}

func (e *dgdrRunPermanentExecutionError) Error() string { return e.cause.Error() }
func (e *dgdrRunPermanentExecutionError) Unwrap() error { return e.cause }

type dgdrRunRetryableError struct {
	cause error
}

func (e *dgdrRunRetryableError) Error() string { return e.cause.Error() }
func (e *dgdrRunRetryableError) Unwrap() error { return e.cause }

type dgdrRunHardwareDiscoverer interface {
	Discover(context.Context, nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error)
}

// DynamoGraphDeploymentRunV1beta2Reconciler executes one immutable Run as an
// AISimulate Job. The publisher sidecar owns progress and candidate publication;
// this controller owns the Job reference, execution phase, timestamps and Completed.
type DynamoGraphDeploymentRunV1beta2Reconciler struct {
	client.Client
	APIReader               client.Reader
	Config                  *configv1alpha1.OperatorConfiguration
	RuntimeConfig           *commoncontroller.RuntimeConfig
	Hardware                dgdrRunHardwareDiscoverer
	PlannerImage            string
	DynamoVersion           string
	OperatorImage           string
	OperatorImagePullPolicy corev1.PullPolicy
	DockerSecretRetriever   DockerSecretRetriever
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentcandidates,verbs=get;list;watch;create;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentcandidates/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps;serviceaccounts,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update;patch

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := r.Get(ctx, req.NamespacedName, run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if run.DeletionTimestamp != nil || meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted) {
		return ctrl.Result{}, nil
	}
	result, err := r.reconcileRun(ctx, run)
	if err == nil || !dgdrRunIsPermanentError(err) {
		return result, err
	}
	reason := dgdrRunReasonInvalidExecution
	if _, ok := errors.AsType[*commoncontroller.OwnershipConflictError](err); ok {
		reason = commoncontroller.EventReasonOwnershipConflict
	} else if apierrors.IsInvalid(err) {
		reason = dgdrRunReasonInvalidResource
	}
	if statusErr := r.completeRun(ctx, run, reason, err.Error()); statusErr != nil {
		return ctrl.Result{}, statusErr
	}
	return ctrl.Result{}, reconcile.TerminalError(err)
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcileRun(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) (ctrl.Result, error) {
	jobs, err := r.ownedJobs(ctx, run)
	if err != nil {
		return ctrl.Result{}, err
	}
	if len(jobs) > 1 {
		return ctrl.Result{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf(
			"Run %q controls %d Jobs; expected at most one", run.Name, len(jobs))}
	}

	var job *batchv1.Job
	if len(jobs) == 1 {
		job = &jobs[0]
	}
	if supersedingGeneration, cancelRequested := dgdrRunSupersedingGeneration(run); cancelRequested {
		return ctrl.Result{}, r.cancelSupersededRun(ctx, run, job, supersedingGeneration)
	}
	resourceName := dgdrRunResourceName(run.Name)
	if err := r.reconcilePublisherAccess(ctx, run, resourceName); err != nil {
		return ctrl.Result{}, err
	}
	execution, err := r.reconcileInput(ctx, run, job, resourceName)
	if err != nil {
		if errors.Is(err, errDGDRRunExecutionPending) {
			return ctrl.Result{RequeueAfter: dgdrRunExecutionCacheRetry}, nil
		}
		return ctrl.Result{}, err
	}

	if job == nil {
		job = r.buildJob(run, resourceName, execution)
		if err := controllerutil.SetControllerReference(run, job, r.Scheme()); err != nil {
			return ctrl.Result{}, fmt.Errorf("set Job owner: %w", err)
		}
		if err := r.Create(ctx, job); err != nil {
			// A cache miss followed by AlreadyExists is expected to heal on the next
			// reconciliation once the informer observes the Job. Never bypass the cache.
			return ctrl.Result{}, fmt.Errorf("create Job %q: %w", job.Name, err)
		}
	}

	return ctrl.Result{}, r.reconcileRunStatus(ctx, run, job, execution)
}

func dgdrRunSupersedingGeneration(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) (string, bool) {
	runGeneration, err := dgdrV1beta2RunGeneration(run)
	if err != nil {
		return "", false
	}
	cancelGenerationValue, ok := run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]
	if !ok {
		return "", false
	}
	cancelGeneration, err := strconv.ParseInt(cancelGenerationValue, 10, 64)
	return cancelGenerationValue, err == nil && cancelGeneration > runGeneration
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) cancelSupersededRun(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
	supersedingGeneration string,
) error {
	if job != nil {
		if job.DeletionTimestamp.IsZero() {
			err := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationForeground))
			if err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("stop Job %q for superseded Run %q: %w", job.Name, run.Name, err)
			}
			if err == nil {
				return nil
			}
		} else {
			return nil
		}
	}

	return r.completeRun(ctx, run, dgdrRunReasonSuperseded,
		fmt.Sprintf("Run was superseded by Request generation %s", supersedingGeneration))
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) completeRun(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	reason, message string,
) error {
	original := run.DeepCopy()
	run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseFailed
	now := metav1.Now()
	run.Status.CompletionTime = &now
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status:             metav1.ConditionTrue,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: run.Generation,
	})
	if apiequality.Semantic.DeepEqual(original.Status, run.Status) {
		return nil
	}
	if err := r.Status().Patch(ctx, run, client.MergeFrom(original)); err != nil {
		return &dgdrRunRetryableError{cause: fmt.Errorf("complete Run %q: %w", run.Name, err)}
	}
	return nil
}

func dgdrRunIsPermanentError(err error) bool {
	if _, ok := errors.AsType[*dgdrRunRetryableError](err); ok {
		return false
	}
	if _, ok := errors.AsType[*dgdrRunPermanentExecutionError](err); ok {
		return true
	}
	_, ownershipConflict := errors.AsType[*commoncontroller.OwnershipConflictError](err)
	return ownershipConflict || apierrors.IsInvalid(err)
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) ownedJobs(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) ([]batchv1.Job, error) {
	list := &batchv1.JobList{}
	if err := r.List(ctx, list, client.InNamespace(run.Namespace),
		client.MatchingFields{dgdrRunJobControllerUIDIndex: string(run.UID)}); err != nil {
		return nil, fmt.Errorf("list Jobs of Run %q: %w", run.Name, err)
	}
	owned := make([]batchv1.Job, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], run) {
			owned = append(owned, list.Items[i])
		}
	}
	return owned, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcilePublisherAccess(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	name string,
) error {
	sa := &corev1.ServiceAccount{Name: name, Namespace: run.Namespace}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, sa, func() error {
		if sa.ResourceVersion != "" {
			if err := commoncontroller.CheckControllerOwnership(sa, run, r.Scheme()); err != nil {
				return err
			}
		}
		return controllerutil.SetControllerReference(run, sa, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile ServiceAccount %q: %w", name, err)
	}

	role := &rbacv1.Role{Name: name, Namespace: run.Namespace}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, role, func() error {
		if role.ResourceVersion != "" {
			if err := commoncontroller.CheckControllerOwnership(role, run, r.Scheme()); err != nil {
				return err
			}
		}
		role.Rules = dgdrRunPublisherRules(run.Name)
		return controllerutil.SetControllerReference(run, role, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile Role %q: %w", name, err)
	}

	binding := &rbacv1.RoleBinding{Name: name, Namespace: run.Namespace}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, binding, func() error {
		if binding.ResourceVersion != "" {
			if err := commoncontroller.CheckControllerOwnership(binding, run, r.Scheme()); err != nil {
				return err
			}
		}
		binding.RoleRef = rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: name}
		binding.Subjects = []rbacv1.Subject{{Kind: "ServiceAccount", Name: name, Namespace: run.Namespace}}
		return controllerutil.SetControllerReference(run, binding, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile RoleBinding %q: %w", name, err)
	}
	return nil
}

func dgdrRunPublisherRules(runName string) []rbacv1.PolicyRule {
	return []rbacv1.PolicyRule{
		{
			APIGroups:     []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources:     []string{"dynamographdeploymentruns"},
			ResourceNames: []string{runName},
			Verbs:         []string{"get", "list", "watch"},
		},
		{
			APIGroups:     []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources:     []string{"dynamographdeploymentruns/finalizers"},
			ResourceNames: []string{runName},
			Verbs:         []string{"update"},
		},
		{
			APIGroups:     []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources:     []string{"dynamographdeploymentruns/status"},
			ResourceNames: []string{runName},
			Verbs:         []string{"get", "update", "patch"},
		},
		{
			APIGroups: []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources: []string{"dynamographdeploymentcandidates"},
			Verbs:     []string{"get", "list", "watch", "create", "patch", "delete"},
		},
		{
			APIGroups: []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources: []string{"dynamographdeploymentcandidates/status"},
			Verbs:     []string{"get", "update", "patch"},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list", "watch"},
		},
	}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcileInput(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
	name string,
) (dgdrRunExecution, error) {
	configMap := &corev1.ConfigMap{}
	key := types.NamespacedName{Namespace: run.Namespace, Name: name}
	err := r.Get(ctx, key, configMap)
	if err == nil {
		return r.existingRunExecution(configMap, run, job)
	}
	if !apierrors.IsNotFound(err) {
		return dgdrRunExecution{}, fmt.Errorf("get ConfigMap %q: %w", name, err)
	}

	if job != nil {
		if r.APIReader != nil {
			persisted := &corev1.ConfigMap{}
			apiErr := r.APIReader.Get(ctx, key, persisted)
			if apiErr == nil {
				return dgdrRunExecution{}, fmt.Errorf(
					"%w: ConfigMap %q exists in the API but is not visible in the controller cache",
					errDGDRRunExecutionPending, name,
				)
			}
			if !apierrors.IsNotFound(apiErr) {
				return dgdrRunExecution{}, fmt.Errorf("read ConfigMap %q directly from the API: %w", name, apiErr)
			}
		}
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf(
			"immutable execution ConfigMap %q is missing while Job %q exists; refusing to reconstruct it from current operator configuration",
			name, job.Name,
		)}
	}

	// Resolve every execution value before creating the immutable artifact.
	execution := dgdrRunExecution{
		SnapshotPath:        dgdrRunSnapshotPath,
		PlannerImage:        r.PlannerImage,
		PlannerPullPolicy:   corev1.PullIfNotPresent,
		PublisherImage:      r.OperatorImage,
		PublisherPullPolicy: r.OperatorImagePullPolicy,
	}
	if execution.PublisherPullPolicy == "" {
		execution.PublisherPullPolicy = corev1.PullIfNotPresent
	}
	execution.GPUsPerNode, err = r.discoverGPUsPerNode(ctx, run)
	if err != nil {
		return dgdrRunExecution{}, err
	}
	execution.DynamoVersion, err = dgdrRunDynamoVersion(execution.PlannerImage, r.DynamoVersion)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	execution.RuntimeImageSource, err = dgdrRunRuntimeImageSource(execution.PlannerImage)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	execution.ImagePullSecrets, err = r.dgdrRunImagePullSecrets(run.Namespace, execution.PlannerImage, execution.PublisherImage)
	if err != nil {
		return dgdrRunExecution{}, err
	}

	input, err := dgdrRunInput(run, execution.RuntimeImageSource, execution.DynamoVersion, execution.GPUsPerNode)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	contents, err := yaml.Marshal(input)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf("encode AISimulate input: %w", err)}
	}
	execution.Input, err = dgdrRunStatusInput(string(contents))
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	pullSecrets, err := json.Marshal(execution.ImagePullSecrets)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf("encode image pull secrets: %w", err)}
	}

	configMap = &corev1.ConfigMap{
		Name:      name,
		Namespace: run.Namespace,
		Annotations: map[string]string{
			dgdrRunGPUsPerNodeAnnotation:  strconv.Itoa(execution.GPUsPerNode),
			dgdrRunSnapshotPathAnnotation: execution.SnapshotPath,
		},
		Immutable: new(true),
		Data: map[string]string{
			dgdrRunInputKey:               string(contents),
			dgdrRunPlannerImageKey:        execution.PlannerImage,
			dgdrRunPlannerPullPolicyKey:   string(execution.PlannerPullPolicy),
			dgdrRunRuntimeImageSourceKey:  execution.RuntimeImageSource,
			dgdrRunPublisherImageKey:      execution.PublisherImage,
			dgdrRunPublisherPullPolicyKey: string(execution.PublisherPullPolicy),
			dgdrRunDynamoVersionKey:       execution.DynamoVersion,
			dgdrRunGPUsPerNodeKey:         strconv.Itoa(execution.GPUsPerNode),
			dgdrRunSnapshotPathKey:        execution.SnapshotPath,
			dgdrRunImagePullSecretsKey:    string(pullSecrets),
		},
	}
	desired := configMap
	configMap = &corev1.ConfigMap{Name: name, Namespace: run.Namespace}
	operation, err := controllerutil.CreateOrUpdate(ctx, r.Client, configMap, func() error {
		if configMap.ResourceVersion != "" {
			return commoncontroller.CheckControllerOwnership(configMap, run, r.Scheme())
		}
		configMap.Annotations = desired.Annotations
		configMap.Immutable = desired.Immutable
		configMap.Data = desired.Data
		return controllerutil.SetControllerReference(run, configMap, r.Scheme())
	})
	if err != nil {
		return dgdrRunExecution{}, fmt.Errorf("reconcile ConfigMap %q: %w", name, err)
	}
	if operation != controllerutil.OperationResultCreated {
		return r.existingRunExecution(configMap, run, job)
	}
	return execution, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) existingRunExecution(
	configMap *corev1.ConfigMap,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
) (dgdrRunExecution, error) {
	if err := commoncontroller.CheckControllerOwnership(configMap, run, r.Scheme()); err != nil {
		return dgdrRunExecution{}, err
	}
	execution, err := dgdrRunExecutionFromConfigMap(configMap)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf("read ConfigMap %q: %w", configMap.Name, err)}
	}
	if job != nil {
		if err := validateDGDRRunJobExecution(job, execution); err != nil {
			return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf(
				"Job %q and ConfigMap %q disagree: %w", job.Name, configMap.Name, err)}
		}
	}
	return execution, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) discoverGPUsPerNode(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) (int, error) {
	if r.Hardware == nil {
		return 0, &dgdrRunRetryableError{cause: errors.New("discover GPUs per node: hardware discoverer is not configured")}
	}
	if run.Spec.Hardware == nil || run.Spec.Hardware.GPU == nil || run.Spec.Hardware.GPU.SKU == nil {
		return 0, &dgdrRunPermanentExecutionError{cause: errors.New(
			"discover GPUs per node: spec.hardware.gpu.sku was not resolved by Request admission")}
	}
	sku := *run.Spec.Hardware.GPU.SKU
	filterSKU := nvidiacomv1beta1.GPUSKUType("")
	if aisimulate.IsKnownHardwareSKU(sku) {
		filterSKU = nvidiacomv1beta1.GPUSKUType(sku)
	}
	info, err := r.Hardware.Discover(ctx, filterSKU)
	if err != nil {
		return 0, &dgdrRunRetryableError{cause: fmt.Errorf("discover GPUs per node: %w", err)}
	}
	if filterSKU == "" && info != nil && len(info.OtherModels) > 0 {
		return 0, &dgdrRunRetryableError{cause: fmt.Errorf(
			"discover GPUs per node for external AISimulate SKU %q: cluster discovery selected %q among multiple GPU models (%s)",
			sku, info.Model, strings.Join(info.OtherModels, ", "),
		)}
	}
	if info == nil || info.GPUsPerNode < 1 {
		return 0, &dgdrRunRetryableError{cause: errors.New("discover GPUs per node: discovery returned no positive GPU count")}
	}
	return info.GPUsPerNode, nil
}

func dgdrRunInput(
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	runtimeImageSource string,
	dynamoVersion string,
	gpusPerNode int,
) (map[string]any, error) {
	if run.Spec.Recommendation == nil || run.Spec.Recommendation.MaxCandidates < 1 {
		return nil, errors.New("spec.recommendation.maxCandidates must be defaulted before the Run is created")
	}
	output := map[string]any{
		"name":           run.Name,
		"max_candidates": int64(run.Spec.Recommendation.MaxCandidates),
	}
	underlay := map[string]any{"dgdr_run": output}
	input, err := aisimulate.LowerWithUnderlay(&run.Spec.NativeSearchInput, underlay)
	if err != nil {
		return nil, err
	}

	// Apply operator-owned execution values after user overrides have won over public search fields.
	reserved := map[string]any{
		"name":                     run.Name,
		"max_candidates":           int64(run.Spec.Recommendation.MaxCandidates),
		"snapshot_dir":             dgdrRunSnapshotPath,
		"namespace":                run.Namespace,
		"runtime_image":            runtimeImageSource,
		"runtime_version_override": dynamoVersion,
		"derive_runtime_images":    true,
		"num_gpus_per_node":        int64(gpusPerNode),
	}
	for key, value := range reserved {
		if err := unstructured.SetNestedField(input, value, "dgdr_run", key); err != nil {
			return nil, fmt.Errorf("set operator-owned dgdr_run.%s: %w", key, err)
		}
	}
	return input, nil
}

type dgdrRunExecution struct {
	GPUsPerNode         int
	SnapshotPath        string
	PlannerImage        string
	PlannerPullPolicy   corev1.PullPolicy
	RuntimeImageSource  string
	PublisherImage      string
	PublisherPullPolicy corev1.PullPolicy
	DynamoVersion       string
	ImagePullSecrets    []corev1.LocalObjectReference
	Input               *runtime.RawExtension
}

func dgdrRunDynamoVersion(plannerImage, dynamoVersion string) (string, error) {
	if version, err := runtimeversion.ParseImageVersion(plannerImage); err == nil {
		return version.String(), nil
	}
	version, err := runtimeversion.Parse(dynamoVersion)
	if err != nil {
		return "", fmt.Errorf("resolve Dynamo version for planner image %q: %w", plannerImage, err)
	}
	return version.String(), nil
}

func dgdrRunRuntimeImageSource(plannerImage string) (string, error) {
	ref := strings.TrimSpace(plannerImage)
	nameAndTag, _, hasDigest := strings.Cut(ref, "@")
	lastSlash, lastColon := strings.LastIndex(nameAndTag, "/"), strings.LastIndex(nameAndTag, ":")
	if lastColon > lastSlash && lastColon < len(nameAndTag)-1 {
		return ref, nil
	}
	if hasDigest {
		return "", fmt.Errorf(
			"resolve runtime images for planner image %q: a digest-only planner image cannot identify sibling runtime images; configure a shared release tag, optionally with a digest",
			plannerImage,
		)
	}
	return "", fmt.Errorf(
		"resolve runtime images for planner image %q: a shared release tag is required, optionally with a repository-specific digest",
		plannerImage,
	)
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) dgdrRunImagePullSecrets(
	namespace string,
	images ...string,
) ([]corev1.LocalObjectReference, error) {
	if r.DockerSecretRetriever == nil {
		return nil, nil
	}

	// Resolve every execution image against secrets visible in the Run namespace.
	var refs []corev1.LocalObjectReference
	for _, image := range images {
		names, err := r.DockerSecretRetriever.GetSecrets(namespace, image)
		if err != nil {
			return nil, &dgdrRunRetryableError{cause: fmt.Errorf(
				"resolve image pull secrets for %q in namespace %q: %w", image, namespace, err)}
		}
		additional := make([]corev1.LocalObjectReference, 0, len(names))
		for _, name := range names {
			additional = append(additional, corev1.LocalObjectReference{Name: name})
		}
		refs = commoncontroller.AppendUniqueImagePullSecrets(refs, additional)
	}
	return refs, nil
}

func dgdrRunExecutionFromConfigMap(configMap *corev1.ConfigMap) (dgdrRunExecution, error) {
	execution := dgdrRunExecution{
		PlannerImage:        configMap.Data[dgdrRunPlannerImageKey],
		PlannerPullPolicy:   corev1.PullPolicy(configMap.Data[dgdrRunPlannerPullPolicyKey]),
		RuntimeImageSource:  configMap.Data[dgdrRunRuntimeImageSourceKey],
		PublisherImage:      configMap.Data[dgdrRunPublisherImageKey],
		PublisherPullPolicy: corev1.PullPolicy(configMap.Data[dgdrRunPublisherPullPolicyKey]),
		DynamoVersion:       configMap.Data[dgdrRunDynamoVersionKey],
		SnapshotPath:        configMap.Data[dgdrRunSnapshotPathKey],
	}
	for key, value := range map[string]string{
		dgdrRunPlannerImageKey:        execution.PlannerImage,
		dgdrRunPlannerPullPolicyKey:   string(execution.PlannerPullPolicy),
		dgdrRunRuntimeImageSourceKey:  execution.RuntimeImageSource,
		dgdrRunPublisherImageKey:      execution.PublisherImage,
		dgdrRunPublisherPullPolicyKey: string(execution.PublisherPullPolicy),
		dgdrRunDynamoVersionKey:       execution.DynamoVersion,
		dgdrRunSnapshotPathKey:        execution.SnapshotPath,
	} {
		if value == "" {
			return dgdrRunExecution{}, fmt.Errorf("data key %q is empty", key)
		}
	}
	if execution.PlannerPullPolicy != corev1.PullAlways &&
		execution.PlannerPullPolicy != corev1.PullIfNotPresent &&
		execution.PlannerPullPolicy != corev1.PullNever {
		return dgdrRunExecution{}, fmt.Errorf("data key %q=%q is not a valid image pull policy", dgdrRunPlannerPullPolicyKey, execution.PlannerPullPolicy)
	}
	if execution.PublisherPullPolicy != corev1.PullAlways &&
		execution.PublisherPullPolicy != corev1.PullIfNotPresent &&
		execution.PublisherPullPolicy != corev1.PullNever {
		return dgdrRunExecution{}, fmt.Errorf("data key %q=%q is not a valid image pull policy", dgdrRunPublisherPullPolicyKey, execution.PublisherPullPolicy)
	}

	// Decode the immutable topology, credentials and effective input.
	var err error
	execution.GPUsPerNode, err = positiveInteger(configMap.Data[dgdrRunGPUsPerNodeKey], "data key "+dgdrRunGPUsPerNodeKey)
	if err != nil {
		return dgdrRunExecution{}, err
	}
	if err := json.Unmarshal([]byte(configMap.Data[dgdrRunImagePullSecretsKey]), &execution.ImagePullSecrets); err != nil {
		return dgdrRunExecution{}, fmt.Errorf("decode data key %q: %w", dgdrRunImagePullSecretsKey, err)
	}
	execution.Input, err = dgdrRunStatusInput(configMap.Data[dgdrRunInputKey])
	if err != nil {
		return dgdrRunExecution{}, fmt.Errorf("decode data key %q: %w", dgdrRunInputKey, err)
	}
	return execution, nil
}

func validateDGDRRunJobExecution(job *batchv1.Job, execution dgdrRunExecution) error {
	jobGPUs, err := positiveAnnotation(job.Annotations, dgdrRunGPUsPerNodeAnnotation)
	if err != nil {
		return err
	}
	if jobGPUs != execution.GPUsPerNode {
		return errors.New("GPU counts differ")
	}
	if job.Annotations[dgdrRunSnapshotPathAnnotation] != execution.SnapshotPath {
		return errors.New("snapshot paths differ")
	}

	// Match both container identities and all credentials against the frozen artifact.
	planner, found := dgdrRunContainer(job.Spec.Template.Spec.Containers, dgdrRunSweeperContainer)
	if !found || planner.Image != execution.PlannerImage || planner.ImagePullPolicy != execution.PlannerPullPolicy {
		return errors.New("planner image identities differ")
	}
	publisher, found := dgdrRunContainer(job.Spec.Template.Spec.Containers, dgdrRunPublisherContainer)
	if !found || publisher.Image != execution.PublisherImage || publisher.ImagePullPolicy != execution.PublisherPullPolicy {
		return errors.New("publisher image identities differ")
	}
	if !apiequality.Semantic.DeepEqual(job.Spec.Template.Spec.ImagePullSecrets, execution.ImagePullSecrets) {
		return errors.New("image pull secrets differ")
	}
	return nil
}

func dgdrRunContainer(containers []corev1.Container, name string) (corev1.Container, bool) {
	for _, container := range containers {
		if container.Name == name {
			return container, true
		}
	}
	return corev1.Container{}, false
}

func dgdrRunStatusInput(contents string) (*runtime.RawExtension, error) {
	raw, err := yaml.YAMLToJSON([]byte(contents))
	if err != nil {
		return nil, fmt.Errorf("decode AISimulate input: %w", err)
	}
	var input map[string]json.RawMessage
	if err := json.Unmarshal(raw, &input); err != nil {
		return nil, fmt.Errorf("decode AISimulate input: %w", err)
	}
	if input == nil {
		return nil, errors.New("decode AISimulate input: expected an object")
	}
	return &runtime.RawExtension{Raw: raw}, nil
}

func positiveAnnotation(annotations map[string]string, key string) (int, error) {
	return positiveInteger(annotations[key], "annotation "+key)
}

func positiveInteger(value, field string) (int, error) {
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 1 {
		return 0, fmt.Errorf("%s=%q is not a positive integer", field, value)
	}
	return parsed, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) buildJob(
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	name string,
	execution dgdrRunExecution,
) *batchv1.Job {
	// Pin the named planner-image user to its numeric UID so kubelet can enforce runAsNonRoot.
	sweeperSecurityContext := dgdrRunContainerSecurityContext()
	sweeperSecurityContext.RunAsUser = ptr.To[int64](1000)

	job := &batchv1.Job{
		Name:      name,
		Namespace: run.Namespace,
		Annotations: map[string]string{
			dgdrRunGPUsPerNodeAnnotation:  strconv.Itoa(execution.GPUsPerNode),
			dgdrRunSnapshotPathAnnotation: execution.SnapshotPath,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: ptr.To[int32](0),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName:           name,
					AutomountServiceAccountToken: new(false),
					RestartPolicy:                corev1.RestartPolicyNever,
					SecurityContext: &corev1.PodSecurityContext{
						FSGroup:      ptr.To[int64](65532),
						RunAsNonRoot: new(true),
						SeccompProfile: &corev1.SeccompProfile{
							Type: corev1.SeccompProfileTypeRuntimeDefault,
						},
					},
					ImagePullSecrets: append([]corev1.LocalObjectReference(nil), execution.ImagePullSecrets...),
					Containers: []corev1.Container{
						{
							Name:            dgdrRunSweeperContainer,
							Image:           execution.PlannerImage,
							ImagePullPolicy: execution.PlannerPullPolicy,
							Command:         []string{"aisimulate"},
							Args: []string{
								"recommend", "--stack", "dynamo",
								"--config", dgdrRunInputPath,
								"--output", "dgdr_run",
								"--output-dir", execution.SnapshotPath,
								"--overwrite",
							},
							SecurityContext: sweeperSecurityContext,
							VolumeMounts: []corev1.VolumeMount{
								{Name: dgdrRunInputVolume, MountPath: dgdrRunInputMountPath, ReadOnly: true},
								{Name: dgdrRunSnapshotVolume, MountPath: execution.SnapshotPath},
							},
						},
						{
							Name:            dgdrRunPublisherContainer,
							Image:           execution.PublisherImage,
							ImagePullPolicy: execution.PublisherPullPolicy,
							Command:         []string{dgdrRunPublisherBinary},
							Args: []string{
								"--snapshot-dir", execution.SnapshotPath,
								"--run-name", run.Name,
								"--run-uid", string(run.UID),
								"--sweeper-container", dgdrRunSweeperContainer,
							},
							Env: []corev1.EnvVar{
								{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
								{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
							},
							SecurityContext: dgdrRunContainerSecurityContext(),
							VolumeMounts: []corev1.VolumeMount{
								{Name: dgdrRunSnapshotVolume, MountPath: execution.SnapshotPath},
								{Name: dgdrRunServiceAccountVolume, MountPath: dgdrRunServiceAccountMountPath, ReadOnly: true},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: dgdrRunInputVolume,
							ConfigMap: &corev1.ConfigMapVolumeSource{
								Name: name,
							},
						},
						{Name: dgdrRunSnapshotVolume, EmptyDir: &corev1.EmptyDirVolumeSource{}},
						dgdrRunPublisherKubeAPIVolume(),
					},
				},
			},
		},
	}
	return job
}

func dgdrRunContainerSecurityContext() *corev1.SecurityContext {
	return &corev1.SecurityContext{
		AllowPrivilegeEscalation: new(false),
		Capabilities: &corev1.Capabilities{
			Drop: []corev1.Capability{"ALL"},
		},
	}
}

func dgdrRunPublisherKubeAPIVolume() corev1.Volume {
	return corev1.Volume{
		Name: dgdrRunServiceAccountVolume,
		Projected: &corev1.ProjectedVolumeSource{
			Sources: []corev1.VolumeProjection{
				{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
					Path: "token", ExpirationSeconds: new(dgdrRunServiceAccountTokenTTL),
				}},
				{ConfigMap: &corev1.ConfigMapProjection{
					Name:  ConfigMapNameKubeRootCA,
					Items: []corev1.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}},
				}},
				{DownwardAPI: &corev1.DownwardAPIProjection{Items: []corev1.DownwardAPIVolumeFile{{
					Path: "namespace", FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.namespace"},
				}}}},
			},
		},
	}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcileRunStatus(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
	execution dgdrRunExecution,
) error {
	if execution.Input == nil {
		return &dgdrRunPermanentExecutionError{cause: errors.New(
			"update Run status: effective AISimulate input is missing")}
	}
	original := run.DeepCopy()
	run.Status.AISimulate = &nvidiacomv1beta2.AISimulateRunStatus{Input: execution.Input.DeepCopy()}
	run.Status.JobRef = &nvidiacomv1beta2.ResourceReference{Name: job.Name, UID: job.UID}
	run.Status.Phase = nvidiacomv1beta2.DGDRRunPhasePending
	reason, message := dgdrRunReasonPending, dgdrRunMessagePending
	completed := metav1.ConditionFalse

	if job.Status.StartTime != nil {
		run.Status.StartTime = job.Status.StartTime.DeepCopy()
	}
	if condition := dgdrRunJobCondition(job, batchv1.JobComplete); condition != nil && condition.Status == corev1.ConditionTrue {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseSucceeded
		reason, message = dgdrRunReasonSucceeded, dgdrRunMessageSucceeded
		completed = metav1.ConditionTrue
		run.Status.CompletionTime = dgdrRunCompletionTime(job)
	} else if condition := dgdrRunJobCondition(job, batchv1.JobFailed); condition != nil && condition.Status == corev1.ConditionTrue {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseFailed
		reason, message = r.dgdrRunFailure(ctx, job, condition)
		completed = metav1.ConditionTrue
		run.Status.CompletionTime = dgdrRunCompletionTime(job)
	} else if job.Status.Active > 0 || job.Status.StartTime != nil {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseRunning
		reason, message = dgdrRunReasonRunning, dgdrRunMessageRunning
	}

	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status:             completed,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: run.Generation,
	})
	if apiequality.Semantic.DeepEqual(original.Status, run.Status) {
		return nil
	}
	if err := r.Status().Patch(ctx, run, client.MergeFrom(original)); err != nil {
		return &dgdrRunRetryableError{cause: fmt.Errorf("update Run %q status: %w", run.Name, err)}
	}
	return nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) dgdrRunFailure(
	ctx context.Context,
	job *batchv1.Job,
	condition *batchv1.JobCondition,
) (string, string) {
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(job.Namespace), client.MatchingLabels{batchv1.JobNameLabel: job.Name}); err != nil {
		log.FromContext(ctx).Error(err, "Could not inspect failed AISimulate Job pods", "job", job.Name)
		return dgdrRunReasonJobFailed, condition.Message
	}
	if terminated := dgdrRunFailedContainer(pods.Items, job, dgdrRunPublisherContainer); terminated != nil {
		message := dgdrRunPublisherFailureMessage(terminated)
		switch int(terminated.ExitCode) {
		case dgdrrunpublisher.ExitRunFailed:
			return dgdrRunReasonSearchFailed, message
		case dgdrrunpublisher.ExitProtocolViolation:
			return dgdrRunReasonProtocolViolation, message
		case dgdrrunpublisher.ExitReconcileFailed:
			return dgdrRunReasonPublicationFailed, message
		default:
			return dgdrRunReasonJobFailed, message
		}
	}
	if terminated := dgdrRunFailedContainer(pods.Items, job, dgdrRunSweeperContainer); terminated != nil {
		return dgdrRunReasonSearchFailed, dgdrRunFailureMessage(dgdrRunSweeperContainer, terminated)
	}
	message := condition.Message
	if message == "" {
		message = "AISimulate Job failed"
	}
	return dgdrRunReasonJobFailed, message
}

func dgdrRunFailedContainer(
	pods []corev1.Pod,
	job *batchv1.Job,
	containerName string,
) *corev1.ContainerStateTerminated {
	for i := range pods {
		if !metav1.IsControlledBy(&pods[i], job) {
			continue
		}
		for j := range pods[i].Status.ContainerStatuses {
			status := &pods[i].Status.ContainerStatuses[j]
			if status.Name == containerName && status.State.Terminated != nil && status.State.Terminated.ExitCode != 0 {
				return status.State.Terminated
			}
		}
	}
	return nil
}

func dgdrRunPublisherFailureMessage(terminated *corev1.ContainerStateTerminated) string {
	var outcome dgdrrunpublisher.TerminalOutcome
	if err := json.Unmarshal([]byte(terminated.Message), &outcome); err == nil {
		parts := make([]string, 0, 2)
		for _, part := range []string{outcome.Message, outcome.Error} {
			if part = strings.TrimSpace(part); part != "" {
				parts = append(parts, part)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, ": ")
		}
	}
	return dgdrRunFailureMessage(dgdrRunPublisherContainer, terminated)
}

func dgdrRunFailureMessage(containerName string, terminated *corev1.ContainerStateTerminated) string {
	if terminated.Message != "" {
		return terminated.Message
	}
	if terminated.Reason != "" {
		return terminated.Reason
	}
	return fmt.Sprintf("container %s exited with code %d", containerName, terminated.ExitCode)
}

func dgdrRunCompletionTime(job *batchv1.Job) *metav1.Time {
	if job.Status.CompletionTime != nil {
		return job.Status.CompletionTime.DeepCopy()
	}
	for _, condition := range job.Status.Conditions {
		if (condition.Type == batchv1.JobComplete || condition.Type == batchv1.JobFailed) && condition.Status == corev1.ConditionTrue {
			return condition.LastTransitionTime.DeepCopy()
		}
	}
	return nil
}

func dgdrRunJobCondition(job *batchv1.Job, conditionType batchv1.JobConditionType) *batchv1.JobCondition {
	for i := range job.Status.Conditions {
		if job.Status.Conditions[i].Type == conditionType {
			return &job.Status.Conditions[i]
		}
	}
	return nil
}

func dgdrRunResourceName(runName string) string {
	suffix := "-" + dgdrRunResourceComponent
	if len(runName)+len(suffix) <= dgdrRunMaxJobNameLength {
		return runName + suffix
	}
	sum := sha256.Sum256([]byte(runName))
	hash := hex.EncodeToString(sum[:6])
	room := dgdrRunMaxJobNameLength - len(suffix) - len(hash) - 1
	prefix := strings.TrimRight(runName[:room], "-.")
	return prefix + "-" + hash + suffix
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.APIReader == nil {
		r.APIReader = mgr.GetAPIReader()
	}
	if r.PlannerImage == "" {
		return errors.New("planner image is required for the DGDR Run sweeper")
	}
	if r.OperatorImage == "" {
		return errors.New("operator image is required for the DGDR Run publisher")
	}
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(), &batchv1.Job{}, dgdrRunJobControllerUIDIndex, dgdrRunJobControllerUIDs,
	); err != nil {
		return fmt.Errorf("index Jobs by Run controller UID: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrun-v1beta2").
		For(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}

func dgdrRunJobControllerUIDs(obj client.Object) []string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil || owner.Kind != "DynamoGraphDeploymentRun" || owner.APIVersion != nvidiacomv1beta2.GroupVersion.String() {
		return nil
	}
	return []string{string(owner.UID)}
}
