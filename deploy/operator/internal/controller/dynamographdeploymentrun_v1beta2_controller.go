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
	"maps"
	"strconv"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
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
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/yaml"
)

const (
	dgdrRunJobControllerUIDIndex = ".metadata.dgdrRunControllerUID"
	dgdrRunTraceClaimIndex       = ".spec.workload.trace.source.pvc.claimName"
	dgdrRunMaxJobNameLength      = 63

	dgdrRunInputKey                = "input.yaml"
	dgdrRunPlannerImageKey         = "planner-image"
	dgdrRunPlannerPullPolicyKey    = "planner-image-pull-policy"
	dgdrRunRuntimeImageSourceKey   = "runtime-image-source"
	dgdrRunPublisherImageKey       = "publisher-image"
	dgdrRunPublisherPullPolicyKey  = "publisher-image-pull-policy"
	dgdrRunDynamoVersionKey        = "dynamo-version"
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
	dgdrRunTraceVolume             = "trace"
	dgdrRunServiceAccountVolume    = "publisher-kube-api"
	dgdrRunSnapshotPathAnnotation  = "nvidia.com/dgdr-snapshot-path"
	dgdrRunResourceComponent       = "aisimulate"
	dgdrRunServiceAccountTokenTTL  = int64(3600)
	dgdrRunTraceGroup              = int64(65532)
	dgdrRunLegacyFSGroup           = int64(65532)
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

// DynamoGraphDeploymentRunV1beta2Reconciler executes one immutable Run as an
// AISimulate Job. The publisher sidecar owns progress and candidate publication;
// this controller owns the Job reference, execution phase, timestamps and Completed.
type DynamoGraphDeploymentRunV1beta2Reconciler struct {
	client.Client
	APIReader               client.Reader
	Config                  *configv1alpha1.OperatorConfiguration
	RuntimeConfig           *commoncontroller.RuntimeConfig
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
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch
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
	if run.DeletionTimestamp != nil {
		return ctrl.Result{}, nil
	}

	// A Run fails for its trace while its Job may still run, because the failure is recorded before
	// the Job is deleted: a failed or stale write then leaves the Job in place rather than gone and
	// recreated. Deleting it is the one step left for such a completed Run.
	if completed := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted); completed != nil &&
		completed.Status == metav1.ConditionTrue {
		if completed.Reason != dgdrRunReasonTraceSourceUnavailable {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, r.deleteUnfinishedJobs(ctx, run)
	}

	if condition := dgdrRunPendingPermanentFailure(run); condition != nil {
		return r.finishPermanentFailure(ctx, run, condition.Reason, errors.New(condition.Message))
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
	return r.beginPermanentFailure(ctx, run, reason, err)
}

func dgdrRunPendingPermanentFailure(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) *metav1.Condition {
	condition := meta.FindStatusCondition(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
	if condition == nil || condition.Status != metav1.ConditionFalse {
		return nil
	}
	switch condition.Reason {
	case dgdrRunReasonInvalidExecution, dgdrRunReasonInvalidResource, commoncontroller.EventReasonOwnershipConflict:
		return condition
	default:
		return nil
	}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) beginPermanentFailure(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	reason string,
	cause error,
) (ctrl.Result, error) {
	original := run.DeepCopy()
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status:             metav1.ConditionFalse,
		Reason:             reason,
		Message:            cause.Error(),
		ObservedGeneration: run.Generation,
	})
	// Like every Run status write, the marker carries the resourceVersion the cache showed, so a
	// stale Run cannot overwrite a newer status, such as a recorded trace failure.
	if !apiequality.Semantic.DeepEqual(original.Status, run.Status) {
		if err := r.Status().Patch(ctx, run, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
			return ctrl.Result{}, &dgdrRunRetryableError{cause: fmt.Errorf("record permanent failure for Run %q: %w", run.Name, err)}
		}
	}
	// Deletion is a later observed-state transition. Returning here ensures the
	// controller never acts on a marker that only exists in this local object.
	return ctrl.Result{RequeueAfter: dgdrRunExecutionCacheRetry}, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) finishPermanentFailure(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	reason string,
	cause error,
) (ctrl.Result, error) {
	jobs, err := r.ownedJobs(ctx, run)
	if err != nil {
		return ctrl.Result{}, err
	}
	stopped, err := r.stopOwnedJobs(ctx, run, jobs, "after permanent failure")
	if err != nil {
		return ctrl.Result{}, err
	}
	if !stopped {
		return ctrl.Result{RequeueAfter: dgdrRunExecutionCacheRetry}, nil
	}
	if err := r.completeRun(ctx, run, reason, cause.Error()); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, reconcile.TerminalError(cause)
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
		return r.cancelSupersededRun(ctx, run, job, supersedingGeneration)
	}

	// A Job that can never mount its trace would keep the Run active forever, so a Run that has not
	// created its Job checks the claim before anything is created for it. A recorded jobRef, or a Job
	// the API server already holds, means the Job was created even if the cache does not show it yet;
	// it is checked below once the cache shows it.
	resourceName := dgdrRunResourceName(run.Name)
	if job == nil && run.Status.JobRef == nil {
		unavailable, err := r.traceSourceUnavailable(ctx, run, false)
		if err != nil {
			return ctrl.Result{}, err
		}
		if unavailable != "" {
			created, err := r.runJobExists(ctx, run, resourceName)
			if err != nil {
				return ctrl.Result{}, err
			}
			if !created {
				return ctrl.Result{}, r.completeRun(ctx, run, dgdrRunReasonTraceSourceUnavailable, unavailable)
			}
		}
	}

	// A Job whose sweeper can never read the trace would stay pending with nothing to end it: the
	// Run fails, and the completed Run then deletes the Job. Claim and pod events wake the Run for
	// this check.
	if job != nil && dgdrRunTraceClaim(run) != nil && !dgdrRunJobFinished(job) {
		unavailable, err := r.jobTraceUnavailable(ctx, run, job)
		if err != nil {
			return ctrl.Result{}, err
		}
		if unavailable != "" {
			return ctrl.Result{}, r.completeRun(ctx, run, dgdrRunReasonTraceSourceUnavailable,
				fmt.Sprintf("%s; Job %q is deleted", unavailable, job.Name))
		}
	}

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
) (ctrl.Result, error) {
	var cached []batchv1.Job
	if job != nil {
		cached = []batchv1.Job{*job}
	}
	stopped, err := r.stopOwnedJobs(ctx, run, cached, "for supersession")
	if err != nil {
		return ctrl.Result{}, err
	}
	if !stopped {
		return ctrl.Result{RequeueAfter: dgdrRunExecutionCacheRetry}, nil
	}

	return ctrl.Result{}, r.completeRun(ctx, run, dgdrRunReasonSuperseded,
		fmt.Sprintf("Run was superseded by Request generation %s", supersedingGeneration))
}

// stopOwnedJobs stops every execution visible through either the informer cache or the API
// server. Completion is safe only after the uncached reader also confirms absence.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) stopOwnedJobs(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	cached []batchv1.Job,
	action string,
) (bool, error) {
	jobs := cached
	if len(jobs) == 0 {
		var err error
		jobs, err = r.uncachedOwnedJobs(ctx, run)
		if err != nil {
			return false, err
		}
	}
	if len(jobs) == 0 {
		return true, nil
	}
	for i := range jobs {
		if !jobs[i].DeletionTimestamp.IsZero() {
			continue
		}
		if err := r.Delete(ctx, &jobs[i], client.PropagationPolicy(metav1.DeletePropagationForeground)); err != nil && !apierrors.IsNotFound(err) {
			return false, fmt.Errorf("stop Job %q %s of Run %q: %w", jobs[i].Name, action, run.Name, err)
		}
	}
	return false, nil
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
	// The write carries the resourceVersion the cache showed, so a stale Run conflicts instead of
	// overwriting a newer status.
	if err := r.Status().Patch(ctx, run, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
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

const (
	dgdrRunReasonTraceSourceUnavailable = "TraceSourceUnavailable"
	dgdrRunClaimProtectionFinalizer     = "kubernetes.io/pvc-protection"
)

// traceSourceUnavailable returns why a Job could not mount the Run's trace claim, or "" when it
// can or the Run replays no trace. scheduled tells whether a pod of the Job is already bound to a
// node: the kubelet still mounts a claim being deleted while the pvc-protection finalizer keeps it
// for that pod. The answer fails the Run terminally, so the claim is read through the uncached API
// reader: claim events only wake the Run, and a claim the cache has not observed yet, such as one
// created just before the Run, must not end it. A Pending claim is accepted: a WaitForFirstConsumer
// claim binds only once its first pod is scheduled. Access modes are not checked: whether another
// pod holds a ReadWriteOnce or ReadWriteOncePod claim is a scheduling question, not a terminal one.
// Other read errors are returned for a retry. run must not be nil.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) traceSourceUnavailable(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	scheduled bool,
) (string, error) {
	source := dgdrRunTraceClaim(run)
	if source == nil {
		return "", nil
	}
	name := source.ClaimName
	claim := &corev1.PersistentVolumeClaim{}
	err := r.APIReader.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, claim)
	if apierrors.IsNotFound(err) {
		return fmt.Sprintf("trace PersistentVolumeClaim %q does not exist in namespace %q", name, run.Namespace), nil
	}
	if err != nil {
		return "", fmt.Errorf("get trace PersistentVolumeClaim %q in namespace %q: %w", name, run.Namespace, err)
	}

	// Each state below keeps the Job from ever mounting the claim read-only as a filesystem.
	reason := ""
	switch {
	case claim.DeletionTimestamp != nil && !(scheduled && controllerutil.ContainsFinalizer(claim, dgdrRunClaimProtectionFinalizer)):
		reason = "is being deleted"
	case claim.Spec.VolumeMode != nil && *claim.Spec.VolumeMode == corev1.PersistentVolumeBlock:
		reason = "is a raw block volume, not a filesystem"
	case claim.Status.Phase == corev1.ClaimLost:
		reason = "has lost its PersistentVolume"
	default:
		return "", nil
	}
	return fmt.Sprintf("trace PersistentVolumeClaim %q in namespace %q %s", name, run.Namespace, reason), nil
}

// runJobExists reports whether the API server holds the Job this Run controls under name. Creating
// the Job and recording its jobRef are separate writes, so a Job can exist while both the cache and
// status.jobRef miss it. The Job is therefore read through the uncached API reader, and only right
// before the Run would be failed terminally for its trace claim.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) runJobExists(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	name string,
) (bool, error) {
	job := &batchv1.Job{}
	err := r.APIReader.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, job)
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("get Job %q of Run %q: %w", name, run.Name, err)
	}
	return metav1.IsControlledBy(job, run), nil
}

// dgdrRunTraceClaim returns the claim the Run's trace is read from, or nil when it replays no trace.
// run must not be nil.
func dgdrRunTraceClaim(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) *nvidiacomv1beta2.TracePVCSource {
	if run.Spec.Workload == nil || run.Spec.Workload.Trace == nil {
		return nil
	}
	return run.Spec.Workload.Trace.Source.PVC
}

// dgdrRunTraceClaimNames indexes a Run by the claim its trace is read from.
func dgdrRunTraceClaimNames(obj client.Object) []string {
	run, ok := obj.(*nvidiacomv1beta2.DynamoGraphDeploymentRun)
	if !ok {
		return nil
	}
	if source := dgdrRunTraceClaim(run); source != nil {
		return []string{source.ClaimName}
	}
	return nil
}

// runsForTraceClaim maps a claim event to the Runs in the claim's namespace that read their trace
// from it.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) runsForTraceClaim(ctx context.Context, claim client.Object) []reconcile.Request {
	runs := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := r.List(ctx, runs, client.InNamespace(claim.GetNamespace()),
		client.MatchingFields{dgdrRunTraceClaimIndex: claim.GetName()}); err != nil {
		log.FromContext(ctx).Error(err, "Could not list the Runs that read a trace claim", "claim", claim.GetName())
		return nil
	}
	requests := make([]reconcile.Request, 0, len(runs.Items))
	for i := range runs.Items {
		requests = append(requests, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(&runs.Items[i])})
	}
	return requests
}

// dgdrRunJobFinished reports whether job reports a finished pod or a terminal or interim terminal
// condition, so its outcome no longer depends on the trace. The Job runs one pod, because its
// backoffLimit is 0. job must not be nil.
func dgdrRunJobFinished(job *batchv1.Job) bool {
	uncounted := job.Status.UncountedTerminatedPods
	if job.Status.Succeeded > 0 || job.Status.Failed > 0 ||
		(uncounted != nil && len(uncounted.Succeeded)+len(uncounted.Failed) > 0) {
		return true
	}
	for _, conditionType := range []batchv1.JobConditionType{
		batchv1.JobComplete, batchv1.JobFailed, batchv1.JobSuccessCriteriaMet, batchv1.JobFailureTarget,
	} {
		if condition := dgdrRunJobCondition(job, conditionType); condition != nil && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

// jobTraceUnavailable returns why job, which has not finished, can never read the Run's trace, or
// "" when it can or still may.
//
// A sweeper that cannot mount the trace file waits in CreateContainerConfigError until the pod is
// deleted, so that is reported whatever the claim does. The claim itself matters until the sweeper
// has started. Before its pod is scheduled, the pod cannot get a claim that is unusable. A scheduled
// pod still cannot mount a claim that is gone, lost or a block volume, or one being deleted without
// pvc-protection, as when the claim goes while the pod is bound; pvc-protection keeps a claim being
// deleted for the pods bound to it, and the kubelet mounts it. While the cache shows fewer pods than
// the Job counts as active, whether one is scheduled or started is unknown and nothing is reported;
// the pod's event wakes the Run again. The pod cache can lag a binding by a moment, and a claim
// deleted in that moment fails the Run although its pod just got the claim. run and job must not
// be nil.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) jobTraceUnavailable(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
) (string, error) {
	pods, err := r.jobPods(ctx, job)
	if err != nil {
		return "", err
	}

	// A trace file the sweeper cannot mount, whether or not the claim is usable.
	source := dgdrRunTraceClaim(run)
	for i := range pods {
		if message := dgdrRunTraceMountFailure(&pods[i]); message != "" {
			return fmt.Sprintf("trace file %q on PersistentVolumeClaim %q cannot be mounted: %s",
				source.Path, source.ClaimName, message), nil
		}
	}

	// The claim, until the sweeper has started.
	if ptr.Deref(job.Status.Ready, 0) > 0 || int(job.Status.Active) > len(pods) {
		return "", nil
	}
	scheduled := false
	for i := range pods {
		if dgdrRunSweeperStarted(&pods[i]) {
			return "", nil
		}
		scheduled = scheduled || pods[i].Spec.NodeName != ""
	}
	return r.traceSourceUnavailable(ctx, run, scheduled)
}

// jobPods returns the pods job controls, as the cache shows them. job must not be nil.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) jobPods(ctx context.Context, job *batchv1.Job) ([]corev1.Pod, error) {
	list := &corev1.PodList{}
	if err := r.List(ctx, list, client.InNamespace(job.Namespace), client.MatchingLabels{batchv1.JobNameLabel: job.Name}); err != nil {
		return nil, fmt.Errorf("list pods of Job %q: %w", job.Name, err)
	}
	pods := make([]corev1.Pod, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], job) {
			pods = append(pods, list.Items[i])
		}
	}
	return pods, nil
}

// dgdrRunSweeperStarted reports whether pod's sweeper container has been created and started, so it
// mounted the trace. The pod cache drops a running state, so a reported sweeper that is not waiting
// has started. pod must not be nil.
func dgdrRunSweeperStarted(pod *corev1.Pod) bool {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == dgdrRunSweeperContainer &&
			(status.State.Waiting == nil || status.LastTerminationState.Terminated != nil) {
			return true
		}
	}
	return false
}

// dgdrRunTraceMountFailure returns the kubelet's message when pod's sweeper cannot start because
// the trace file cannot be mounted, such as a missing file on a read-only claim or a symlink that
// leaves the claim, or "" otherwise. pod must not be nil.
func dgdrRunTraceMountFailure(pod *corev1.Pod) string {
	for _, status := range pod.Status.ContainerStatuses {
		waiting := status.State.Waiting
		if status.Name == dgdrRunSweeperContainer && waiting != nil && waiting.Reason == "CreateContainerConfigError" &&
			strings.Contains(waiting.Message, fmt.Sprintf("volumeMount %q", dgdrRunTraceVolume)) {
			return waiting.Message
		}
	}
	return ""
}

// deleteUnfinishedJobs deletes the Jobs run controls that have not finished, with their pods. run
// failed because those Jobs could never read its trace. The UID precondition keeps a Job created
// under the same name since the cache showed it. run must not be nil.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) deleteUnfinishedJobs(ctx context.Context, run *nvidiacomv1beta2.DynamoGraphDeploymentRun) error {
	jobs, err := r.ownedJobs(ctx, run)
	if err != nil {
		return err
	}
	for i := range jobs {
		job := &jobs[i]
		if dgdrRunJobFinished(job) {
			continue
		}
		err := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationBackground), client.Preconditions{UID: &job.UID})
		if client.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete Job %q: %w", job.Name, err)
		}
	}
	return nil
}

// dgdrRunJobPod reports whether obj is a pod of a Job named like a Run's AISimulate Job.
func dgdrRunJobPod(obj client.Object) bool {
	owner := metav1.GetControllerOf(obj)
	return owner != nil && owner.Kind == "Job" && owner.APIVersion == batchv1.SchemeGroupVersion.String() &&
		strings.HasSuffix(owner.Name, "-"+dgdrRunResourceComponent)
}

// runForJobPod maps a pod event to the Run that controls the pod's Job.
func (r *DynamoGraphDeploymentRunV1beta2Reconciler) runForJobPod(ctx context.Context, pod client.Object) []reconcile.Request {
	owner := metav1.GetControllerOf(pod)
	if owner == nil {
		return nil
	}
	job := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: pod.GetNamespace(), Name: owner.Name}, job); err != nil {
		if !apierrors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "Could not get the Job of a pod", "pod", pod.GetName(), "job", owner.Name)
		}
		return nil
	}
	run := metav1.GetControllerOf(job)
	if job.UID != owner.UID || run == nil ||
		run.Kind != "DynamoGraphDeploymentRun" || run.APIVersion != nvidiacomv1beta2.GroupVersion.String() {
		return nil
	}
	return []reconcile.Request{{Namespace: job.Namespace, Name: run.Name}}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) uncachedOwnedJobs(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) ([]batchv1.Job, error) {
	if r.APIReader == nil {
		return nil, errors.New("API reader is required to confirm Job absence")
	}
	list := &batchv1.JobList{}
	if err := r.APIReader.List(ctx, list, client.InNamespace(run.Namespace)); err != nil {
		return nil, fmt.Errorf("list Jobs of Run %q from the API: %w", run.Name, err)
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
	execution.DynamoVersion, err = dgdrRunDynamoVersion(execution.PlannerImage, r.DynamoVersion)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	execution.RuntimeImageSource, err = dgdrRunRuntimeImageSource(execution.PlannerImage, execution.DynamoVersion)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: err}
	}
	execution.ImagePullSecrets, err = r.dgdrRunImagePullSecrets(run.Namespace, execution.PlannerImage, execution.PublisherImage)
	if err != nil {
		return dgdrRunExecution{}, err
	}

	input, err := dgdrRunInput(run, execution.RuntimeImageSource, execution.DynamoVersion)
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
	if configMap.Immutable == nil || !*configMap.Immutable {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf(
			"execution ConfigMap %q must be immutable", configMap.Name)}
	}
	execution, err := dgdrRunExecutionFromConfigMap(configMap)
	if err != nil {
		return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf("read ConfigMap %q: %w", configMap.Name, err)}
	}
	if job != nil {
		if err := validateDGDRRunJobExecution(job, run, configMap.Name, execution); err != nil {
			return dgdrRunExecution{}, &dgdrRunPermanentExecutionError{cause: fmt.Errorf(
				"Job %q and ConfigMap %q disagree: %w", job.Name, configMap.Name, err)}
		}
	}
	return execution, nil
}

func dgdrRunInput(
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	runtimeImageSource string,
	dynamoVersion string,
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
	}
	if run.Spec.Hardware != nil && run.Spec.Hardware.NumGPUsPerNode != nil {
		reserved["num_gpus_per_node"] = int64(*run.Spec.Hardware.NumGPUsPerNode)
	}
	for key, value := range reserved {
		if err := unstructured.SetNestedField(input, value, "dgdr_run", key); err != nil {
			return nil, fmt.Errorf("set operator-owned dgdr_run.%s: %w", key, err)
		}
	}

	// The typed trace owns traffic.source.type, format and paths the same way. Restoring them keeps a
	// Run created without Request admission reading only the file on its read-only mount. Such a Run
	// can also replace traffic or traffic.source with a non-object; that parent is rebuilt rather than
	// failing every reconcile.
	if run.Spec.Workload != nil && run.Spec.Workload.Trace != nil {
		traffic, ok := input["traffic"].(map[string]any)
		if !ok {
			traffic = map[string]any{}
			input["traffic"] = traffic
		}
		source, ok := traffic["source"].(map[string]any)
		if !ok {
			source = map[string]any{}
			traffic["source"] = source
		}
		maps.Copy(source, aisimulate.TraceSource(run.Spec.Workload.Trace))
	}
	return input, nil
}

type dgdrRunExecution struct {
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

func dgdrRunRuntimeImageSource(plannerImage, dynamoVersion string) (string, error) {
	ref := strings.TrimSpace(plannerImage)
	nameAndTag, _, hasDigest := strings.Cut(ref, "@")
	lastSlash, lastColon := strings.LastIndex(nameAndTag, "/"), strings.LastIndex(nameAndTag, ":")
	if lastColon > lastSlash && lastColon < len(nameAndTag)-1 {
		return nameAndTag, nil
	}
	if hasDigest {
		return nameAndTag + ":" + dynamoVersion, nil
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

	// Decode the immutable credentials and effective input.
	var err error
	if err := json.Unmarshal([]byte(configMap.Data[dgdrRunImagePullSecretsKey]), &execution.ImagePullSecrets); err != nil {
		return dgdrRunExecution{}, fmt.Errorf("decode data key %q: %w", dgdrRunImagePullSecretsKey, err)
	}
	execution.Input, err = dgdrRunStatusInput(configMap.Data[dgdrRunInputKey])
	if err != nil {
		return dgdrRunExecution{}, fmt.Errorf("decode data key %q: %w", dgdrRunInputKey, err)
	}
	return execution, nil
}

func validateDGDRRunJobExecution(
	job *batchv1.Job,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	inputConfigMapName string,
	execution dgdrRunExecution,
) error {
	if job.Annotations[dgdrRunSnapshotPathAnnotation] != execution.SnapshotPath {
		return errors.New("snapshot paths differ")
	}
	if err := validateDGDRRunJobSpec(&job.Spec); err != nil {
		return err
	}
	pod := &job.Spec.Template.Spec
	if pod.RestartPolicy != corev1.RestartPolicyNever {
		return errors.New("pod restart policy differs")
	}
	if pod.HostPID || pod.HostIPC || pod.HostNetwork ||
		(pod.ShareProcessNamespace != nil && *pod.ShareProcessNamespace) {
		return errors.New("pod namespace isolation differs")
	}
	if len(pod.InitContainers) != 0 {
		return errors.New("init containers are not allowed")
	}
	if err := validateDGDRRunPodSecurityContext(pod.SecurityContext, run); err != nil {
		return err
	}
	if pod.ServiceAccountName != inputConfigMapName || pod.AutomountServiceAccountToken == nil || *pod.AutomountServiceAccountToken {
		return errors.New("publisher service-account identities differ")
	}
	if len(pod.Containers) != 2 {
		return fmt.Errorf("container count differs: got %d, want 2", len(pod.Containers))
	}

	// Match both container identities, their immutable bindings and all credentials.
	planner, found := dgdrRunContainer(pod.Containers, dgdrRunSweeperContainer)
	if !found {
		return errors.New("planner image identities differ")
	}
	if err := validateDGDRRunPlannerContainer(&planner, run, execution); err != nil {
		return err
	}
	publisher, found := dgdrRunContainer(pod.Containers, dgdrRunPublisherContainer)
	if !found {
		return errors.New("publisher image identities differ")
	}
	if err := validateDGDRRunPublisherContainer(&publisher, run, execution); err != nil {
		return err
	}
	if err := validateDGDRRunVolumes(pod.Volumes, run, inputConfigMapName); err != nil {
		return err
	}
	if !apiequality.Semantic.DeepEqual(pod.ImagePullSecrets, execution.ImagePullSecrets) {
		return errors.New("image pull secrets differ")
	}
	return nil
}

func validateDGDRRunPlannerContainer(
	planner *corev1.Container,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	execution dgdrRunExecution,
) error {
	if planner.Image != execution.PlannerImage || planner.ImagePullPolicy != execution.PlannerPullPolicy {
		return errors.New("planner image identities differ")
	}
	if err := validateDGDRRunContainerSecurityContext(planner.SecurityContext, ptr.To[int64](1000)); err != nil {
		return fmt.Errorf("planner security context differs: %w", err)
	}
	if !apiequality.Semantic.DeepEqual(planner.Command, dgdrRunSweeperCommand()) ||
		!apiequality.Semantic.DeepEqual(planner.Args, dgdrRunSweeperArgs(execution.SnapshotPath)) {
		return errors.New("planner command or arguments differ")
	}
	if len(planner.Env) != 0 || len(planner.EnvFrom) != 0 {
		return errors.New("planner environment differs")
	}
	if planner.Lifecycle != nil {
		return errors.New("planner lifecycle differs")
	}
	if !apiequality.Semantic.DeepEqual(planner.VolumeMounts, dgdrRunSweeperVolumeMounts(run, execution.SnapshotPath)) {
		return errors.New("planner volume mounts differ")
	}
	return nil
}

func validateDGDRRunPublisherContainer(
	publisher *corev1.Container,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	execution dgdrRunExecution,
) error {
	if publisher.Image != execution.PublisherImage || publisher.ImagePullPolicy != execution.PublisherPullPolicy {
		return errors.New("publisher image identities differ")
	}
	if err := validateDGDRRunContainerSecurityContext(publisher.SecurityContext, nil); err != nil {
		return fmt.Errorf("publisher security context differs: %w", err)
	}
	if !apiequality.Semantic.DeepEqual(publisher.Command, dgdrRunPublisherCommand()) ||
		!apiequality.Semantic.DeepEqual(publisher.Args, dgdrRunPublisherArgs(run, execution.SnapshotPath)) {
		return errors.New("publisher command, arguments or identity differ")
	}
	if err := validateDGDRRunPublisherEnv(publisher.Env); err != nil {
		return err
	}
	if len(publisher.EnvFrom) != 0 {
		return errors.New("publisher environment differs")
	}
	if publisher.Lifecycle != nil {
		return errors.New("publisher lifecycle differs")
	}
	if !apiequality.Semantic.DeepEqual(publisher.VolumeMounts, dgdrRunPublisherVolumeMounts(execution.SnapshotPath)) {
		return errors.New("publisher volume mounts differ")
	}
	return nil
}

func validateDGDRRunJobSpec(spec *batchv1.JobSpec) error {
	if spec.BackoffLimit == nil || *spec.BackoffLimit != 0 {
		return errors.New("Job backoff limit differs")
	}
	if (spec.Parallelism != nil && *spec.Parallelism != 1) ||
		(spec.Completions != nil && *spec.Completions != 1) {
		return errors.New("Job cardinality differs")
	}
	if spec.CompletionMode != nil && *spec.CompletionMode != batchv1.NonIndexedCompletion {
		return errors.New("Job completion mode differs")
	}
	if (spec.Suspend != nil && *spec.Suspend) || (spec.ManualSelector != nil && *spec.ManualSelector) {
		return errors.New("Job execution policy differs")
	}
	if spec.ActiveDeadlineSeconds != nil || spec.PodFailurePolicy != nil || spec.SuccessPolicy != nil ||
		spec.BackoffLimitPerIndex != nil || spec.MaxFailedIndexes != nil ||
		spec.TTLSecondsAfterFinished != nil ||
		(spec.PodReplacementPolicy != nil && *spec.PodReplacementPolicy != batchv1.TerminatingOrFailed) ||
		spec.ManagedBy != nil {
		return errors.New("Job execution policy differs")
	}
	return nil
}

func validateDGDRRunPodSecurityContext(security *corev1.PodSecurityContext, run *nvidiacomv1beta2.DynamoGraphDeploymentRun) error {
	if security == nil {
		return errors.New("pod fsGroup differs")
	}

	// The pod requests no fsGroup. Earlier operators set fsGroup 65532 on every Run Job; such a Job
	// mounts no trace claim, so a Run without a trace keeps accepting it and an operator upgrade does
	// not fail its in-flight sweep.
	if security.FSGroup != nil && (dgdrRunTraceClaim(run) != nil || *security.FSGroup != dgdrRunLegacyFSGroup) {
		return errors.New("pod fsGroup differs")
	}
	if !apiequality.Semantic.DeepEqual(security.SupplementalGroups, dgdrRunPodSecurityContext(run).SupplementalGroups) {
		return errors.New("pod supplemental groups differ")
	}
	if security.RunAsNonRoot == nil || !*security.RunAsNonRoot {
		return errors.New("pod runAsNonRoot differs")
	}
	if security.SeccompProfile == nil || security.SeccompProfile.Type != corev1.SeccompProfileTypeRuntimeDefault {
		return errors.New("pod seccomp profile differs")
	}
	return nil
}

func validateDGDRRunContainerSecurityContext(security *corev1.SecurityContext, runAsUser *int64) error {
	if security == nil || security.AllowPrivilegeEscalation == nil || *security.AllowPrivilegeEscalation {
		return errors.New("allowPrivilegeEscalation must be false")
	}
	if security.Privileged != nil && *security.Privileged {
		return errors.New("privileged must be false")
	}
	if security.RunAsNonRoot != nil && !*security.RunAsNonRoot {
		return errors.New("runAsNonRoot cannot be disabled")
	}
	if security.SeccompProfile != nil && security.SeccompProfile.Type != corev1.SeccompProfileTypeRuntimeDefault {
		return errors.New("seccomp profile cannot weaken the pod default")
	}
	if security.ProcMount != nil && *security.ProcMount != corev1.DefaultProcMount {
		return errors.New("procMount cannot weaken the container default")
	}
	if security.Capabilities == nil || len(security.Capabilities.Add) != 0 ||
		!apiequality.Semantic.DeepEqual(security.Capabilities.Drop, []corev1.Capability{"ALL"}) {
		return errors.New("all capabilities must be dropped")
	}
	if !apiequality.Semantic.DeepEqual(security.RunAsUser, runAsUser) {
		return errors.New("runAsUser differs")
	}
	return nil
}

func validateDGDRRunPublisherEnv(env []corev1.EnvVar) error {
	if len(env) != 2 {
		return errors.New("publisher pod identity environment differs")
	}
	for name, fieldPath := range map[string]string{
		"POD_NAMESPACE": "metadata.namespace",
		"POD_NAME":      "metadata.name",
	} {
		found := false
		for i := range env {
			if env[i].Name == name && env[i].Value == "" && env[i].ValueFrom != nil &&
				env[i].ValueFrom.FieldRef != nil && env[i].ValueFrom.FieldRef.FieldPath == fieldPath {
				found = true
				break
			}
		}
		if !found {
			return errors.New("publisher pod identity environment differs")
		}
	}
	return nil
}

func validateDGDRRunVolumes(
	volumes []corev1.Volume,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	inputConfigMapName string,
) error {
	if len(volumes) != len(dgdrRunVolumes(run, inputConfigMapName)) {
		return errors.New("input, snapshot or publisher credential volumes differ")
	}
	input, found := dgdrRunVolume(volumes, dgdrRunInputVolume)
	if !found || input.ConfigMap == nil || input.ConfigMap.Name != inputConfigMapName ||
		len(input.ConfigMap.Items) != 0 || input.ConfigMap.Optional != nil || !dgdrRunDefaultMode(input.ConfigMap.DefaultMode) {
		return errors.New("input, snapshot or publisher credential volumes differ")
	}
	snapshot, found := dgdrRunVolume(volumes, dgdrRunSnapshotVolume)
	if !found || snapshot.EmptyDir == nil || !apiequality.Semantic.DeepEqual(snapshot.EmptyDir, &corev1.EmptyDirVolumeSource{}) {
		return errors.New("input, snapshot or publisher credential volumes differ")
	}
	publisher, found := dgdrRunVolume(volumes, dgdrRunServiceAccountVolume)
	if !found || publisher.Projected == nil ||
		!dgdrRunDefaultMode(publisher.Projected.DefaultMode) ||
		!apiequality.Semantic.DeepEqual(publisher.Projected.Sources, dgdrRunPublisherKubeAPIVolume().Projected.Sources) {
		return errors.New("input, snapshot or publisher credential volumes differ")
	}

	// A Run that replays a trace mounts exactly its claim, read-only.
	if source := dgdrRunTraceClaim(run); source != nil {
		trace, found := dgdrRunVolume(volumes, dgdrRunTraceVolume)
		if !found || !apiequality.Semantic.DeepEqual(trace, dgdrRunTraceClaimVolume(source)) {
			return errors.New("trace volume differs")
		}
	}
	return nil
}

func dgdrRunDefaultMode(mode *int32) bool {
	return mode == nil || *mode == int32(0o644)
}

func dgdrRunVolume(volumes []corev1.Volume, name string) (corev1.Volume, bool) {
	var found *corev1.Volume
	for i := range volumes {
		if volumes[i].Name != name {
			continue
		}
		if found != nil {
			return corev1.Volume{}, false
		}
		found = &volumes[i]
	}
	if found == nil {
		return corev1.Volume{}, false
	}
	return *found, true
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
			dgdrRunSnapshotPathAnnotation: execution.SnapshotPath,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: ptr.To[int32](0),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName:           name,
					AutomountServiceAccountToken: new(false),
					RestartPolicy:                corev1.RestartPolicyNever,
					SecurityContext:              dgdrRunPodSecurityContext(run),
					ImagePullSecrets:             append([]corev1.LocalObjectReference(nil), execution.ImagePullSecrets...),
					Containers: []corev1.Container{
						{
							Name:            dgdrRunSweeperContainer,
							Image:           execution.PlannerImage,
							ImagePullPolicy: execution.PlannerPullPolicy,
							Command:         dgdrRunSweeperCommand(),
							Args:            dgdrRunSweeperArgs(execution.SnapshotPath),
							SecurityContext: sweeperSecurityContext,
							VolumeMounts:    dgdrRunSweeperVolumeMounts(run, execution.SnapshotPath),
						},
						{
							Name:            dgdrRunPublisherContainer,
							Image:           execution.PublisherImage,
							ImagePullPolicy: execution.PublisherPullPolicy,
							Command:         dgdrRunPublisherCommand(),
							Args:            dgdrRunPublisherArgs(run, execution.SnapshotPath),
							Env:             dgdrRunPublisherEnv(),
							SecurityContext: dgdrRunContainerSecurityContext(),
							VolumeMounts:    dgdrRunPublisherVolumeMounts(execution.SnapshotPath),
						},
					},
					Volumes: dgdrRunVolumes(run, name),
				},
			},
		},
	}
	return job
}

func dgdrRunSweeperCommand() []string {
	return []string{"aisimulate"}
}

func dgdrRunSweeperArgs(snapshotPath string) []string {
	return []string{
		"recommend", "--stack", "dynamo",
		"--config", dgdrRunInputPath,
		"--output", "dgdr_run",
		"--output-dir", snapshotPath,
		"--overwrite",
	}
}

func dgdrRunPublisherCommand() []string {
	return []string{dgdrRunPublisherBinary}
}

func dgdrRunPublisherArgs(run *nvidiacomv1beta2.DynamoGraphDeploymentRun, snapshotPath string) []string {
	return []string{
		"--snapshot-dir", snapshotPath,
		"--run-name", run.Name,
		"--run-uid", string(run.UID),
		"--sweeper-container", dgdrRunSweeperContainer,
	}
}

func dgdrRunPublisherEnv() []corev1.EnvVar {
	return []corev1.EnvVar{
		{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
		{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
}

// dgdrRunPodSecurityContext returns the pod security context of run's Job. The pod sets no
// fsGroup, for which kubelet and CSI drivers may chown and chmod a volume recursively; a Run that
// replays a trace gets the trace group as a supplemental group instead, which changes no volume.
// run must not be nil.
func dgdrRunPodSecurityContext(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) *corev1.PodSecurityContext {
	security := &corev1.PodSecurityContext{
		RunAsNonRoot: new(true),
		SeccompProfile: &corev1.SeccompProfile{
			Type: corev1.SeccompProfileTypeRuntimeDefault,
		},
	}
	if dgdrRunTraceClaim(run) != nil {
		security.SupplementalGroups = []int64{dgdrRunTraceGroup}
	}
	return security
}

// dgdrRunSweeperVolumeMounts returns the sweeper's mounts. Only AISimulate reads the trace: it
// mounts the one trace file read-only through a subPath, which the kubelet resolves inside the
// claim, refusing a symlink that resolves outside it. run must not be nil.
func dgdrRunSweeperVolumeMounts(run *nvidiacomv1beta2.DynamoGraphDeploymentRun, snapshotPath string) []corev1.VolumeMount {
	mounts := []corev1.VolumeMount{
		{Name: dgdrRunInputVolume, MountPath: dgdrRunInputMountPath, ReadOnly: true},
		{Name: dgdrRunSnapshotVolume, MountPath: snapshotPath},
	}
	if source := dgdrRunTraceClaim(run); source != nil {
		mounts = append(mounts, corev1.VolumeMount{
			Name: dgdrRunTraceVolume, MountPath: aisimulate.TracePath(source.Path), SubPath: source.Path, ReadOnly: true,
		})
	}
	return mounts
}

func dgdrRunPublisherVolumeMounts(snapshotPath string) []corev1.VolumeMount {
	return []corev1.VolumeMount{
		{Name: dgdrRunSnapshotVolume, MountPath: snapshotPath},
		{Name: dgdrRunServiceAccountVolume, MountPath: dgdrRunServiceAccountMountPath, ReadOnly: true},
	}
}

// dgdrRunVolumes returns the volumes of run's Job, with the trace claim mounted read-only when run
// replays a trace. run must not be nil.
func dgdrRunVolumes(run *nvidiacomv1beta2.DynamoGraphDeploymentRun, inputConfigMapName string) []corev1.Volume {
	volumes := []corev1.Volume{
		{Name: dgdrRunInputVolume, ConfigMap: &corev1.ConfigMapVolumeSource{Name: inputConfigMapName}},
		{Name: dgdrRunSnapshotVolume, EmptyDir: &corev1.EmptyDirVolumeSource{}},
		dgdrRunPublisherKubeAPIVolume(),
	}
	if source := dgdrRunTraceClaim(run); source != nil {
		volumes = append(volumes, dgdrRunTraceClaimVolume(source))
	}
	return volumes
}

// dgdrRunTraceClaimVolume returns the read-only volume of the claim source names. source must not
// be nil.
func dgdrRunTraceClaimVolume(source *nvidiacomv1beta2.TracePVCSource) corev1.Volume {
	return corev1.Volume{
		Name: dgdrRunTraceVolume,
		PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
			ClaimName: source.ClaimName,
			ReadOnly:  true,
		},
	}
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

	// A stale Run must not overwrite a newer status, such as a trace failure recorded while the Job
	// was still pending, so the write carries the resourceVersion the cache showed.
	if apiequality.Semantic.DeepEqual(original.Status, run.Status) {
		return nil
	}
	if err := r.Status().Patch(ctx, run, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
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
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(), &nvidiacomv1beta2.DynamoGraphDeploymentRun{}, dgdrRunTraceClaimIndex, dgdrRunTraceClaimNames,
	); err != nil {
		return fmt.Errorf("index Runs by trace claim: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrun-v1beta2").
		For(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		Watches(&corev1.PersistentVolumeClaim{}, handler.EnqueueRequestsFromMapFunc(r.runsForTraceClaim)).
		Watches(&corev1.Pod{}, handler.EnqueueRequestsFromMapFunc(r.runForJobPod),
			builder.WithPredicates(predicate.NewPredicateFuncs(dgdrRunJobPod))).
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
