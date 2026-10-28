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
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// dgdrV1beta2GenerationAnnotation records the Request generation a Run was created for. It is
	// the identity of a Run within its Request. metadata.name is only the convention used when a Run is
	// created, so the controller never reconstructs or validates it for an existing Run.
	dgdrV1beta2GenerationAnnotation = "nvidia.com/dgdr-request-generation"

	// dgdrV1beta2CancelRequestedByGenerationAnnotation records the newer Request generation that
	// superseded a non-terminal Run. The Request controller owns this intent; the Run controller
	// stops the execution and publishes the terminal outcome without deleting the Run.
	dgdrV1beta2CancelRequestedByGenerationAnnotation = "nvidia.com/dgdr-cancel-requested-by-generation"

	// dgdrRunControllerUIDIndex indexes Runs by the UID of their controller OwnerReference. The
	// OwnerReference is the only definition of which Runs belong to a Request.
	dgdrRunControllerUIDIndex = "dynamo.nvidia.com/dgdr-run-controller-uid"

	// dgdrV1beta2RetainedRuns is how many terminal Runs a Request keeps. Older ones are deleted.
	dgdrV1beta2RetainedRuns = 5

	dgdrReasonRunCreated        = "RunCreated"
	dgdrReasonRunCreating       = "RunCreating"
	dgdrReasonCancelingStaleRun = "CancelingSupersededRun"
	dgdrReasonInvalidSpec       = "InvalidSpec"
	dgdrReasonRunNameConflict   = "RunNameConflict"

	// dgdrV1beta2MaxRunNameLength is the longest object name Kubernetes accepts.
	dgdrV1beta2MaxRunNameLength = 253

	// dgdrV1beta2ConflictRetry is how long to wait before looking again at a Run name held by another owner.
	dgdrV1beta2ConflictRetry = 5 * time.Second
)

// DynamoGraphDeploymentRequestV1beta2Reconciler turns an accepted native v1beta2 Request into an
// immutable Run. It never reads the cluster for hardware: admission has already supplied it, and a
// Request that still lacks it is reported instead of searched for. The Run controller executes the Run.
type DynamoGraphDeploymentRequestV1beta2Reconciler struct {
	client.Client
	APIReader     client.Reader
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns,verbs=get;list;watch;create;patch;delete

// Reconcile is level based: once no older Run is active it creates the Run of the latest generation
// only, so generations edited in between are coalesced and never get a Run of their own. It records
// the active and retained Runs in the Request status before it deletes any retired Run.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
	if err := r.Get(ctx, req.NamespacedName, request); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	// A native v1beta1 Request is a read-only carrier here; the v1beta1 reconciler owns it.
	if request.DeletionTimestamp != nil || request.Spec.V1Beta1 != nil {
		return ctrl.Result{}, nil
	}
	original := request.DeepCopy()
	result := ctrl.Result{}

	runs, err := r.ownedRuns(ctx, request)
	if err != nil {
		return ctrl.Result{}, err
	}

	wantName := dgdrV1beta2RunName(request)
	entries, err := dgdrV1beta2RunEntries(runs)
	if err != nil {
		return ctrl.Result{}, err
	}
	// The generation annotation identifies the Run of the current generation, whatever its name.
	var current *nvidiacomv1beta2.DynamoGraphDeploymentRun
	for i := range entries {
		if entries[i].generation == request.Generation {
			current = entries[i].run.DeepCopy()
		}
	}
	active, _ := dgdrV1beta2SplitRuns(entries)

	accepted := current != nil
	runName := wantName
	if current != nil {
		runName = current.Name
	}
	condition := dgdrV1beta2Condition(request, metav1.ConditionTrue, dgdrReasonRunCreated,
		fmt.Sprintf("Run %q exists for generation %d", runName, request.Generation))
	if current == nil {
		if active != nil {
			if active.DeletionTimestamp.IsZero() {
				if err := r.requestRunCancellation(ctx, active, request.Generation); err != nil {
					return ctrl.Result{}, err
				}
			}
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonCancelingStaleRun,
				fmt.Sprintf("Waiting for superseded Run %q to stop before generation %d starts", active.Name, request.Generation))
			request.Status.ActiveRunRef = &corev1.LocalObjectReference{Name: active.Name}
			meta.SetStatusCondition(&request.Status.Conditions, condition)
			if err := r.patchStatus(ctx, request, original); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, nil
		} else if run, buildErr := r.buildRun(request); buildErr != nil {
			logger.Info("Request cannot be turned into a Run", "reason", buildErr.Error())
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonInvalidSpec, buildErr.Error())
		} else {
			marker := meta.FindStatusCondition(request.Status.Conditions, nvidiacomv1beta2.DGDRRequestConditionAccepted)
			markerMatches := request.Status.ActiveRunRef != nil && request.Status.ActiveRunRef.Name == wantName &&
				marker != nil && marker.ObservedGeneration == request.Generation

			switch {
			case markerMatches && (marker.Reason == dgdrReasonRunCreating || marker.Reason == dgdrReasonRunCreated):
				return r.createReservedRun(ctx, request, run, original)
			case request.Status.ActiveRunRef != nil && marker != nil && marker.Reason == dgdrReasonRunCreating:
				// The Request informer has observed a newer spec before the Run informer observed
				// the reserved create. Resolve its unknown outcome directly: an owned Run waits for
				// the cache before cancellation, while authoritative absence releases the stale
				// reservation. This is the only uncached read in the Request transaction.
				pending, err := r.reservedRunPending(ctx, request, marker.ObservedGeneration)
				if err != nil {
					return ctrl.Result{}, err
				}
				if pending {
					return ctrl.Result{RequeueAfter: dgdrV1beta2ConflictRetry}, nil
				}
				fallthrough
			default:
				condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonRunCreating,
					fmt.Sprintf("Creating Run %q for generation %d", wantName, request.Generation))
				request.Status.ActiveRunRef = &corev1.LocalObjectReference{Name: wantName}
				meta.SetStatusCondition(&request.Status.Conditions, condition)
				if err := r.patchStatus(ctx, request, original); err != nil {
					return ctrl.Result{}, err
				}
				return r.createReservedRun(ctx, request, run, request.DeepCopy())
			}
		}
	}

	entries, err = dgdrV1beta2RunEntries(runs)
	if err != nil {
		return ctrl.Result{}, err
	}
	active, terminal := dgdrV1beta2SplitRuns(entries)
	keep := min(len(terminal), dgdrV1beta2RetainedRuns)
	retired := terminal[keep:]

	if accepted {
		request.Status.ObservedGeneration = request.Generation
	}
	request.Status.ActiveRunRef = nil
	if active != nil {
		request.Status.ActiveRunRef = &corev1.LocalObjectReference{Name: active.Name}
	}
	request.Status.RecentRunRefs = nil
	for i := range keep {
		request.Status.RecentRunRefs = append(request.Status.RecentRunRefs, corev1.LocalObjectReference{Name: terminal[i].Name})
	}
	meta.SetStatusCondition(&request.Status.Conditions, condition)

	if !apiequality.Semantic.DeepEqual(original.Status, request.Status) {
		if err := r.patchStatus(ctx, request, original); err != nil {
			return ctrl.Result{}, err
		}
	}

	// The status no longer references these Runs, so they may go. If a deletion fails, an unreferenced
	// child stays for the next pass; the status never points at a Run that was already deleted.
	for i := range retired {
		if err := r.Delete(ctx, &retired[i]); err != nil && !apierrors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("delete retired Run %q: %w", retired[i].Name, err)
		}
	}
	return result, nil
}

func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) requestRunCancellation(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	supersedingGeneration int64,
) error {
	if existing := run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation]; existing != "" {
		generation, err := strconv.ParseInt(existing, 10, 64)
		if err == nil && generation >= supersedingGeneration {
			return nil
		}
	}
	want := strconv.FormatInt(supersedingGeneration, 10)
	original := run.DeepCopy()
	if run.Annotations == nil {
		run.Annotations = map[string]string{}
	}
	run.Annotations[dgdrV1beta2CancelRequestedByGenerationAnnotation] = want
	if err := r.Patch(ctx, run, client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})); err != nil {
		return fmt.Errorf("request cancellation of superseded Run %q: %w", run.Name, err)
	}
	return nil
}

func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) patchStatus(
	ctx context.Context,
	request, original *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) error {
	patch := client.MergeFromWithOptions(original, client.MergeFromWithOptimisticLock{})
	if err := r.Status().Patch(ctx, request, patch); err != nil {
		return fmt.Errorf("update Request status: %w", err)
	}
	return nil
}

func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) reservedRunPending(
	ctx context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	generation int64,
) (bool, error) {
	if r.APIReader == nil {
		return false, errors.New("resolve reserved Run create: APIReader is not configured")
	}
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	key := client.ObjectKey{Namespace: request.Namespace, Name: request.Status.ActiveRunRef.Name}
	if err := r.APIReader.Get(ctx, key, run); apierrors.IsNotFound(err) {
		return false, nil
	} else if err != nil {
		return false, fmt.Errorf("resolve reserved Run %q: %w", key.Name, err)
	}
	return metav1.IsControlledBy(run, request) && dgdrV1beta2RunIsGeneration(run, generation), nil
}

func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) createReservedRun(
	ctx context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	original *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (ctrl.Result, error) {
	err := r.Create(ctx, run)
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return ctrl.Result{}, fmt.Errorf("create Run %q: %w", run.Name, err)
	}

	// A successful create reaches the API server before its watch event necessarily reaches this
	// controller. The fixed Run name makes retry safe; only publish RunCreated once the cached Get
	// proves that this informer has observed the owned Run.
	existing := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := r.Get(ctx, client.ObjectKeyFromObject(run), existing); apierrors.IsNotFound(err) {
		return ctrl.Result{RequeueAfter: dgdrV1beta2ConflictRetry}, nil
	} else if err != nil {
		return ctrl.Result{}, fmt.Errorf("get existing Run %q: %w", run.Name, err)
	}
	if metav1.IsControlledBy(existing, request) && dgdrV1beta2RunIsGeneration(existing, request.Generation) {
		return r.markRunCreated(ctx, request, existing.Name, original)
	}

	condition := dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonRunNameConflict,
		fmt.Sprintf("Run %q exists and is not the Run of generation %d of this Request; trying again once it is gone", run.Name, request.Generation))
	request.Status.ActiveRunRef = nil
	meta.SetStatusCondition(&request.Status.Conditions, condition)
	if err := r.patchStatus(ctx, request, original); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: dgdrV1beta2ConflictRetry}, nil
}

func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) markRunCreated(
	ctx context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	runName string,
	original *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (ctrl.Result, error) {
	request.Status.ObservedGeneration = request.Generation
	request.Status.ActiveRunRef = &corev1.LocalObjectReference{Name: runName}
	condition := dgdrV1beta2Condition(request, metav1.ConditionTrue, dgdrReasonRunCreated,
		fmt.Sprintf("Run %q exists for generation %d", runName, request.Generation))
	meta.SetStatusCondition(&request.Status.Conditions, condition)
	if err := r.patchStatus(ctx, request, original); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// buildRun freezes the accepted Request input into an immutable Run spec. The Run reconciler lowers
// this snapshot once, records the effective public recommendation input in status.aisimulate.input,
// and passes that same document to the Job.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) buildRun(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRun, error) {
	frozen := request.Spec.NativeSearchInput.DeepCopy()
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		Name:        dgdrV1beta2RunName(request),
		Namespace:   request.Namespace,
		Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: strconv.FormatInt(request.Generation, 10)},
		Spec:        nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: *frozen},
	}
	if err := controllerutil.SetControllerReference(request, run, r.Scheme()); err != nil {
		return nil, fmt.Errorf("set Run owner: %w", err)
	}
	return run, nil
}

// ownedRuns returns the Runs this Request controls. The controller OwnerReference is the only
// source of ownership: the cache index selects candidates and IsControlledBy is the exact check.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) ownedRuns(
	ctx context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) ([]nvidiacomv1beta2.DynamoGraphDeploymentRun, error) {
	list := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := r.List(ctx, list, client.InNamespace(request.Namespace),
		client.MatchingFields{dgdrRunControllerUIDIndex: string(request.UID)}); err != nil {
		return nil, fmt.Errorf("list Runs: %w", err)
	}
	owned := make([]nvidiacomv1beta2.DynamoGraphDeploymentRun, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], request) {
			owned = append(owned, list.Items[i])
		}
	}
	return owned, nil
}

// SetupWithManager registers the controller. It watches the Runs it owns, so a finished Run wakes
// the Request that was waiting for it.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.APIReader == nil {
		r.APIReader = mgr.GetAPIReader()
	}
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&nvidiacomv1beta2.DynamoGraphDeploymentRun{},
		dgdrRunControllerUIDIndex,
		dgdrRunControllerUIDs,
	); err != nil {
		return fmt.Errorf("index Runs by controller UID: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrequest-v1beta2").
		For(&nvidiacomv1beta2.DynamoGraphDeploymentRequest{}).
		Owns(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}

// dgdrRunControllerUIDs is the index function behind dgdrRunControllerUIDIndex.
func dgdrRunControllerUIDs(obj client.Object) []string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil {
		return nil
	}
	return []string{string(owner.UID)}
}

// dgdrV1beta2RunName names the Run of one Request generation. A metadata-only update keeps the
// generation, so it never creates a second Run.
func dgdrV1beta2RunName(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) string {
	return dgdrV1beta2RunNameFor(request.Name, request.Generation)
}

// dgdrV1beta2RunNameFor is <request-name>-<generation> whenever that fits in an object name. A longer
// name keeps the generation suffix and a stable hash of the full Request name, and truncates only the
// Request-name part, so every valid Request name has a valid Run name and two long names that share a
// prefix still get different Run names.
func dgdrV1beta2RunNameFor(requestName string, generation int64) string {
	suffix := "-" + strconv.FormatInt(generation, 10)
	if len(requestName)+len(suffix) <= dgdrV1beta2MaxRunNameLength {
		return requestName + suffix
	}
	sum := sha256.Sum256([]byte(requestName))
	hash := hex.EncodeToString(sum[:6])
	maxPrefix := dgdrV1beta2MaxRunNameLength - len(suffix) - 1 - len(hash)
	prefix := strings.TrimRight(requestName[:maxPrefix], "-.")
	return prefix + "-" + hash + suffix
}

func dgdrV1beta2Condition(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	status metav1.ConditionStatus,
	reason, message string,
) metav1.Condition {
	return metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRequestConditionAccepted,
		Status:             status,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: request.Generation,
	}
}

// dgdrV1beta2IsTerminal reports whether the Run has published its Completed condition. The conditions
// are authoritative; the phase only describes how a completed Run ended.
func dgdrV1beta2IsTerminal(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) bool {
	return meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted)
}

// dgdrV1beta2RunGeneration reads the Request generation a Run was created for from its generation
// annotation. The annotation, not metadata.name, identifies a Run within its Request: the name is only
// the convention used when a Run is created, so a later change of the naming scheme never orphans a
// Run that already exists. A missing or invalid annotation is an error, never generation zero.
func dgdrV1beta2RunGeneration(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) (int64, error) {
	value, ok := run.Annotations[dgdrV1beta2GenerationAnnotation]
	if !ok {
		return 0, fmt.Errorf("run %q has no Request generation annotation", run.Name)
	}
	generation, err := strconv.ParseInt(value, 10, 64)
	if err != nil || generation <= 0 {
		return 0, fmt.Errorf("run %q has invalid Request generation %q", run.Name, value)
	}
	return generation, nil
}

// dgdrV1beta2RunIsGeneration reports whether the Run's annotation names the given generation.
func dgdrV1beta2RunIsGeneration(run *nvidiacomv1beta2.DynamoGraphDeploymentRun, generation int64) bool {
	got, err := dgdrV1beta2RunGeneration(run)
	return err == nil && got == generation
}

type dgdrV1beta2RunEntry struct {
	run        nvidiacomv1beta2.DynamoGraphDeploymentRun
	generation int64
}

// dgdrV1beta2Newer orders Runs newest first: by Request generation, then creation time, then name.
func dgdrV1beta2Newer(a, b *dgdrV1beta2RunEntry) bool {
	if a.generation != b.generation {
		return a.generation > b.generation
	}
	if !a.run.CreationTimestamp.Equal(&b.run.CreationTimestamp) {
		return b.run.CreationTimestamp.Before(&a.run.CreationTimestamp)
	}
	return a.run.Name > b.run.Name
}

// dgdrV1beta2RunEntries pairs every owned Run with the Request generation its annotation names.
func dgdrV1beta2RunEntries(runs []nvidiacomv1beta2.DynamoGraphDeploymentRun) ([]dgdrV1beta2RunEntry, error) {
	entries := make([]dgdrV1beta2RunEntry, 0, len(runs))
	for i := range runs {
		generation, err := dgdrV1beta2RunGeneration(&runs[i])
		if err != nil {
			return nil, err
		}
		entries = append(entries, dgdrV1beta2RunEntry{run: runs[i], generation: generation})
	}
	return entries, nil
}

// dgdrV1beta2SplitRuns returns the newest non-terminal Run, if any, and the terminal Runs newest first.
func dgdrV1beta2SplitRuns(
	entries []dgdrV1beta2RunEntry,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRun, []nvidiacomv1beta2.DynamoGraphDeploymentRun) {
	sorted := append([]dgdrV1beta2RunEntry(nil), entries...)
	sort.SliceStable(sorted, func(i, j int) bool { return dgdrV1beta2Newer(&sorted[i], &sorted[j]) })
	var active *nvidiacomv1beta2.DynamoGraphDeploymentRun
	var terminal []nvidiacomv1beta2.DynamoGraphDeploymentRun
	for i := range sorted {
		switch {
		case !dgdrV1beta2IsTerminal(&sorted[i].run):
			if active == nil {
				active = &sorted[i].run
			}
		default:
			terminal = append(terminal, sorted[i].run)
		}
	}
	return active, terminal
}

// requireDGDRV1beta2API fails when the cluster does not serve the complete v1beta2 execution API.
// Without them the controller's caches never sync and the manager would hang at startup.
func requireDGDRV1beta2API(mgr ctrl.Manager) error {
	return requireDGDRV1beta2RESTMappings(mgr.GetRESTMapper())
}

func requireDGDRV1beta2RESTMappings(mapper meta.RESTMapper) error {
	group, version := nvidiacomv1beta2.GroupVersion.Group, nvidiacomv1beta2.GroupVersion.Version
	for _, kind := range []string{
		"DynamoGraphDeploymentRequest",
		"DynamoGraphDeploymentRun",
		"DynamoGraphDeploymentCandidate",
	} {
		_, err := mapper.RESTMapping(schema.GroupKind{Group: group, Kind: kind}, version)
		if meta.IsNoMatchError(err) {
			return fmt.Errorf("--dgdr-v1beta2-enabled is set but the cluster does not serve %s/%s %s; "+
				"install the v1beta2 CRDs (dgdrV1Beta2.enabled)", group, version, kind)
		}
		if err != nil {
			return fmt.Errorf("check that %s/%s %s is served: %w", group, version, kind, err)
		}
	}
	return nil
}
