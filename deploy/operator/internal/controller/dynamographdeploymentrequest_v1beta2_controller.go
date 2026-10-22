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
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// dgdrV1beta2GenerationAnnotation records the Request generation a Run was created for. It is
	// diagnostic only: the generation the controller acts on is the one in the immutable Run name.
	dgdrV1beta2GenerationAnnotation = "nvidia.com/dgdr-request-generation"

	// dgdrRunControllerUIDIndex indexes Runs by the UID of their controller OwnerReference. The
	// OwnerReference is the only definition of which Runs belong to a Request.
	dgdrRunControllerUIDIndex = "dynamo.nvidia.com/dgdr-run-controller-uid"

	// dgdrV1beta2RetainedRuns is how many terminal Runs a Request keeps. Older ones are deleted.
	dgdrV1beta2RetainedRuns = 5

	// DGDRAcceptedConditionV1beta2 reports whether the current Request generation has been
	// materialized as an immutable Run. Execution state belongs to the Run, not to this condition.
	DGDRAcceptedConditionV1beta2 = "Accepted"

	dgdrReasonRunCreated          = "RunCreated"
	dgdrReasonInvalidSpec         = "InvalidSpec"
	dgdrReasonWaitingForActiveRun = "WaitingForActiveRun"
	dgdrReasonRunNameConflict     = "RunNameConflict"

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
	Config        *configv1alpha1.OperatorConfiguration
	RuntimeConfig *commonController.RuntimeConfig
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns,verbs=get;list;watch;create;delete

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
	var current *nvidiacomv1beta2.DynamoGraphDeploymentRun
	for i := range runs {
		if runs[i].Name == wantName {
			current = &runs[i]
		}
	}

	active, _, err := dgdrV1beta2SplitRuns(request.Name, runs)
	if err != nil {
		return ctrl.Result{}, err
	}

	accepted := current != nil
	condition := dgdrV1beta2Condition(request, metav1.ConditionTrue, dgdrReasonRunCreated,
		fmt.Sprintf("Run %q exists for generation %d", wantName, request.Generation))
	if current == nil {
		if active != nil {
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonWaitingForActiveRun,
				fmt.Sprintf("Run %q is still active; the Run for generation %d starts when it finishes",
					active.Name, request.Generation))
		} else if run, buildErr := r.buildRun(request); buildErr != nil {
			logger.Info("Request cannot be turned into a Run", "reason", buildErr.Error())
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonInvalidSpec, buildErr.Error())
		} else {
			err := r.Create(ctx, run)
			switch {
			case err == nil:
				runs = append(runs, *run)
				accepted = true
			case apierrors.IsAlreadyExists(err):
				// A stale cache, or a Run left by an earlier Request of the same name that is still being
				// garbage collected. Only a Run this Request controls counts as ours.
				existing := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
				if getErr := r.Get(ctx, client.ObjectKeyFromObject(run), existing); getErr != nil {
					return ctrl.Result{}, fmt.Errorf("get existing Run %q: %w", wantName, getErr)
				}
				if metav1.IsControlledBy(existing, request) {
					runs = append(runs, *existing)
					accepted = true
				} else {
					condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonRunNameConflict,
						fmt.Sprintf("Run %q exists and is owned by another object; trying again once it is gone", wantName))
					result.RequeueAfter = dgdrV1beta2ConflictRetry
				}
			default:
				return ctrl.Result{}, fmt.Errorf("create Run %q: %w", wantName, err)
			}
		}
	}

	active, terminal, err := dgdrV1beta2SplitRuns(request.Name, runs)
	if err != nil {
		return ctrl.Result{}, err
	}
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
	for i := 0; i < keep; i++ {
		request.Status.RecentRunRefs = append(request.Status.RecentRunRefs, corev1.LocalObjectReference{Name: terminal[i].Name})
	}
	meta.SetStatusCondition(&request.Status.Conditions, condition)

	if !apiequality.Semantic.DeepEqual(original.Status, request.Status) {
		if err := r.Status().Patch(ctx, request, client.MergeFrom(original)); err != nil {
			return ctrl.Result{}, fmt.Errorf("update Request status: %w", err)
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

// buildRun resolves the optional inputs of the Request into an immutable Run spec. The effective
// AISimulate input is frozen into aisimulate.parameters: that object is the complete resolved input.
// Run execution reads it with aisimulate.InputForRun and never lowers the Run again, because the
// merge-patch deletions (null) that shaped it are not part of the stored result.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) buildRun(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRun, error) {
	spec := &request.Spec
	if spec.AISimulate == nil || spec.AISimulate.Image == "" {
		return nil, errors.New("spec.aisimulate.image is required")
	}
	version, err := aisimulate.ResolveVersion(spec.AISimulate.Image, spec.AISimulate.VersionOverride)
	if err != nil {
		return nil, err
	}
	input, err := aisimulate.Lower(spec)
	if err != nil {
		return nil, err
	}
	if !dgdrV1beta2HasEffectiveValue(input, aisimulate.SearchSpaceHardwareSKU) ||
		!dgdrV1beta2HasEffectiveValue(input, aisimulate.SearchSpaceGPUBudget) {
		return nil, errors.New("the effective AISimulate input names no GPU SKU or budget; set spec.hardware.gpu")
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode the effective AISimulate input: %w", err)
	}

	frozen := spec.NativeSearchInput.DeepCopy()
	frozen.AISimulate = &nvidiacomv1beta2.AISimulateSpec{
		Image:           spec.AISimulate.Image,
		VersionOverride: version,
		Parameters:      &runtime.RawExtension{Raw: raw},
	}
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:        dgdrV1beta2RunName(request),
			Namespace:   request.Namespace,
			Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: strconv.FormatInt(request.Generation, 10)},
		},
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: *frozen},
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
		Type:               DGDRAcceptedConditionV1beta2,
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

// dgdrV1beta2RunGeneration reads the Request generation from the immutable Run name. The name is
// accepted only if it is exactly the name this Request derives for that generation; anything else is an
// error, never generation zero.
func dgdrV1beta2RunGeneration(requestName string, run *nvidiacomv1beta2.DynamoGraphDeploymentRun) (int64, error) {
	index := strings.LastIndex(run.Name, "-")
	if index < 0 {
		return 0, fmt.Errorf("run %q has no Request generation suffix", run.Name)
	}
	generation, err := strconv.ParseInt(run.Name[index+1:], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("run %q has no valid Request generation: %w", run.Name, err)
	}
	if expected := dgdrV1beta2RunNameFor(requestName, generation); run.Name != expected {
		return 0, fmt.Errorf("run %q is not the Run name of generation %d of Request %q (expected %q)",
			run.Name, generation, requestName, expected)
	}
	return generation, nil
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

// dgdrV1beta2SplitRuns returns the newest non-terminal Run, if any, and the terminal Runs newest first.
func dgdrV1beta2SplitRuns(
	requestName string,
	runs []nvidiacomv1beta2.DynamoGraphDeploymentRun,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRun, []nvidiacomv1beta2.DynamoGraphDeploymentRun, error) {
	entries := make([]dgdrV1beta2RunEntry, 0, len(runs))
	for i := range runs {
		generation, err := dgdrV1beta2RunGeneration(requestName, &runs[i])
		if err != nil {
			return nil, nil, err
		}
		entries = append(entries, dgdrV1beta2RunEntry{run: runs[i], generation: generation})
	}
	sort.SliceStable(entries, func(i, j int) bool { return dgdrV1beta2Newer(&entries[i], &entries[j]) })
	var active *nvidiacomv1beta2.DynamoGraphDeploymentRun
	var terminal []nvidiacomv1beta2.DynamoGraphDeploymentRun
	for i := range entries {
		switch {
		case !dgdrV1beta2IsTerminal(&entries[i].run):
			if active == nil {
				active = &entries[i].run
			}
		default:
			terminal = append(terminal, entries[i].run)
		}
	}
	return active, terminal, nil
}

// dgdrV1beta2HasEffectiveValue reports whether input holds a non-null, non-empty value at path.
func dgdrV1beta2HasEffectiveValue(input map[string]any, path []string) bool {
	value, found, err := unstructured.NestedFieldNoCopy(input, path...)
	if err != nil || !found || value == nil {
		return false
	}
	text, isString := value.(string)
	return !isString || text != ""
}

// requireDGDRV1beta2API fails when the cluster does not serve the v1beta2 Request and Run APIs.
// Without them the controller's caches never sync and the manager would hang at startup.
func requireDGDRV1beta2API(mgr ctrl.Manager) error {
	group, version := nvidiacomv1beta2.GroupVersion.Group, nvidiacomv1beta2.GroupVersion.Version
	for _, kind := range []string{"DynamoGraphDeploymentRequest", "DynamoGraphDeploymentRun"} {
		_, err := mgr.GetRESTMapper().RESTMapping(schema.GroupKind{Group: group, Kind: kind}, version)
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
