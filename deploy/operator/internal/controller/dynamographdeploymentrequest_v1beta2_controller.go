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
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"

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
	// dgdrV1beta2RequestLabel names the Request that owns a Run.
	dgdrV1beta2RequestLabel = "nvidia.com/dgdr-request"
	// dgdrV1beta2GenerationAnnotation records the Request generation a Run was created for.
	dgdrV1beta2GenerationAnnotation = "nvidia.com/dgdr-request-generation"

	// dgdrV1beta2RetainedRuns is how many terminal Runs a Request keeps. Older ones are deleted.
	dgdrV1beta2RetainedRuns = 5

	// DGDRReconciledConditionV1beta2 reports whether the controller accepted the current generation.
	DGDRReconciledConditionV1beta2 = "Reconciled"

	dgdrReasonRunCreated          = "RunCreated"
	dgdrReasonInvalidSpec         = "InvalidSpec"
	dgdrReasonWaitingForActiveRun = "WaitingForActiveRun"

	dgdrRunPhaseSucceeded = "Succeeded"
	dgdrRunPhaseFailed    = "Failed"
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

// Reconcile creates the Run for the current generation of a native v1beta2 Request once no other Run
// is active, and records the active and retained Runs in the Request status.
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

	accepted := current != nil
	condition := dgdrV1beta2Condition(request, metav1.ConditionTrue, dgdrReasonRunCreated,
		fmt.Sprintf("Run %q exists for generation %d", wantName, request.Generation))
	if current == nil {
		if active := dgdrV1beta2NewestActive(runs); active != nil {
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonWaitingForActiveRun,
				fmt.Sprintf("Run %q is still active; the Run for generation %d starts when it finishes",
					active.Name, request.Generation))
		} else if run, buildErr := r.buildRun(request); buildErr != nil {
			logger.Info("Request cannot be turned into a Run", "reason", buildErr.Error())
			condition = dgdrV1beta2Condition(request, metav1.ConditionFalse, dgdrReasonInvalidSpec, buildErr.Error())
		} else {
			if err := r.Create(ctx, run); err != nil && !apierrors.IsAlreadyExists(err) {
				return ctrl.Result{}, fmt.Errorf("create Run %q: %w", wantName, err)
			}
			runs = append(runs, *run)
			accepted = true
		}
	}

	active, terminal := dgdrV1beta2SplitRuns(runs)
	keep := min(len(terminal), dgdrV1beta2RetainedRuns)
	for i := keep; i < len(terminal); i++ {
		if err := r.Delete(ctx, &terminal[i]); err != nil && !apierrors.IsNotFound(err) {
			return ctrl.Result{}, fmt.Errorf("delete retired Run %q: %w", terminal[i].Name, err)
		}
	}

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
	return ctrl.Result{}, nil
}

// buildRun resolves the optional inputs of the Request into an immutable Run spec. The effective
// AISimulate input is frozen into aisimulate.parameters, which is authoritative, so lowering the Run
// again yields the same bytes.
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
			Labels:      map[string]string{dgdrV1beta2RequestLabel: request.Name},
			Annotations: map[string]string{dgdrV1beta2GenerationAnnotation: strconv.FormatInt(request.Generation, 10)},
		},
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: *frozen},
	}
	if err := controllerutil.SetControllerReference(request, run, r.Scheme()); err != nil {
		return nil, fmt.Errorf("set Run owner: %w", err)
	}
	return run, nil
}

// ownedRuns returns the Runs this Request controls.
func (r *DynamoGraphDeploymentRequestV1beta2Reconciler) ownedRuns(
	ctx context.Context,
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
) ([]nvidiacomv1beta2.DynamoGraphDeploymentRun, error) {
	list := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
	if err := r.List(ctx, list, client.InNamespace(request.Namespace),
		client.MatchingLabels{dgdrV1beta2RequestLabel: request.Name}); err != nil {
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
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrequest-v1beta2").
		For(&nvidiacomv1beta2.DynamoGraphDeploymentRequest{}).
		Owns(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}

// dgdrV1beta2RunName names the Run of one Request generation. A metadata-only update keeps the
// generation, so it never creates a second Run.
func dgdrV1beta2RunName(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) string {
	return fmt.Sprintf("%s-%d", request.Name, request.Generation)
}

func dgdrV1beta2Condition(
	request *nvidiacomv1beta2.DynamoGraphDeploymentRequest,
	status metav1.ConditionStatus,
	reason, message string,
) metav1.Condition {
	return metav1.Condition{
		Type:               DGDRReconciledConditionV1beta2,
		Status:             status,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: request.Generation,
	}
}

func dgdrV1beta2IsTerminal(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) bool {
	phase := string(run.Status.Phase)
	return phase == dgdrRunPhaseSucceeded || phase == dgdrRunPhaseFailed
}

// dgdrV1beta2Newer orders Runs newest first: by Request generation, then creation time, then name.
func dgdrV1beta2Newer(a, b *nvidiacomv1beta2.DynamoGraphDeploymentRun) bool {
	ga, gb := dgdrV1beta2RunGeneration(a), dgdrV1beta2RunGeneration(b)
	if ga != gb {
		return ga > gb
	}
	if !a.CreationTimestamp.Equal(&b.CreationTimestamp) {
		return b.CreationTimestamp.Before(&a.CreationTimestamp)
	}
	return a.Name > b.Name
}

func dgdrV1beta2RunGeneration(run *nvidiacomv1beta2.DynamoGraphDeploymentRun) int64 {
	generation, _ := strconv.ParseInt(run.Annotations[dgdrV1beta2GenerationAnnotation], 10, 64)
	return generation
}

// dgdrV1beta2SplitRuns returns the newest non-terminal Run, if any, and the terminal Runs newest first.
func dgdrV1beta2SplitRuns(
	runs []nvidiacomv1beta2.DynamoGraphDeploymentRun,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRun, []nvidiacomv1beta2.DynamoGraphDeploymentRun) {
	sorted := append([]nvidiacomv1beta2.DynamoGraphDeploymentRun(nil), runs...)
	sort.SliceStable(sorted, func(i, j int) bool { return dgdrV1beta2Newer(&sorted[i], &sorted[j]) })
	var active *nvidiacomv1beta2.DynamoGraphDeploymentRun
	var terminal []nvidiacomv1beta2.DynamoGraphDeploymentRun
	for i := range sorted {
		switch {
		case !dgdrV1beta2IsTerminal(&sorted[i]):
			if active == nil {
				active = &sorted[i]
			}
		default:
			terminal = append(terminal, sorted[i])
		}
	}
	return active, terminal
}

func dgdrV1beta2NewestActive(
	runs []nvidiacomv1beta2.DynamoGraphDeploymentRun,
) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	active, _ := dgdrV1beta2SplitRuns(runs)
	return active
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
