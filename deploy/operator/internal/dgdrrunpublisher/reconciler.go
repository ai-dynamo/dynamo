/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package dgdrrunpublisher

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	dgdv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	// LabelRunUID ties a candidate to one incarnation of its run. The run name can be
	// reused after a delete and recreate; the UID cannot, so candidates left over from a
	// previous run of the same name are never mistaken for current ones. The manager's
	// cache selects candidates by this label.
	LabelRunUID = "nvidia.com/dgdr-run-uid"
	// LabelCandidateID records the stable candidate id (the identity read back on list).
	LabelCandidateID = "nvidia.com/dgdr-candidate-id"

	evaluatedCondition = "Evaluated"

	// AnnotationAdditionalResources carries the companion resources (for example
	// generated ConfigMaps) rendered next to the DGD, as multi-document YAML. It is the
	// same key and format the DGDR controller uses.
	AnnotationAdditionalResources = "dgdr.nvidia.com/additional-resources"

	// fieldOwner is the server-side-apply field manager of the publisher.
	fieldOwner = "dgdr-run-publisher"

	// finalReconcileAttempts bounds the retries once the Sweeper has exited: nothing will
	// rewrite the snapshot, so a persistent failure must end the process.
	finalReconcileAttempts = 3
)

// Reconciler reconciles the one DynamoGraphDeploymentRun this publisher serves.
//
// Client is the manager's client: reads come from the scoped informer cache (this Run,
// its candidates, this Pod), writes go to the API server. It needs RBAC (namespaced Role
// bound to the run's ServiceAccount): runs get/list/watch and runs/status patch (server-
// side apply); candidates get/list/watch/create/patch/delete and candidates/status patch;
// pods get/list/watch (own pod, for Sweeper crash detection; resourceNames cannot be used
// with list/watch of generated Job pod names, so the cache narrows with a field selector).
type Reconciler struct {
	Client           client.Client
	Namespace        string
	RunName          string
	RunUID           string
	PodName          string
	SweeperContainer string

	Observer   *Observer
	Store      *InputStore
	Completion *Completion

	// finalFailures counts failed reconciles after the Sweeper exited. The controller
	// runs one worker, so it needs no locking.
	finalFailures int
}

func (r *Reconciler) key() types.NamespacedName {
	return types.NamespacedName{Namespace: r.Namespace, Name: r.RunName}
}

// Reconcile handles the single run key. Both the input observer and informer events
// enqueue it; the returned error makes the workqueue retry with backoff.
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	if req.NamespacedName != r.key() || r.Completion.Finished() {
		return ctrl.Result{}, nil
	}
	return ctrl.Result{}, r.reconcile(ctx)
}

func (r *Reconciler) reconcile(ctx context.Context) error {
	var run v1beta2.DynamoGraphDeploymentRun
	if err := r.Client.Get(ctx, r.key(), &run); err != nil {
		return fmt.Errorf("getting run: %w", err)
	}
	if string(run.UID) != r.RunUID {
		// The run was deleted and recreated under the same name; this pod belongs to the
		// previous incarnation and must not touch the new one.
		r.Completion.Finish(fmt.Errorf("run %s has uid %s, this publisher serves uid %s", r.RunName, run.UID, r.RunUID))
		return nil
	}

	// Read the container state BEFORE the snapshot: the Sweeper's last write precedes its
	// exit, so an exit observed here means that write is visible to the check below.
	state, stateErr := r.sweeperState(ctx)
	if stateErr != nil {
		log.Printf("reading sweeper container state: %v", stateErr)
	}
	exited := stateErr == nil && state.Exited
	if exited {
		r.Observer.Check()
	}

	input := r.Store.Load()
	if input.Err != nil && exited {
		if errors.Is(input.Err, ErrProtocolViolation) {
			r.Completion.Finish(input.Err)
			return nil
		}
		return r.retryOrFail(exited, input.Err)
	}
	if input.Snapshot == nil {
		if exited {
			r.Completion.Finish(exitResult(state))
		}
		return nil // the observer enqueues the key when the first snapshot arrives
	}

	desired := Desire(r.RunName, r.RunUID, input.Snapshot)
	if err := r.apply(ctx, &run, desired); err != nil {
		return r.retryOrFail(exited, err)
	}
	switch {
	case desired.Terminal:
		r.Completion.Finish(terminalResult(desired.Phase))
	case exited:
		r.Completion.Finish(exitResult(state))
	}
	return nil
}

// retryOrFail hands a failure to the workqueue's backoff, except where retrying cannot
// help (malformed projection) or the Sweeper is gone and the attempts are used up.
func (r *Reconciler) retryOrFail(exited bool, err error) error {
	var inputErr *dgdcreconcile.DiffInputError
	if errors.As(err, &inputErr) {
		r.Completion.Finish(err) // retrying cannot fix it
		return nil
	}
	if exited {
		r.finalFailures++
		if r.finalFailures >= finalReconcileAttempts {
			r.Completion.Finish(fmt.Errorf("reconciling last snapshot: %w", err))
			return nil
		}
	}
	return err
}

// apply drives the cluster toward the desired state: create missing candidates, finish
// the status of candidates that lack it, update the run, then delete obsolete candidates.
// It is idempotent and safe to retry; existing candidate specs are never touched.
func (r *Reconciler) apply(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, desired DesiredState) error {
	var list v1beta2.DynamoGraphDeploymentCandidateList
	if err := r.Client.List(ctx, &list, client.InNamespace(r.Namespace), client.MatchingLabels{LabelRunUID: r.RunUID}); err != nil {
		return fmt.Errorf("listing candidates: %w", err)
	}
	current := make([]dgdcreconcile.CurrentDGDC, 0, len(list.Items))
	uids := make(map[string]types.UID, len(list.Items))
	for i := range list.Items {
		item := &list.Items[i]
		current = append(current, dgdcreconcile.CurrentDGDC{
			Name:       item.Name,
			ID:         item.Labels[LabelCandidateID],
			Incomplete: !meta.IsStatusConditionTrue(item.Status.Conditions, evaluatedCondition),
		})
		uids[item.Name] = item.UID
	}
	actions, err := dgdcreconcile.ComputeActions(desired.Candidates, current)
	if err != nil {
		return err
	}

	for _, candidate := range actions.Creates {
		if err := r.createCandidate(ctx, run, desired.Names[candidate.ID], candidate); err != nil {
			return fmt.Errorf("creating candidate %s: %w", candidate.ID, err)
		}
	}
	unpopulated := make([]dgdcreconcile.DesiredCandidate, 0, len(actions.Creates)+len(actions.Incomplete))
	unpopulated = append(unpopulated, actions.Creates...)
	unpopulated = append(unpopulated, actions.Incomplete...)
	for _, candidate := range unpopulated {
		if err := r.populateStatus(ctx, desired.Names[candidate.ID], candidate); err != nil {
			return fmt.Errorf("populating status of candidate %s: %w", candidate.ID, err)
		}
	}

	if desired.Status.NeedsUpdate(observedStatus(run)) {
		if err := r.updateRunStatus(ctx, desired.Status); err != nil {
			return fmt.Errorf("updating run status: %w", err)
		}
	}

	for _, name := range actions.Deletes {
		uid := uids[name]
		object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{Namespace: r.Namespace, Name: name}}
		if err := client.IgnoreNotFound(r.Client.Delete(ctx, object, client.Preconditions{UID: &uid})); err != nil {
			return fmt.Errorf("deleting candidate %s: %w", name, err)
		}
	}
	return nil
}

// createCandidate creates the candidate with server-side apply. It is only called for a
// candidate that is missing from the cache, and re-applying the same content after a
// stale cache read is a no-op, so no spec comparison is needed.
func (r *Reconciler) createCandidate(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) error {
	spec, companions, err := decodeManifest(candidate)
	if err != nil {
		return err
	}
	parameters, err := normalizeJSON(candidate.Parameters)
	if err != nil {
		return err
	}
	object := &v1beta2.DynamoGraphDeploymentCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: r.Namespace,
			Name:      name,
			Labels:    map[string]string{LabelRunUID: r.RunUID, LabelCandidateID: candidate.ID},
		},
		Spec: v1beta2.DynamoGraphDeploymentCandidateSpec{DynamoGraphDeploymentSpec: *spec},
	}
	if companions != "" {
		object.Annotations = map[string]string{AnnotationAdditionalResources: companions}
	}
	if err := controllerutil.SetControllerReference(run, object, r.Client.Scheme()); err != nil {
		return err
	}
	content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		return err
	}
	// The status is written separately, through the status subresource.
	delete(content, "status")
	unstructured.RemoveNestedField(content, "metadata", "creationTimestamp")
	if err := unstructured.SetNestedField(content, parameters, "spec", "parameters"); err != nil {
		return err
	}
	apply := &unstructured.Unstructured{Object: content}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentCandidate"))
	return r.Client.Apply(ctx, client.ApplyConfigurationFromUnstructured(apply), client.FieldOwner(fieldOwner), client.ForceOwnership)
}

// populateStatus records the evaluation facts once. The candidate is referenced from the
// run only after this, and its status is immutable afterwards.
func (r *Reconciler) populateStatus(ctx context.Context, name string, candidate dgdcreconcile.DesiredCandidate) error {
	metrics, err := normalizeJSON(candidate.Metrics)
	if err != nil {
		return err
	}
	apply := &unstructured.Unstructured{Object: map[string]any{
		"metadata": map[string]any{"namespace": r.Namespace, "name": name},
		"status": map[string]any{
			"metrics": metrics,
			"conditions": []any{map[string]any{
				"type":               evaluatedCondition,
				"status":             string(metav1.ConditionTrue),
				"reason":             evaluatedCondition,
				"message":            "evaluated by the Sweeper",
				"lastTransitionTime": time.Now().UTC().Format(time.RFC3339),
			}},
		},
	}}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentCandidate"))
	return r.Client.Status().Apply(ctx, client.ApplyConfigurationFromUnstructured(apply), client.FieldOwner(fieldOwner), client.ForceOwnership)
}

// updateRunStatus writes the publisher-owned run status fields with server-side apply, so
// the run controller's own status fields (conditions) are left alone.
func (r *Reconciler) updateRunStatus(ctx context.Context, status RunStatus) error {
	refs := make([]any, 0, len(status.CandidateNames))
	for _, name := range status.CandidateNames {
		refs = append(refs, map[string]any{"name": name})
	}
	fields := map[string]any{
		"message":       status.Message,
		"candidateRefs": refs,
	}
	if status.HasProgress {
		fields["progress"] = map[string]any{"round": int64(status.Round), "evaluated": int64(status.Evaluated)}
	}
	if !status.LastProgress.IsZero() {
		fields["lastProgressTime"] = status.LastProgress.UTC().Format(time.RFC3339)
	}
	apply := &unstructured.Unstructured{Object: map[string]any{
		"metadata": map[string]any{"namespace": r.Namespace, "name": r.RunName},
		"status":   fields,
	}}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentRun"))
	return r.Client.Status().Apply(ctx, client.ApplyConfigurationFromUnstructured(apply), client.FieldOwner(fieldOwner), client.ForceOwnership)
}

// observedStatus is the publisher-owned part of the run's current (cached) status.
func observedStatus(run *v1beta2.DynamoGraphDeploymentRun) RunStatus {
	observed := RunStatus{Message: run.Status.Message}
	if progress := run.Status.Progress; progress != nil {
		observed.HasProgress = true
		observed.Round = progress.Round
		observed.Evaluated = progress.Evaluated
	}
	if run.Status.LastProgressTime != nil {
		observed.LastProgress = run.Status.LastProgressTime.UTC()
	}
	for _, ref := range run.Status.CandidateRefs {
		observed.CandidateNames = append(observed.CandidateNames, ref.Name)
	}
	return observed
}

// sweeperState reads the Sweeper container's status from this pod, via the cache.
func (r *Reconciler) sweeperState(ctx context.Context) (SweeperState, error) {
	var pod corev1.Pod
	if err := r.Client.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: r.PodName}, &pod); err != nil {
		return SweeperState{}, err
	}
	return sweeperStateOf(&pod, r.SweeperContainer), nil
}

func sweeperStateOf(pod *corev1.Pod, container string) SweeperState {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name != container {
			continue
		}
		if terminated := status.State.Terminated; terminated != nil {
			return SweeperState{Exited: true, ExitCode: terminated.ExitCode}
		}
		return SweeperState{}
	}
	return SweeperState{}
}

// normalizeJSON round-trips a value through JSON so it only holds the types unstructured
// objects allow (int64, float64, string, bool, maps, slices).
func normalizeJSON(value map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if value == nil {
		return out, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// decodeManifest extracts the DGD spec and the companion resources from a rendered
// manifest. A DGD in either supported apiVersion is converted to v1beta1, the same way
// the DGDR controller does it.
func decodeManifest(candidate dgdcreconcile.DesiredCandidate) (*v1beta1.DynamoGraphDeploymentSpec, string, error) {
	manifest, err := ParseManifest(candidate.Spec)
	if err != nil {
		return nil, "", fmt.Errorf("candidate %s: %w", candidate.ID, err)
	}
	obj := &unstructured.Unstructured{Object: manifest.DGD}
	var beta v1beta1.DynamoGraphDeployment
	switch obj.GetAPIVersion() {
	case dgdv1alpha1.GroupVersion.String():
		var alpha dgdv1alpha1.DynamoGraphDeployment
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &alpha); err != nil {
			return nil, "", fmt.Errorf("candidate %s: %w", candidate.ID, err)
		}
		if err := alpha.ConvertTo(&beta); err != nil {
			return nil, "", fmt.Errorf("candidate %s: %w", candidate.ID, err)
		}
	case v1beta1.GroupVersion.String():
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &beta); err != nil {
			return nil, "", fmt.Errorf("candidate %s: %w", candidate.ID, err)
		}
	default:
		return nil, "", fmt.Errorf("candidate %s: unsupported DynamoGraphDeployment apiVersion %q", candidate.ID, obj.GetAPIVersion())
	}
	return &beta.Spec, strings.Join(manifest.Companions, "\n---\n"), nil
}
