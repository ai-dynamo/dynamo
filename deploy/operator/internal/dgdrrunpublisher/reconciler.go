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
	"errors"
	"fmt"
	"strings"

	dgdv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// LabelRunUID ties a candidate to one incarnation of its run. The run name can be
	// reused after a delete and recreate; the UID cannot, so candidates left over from a
	// previous run of the same name are never mistaken for current ones. The manager's
	// cache selects candidates by this label.
	LabelRunUID = "nvidia.com/dgdr-run-uid"
	// LabelCandidateID records the stable candidate id (the identity read back on list).
	LabelCandidateID = "nvidia.com/dgdr-candidate-id"

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
// its candidates, this Pod). Writes go through Writer to the API server. It needs RBAC
// (namespaced Role bound to the run's ServiceAccount): runs get/list/watch and
// runs/status patch (server-side apply); candidates get/list/watch/create/patch/delete
// and candidates/status patch; pods get/list/watch (own pod, for Sweeper crash
// detection; the cache narrows list/watch to it with a field selector because generated
// Job pod names cannot be listed in resourceNames).
//
// Write echoes and cache lag: the publisher's own writes come back as watch events, and
// the cache can trail them. The reconciler keeps no record of its writes. Every pass
// decides from the current observations (the cached run and candidates, the latest input
// snapshot), so a repeated pass while the cache catches up may repeat an idempotent
// write, which is harmless, and an echo of a write finds nothing left to do. Anything
// that changes behind the publisher's back (a deleted candidate, a modified run status)
// is repaired on the next pass.
//
// Stale reads: every status write is constrained to the resourceVersion the decision was
// based on (the cached run, the cached candidate, or the version read back from the API
// server for a candidate applied a moment ago). A decision made from an older cache
// observation fails with a conflict instead of overwriting newer state, and is retried
// through the workqueue once the informer has delivered the newer version. A candidate's
// status is only populated when the API server's own copy, read right after the apply,
// has none yet, so a stale cache can never make the publisher rewrite populated status.
type Reconciler struct {
	Client           client.Client
	Writer           writer
	Namespace        string
	RunName          string
	RunUID           string
	PodName          string
	SweeperContainer string

	Observer   *Observer
	Store      *InputStore
	Completion *Completion

	// The fields below are touched only by Reconcile; the controller runs one worker, so
	// they need no locking.

	// finalFailures counts failed reconciles after the Sweeper exited.
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
		ctrl.LoggerFrom(ctx).Error(stateErr, "reading sweeper container state")
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
	if apierrors.IsConflict(err) {
		// A write was based on a stale cached read. The informer delivers the newer
		// version and the retry decides again, so this is not a failed attempt.
		return err
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
	if desired.Failed > 0 {
		ctrl.LoggerFrom(ctx).V(1).Info("candidates failed materialization and are not published", "count", desired.Failed)
	}
	var list v1beta2.DynamoGraphDeploymentCandidateList
	if err := r.Client.List(ctx, &list, client.InNamespace(r.Namespace), client.MatchingLabels{LabelRunUID: r.RunUID}); err != nil {
		return fmt.Errorf("listing candidates: %w", err)
	}
	current := make([]dgdcreconcile.CurrentDGDC, 0, len(list.Items))
	uids := make(map[string]types.UID, len(list.Items))
	versions := make(map[string]string, len(list.Items)) // cached resourceVersion per candidate
	for i := range list.Items {
		item := &list.Items[i]
		current = append(current, dgdcreconcile.CurrentDGDC{
			Name:       item.Name,
			ID:         item.Labels[LabelCandidateID],
			Incomplete: item.Status.Metrics == nil,
		})
		uids[item.Name] = item.UID
		versions[item.Name] = item.ResourceVersion
	}
	actions, err := dgdcreconcile.ComputeActions(desired.Candidates, current)
	if err != nil {
		return err
	}
	// populated names the candidates whose status the API server already has; they are
	// known from the live read that follows each apply, not from the cache.
	populated := map[string]struct{}{}
	for _, candidate := range actions.Creates {
		name := desired.Names[candidate.ID]
		stored, err := r.Writer.CreateCandidate(ctx, run, name, candidate)
		if err != nil {
			return fmt.Errorf("creating candidate %s: %w", candidate.ID, err)
		}
		versions[name] = stored.ResourceVersion // a new candidate is not in the cache; use the version the API server returned
		if stored.Populated {
			populated[name] = struct{}{}
		}
	}
	unpopulated := make([]dgdcreconcile.DesiredCandidate, 0, len(actions.Creates)+len(actions.Incomplete))
	unpopulated = append(unpopulated, actions.Creates...)
	unpopulated = append(unpopulated, actions.Incomplete...)
	for _, candidate := range unpopulated {
		name := desired.Names[candidate.ID]
		if _, done := populated[name]; done {
			continue
		}
		if err := r.Writer.PopulateStatus(ctx, name, versions[name], candidate); err != nil {
			return fmt.Errorf("populating status of candidate %s: %w", candidate.ID, err)
		}
	}

	if err := r.updateRunStatus(ctx, run, desired.Status); err != nil {
		return err
	}

	for _, name := range actions.Deletes {
		if err := r.Writer.DeleteCandidate(ctx, name, uids[name]); err != nil {
			return fmt.Errorf("deleting candidate %s: %w", name, err)
		}
	}
	return nil
}

// updateRunStatus writes the run status when it differs from what the cache shows. A
// snapshot without a usable timestamp keeps the progress time the cached run already has,
// because a server-side-apply payload that omits the field would remove it. The write is
// constrained to the cached run's resourceVersion, so a decision made from a stale cache
// fails with a conflict instead of overwriting newer status.
func (r *Reconciler) updateRunStatus(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, desired RunStatus) error {
	observed := observedStatus(run)
	if !desired.NeedsUpdate(observed) {
		return nil
	}
	write := desired.WithPreservedProgress(observed.LastProgress)
	if err := r.Writer.UpdateRunStatus(ctx, run.ResourceVersion, write); err != nil {
		return fmt.Errorf("updating run status: %w", err)
	}
	return nil
}

// observedStatus is the publisher-owned part of the run's current (cached) status.
func observedStatus(run *v1beta2.DynamoGraphDeploymentRun) RunStatus {
	var observed RunStatus
	if progress := run.Status.Progress; progress != nil && progress.Candidates != nil {
		observed.Published = progress.Candidates.Published
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
