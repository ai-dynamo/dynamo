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
	"slices"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// LabelRunUID ties a candidate to one incarnation of its run. The run name can be
	// reused after a delete and recreate; the UID cannot, so candidates left over from a
	// previous run of the same name are never mistaken for current ones. It is a repairable
	// index; the controller owner UID remains the authoritative ownership relation.
	LabelRunUID = "nvidia.com/dgdr-run-uid"
	// LabelCandidateID records the stable candidate id (the identity read back on list).
	LabelCandidateID = "nvidia.com/dgdr-candidate-id"

	// AnnotationAdditionalResources carries the companion resources (for example
	// generated ConfigMaps) rendered next to the DGD, as multi-document YAML. It is the
	// same key and format the DGDR controller uses.
	AnnotationAdditionalResources = "dgdr.nvidia.com/additional-resources"

	// fieldOwner is the server-side-apply field manager of the publisher.
	fieldOwner = "dgdr-run-publisher"
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
// the cache can trail them. Every pass still decides from current cached observations,
// so repeated idempotent writes are harmless. The reconciler remembers only writes that
// have not yet appeared in the cache; this is a completion barrier, not desired state.
// A terminal result is acknowledged only after those echoes arrive and a final pass
// observes that the desired state needs no more writes.
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

	pendingCandidates map[string]candidateObservation
	pendingRun        *runObservation
}

// candidateObservation describes the cache echo that proves a candidate write has
// become visible. A deletion is observed when the written UID is absent; a create or
// status write is observed only on the exact candidate incarnation.
type candidateObservation struct {
	UID       types.UID
	ID        string
	Present   bool
	Populated bool
}

type runObservation struct {
	BaseResourceVersion string
	Status              RunStatus
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
		r.Completion.Finish(fmt.Errorf("reading last snapshot: %w", input.Err))
		return nil
	}
	if input.Snapshot == nil {
		if exited {
			r.Completion.Finish(exitResult(state))
		}
		return nil // the observer enqueues the key when the first snapshot arrives
	}

	desired := Desire(r.RunName, r.RunUID, input.Snapshot)
	quiescent, err := r.apply(ctx, &run, desired)
	if err != nil {
		return r.retryOrFail(err)
	}
	if !quiescent {
		return nil
	}
	switch {
	case desired.Terminal:
		r.Completion.Finish(terminalResult(desired))
	case exited:
		r.Completion.Finish(exitResult(state))
	}
	return nil
}

// retryOrFail hands a transient failure to the workqueue's backoff. A malformed
// projection is permanent and completes the publisher immediately. Kubernetes API and
// network failures remain retryable after the Sweeper exits because the terminal snapshot
// is durable and the publisher Pod's lifecycle is the cancellation boundary.
func (r *Reconciler) retryOrFail(err error) error {
	if _, ok := errors.AsType[*dgdcreconcile.DiffInputError](err); ok {
		r.Completion.Finish(err) // retrying cannot fix it
		return nil
	}
	return err
}

// apply drives the cluster toward the desired state: create missing candidates, finish
// the status of candidates that lack it, update the run, then delete obsolete candidates.
// It is idempotent and safe to retry; existing candidate specs are never touched.
func (r *Reconciler) apply(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, desired DesiredState) (bool, error) {
	if desired.Failed > 0 {
		ctrl.LoggerFrom(ctx).V(1).Info("candidates failed materialization and are not published", "count", desired.Failed)
	}
	var list v1beta2.DynamoGraphDeploymentCandidateList
	if err := r.Client.List(ctx, &list, client.InNamespace(r.Namespace)); err != nil {
		return false, fmt.Errorf("listing candidates: %w", err)
	}
	r.observeWrites(run, list.Items)
	wrote := false
	current := make([]dgdcreconcile.CurrentDGDC, 0, len(list.Items))
	ownedObsolete := make([]string, 0, len(list.Items))
	uids := make(map[string]types.UID, len(list.Items))
	versions := make(map[string]string, len(list.Items)) // cached resourceVersion per candidate
	expectedIDs := make(map[string]string, len(desired.Names))
	for id, name := range desired.Names {
		expectedIDs[name] = id
	}
	for i := range list.Items {
		item := &list.Items[i]

		// A label match scopes the cache but does not establish ownership. Foreign
		// candidates must never satisfy desired state or become deletion targets.
		owner := metav1.GetControllerOf(item)
		if owner == nil || owner.UID != run.UID {
			continue
		}

		// The deterministic name is the authoritative identity. An owned object under
		// another name is obsolete; an expected object with a stale identity label is
		// reapplied in place instead of deleted.
		expectedID, isExpected := expectedIDs[item.Name]
		if !isExpected {
			ownedObsolete = append(ownedObsolete, item.Name)
			uids[item.Name] = item.UID
			continue
		}
		if item.Labels[LabelRunUID] != r.RunUID || item.Labels[LabelCandidateID] != expectedID {
			continue
		}
		current = append(current, dgdcreconcile.CurrentDGDC{
			Name:       item.Name,
			ID:         expectedID,
			Incomplete: item.Status.Metrics == nil,
		})
		uids[item.Name] = item.UID
		versions[item.Name] = item.ResourceVersion
	}
	actions, err := dgdcreconcile.ComputeActions(desired.Candidates, current)
	if err != nil {
		return false, err
	}
	actions.Deletes = append(actions.Deletes, ownedObsolete...)
	// populated names the candidates whose status the API server already has; they are
	// known from the live read that follows each apply, not from the cache.
	populated := map[string]struct{}{}
	for _, candidate := range actions.Creates {
		name := desired.Names[candidate.ID]
		stored, err := r.Writer.CreateCandidate(ctx, run, name, candidate)
		if err != nil {
			return false, fmt.Errorf("creating candidate %s: %w", candidate.ID, err)
		}
		wrote = true
		r.expectCandidate(name, candidateObservation{UID: stored.UID, ID: candidate.ID, Present: true, Populated: stored.Populated})
		uids[name] = stored.UID
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
			return false, fmt.Errorf("populating status of candidate %s: %w", candidate.ID, err)
		}
		wrote = true
		r.expectCandidate(name, candidateObservation{UID: uids[name], ID: candidate.ID, Present: true, Populated: true})
	}

	updated, err := r.updateRunStatus(ctx, run, desired.Status)
	if err != nil {
		return false, err
	}
	wrote = wrote || updated

	for _, name := range actions.Deletes {
		if err := r.Writer.DeleteCandidate(ctx, name, uids[name]); err != nil {
			return false, fmt.Errorf("deleting candidate %s: %w", name, err)
		}
		wrote = true
		r.expectCandidate(name, candidateObservation{UID: uids[name]})
	}
	return !wrote && len(r.pendingCandidates) == 0 && r.pendingRun == nil, nil
}

// updateRunStatus writes the run status when it differs from what the cache shows. A
// snapshot without a usable timestamp keeps the progress time the cached run already has,
// because a server-side-apply payload that omits the field would remove it. The write is
// constrained to the cached run's resourceVersion, so a decision made from a stale cache
// fails with a conflict instead of overwriting newer status.
func (r *Reconciler) updateRunStatus(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, desired RunStatus) (bool, error) {
	observed := observedStatus(run)
	if !desired.NeedsUpdate(observed) {
		return false, nil
	}
	write := desired.WithPreservedProgress(observed.LastProgress)
	if err := r.Writer.UpdateRunStatus(ctx, run.ResourceVersion, write); err != nil {
		return false, fmt.Errorf("updating run status: %w", err)
	}
	r.pendingRun = &runObservation{BaseResourceVersion: run.ResourceVersion, Status: write}
	return true, nil
}

func (r *Reconciler) expectCandidate(name string, observation candidateObservation) {
	if r.pendingCandidates == nil {
		r.pendingCandidates = map[string]candidateObservation{}
	}
	r.pendingCandidates[name] = observation
}

// observeWrites removes completion-barrier entries only when the corresponding write is
// visible through the informer cache. API-server read-backs are deliberately insufficient:
// completion stops the process, and therefore must not race the watch echoes that trigger
// repairs for a later terminal snapshot.
func (r *Reconciler) observeWrites(run *v1beta2.DynamoGraphDeploymentRun, candidates []v1beta2.DynamoGraphDeploymentCandidate) {
	byName := make(map[string]*v1beta2.DynamoGraphDeploymentCandidate, len(candidates))
	for i := range candidates {
		byName[candidates[i].Name] = &candidates[i]
	}
	for name, expected := range r.pendingCandidates {
		candidate := byName[name]
		if !expected.Present {
			if candidate == nil || candidate.UID != expected.UID {
				delete(r.pendingCandidates, name)
			}
			continue
		}
		if candidate == nil || candidate.UID != expected.UID {
			continue
		}
		owner := metav1.GetControllerOf(candidate)
		if owner == nil || owner.UID != run.UID || candidate.Labels[LabelRunUID] != r.RunUID || candidate.Labels[LabelCandidateID] != expected.ID {
			continue
		}
		if !expected.Populated || candidate.Status.Metrics != nil {
			delete(r.pendingCandidates, name)
		}
	}
	if r.pendingRun != nil && run.ResourceVersion != r.pendingRun.BaseResourceVersion && runStatusEqual(observedStatus(run), r.pendingRun.Status) {
		r.pendingRun = nil
	}
}

func runStatusEqual(left, right RunStatus) bool {
	return left.Published == right.Published && left.LastProgress.Equal(right.LastProgress) && slices.Equal(left.CandidateNames, right.CandidateNames)
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
