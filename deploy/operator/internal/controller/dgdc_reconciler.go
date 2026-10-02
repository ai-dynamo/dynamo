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

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	sigsyaml "sigs.k8s.io/yaml"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
)

// -- relayed sweeper_status.yaml schema -------------------------------------
//
// These mirror dynamo.aisimulate.output.dgd.kube_status's dataclasses
// field-for-field (candidate_id -> "id", outcome, manifest, path, error; and
// the snapshot's status/round_no/cumulative_evaluated/message/error/
// candidates). Kept private to this file and hand-written rather than
// generated, the same way the rest of this controller already parses
// sidecar-relayed YAML (see generateDGDSpec's extractResourcesFromYAML).

// sweeperCandidateOutcome mirrors kube_status.CandidateOutcome's string values.
const (
	sweeperCandidateOutcomeMaterialized          = "materialized"
	sweeperCandidateOutcomeMaterializationFailed = "materialization_failed"
)

type sweeperCandidateEntry struct {
	ID       string `json:"id"`
	Outcome  string `json:"outcome"`
	Manifest string `json:"manifest,omitempty"`
	Path     string `json:"path,omitempty"`
	Error    string `json:"error,omitempty"`
}

type sweeperStatusSnapshot struct {
	Status              string                  `json:"status"`
	Timestamp           string                  `json:"timestamp,omitempty"`
	RoundNo             int                     `json:"round_no"`
	CumulativeEvaluated int                     `json:"cumulative_evaluated"`
	Message             string                  `json:"message,omitempty"`
	Error               string                  `json:"error,omitempty"`
	Candidates          []sweeperCandidateEntry `json:"candidates,omitempty"`
}

// renderedManifest is the minimal shape read back out of a candidate's
// inlined full-manifest YAML -- only .spec is needed to build the DGDC's
// Spec and to compute the candidate's content-addressed identity; apiVersion
// /kind/metadata are the rendered DGD's own, not the DGDC's.
type renderedManifest struct {
	Spec nvidiacomv1beta1.DynamoGraphDeploymentSpec `json:"spec"`
}

const (
	// SweeperStatusConfigMapKey is the ConfigMap data key the sidecar relays
	// sweeper_status.yaml under -- must match
	// dynamo.aisimulate.output.dgd.kube_status.STATUS_FILE_NAME exactly,
	// since that Python constant is what names the file the shell template
	// in sidecarScriptTemplate greps for and blob-dumps verbatim.
	SweeperStatusConfigMapKey = "sweeper_status.yaml"

	// DGDCIdentityLabel records the content-addressed identity
	// (dgdcreconcile.ComputeIdentity) a DGDC was created with, so a later
	// reconcile can read it back without re-deriving it from a live Spec
	// (see dgdcreconcile.CurrentDGDC's doc comment on why identity is never
	// recomputed from cluster state).
	DGDCIdentityLabel = "nvidia.com/dgdc-identity"

	// DGDCOwnerDGDRLabel records which DGDR a DGDC belongs to, so List calls
	// can be scoped to one DGDR's candidates without relying on an owner
	// reference index. Applied by this reconciler at Create time.
	DGDCOwnerDGDRLabel = "nvidia.com/dgdr-name"
)

// errSweeperStatusNotRelayedYet means the output ConfigMap exists but the
// sidecar hasn't relayed a sweeper_status.yaml blob into it yet (or this DGDR
// isn't running a Sweeper(v2) search at all) -- not an error, just "nothing
// to reconcile this pass".
var errSweeperStatusNotRelayedYet = errors.New("sweeper_status.yaml not yet relayed")

// DynamoGraphDeploymentCandidateReconciler reconciles the set of
// DynamoGraphDeploymentCandidate (DGDC) objects for one DGDR's Sweeper(v2)
// run against the live-progress state relayed into its output ConfigMap.
//
// This is deliberately its own controller rather than another branch of
// DynamoGraphDeploymentRequestReconciler.Reconcile's phase switch: that
// switch is v1beta1's single-shot "profile once, generate one DGD spec"
// model (DGDRPhasePending -> Profiling -> Deploying -> ...), which doesn't
// fit a long-running search that continuously creates, re-ranks, and retires
// many candidates while it runs. Once DynamoGraphDeploymentRun (the real
// v1beta2 type from PRs #13603/#13744) lands, this should watch that type
// instead of DynamoGraphDeploymentRequest -- the switch below is written
// against v1beta1.DynamoGraphDeploymentRequest only because that's the real,
// already-merged type with a Namespace/Name identity and the existing
// getOutputConfigMapName/ConfigMapOutputPrefix convention this reconciler
// needs to find the right ConfigMap; nothing else here depends on DGDR's
// v1beta1-specific fields.
type DynamoGraphDeploymentCandidateReconciler struct {
	client.Client
	Recorder events.EventRecorder
}

// Reconcile implements reconcile.Reconciler. req names a
// DynamoGraphDeploymentRequest (see the type doc comment for why); it is
// triggered by the output ConfigMap watch wired in SetupWithManager, so it
// fires on every sidecar relay, not on a fixed poll interval.
func (r *DynamoGraphDeploymentCandidateReconciler) Reconcile(ctx context.Context, req reconcile.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	if err := r.Get(ctx, req.NamespacedName, dgdr); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, fmt.Errorf("failed to get DGDR %s: %w", req.NamespacedName, err)
	}

	snapshot, err := r.readSweeperStatus(ctx, dgdr)
	if err != nil {
		if errors.Is(err, errSweeperStatusNotRelayedYet) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	desired, err := desiredCandidatesFromSnapshot(snapshot)
	if err != nil {
		// A malformed relayed manifest is a data problem with this round's
		// output, not an infrastructure error -- requeueing immediately
		// would just see the same bad blob again. Record it and wait for
		// the next relay (a new round, or a retry) to produce a better one.
		logger.Error(err, "Failed to parse relayed sweeper candidates; waiting for next relay", "dgdr", dgdr.Name)
		r.Recorder.Eventf(dgdr, nil, corev1.EventTypeWarning, "DGDCParseFailed", "Reconcile", "%s", err.Error())
		return ctrl.Result{}, nil
	}

	current, err := r.listCurrentDGDCs(ctx, dgdr)
	if err != nil {
		return ctrl.Result{}, err
	}

	actions, err := dgdcreconcile.ComputeActions(desired, current)
	if err != nil {
		var diffErr *dgdcreconcile.DiffInputError
		if errors.As(err, &diffErr) {
			// Terminal per dgdcreconcile's own contract (see ComputeActions'
			// doc comment): duplicate identities mean the diff cannot safely
			// decide ownership. Surface it and stop -- retrying the same
			// malformed input won't fix it either.
			logger.Error(err, "Malformed candidate set; cannot reconcile DGDCs", "dgdr", dgdr.Name)
			r.Recorder.Eventf(dgdr, nil, corev1.EventTypeWarning, "DGDCDiffFailed", "Reconcile", "%s", err.Error())
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if err := r.applyActions(ctx, dgdr, actions); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info("Reconciled DGDC candidates", "dgdr", dgdr.Name,
		"creates", len(actions.Creates), "deletes", len(actions.Deletes), "statusUpdates", len(actions.StatusUpdates))
	return ctrl.Result{}, nil
}

// readSweeperStatus fetches and parses the relayed sweeper_status.yaml blob
// from this DGDR's output ConfigMap. Mirrors generateDGDSpec's existing
// read-ConfigMap pattern.
func (r *DynamoGraphDeploymentCandidateReconciler) readSweeperStatus(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (*sweeperStatusSnapshot, error) {
	outputConfigMapName := getOutputConfigMapName(dgdr)
	cm := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Name: outputConfigMapName, Namespace: dgdr.Namespace}, cm); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, errSweeperStatusNotRelayedYet
		}
		return nil, fmt.Errorf("failed to get output ConfigMap %s: %w", outputConfigMapName, err)
	}

	raw, exists := cm.Data[SweeperStatusConfigMapKey]
	if !exists {
		return nil, errSweeperStatusNotRelayedYet
	}

	var snapshot sweeperStatusSnapshot
	if err := sigsyaml.Unmarshal([]byte(raw), &snapshot); err != nil {
		return nil, fmt.Errorf("failed to parse %s from ConfigMap %s: %w", SweeperStatusConfigMapKey, outputConfigMapName, err)
	}
	return &snapshot, nil
}

// desiredCandidatesFromSnapshot converts the relayed, best-first candidate
// list into dgdcreconcile.DesiredCandidate, deriving each one's one-based
// Rank from list position.
//
// This depends on DGDRAdapter._write_snapshot sorting `candidates` by score
// before serializing (fixed alongside this reconciler -- it previously
// iterated a plain dict in insertion/update order, which is not rank order).
// Materialization-failed entries carry no spec and are skipped: they were
// never a candidate for creation, so they correctly fall out of the desired
// set rather than producing a Create with an empty Spec.
//
// Experimental is left empty for every candidate: kube_status.
// CandidateStatusEntry does not currently carry a separate experimental-context
// field alongside manifest, so there's nothing yet to distinguish identical
// manifests rendered under different experimental context (e.g. PR #14993's
// runtime-binding diagnostics). ComputeIdentity still hashes an explicit
// (always-empty-for-now) experimental map rather than omitting it, so that
// once the adapter does emit one, only this conversion needs to change.
func desiredCandidatesFromSnapshot(snapshot *sweeperStatusSnapshot) ([]dgdcreconcile.DesiredCandidate, error) {
	desired := make([]dgdcreconcile.DesiredCandidate, 0, len(snapshot.Candidates))
	rank := int32(0)
	for _, entry := range snapshot.Candidates {
		if entry.Outcome != sweeperCandidateOutcomeMaterialized {
			continue
		}
		if entry.Manifest == "" {
			return nil, fmt.Errorf("candidate %q is materialized but carries no manifest", entry.ID)
		}

		var manifest renderedManifest
		if err := sigsyaml.Unmarshal([]byte(entry.Manifest), &manifest); err != nil {
			return nil, fmt.Errorf("failed to parse manifest for candidate %q: %w", entry.ID, err)
		}

		specMap, err := toStringAnyMap(manifest.Spec)
		if err != nil {
			return nil, fmt.Errorf("failed to normalize spec for candidate %q: %w", entry.ID, err)
		}

		rank++
		desired = append(desired, dgdcreconcile.DesiredCandidate{
			Spec:         specMap,
			Experimental: map[string]any{},
			Rank:         ptr.To(rank),
		})
	}
	return desired, nil
}

// toStringAnyMap round-trips a typed struct through JSON into a
// map[string]any, so dgdcreconcile.ComputeIdentity's canonical-JSON hashing
// sees the same shape regardless of whether a caller built it from a live
// typed Spec (this reconciler) or a hand-built map (dgdcreconcile's own
// tests) -- the package is deliberately typed-API-independent (see diff.go's
// package doc comment), so this conversion belongs here, not in dgdcreconcile.
func toStringAnyMap(spec nvidiacomv1beta1.DynamoGraphDeploymentSpec) (map[string]any, error) {
	raw, err := json.Marshal(spec)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// listCurrentDGDCs lists the DGDCs this reconciler previously created for
// dgdr, keyed by the identity label each was created with.
func (r *DynamoGraphDeploymentCandidateReconciler) listCurrentDGDCs(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) ([]dgdcreconcile.CurrentDGDC, error) {
	list := &v1beta2.DynamoGraphDeploymentCandidateList{}
	if err := r.List(ctx, list,
		client.InNamespace(dgdr.Namespace),
		client.MatchingLabels{DGDCOwnerDGDRLabel: dgdr.Name},
	); err != nil {
		return nil, fmt.Errorf("failed to list DynamoGraphDeploymentCandidates for DGDR %s: %w", dgdr.Name, err)
	}

	current := make([]dgdcreconcile.CurrentDGDC, 0, len(list.Items))
	for _, item := range list.Items {
		identity := item.Labels[DGDCIdentityLabel]
		if identity == "" {
			// Not one of ours (or created before this label existed) --
			// skip rather than guess; ComputeActions would otherwise see a
			// bogus empty-string identity and could misreport it as a
			// duplicate against a legitimately-unset case.
			continue
		}
		current = append(current, dgdcreconcile.CurrentDGDC{
			Name:     item.Name,
			Identity: identity,
			Rank:     item.Status.Rank,
		})
	}
	return current, nil
}

// applyActions performs the Create/Delete/status-Patch calls dgdcreconcile
// decided on. Each Create embeds its own identity so a concurrent duplicate
// create (e.g. two reconciles racing on a slow informer cache) fails on the
// label-derived deterministic name below rather than producing two live
// DGDCs with the same identity -- ComputeActions' own duplicate-identity
// guard only protects the maps it is given, not the cluster.
func (r *DynamoGraphDeploymentCandidateReconciler) applyActions(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, actions dgdcreconcile.Actions) error {
	logger := log.FromContext(ctx)

	for _, candidate := range actions.Creates {
		identity, err := candidate.Identity()
		if err != nil {
			return fmt.Errorf("failed to compute identity for a candidate to create: %w", err)
		}

		var spec nvidiacomv1beta1.DynamoGraphDeploymentSpec
		specJSON, err := json.Marshal(candidate.Spec)
		if err != nil {
			return fmt.Errorf("failed to marshal spec for candidate %s: %w", identity, err)
		}
		if err := json.Unmarshal(specJSON, &spec); err != nil {
			return fmt.Errorf("failed to decode spec for candidate %s: %w", identity, err)
		}

		dgdc := &v1beta2.DynamoGraphDeploymentCandidate{
			ObjectMeta: metav1.ObjectMeta{
				// Identity-derived, not generated: makes Create idempotent
				// across reconciles/replays (a retry after a successful but
				// unobserved Create hits AlreadyExists instead of creating
				// a second object for the same candidate).
				Name:      dgdcName(dgdr.Name, identity),
				Namespace: dgdr.Namespace,
				Labels: map[string]string{
					DGDCOwnerDGDRLabel: dgdr.Name,
					DGDCIdentityLabel:  identity,
				},
			},
			Spec: spec,
		}
		if candidate.Rank != nil {
			dgdc.Status.Rank = candidate.Rank
		}
		if err := ctrl.SetControllerReference(dgdr, dgdc, r.Scheme()); err != nil {
			return fmt.Errorf("failed to set owner reference on DGDC %s: %w", dgdc.Name, err)
		}

		if err := r.Create(ctx, dgdc); err != nil {
			if apierrors.IsAlreadyExists(err) {
				logger.Info("DGDC already exists, skipping create", "name", dgdc.Name)
				continue
			}
			return fmt.Errorf("failed to create DGDC %s: %w", dgdc.Name, err)
		}
		// The initial Rank set above is part of the create payload for
		// convenience, but Status is a separate subresource on a real CRD
		// -- some API servers/clients don't persist Status from the main
		// Create call. Patch it explicitly so a scalar goal's first-seen
		// Rank is never silently dropped.
		if candidate.Rank != nil {
			if err := r.Status().Update(ctx, dgdc); err != nil {
				return fmt.Errorf("failed to set initial rank on DGDC %s: %w", dgdc.Name, err)
			}
		}
	}

	for _, name := range actions.Deletes {
		dgdc := &v1beta2.DynamoGraphDeploymentCandidate{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: dgdr.Namespace},
		}
		if err := r.Delete(ctx, dgdc); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete DGDC %s: %w", name, err)
		}
	}

	for _, update := range actions.StatusUpdates {
		dgdc := &v1beta2.DynamoGraphDeploymentCandidate{}
		if err := r.Get(ctx, types.NamespacedName{Name: update.Name, Namespace: dgdr.Namespace}, dgdc); err != nil {
			if apierrors.IsNotFound(err) {
				// Raced with an external delete; nothing to update.
				continue
			}
			return fmt.Errorf("failed to get DGDC %s for status update: %w", update.Name, err)
		}
		dgdc.Status.Rank = update.NewRank
		if err := r.Status().Update(ctx, dgdc); err != nil {
			return fmt.Errorf("failed to update rank on DGDC %s: %w", update.Name, err)
		}
	}

	return nil
}

// dgdcName derives a stable, DNS-1123-safe DGDC name from the owning DGDR's
// name and the candidate's content-addressed identity. The identity is
// already a lowercase hex string (see dgdcreconcile.ComputeIdentity), so no
// further sanitizing is needed; the result is deterministic across
// reconciles, which is what makes Create idempotent.
func dgdcName(dgdrName, identity string) string {
	name := fmt.Sprintf("%s-%s", dgdrName, identity)
	// K8s object names are capped at 253 chars; stay well clear of that
	// without needing to special-case truncation collisions, since
	// identity's full 16 hex chars are always retained verbatim.
	const maxLen = 200
	if len(name) > maxLen {
		name = fmt.Sprintf("%s-%s", strings.TrimSuffix(dgdrName[:maxLen-17], "-"), identity)
	}
	return name
}

// SetupWithManager wires this reconciler to fire whenever a DGDR's output
// ConfigMap changes (the sidecar relay this whole feature depends on), by
// mapping the ConfigMap back to its owning DGDR's NamespacedName via the
// ConfigMapOutputPrefix naming convention -- the same one
// getOutputConfigMapName already uses, so no new owner-reference wiring is
// needed on the ConfigMap itself.
//
// Deliberately NOT also calling Owns(&v1beta2.DynamoGraphDeploymentCandidate{}):
// that would require the placeholder v1beta2 scheme to be registered with
// the manager's scheme builder, which belongs in main.go, not here -- left
// as a one-line TODO for whoever wires this reconciler into main.go, since
// this PR does not modify main.go's scheme registration.
func (r *DynamoGraphDeploymentCandidateReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("dgdc-candidate").
		For(&nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
		Watches(
			&corev1.ConfigMap{},
			handler.EnqueueRequestsFromMapFunc(mapOutputConfigMapToDGDRRequest),
		).
		Complete(r)
}

// mapOutputConfigMapToDGDRRequest maps a ConfigMap event back to its owning
// DGDR's NamespacedName via the ConfigMapOutputPrefix naming convention (the
// same one getOutputConfigMapName uses to build the name in the first
// place), so no owner-reference wiring on the ConfigMap itself is needed.
// A ConfigMap whose name doesn't carry the prefix isn't one of ours -- nil
// means "nothing to enqueue", not an error. Extracted as a named, directly
// testable function rather than inlined in SetupWithManager.
func mapOutputConfigMapToDGDRRequest(_ context.Context, obj client.Object) []reconcile.Request {
	name, ok := strings.CutPrefix(obj.GetName(), ConfigMapOutputPrefix)
	if !ok || name == "" {
		return nil
	}
	return []reconcile.Request{{NamespacedName: types.NamespacedName{
		Name:      name,
		Namespace: obj.GetNamespace(),
	}}}
}
