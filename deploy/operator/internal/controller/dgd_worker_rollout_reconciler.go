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
	"fmt"
	"slices"
	"sort"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

type workerGenerationHashes struct {
	v1 string
	v2 string
}

// unsupportedWorkerHashTransition is the next DGD worker-hash state for a
// pathway that cannot use managed rolling updates. Planning it is read-only;
// callers commit it only after they have reconciled the workload carrying the
// corresponding generation.
type unsupportedWorkerHashTransition struct {
	next                    workerGenerationHashes
	initialize              bool
	workerGenerationChanged bool
}

func (t unsupportedWorkerHashTransition) needsCommit() bool {
	return t.initialize || t.workerGenerationChanged
}

// dgdWorkerRolloutReconciler owns worker-generation metadata and the managed
// rolling-update state machine. It carries the Kubernetes read/write access
// required by that state machine and event recording, never the complete DGD
// controller.
type dgdWorkerRolloutReconciler struct {
	dgdResourceSyncer
}

func newDGDWorkerRolloutReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
) *dgdWorkerRolloutReconciler {
	return &dgdWorkerRolloutReconciler{
		dgdResourceSyncer: newDGDResourceSyncer(kubeClient, recorder),
	}
}

func (r *dgdWorkerRolloutReconciler) planUnsupportedWorkerHashTransition(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (unsupportedWorkerHashTransition, error) {
	desired, err := desiredWorkerHashes(dgd)
	if err != nil {
		return unsupportedWorkerHashTransition{}, err
	}

	current := r.currentWorkerHashes(dgd)
	if current.empty() {
		return unsupportedWorkerHashTransition{
			next:       workerHashesForCompletedGeneration(desired.v2, desired),
			initialize: true,
		}, nil
	}
	if currentWorkerHashesMatchDesired(current, desired) {
		return unsupportedWorkerHashTransition{}, nil
	}
	return unsupportedWorkerHashTransition{
		next:                    r.workerHashesForUnsupportedPathway(dgd, desired),
		workerGenerationChanged: true,
	}, nil
}

// commitUnsupportedWorkerHashTransition records a transition planned from the
// DGD observation used to render the workload. An optimistic-update conflict
// must be retried from a fresh DGD and PCS observation.
func (r *dgdWorkerRolloutReconciler) commitUnsupportedWorkerHashTransition(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	transition unsupportedWorkerHashTransition,
	isGrove bool,
) error {
	if !transition.needsCommit() {
		return nil
	}

	r.setCurrentWorkerHashes(dgd, transition.next)
	if err := r.Update(ctx, dgd); err != nil {
		return err
	}
	if !transition.workerGenerationChanged {
		return nil
	}

	log.FromContext(ctx).Info(
		"Worker spec change detected but rolling update not supported for this pathway",
		"isGrove", isGrove,
		"hasMultinode", dgd.HasAnyMultinodeComponent(),
	)
	if r.recorder != nil {
		r.recorder.Eventf(
			dgd,
			nil,
			corev1.EventTypeWarning,
			"RollingUpdateNotSupported",
			"Update",
			"%s",
			"Worker spec changed but custom rolling updates are not supported for Grove/multinode deployments",
		)
	}
	return nil
}

func (h workerGenerationHashes) empty() bool {
	return h.v1 == "" && h.v2 == ""
}

func (h workerGenerationHashes) v1Only() bool {
	return h.v1 != "" && h.v2 == ""
}

func (h workerGenerationHashes) contains(hash string) bool {
	if hash == "" {
		return false
	}
	return hash == h.v1 || hash == h.v2
}

func desiredWorkerHashes(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (workerGenerationHashes, error) {
	v2Hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return workerGenerationHashes{}, fmt.Errorf("failed to compute v2 worker hash: %w", err)
	}

	// Preserve v1 as the opaque suffix of an existing worker generation.
	current := currentWorkerHashes(dgd)
	v1Hash := current.v1
	if v1Hash == consts.LegacyWorkerHash {
		v1Hash = v2Hash
	}

	return workerGenerationHashes{v1: v1Hash, v2: v2Hash}, nil
}

func (r *dgdWorkerRolloutReconciler) currentWorkerHashes(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workerGenerationHashes {
	return currentWorkerHashes(dgd)
}

func currentWorkerHashes(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workerGenerationHashes {
	return workerGenerationHashes{
		v1: currentWorkerHash(dgd),
		v2: currentWorkerHashV2(dgd),
	}
}

func currentWorkerHashesMatchDesired(current, desired workerGenerationHashes) bool {
	if current.empty() {
		return true
	}
	if current.v1 != "" {
		if current.v1 != desired.v1 {
			return false
		}
		return current.v2 == "" || current.v2 == desired.v2
	}
	return current.v2 == desired.v2
}

func workerHashForDCDGeneration(current, desired workerGenerationHashes) string {
	if current.v1 != "" {
		if current.v1 == desired.v1 {
			if current.v2 == "" || current.v2 == desired.v2 {
				return desired.v1
			}
			return desired.v2
		}
		return desired.v1
	}
	if current.v2 != "" {
		return desired.v2
	}
	return desired.v1
}

func workerHashesForCompletedGeneration(newWorkerHash string, desired workerGenerationHashes) workerGenerationHashes {
	if newWorkerHash == desired.v2 {
		return workerGenerationHashes{v2: desired.v2}
	}
	return desired
}

func (r *dgdWorkerRolloutReconciler) workerHashesForUnsupportedPathway(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired workerGenerationHashes,
) workerGenerationHashes {
	newWorkerHash := r.activeWorkerHashForDCDGeneration(dgd, desired)
	return workerHashesForCompletedGeneration(newWorkerHash, desired)
}

// shouldTriggerRollingUpdate compares desired worker hashes with the active
// generation recorded on the DGD.
//
// During v1/v2 compatibility a worker DCD is current if its worker-hash label
// matches either current-worker-hash (v1) or current-worker-hash-v2. This keeps
// the existing annotation/label meaning downgrade-safe while allowing the
// controller to record the v2 hash that will become primary later.
func (r *dgdWorkerRolloutReconciler) shouldTriggerRollingUpdate(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	desired, err := desiredWorkerHashes(dgd)
	if err != nil {
		return false, err
	}

	current := r.currentWorkerHashes(dgd)
	return !currentWorkerHashesMatchDesired(current, desired), nil
}

// initializeWorkerHashIfNeeded establishes the DGD's active worker generation.
// New DGDs store only the canonical v2 worker hash. DGDs created before
// managed rolling updates may already have worker DCDs without a hash label; in
// that case we label those DCDs with the legacy sentinel and let the normal
// rolling update path migrate from that sentinel to the desired compatibility hash.
func (r *dgdWorkerRolloutReconciler) initializeWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	if !r.currentWorkerHashes(dgd).empty() {
		return r.migrateCurrentWorkerHashIfNeeded(ctx, dgd)
	}

	// Check for legacy (pre-rolling-update) worker DCDs
	legacyDCDs, err := r.findLegacyWorkerDCDs(ctx, dgd)
	if err != nil {
		return fmt.Errorf("failed to check for legacy worker DCDs: %w", err)
	}

	if len(legacyDCDs) > 0 {
		logger.Info("Found legacy worker DCDs without hash label, initiating migration",
			"count", len(legacyDCDs))

		// Backfill hash label on legacy DCDs so they're manageable by the rolling update machinery
		for i := range legacyDCDs {
			dcd := &legacyDCDs[i]
			patch := client.MergeFrom(dcd.DeepCopy())
			if dcd.Labels == nil {
				dcd.Labels = make(map[string]string)
			}
			dcd.Labels[consts.KubeLabelDynamoWorkerHash] = consts.LegacyWorkerHash
			if err := r.Patch(ctx, dcd, patch); err != nil {
				return fmt.Errorf("failed to backfill hash label on legacy DCD %s: %w", dcd.Name, err)
			}
			logger.Info("Backfilled worker hash label on legacy DCD",
				"dcdName", dcd.Name, "hash", consts.LegacyWorkerHash)
		}

		// Set sentinel hash — next reconcile triggers a real rolling update from "legacy" -> computed hash
		r.setLegacyWorkerHash(dgd)
		if err := r.Update(ctx, dgd); err != nil {
			return fmt.Errorf("failed to set legacy worker hash: %w", err)
		}

		if r.recorder != nil {
			r.recorder.Eventf(dgd, nil, corev1.EventTypeNormal, "LegacyMigrationStarted", "Update",
				"Detected %d legacy worker DCDs, initiating rolling update migration", len(legacyDCDs))
		}
		return nil
	}

	// Normal first deploy — set the canonical v2 hash.
	hashes, err := desiredWorkerHashes(dgd)
	if err != nil {
		return err
	}
	r.setCurrentWorkerHashes(dgd, workerHashesForCompletedGeneration(hashes.v2, hashes))

	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize worker hash: %w", err)
	}

	logger.Info("Initialized current worker hashes", "v1Hash", hashes.v1, "v2Hash", hashes.v2)

	return nil
}

// migrateCurrentWorkerHashIfNeeded fills in additive v2 worker-hash state while
// the v1 hash still represents the active worker generation.
func (r *dgdWorkerRolloutReconciler) migrateCurrentWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	current := r.currentWorkerHashes(dgd)
	if !current.v1Only() || current.v1 == consts.LegacyWorkerHash {
		return nil
	}

	// Let an active v1 rollout converge on v2 before recording it as current.
	if isRollingUpdateInProgress(&dgd.Status) {
		return nil
	}

	desired, err := desiredWorkerHashes(dgd)
	if err != nil {
		return err
	}

	// Record the current desired state as v2 without changing the active v1 suffix.
	next := workerGenerationHashes{v1: current.v1, v2: desired.v2}
	r.setCurrentWorkerHashes(dgd, next)
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to migrate worker hash annotations: %w", err)
	}
	logger.Info("Migrated worker hash annotations",
		"v1Hash", next.v1,
		"v2Hash", next.v2)
	if r.recorder != nil {
		r.recorder.Eventf(dgd, nil, corev1.EventTypeNormal, "WorkerHashMigrated", "Update",
			"Recorded v2 worker hash annotation without rolling workers")
	}

	return nil
}

// activeWorkerHashForDCDGeneration returns the hash used for generated worker
// DCD names and worker-hash labels in this reconcile. Existing bridge generations
// keep their v1 identity until a worker change selects v2. Already v2-labeled
// generations preserve that value.
func (r *dgdWorkerRolloutReconciler) activeWorkerHashForDCDGeneration(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired workerGenerationHashes,
) string {
	return activeWorkerHashForDCDGeneration(dgd, desired)
}

func activeWorkerHashForDCDGeneration(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired workerGenerationHashes,
) string {
	return activeWorkerHashCandidates(dgd, desired)[0]
}

func activeWorkerHashCandidates(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired workerGenerationHashes,
) []string {
	current := currentWorkerHashes(dgd)
	candidates := make([]string, 0, 2)
	generated := workerHashForDCDGeneration(current, desired)

	// Retarget an active v1-only rollout to its canonical v2 generation.
	if current.v1Only() && isRollingUpdateInProgress(&dgd.Status) {
		generated = desired.v2
	}

	candidates = append(candidates, generated)
	if desired.v1 != "" && current.v1 == desired.v1 &&
		(current.v2 == "" || current.v2 == desired.v2) && desired.v1 != generated {
		candidates = append(candidates, desired.v1)
	}
	if current.contains(desired.v2) && desired.v2 != generated && desired.v2 != desired.v1 {
		candidates = append(candidates, desired.v2)
	}
	return candidates
}

// findLegacyWorkerDCDs returns worker DCDs owned by this DGD that lack the worker hash label.
// These are DCDs created by a pre-rolling-update operator version.
func (r *dgdWorkerRolloutReconciler) findLegacyWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	// List all DCDs for this DGD
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, fmt.Errorf("failed to list DCDs for DGD %s: %w", dgd.Name, err)
	}

	var legacyDCDs []nvidiacomv1beta1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			continue
		}
		// Legacy DCDs lack the worker hash label
		if dcd.Labels[consts.KubeLabelDynamoWorkerHash] == "" {
			legacyDCDs = append(legacyDCDs, dcd)
		}
	}

	return legacyDCDs, nil
}

type managedWorkerInventory struct {
	desiredHash string
	desired     map[string]*nvidiacomv1beta1.DynamoComponentDeployment
	targets     map[string]*nvidiacomv1beta1.DynamoComponentDeployment
	old         []nvidiacomv1beta1.DynamoComponentDeployment
}

func (i managedWorkerInventory) targetComplete() bool {
	return len(i.desired) == len(i.targets)
}

type managedWorkerRolloutPlan struct {
	inventory managedWorkerInventory
	context   dynamo.RollingUpdateContext
}

type workerDCDIdentityCollisionError struct {
	component string
	name      string
	detail    string
}

func (e *workerDCDIdentityCollisionError) Error() string {
	return fmt.Sprintf("worker DCD identity collision for component %q at %q: %s", e.component, e.name, e.detail)
}

// buildManagedWorkerRolloutPlan renders the canonical v2 target then compares
// it with DCDs owned by this DGD UID. Parent hash annotations are deliberately
// not an input: their only role is compatibility projection after convergence.
func (r *dgdWorkerRolloutReconciler) buildManagedWorkerRolloutPlan(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (managedWorkerRolloutPlan, error) {
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return managedWorkerRolloutPlan{}, fmt.Errorf("compute desired worker hash: %w", err)
	}

	rendered, err := dynamo.GenerateDynamoComponentsDeployments(
		dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash},
	)
	if err != nil {
		return managedWorkerRolloutPlan{}, fmt.Errorf("render desired worker DCDs: %w", err)
	}

	inventory := managedWorkerInventory{
		desiredHash: hash,
		desired:     make(map[string]*nvidiacomv1beta1.DynamoComponentDeployment),
		targets:     make(map[string]*nvidiacomv1beta1.DynamoComponentDeployment),
	}
	for componentName, dcd := range rendered {
		if dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			inventory.desired[componentName] = dcd
		}
	}

	observed, err := r.listOwnedWorkerDCDs(ctx, dgd)
	if err != nil {
		return managedWorkerRolloutPlan{}, err
	}
	matched := make(map[string]struct{}, len(observed))
	for componentName, expected := range inventory.desired {
		matches := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, 1)
		for i := range observed {
			if dynamo.GetDCDComponentName(&observed[i]) != componentName {
				continue
			}
			if workerDCDSemanticallyEqual(expected, &observed[i]) {
				matches = append(matches, &observed[i])
			}
		}
		switch len(matches) {
		case 0:
			if err := r.rejectDesiredWorkerDCDNameCollision(ctx, dgd, componentName, expected); err != nil {
				return managedWorkerRolloutPlan{}, err
			}
		case 1:
			if matches[0].Name != expected.Name {
				if err := r.rejectDesiredWorkerDCDNameCollision(ctx, dgd, componentName, expected); err != nil {
					return managedWorkerRolloutPlan{}, err
				}
			}
			inventory.targets[componentName] = matches[0]
			matched[matches[0].Name] = struct{}{}
		default:
			return managedWorkerRolloutPlan{}, &workerDCDIdentityCollisionError{
				component: componentName,
				name:      expected.Name,
				detail:    fmt.Sprintf("found %d semantically matching owned DCDs", len(matches)),
			}
		}
	}
	for i := range observed {
		if _, ok := matched[observed[i].Name]; !ok {
			inventory.old = append(inventory.old, observed[i])
		}
	}

	plan := managedWorkerRolloutPlan{inventory: inventory}
	plan.context = dynamo.RollingUpdateContext{
		NewWorkerHash:                      inventory.desiredHash,
		WorkerHashByComponent:              make(map[string]string, len(inventory.desired)),
		TargetDCDNames:                     make(map[string]string, len(inventory.desired)),
		OldWorkerDCDNames:                  make(map[string]struct{}, len(inventory.old)),
		ObservedOldWorkerDCDs:              make(map[string]*nvidiacomv1beta1.DynamoComponentDeployment, len(inventory.old)),
		TargetComplete:                     inventory.targetComplete(),
		MayMutateOld:                       inventory.targetComplete(),
		OldWorkerReplicaTargetsByComponent: make(map[string]int32),
		OldWorkerReplicaTargetsByDCD:       make(map[string]int32),
		NewWorkerReplicaTargetsByComponent: make(map[string]int32),
	}
	for componentName, expected := range inventory.desired {
		plan.context.WorkerHashByComponent[componentName] = inventory.desiredHash
		plan.context.TargetDCDNames[componentName] = expected.Name
	}
	for componentName, target := range inventory.targets {
		plan.context.WorkerHashByComponent[componentName] = target.Labels[consts.KubeLabelDynamoWorkerHash]
		plan.context.TargetDCDNames[componentName] = target.Name
	}
	for i := range inventory.old {
		plan.context.OldWorkerDCDNames[inventory.old[i].Name] = struct{}{}
		plan.context.ObservedOldWorkerDCDs[inventory.old[i].Name] = inventory.old[i].DeepCopy()
	}

	if len(inventory.desired) == 0 {
		plan.context.MayMutateOld = true
		plan.context.WorkerInventoryInProgress = len(inventory.old) > 0
		for i := range inventory.old {
			componentName := dynamo.GetDCDComponentName(&inventory.old[i])
			plan.context.OldWorkerReplicaTargetsByComponent[componentName] = 0
			plan.context.OldWorkerReplicaTargetsByDCD[inventory.old[i].Name] = 0
		}
		return plan, nil
	}

	if !plan.context.TargetComplete {
		plan.context.WorkerInventoryInProgress = true
		return plan, nil
	}
	if len(inventory.old) > 0 {
		if err := r.populateManagedOldWorkerTargets(ctx, dgd, &plan); err != nil {
			return managedWorkerRolloutPlan{}, err
		}
	}
	plan.context.WorkerInventoryInProgress = len(inventory.old) > 0 || !managedWorkerTargetsReady(inventory)
	return plan, nil
}

func (r *dgdWorkerRolloutReconciler) listOwnedWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	list := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.List(ctx, list, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
	}); err != nil {
		return nil, fmt.Errorf("list worker DCDs for %s/%s: %w", dgd.Namespace, dgd.Name, err)
	}
	workers := make([]nvidiacomv1beta1.DynamoComponentDeployment, 0, len(list.Items))
	for i := range list.Items {
		dcd := &list.Items[i]
		if dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) && metav1.IsControlledBy(dcd, dgd) {
			workers = append(workers, *dcd)
		}
	}
	return workers, nil
}

func (r *dgdWorkerRolloutReconciler) rejectDesiredWorkerDCDNameCollision(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	expected *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := r.Get(ctx, types.NamespacedName{Name: expected.Name, Namespace: dgd.Namespace}, existing)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("get desired worker DCD %s: %w", expected.Name, err)
	}
	if metav1.IsControlledBy(existing, dgd) && workerDCDSemanticallyEqual(expected, existing) {
		return nil
	}
	return &workerDCDIdentityCollisionError{
		component: componentName,
		name:      expected.Name,
		detail:    "the deterministic desired name is occupied by a nonmatching DCD",
	}
}

func workerDCDSemanticallyEqual(
	expected, observed *nvidiacomv1beta1.DynamoComponentDeployment,
) bool {
	normalizedExpected := normalizeWorkerDCDForComparison(expected)
	normalizedObserved := normalizeWorkerDCDForComparison(observed)
	if hasControllerManagedCheckpointMetadata(expected) || hasControllerManagedCheckpointMetadata(observed) {
		// waitForCheckpoint temporarily scales the rendered DCD to zero. That
		// gate is controller progress, not a distinct worker generation.
		normalizedObserved.Spec.Replicas = normalizedExpected.Spec.Replicas
	}
	return apiequality.Semantic.DeepEqual(normalizedExpected, normalizedObserved)
}

func normalizeWorkerDCDForComparison(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) *nvidiacomv1beta1.DynamoComponentDeployment {
	if dcd == nil {
		return nil
	}
	normalized := dcd.DeepCopy()
	normalized.TypeMeta = metav1.TypeMeta{}
	normalized.Name = ""
	normalized.UID = ""
	normalized.ResourceVersion = ""
	normalized.Generation = 0
	normalized.CreationTimestamp = metav1.Time{}
	normalized.DeletionTimestamp = nil
	normalized.DeletionGracePeriodSeconds = nil
	normalized.OwnerReferences = nil
	normalized.ManagedFields = nil
	normalized.Status = nvidiacomv1beta1.DynamoComponentDeploymentStatus{}
	if normalized.Spec.Experimental != nil {
		normalized.Spec.Experimental.Checkpoint = nil
		if apiequality.Semantic.DeepEqual(normalized.Spec.Experimental, &nvidiacomv1beta1.ExperimentalSpec{}) {
			normalized.Spec.Experimental = nil
		}
	}
	delete(normalized.Labels, consts.KubeLabelDynamoWorkerHash)

	podTemplate := normalized.Spec.PodTemplate
	if podTemplate == nil {
		return normalized
	}
	delete(podTemplate.Labels, consts.KubeLabelDynamoWorkerHash)
	delete(podTemplate.Annotations, consts.RestartAnnotation)
	delete(podTemplate.Annotations, consts.CheckpointRestoreCandidateAnnotation)
	delete(podTemplate.Annotations, consts.CheckpointNameAnnotation)
	delete(podTemplate.Annotations, consts.CheckpointStartupPolicyAnnotation)
	podTemplate.Spec.Containers = stripWorkerSuffixEnv(podTemplate.Spec.Containers)
	podTemplate.Spec.InitContainers = stripWorkerSuffixEnv(podTemplate.Spec.InitContainers)
	return normalized
}

func hasControllerManagedCheckpointMetadata(dcd *nvidiacomv1beta1.DynamoComponentDeployment) bool {
	if dcd == nil {
		return false
	}
	if dcd.Spec.Experimental != nil && dcd.Spec.Experimental.Checkpoint != nil {
		return true
	}
	podTemplate := dcd.Spec.PodTemplate
	if podTemplate == nil {
		return false
	}
	return podTemplate.Annotations[consts.CheckpointRestoreCandidateAnnotation] != "" ||
		podTemplate.Annotations[consts.CheckpointNameAnnotation] != "" ||
		podTemplate.Annotations[consts.CheckpointStartupPolicyAnnotation] != ""
}

func stripWorkerSuffixEnv(containers []corev1.Container) []corev1.Container {
	for i := range containers {
		env := containers[i].Env[:0]
		for j := range containers[i].Env {
			if containers[i].Env[j].Name != "DYN_NAMESPACE_WORKER_SUFFIX" {
				env = append(env, containers[i].Env[j])
			}
		}
		containers[i].Env = env
	}
	return containers
}

func managedWorkerTargetsReady(inventory managedWorkerInventory) bool {
	if !inventory.targetComplete() {
		return false
	}
	for componentName := range inventory.desired {
		target := inventory.targets[componentName]
		if target == nil || !workerDCDReady(target) {
			return false
		}
	}
	return true
}

func workerDCDReady(dcd *nvidiacomv1beta1.DynamoComponentDeployment) bool {
	if dcd == nil || dcd.Status.ObservedGeneration < dcd.Generation {
		return false
	}
	desired := int32(1)
	if dcd.Spec.Replicas != nil {
		desired = *dcd.Spec.Replicas
	}
	if desired == 0 {
		return true
	}
	return dcd.Status.Component != nil && dcd.Status.Component.AvailableReplicas != nil &&
		*dcd.Status.Component.AvailableReplicas >= desired
}

func (r *dgdWorkerRolloutReconciler) populateManagedOldWorkerTargets(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	plan *managedWorkerRolloutPlan,
) error {
	oldByComponent := make(map[string][]*nvidiacomv1beta1.DynamoComponentDeployment)
	oldStates := make(map[string]dcdComponentState)
	for i := range plan.inventory.old {
		dcd := &plan.inventory.old[i]
		componentName := dynamo.GetDCDComponentName(dcd)
		oldByComponent[componentName] = append(oldByComponent[componentName], dcd)
		state := dcdComponentStateFromDCD(dcd)
		aggregate := oldStates[componentName]
		aggregate.Spec += state.Spec
		aggregate.Available += state.Available
		aggregate.Actual += state.Actual
		oldStates[componentName] = aggregate
	}

	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			continue
		}
		componentName := spec.ComponentName
		desired := int32(1)
		if spec.Replicas != nil {
			desired = *spec.Replicas
		}
		newState := dcdComponentStateFromDCD(plan.inventory.targets[componentName])
		oldState := oldStates[componentName]
		annotations := dynamo.GetDGDComponentResourceAnnotations(dgd, componentName, spec)
		strategy := deploymentStrategyFromAnnotations(annotations)

		var oldTarget, newTarget int32
		if strategy == common.DeploymentStrategyRecreate {
			drained, err := r.oldWorkerComponentDrained(ctx, dgd, componentName, oldByComponent[componentName])
			if err != nil {
				return err
			}
			if drained {
				newTarget = desired
			}
		} else {
			maxSurge, maxUnavailable := resolveRollingUpdateParams(annotations, desired)
			minAvailable := desired - maxUnavailable
			newUnavailable := max(int32(0), newState.Spec-newState.Available)
			maxScaledDown := max(int32(0), (oldState.Spec+newState.Spec)-minAvailable-newUnavailable)
			oldUnhealthy := max(int32(0), oldState.Spec-oldState.Available)
			availableSurplus := max(int32(0), (oldState.Available+newState.Available)-minAvailable)
			oldTarget = max(int32(0), oldState.Spec-min(maxScaledDown, oldUnhealthy+availableSurplus))
			scaleUpBudget := max(int32(0), desired+maxSurge-oldState.Spec-newState.Spec)
			newTarget = min(desired, newState.Spec+scaleUpBudget)
		}
		plan.context.OldWorkerReplicaTargetsByComponent[componentName] = oldTarget
		plan.context.NewWorkerReplicaTargetsByComponent[componentName] = newTarget
		for dcdName, target := range allocateOldWorkerDCDReplicas(oldByComponent[componentName], oldTarget) {
			plan.context.OldWorkerReplicaTargetsByDCD[dcdName] = target
		}
	}
	return nil
}

func (r *dgdWorkerRolloutReconciler) reconcileManagedWorkerInventory(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	plan managedWorkerRolloutPlan,
) error {
	inventory := plan.inventory
	if len(inventory.desired) == 0 {
		if len(inventory.old) == 0 {
			if err := r.clearProjectedWorkerHashes(ctx, dgd); err != nil {
				return err
			}
			completeManagedRollingUpdate(status, nil)
			return nil
		}
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		drained, err := r.managedOldWorkersDrained(ctx, dgd, inventory.old)
		if err != nil {
			return err
		}
		if drained {
			return r.deleteObservedOldWorkerDCDs(ctx, inventory.old)
		}
		return nil
	}

	if !plan.context.TargetComplete {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhasePending)
		return nil
	}

	targetsReady := managedWorkerTargetsReady(inventory)
	if len(inventory.old) > 0 {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		status.RollingUpdate.UpdatedComponents = managedUpdatedWorkerComponents(dgd, inventory)
		if !targetsReady {
			return nil
		}
		drained, err := r.managedOldWorkersDrained(ctx, dgd, inventory.old)
		if err != nil {
			return err
		}
		if drained {
			return r.deleteObservedOldWorkerDCDs(ctx, inventory.old)
		}
		return nil
	}

	if !targetsReady {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		return nil
	}
	if err := r.projectCompletedWorkerHash(ctx, dgd, inventory.desiredHash); err != nil {
		return err
	}
	completeManagedRollingUpdate(status, workerComponentNames(dgd))
	return nil
}

func setManagedRollingUpdatePhase(
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	phase nvidiacomv1beta1.RollingUpdatePhase,
) {
	rolling := (&dgdWorkerRolloutReconciler{}).getOrCreateRollingUpdateStatus(status)
	if rolling.Phase == phase {
		return
	}
	rolling.Phase = phase
	rolling.EndTime = nil
	if rolling.StartTime == nil {
		now := metav1.Now()
		rolling.StartTime = &now
	}
	if phase == nvidiacomv1beta1.RollingUpdatePhasePending {
		rolling.UpdatedComponents = nil
	}
}

func completeManagedRollingUpdate(
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	updatedComponents []string,
) {
	if status.RollingUpdate == nil {
		return
	}
	status.RollingUpdate.Phase = nvidiacomv1beta1.RollingUpdatePhaseCompleted
	status.RollingUpdate.UpdatedComponents = updatedComponents
	now := metav1.Now()
	status.RollingUpdate.EndTime = &now
}

func workerComponentNames(dgd *nvidiacomv1beta1.DynamoGraphDeployment) []string {
	components := make([]string, 0)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			components = append(components, component.ComponentName)
		}
	}
	sort.Strings(components)
	return components
}

func managedUpdatedWorkerComponents(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inventory managedWorkerInventory,
) []string {
	oldReady := make(map[string]int32)
	for i := range inventory.old {
		dcd := &inventory.old[i]
		if dcd.Status.Component != nil && dcd.Status.Component.ReadyReplicas != nil {
			oldReady[dynamo.GetDCDComponentName(dcd)] += *dcd.Status.Component.ReadyReplicas
		}
	}
	updated := make([]string, 0)
	for componentName := range inventory.desired {
		if workerDCDReady(inventory.targets[componentName]) && oldReady[componentName] == 0 {
			updated = append(updated, componentName)
		}
	}
	sort.Strings(updated)
	return updated
}

func (r *dgdWorkerRolloutReconciler) managedOldWorkersDrained(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	old []nvidiacomv1beta1.DynamoComponentDeployment,
) (bool, error) {
	byComponent := make(map[string][]*nvidiacomv1beta1.DynamoComponentDeployment)
	for i := range old {
		dcd := &old[i]
		componentName := dynamo.GetDCDComponentName(dcd)
		byComponent[componentName] = append(byComponent[componentName], dcd)
	}
	for componentName, dcds := range byComponent {
		drained, err := r.oldWorkerComponentDrained(ctx, dgd, componentName, dcds)
		if err != nil || !drained {
			return drained, err
		}
	}
	return true, nil
}

func (r *dgdWorkerRolloutReconciler) deleteObservedOldWorkerDCDs(
	ctx context.Context,
	old []nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	for i := range old {
		uid := old[i].UID
		if err := r.Delete(ctx, &old[i], client.Preconditions{UID: &uid}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete old worker DCD %s: %w", old[i].Name, err)
		}
	}
	return nil
}

func (r *dgdWorkerRolloutReconciler) projectCompletedWorkerHash(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desiredHash string,
) error {
	current := currentWorkerHashes(dgd)
	if current.v1 == "" && current.v2 == desiredHash {
		return nil
	}
	r.setCurrentWorkerHashes(dgd, workerGenerationHashes{v2: desiredHash})
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("project completed worker hash: %w", err)
	}
	return nil
}

func (r *dgdWorkerRolloutReconciler) clearProjectedWorkerHashes(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if currentWorkerHashes(dgd).empty() {
		return nil
	}
	r.setCurrentWorkerHashes(dgd, workerGenerationHashes{})
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("clear projected worker hashes: %w", err)
	}
	return nil
}

// getCurrentWorkerHash returns the v1 worker generation stored on the DGD.
// It is empty after the DGD has converged to a v2-only generation.
func (r *dgdWorkerRolloutReconciler) getCurrentWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) string {
	return currentWorkerHash(dgd)
}

func currentWorkerHash(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHash]
}

func (r *dgdWorkerRolloutReconciler) getCurrentWorkerHashV2(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) string {
	return currentWorkerHashV2(dgd)
}

func currentWorkerHashV2(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHashV2]
}

// setCurrentWorkerHashes stores the active worker hashes for one generation.
// Empty fields are deleted, which is how v2-only generations intentionally drop
// the downgrade-compatible v1 annotation.
func (r *dgdWorkerRolloutReconciler) setCurrentWorkerHashes(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	hashes workerGenerationHashes,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	if hashes.v1 != "" {
		dgd.Annotations[consts.AnnotationCurrentWorkerHash] = hashes.v1
	} else {
		delete(dgd.Annotations, consts.AnnotationCurrentWorkerHash)
	}
	if hashes.v2 != "" {
		dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hashes.v2
	} else {
		delete(dgd.Annotations, consts.AnnotationCurrentWorkerHashV2)
	}
}

// setLegacyWorkerHash marks pre-rolling-update worker DCDs as the active
// generation so the next reconcile can migrate them with the normal rolling
// update lifecycle.
func (r *dgdWorkerRolloutReconciler) setLegacyWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = consts.LegacyWorkerHash
	delete(dgd.Annotations, consts.AnnotationCurrentWorkerHashV2)
}

// getOrCreateRollingUpdateStatus returns the existing rolling update status or creates a new one.
func (r *dgdWorkerRolloutReconciler) getOrCreateRollingUpdateStatus(
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) *nvidiacomv1beta1.RollingUpdateStatus {
	if status.RollingUpdate == nil {
		status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
			Phase: nvidiacomv1beta1.RollingUpdatePhaseNone,
		}
	}
	return status.RollingUpdate
}

// isRollingUpdateInProgress returns true if a rolling update is currently active.
func isRollingUpdateInProgress(
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) bool {
	if status.RollingUpdate == nil {
		return false
	}
	phase := status.RollingUpdate.Phase
	return phase == nvidiacomv1beta1.RollingUpdatePhasePending ||
		phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress
}

// dcdComponentState holds replica signals extracted from a DCD's Spec and Status.
type dcdComponentState struct {
	Spec      int32 `json:"spec"`      // DCD Spec.Replicas (declared intent)
	Available int32 `json:"available"` // Status.Component.AvailableReplicas (serving traffic)
	Actual    int32 `json:"actual"`    // Status.Component.Replicas (non-terminated pods, excludes Terminating)
}

// dcdComponentStateFromDCD extracts replica signals from a single DCD.
func dcdComponentStateFromDCD(dcd *nvidiacomv1beta1.DynamoComponentDeployment) dcdComponentState {
	s := dcdComponentState{Spec: 1}
	if dcd.Spec.Replicas != nil {
		s.Spec = *dcd.Spec.Replicas
	}
	if dcd.Status.Component != nil {
		s.Actual = dcd.Status.Component.Replicas
		if dcd.Status.Component.AvailableReplicas != nil {
			s.Available = *dcd.Status.Component.AvailableReplicas
		}
	}
	return s
}

// getOldWorkerDCDsByComponent returns old DCDs grouped by component and per-component state
// aggregated across all old generations.
func (r *dgdWorkerRolloutReconciler) getOldWorkerDCDsByComponent(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) (map[string][]*nvidiacomv1beta1.DynamoComponentDeployment, map[string]dcdComponentState, error) {
	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, newWorkerHash)
	if err != nil {
		return nil, nil, err
	}

	dcdsByComponent := make(map[string][]*nvidiacomv1beta1.DynamoComponentDeployment)
	states := make(map[string]dcdComponentState)
	for i := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&oldDCDs[i])
		dcdsByComponent[componentName] = append(dcdsByComponent[componentName], &oldDCDs[i])
		s := dcdComponentStateFromDCD(&oldDCDs[i])
		agg := states[componentName]
		agg.Spec += s.Spec
		agg.Available += s.Available
		agg.Actual += s.Actual
		states[componentName] = agg
	}

	return dcdsByComponent, states, nil
}

// oldWorkerDCDsAtZero reports whether every old DCD has observed its desired
// scale-to-zero state and reports no non-terminated replicas. This is the
// controller-status half of the Recreate barrier. Callers must also inspect
// the old workload pods because Deployment status excludes terminating pods.
func oldWorkerDCDsAtZero(dcds []*nvidiacomv1beta1.DynamoComponentDeployment) bool {
	for _, dcd := range dcds {
		if dcd == nil || dcd.Spec.Replicas == nil || *dcd.Spec.Replicas != 0 {
			return false
		}
		if dcd.Status.ObservedGeneration < dcd.Generation {
			return false
		}
		if dcd.Status.Component == nil || dcd.Status.Component.Replicas != 0 {
			return false
		}
	}
	return true
}

// oldWorkerPodsTerminated reports whether every pod belonging to an old DCD
// is in a terminal phase. A deletion timestamp does not make a pod terminal:
// Pending, Running, and Unknown pods continue to block Recreate until they
// reach Failed or Succeeded, or disappear from the informer cache.
func oldWorkerPodsTerminated(dcds []*nvidiacomv1beta1.DynamoComponentDeployment, pods []corev1.Pod) bool {
	oldDCDNames := make(map[string]struct{}, len(dcds))
	for _, dcd := range dcds {
		if dcd != nil && dcd.Name != "" {
			oldDCDNames[dcd.Name] = struct{}{}
		}
	}

	for i := range pods {
		pod := &pods[i]
		if _, old := oldDCDNames[pod.Labels[consts.KubeLabelDynamoSelector]]; !old {
			continue
		}
		if !isTerminalPhase(pod.Status.Phase) {
			return false
		}
	}
	return true
}

func (r *dgdWorkerRolloutReconciler) oldWorkerComponentDrained(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
	oldDCDs []*nvidiacomv1beta1.DynamoComponentDeployment,
) (bool, error) {
	if !oldWorkerDCDsAtZero(oldDCDs) {
		return false, nil
	}
	pods, err := r.listDGDComponentPods(ctx, dgd, componentName)
	if err != nil {
		return false, err
	}
	return oldWorkerPodsTerminated(oldDCDs, pods), nil
}

func (r *dgdWorkerRolloutReconciler) listDGDComponentPods(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) ([]corev1.Pod, error) {
	podList := &corev1.PodList{}
	if err := r.List(
		ctx,
		podList,
		client.InNamespace(dgd.Namespace),
		client.MatchingFields{
			dgdComponentPodIndex: dgdComponentPodIndexValue(dgd.Name, componentName),
		},
	); err != nil {
		return nil, fmt.Errorf("failed to list pods for DGD %s component %s: %w", dgd.Name, componentName, err)
	}
	return podList.Items, nil
}

// getDesiredWorkerReplicas returns the total desired replicas across all worker components.
func (r *dgdWorkerRolloutReconciler) getDesiredWorkerReplicas(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) int32 {
	var total int32
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			if spec.Replicas != nil {
				total += *spec.Replicas
			} else {
				total += 1 // Default to 1 if not specified
			}
		}
	}
	return total
}

type oldWorkerReplicaPlan struct {
	name      string
	createdAt metav1.Time
	spec      int32 // declared intent for this DCD
	target    int32 // desired replica count for this DCD
}

func allocateOldWorkerDCDReplicas(
	dcds []*nvidiacomv1beta1.DynamoComponentDeployment,
	oldTarget int32,
) map[string]int32 {
	plans := buildOldWorkerReplicaPlans(dcds)
	var servingTarget int32
	for i := range plans {
		servingTarget += plans[i].target
	}

	// if we have more available replicas currently than total desired replicas, remove the oldest DCD replicas first
	if servingTarget > oldTarget {
		removeAvailableReplicasOldestFirst(plans, servingTarget-oldTarget)
		// if we have less available replicas currently than total desired replicas, add the newest DCD replicas first
	} else if servingTarget < oldTarget {
		addUnavailableReplicasNewestFirst(plans, oldTarget-servingTarget)
	}

	return replicaTargetsByDCDName(plans)
}

// initializes DCD replicas to available replicas
func buildOldWorkerReplicaPlans(
	dcds []*nvidiacomv1beta1.DynamoComponentDeployment,
) []oldWorkerReplicaPlan {
	plans := make([]oldWorkerReplicaPlan, 0, len(dcds))

	for _, dcd := range dcds {
		state := dcdComponentStateFromDCD(dcd)
		target := min(state.Spec, state.Available)
		plans = append(plans, oldWorkerReplicaPlan{
			name:      dcd.Name,
			createdAt: dcd.CreationTimestamp,
			spec:      state.Spec,
			target:    target,
		})
	}

	return plans
}

func removeAvailableReplicasOldestFirst(plans []oldWorkerReplicaPlan, replicasToRemove int32) {
	sort.Slice(plans, func(i, j int) bool {
		if plans[i].createdAt.Time.Equal(plans[j].createdAt.Time) {
			return plans[i].name < plans[j].name
		}
		return plans[i].createdAt.Time.Before(plans[j].createdAt.Time)
	})

	for i := range plans {
		if replicasToRemove <= 0 {
			break
		}
		removed := min(plans[i].target, replicasToRemove)
		plans[i].target -= removed
		replicasToRemove -= removed
	}
}

func addUnavailableReplicasNewestFirst(plans []oldWorkerReplicaPlan, replicasToAdd int32) {
	sort.Slice(plans, func(i, j int) bool {
		if plans[i].createdAt.Time.Equal(plans[j].createdAt.Time) {
			return plans[i].name < plans[j].name
		}
		return plans[i].createdAt.Time.After(plans[j].createdAt.Time)
	})

	for i := range plans {
		if replicasToAdd <= 0 {
			break
		}
		unavailable := plans[i].spec - plans[i].target
		added := min(unavailable, replicasToAdd)
		plans[i].target += added
		replicasToAdd -= added
	}
}

func replicaTargetsByDCDName(plans []oldWorkerReplicaPlan) map[string]int32 {
	targets := make(map[string]int32, len(plans))
	for i := range plans {
		targets[plans[i].name] = plans[i].target
	}
	return targets
}

// scaleOldWorkerDCDs patches the replicas field on old worker DCDs during a rolling update.
// When multiple old generations exist for the same component, unavailable replicas are removed
// before reducing old generations that are still serving traffic.
func (r *dgdWorkerRolloutReconciler) scaleOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) error {
	logger := log.FromContext(ctx)

	if !rollingUpdateCtx.InProgress() {
		return nil
	}
	if !rollingUpdateCtx.MayMutateOld {
		return nil
	}

	oldDCDs, err := r.oldWorkerDCDsForRollingUpdate(ctx, dgd, rollingUpdateCtx)
	if err != nil {
		return fmt.Errorf("failed to list old worker DCDs: %w", err)
	}

	for i := range oldDCDs {
		dcd := &oldDCDs[i]
		componentName := dynamo.GetDCDComponentName(dcd)
		if _, ok := rollingUpdateCtx.OldWorkerReplicaTargetsByComponent[componentName]; !ok {
			continue
		}

		desiredReplicas, ok := rollingUpdateCtx.OldWorkerReplicaTargetsByDCD[dcd.Name]
		if !ok {
			return fmt.Errorf("missing old worker DCD replica target for %s", dcd.Name)
		}

		currentReplicas := int32(1)
		if dcd.Spec.Replicas != nil {
			currentReplicas = *dcd.Spec.Replicas
		}

		if currentReplicas == desiredReplicas {
			logger.V(1).Info("Old worker DCD replicas already at desired value",
				"dcdName", dcd.Name, "replicas", desiredReplicas)
			continue
		}

		patch := client.MergeFrom(dcd.DeepCopy())
		dcd.Spec.Replicas = &desiredReplicas

		if err := r.Patch(ctx, dcd, patch); err != nil {
			return fmt.Errorf("failed to patch old worker DCD %s replicas: %w", dcd.Name, err)
		}

		logger.Info("Scaled old worker DCD",
			"dcdName", dcd.Name,
			"component", componentName,
			"oldReplicas", currentReplicas,
			"newReplicas", desiredReplicas)
	}

	return nil
}

// oldWorkerDCDsForRollingUpdate returns the inventory snapshot when the
// managed-DCD reconciler has one. Re-listing by hash would turn an old DCD
// back into a target if a stale cache reports the desired suffix on it.
func (r *dgdWorkerRolloutReconciler) oldWorkerDCDsForRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	if rollingUpdateCtx.ObservedOldWorkerDCDs == nil {
		return r.listOldWorkerDCDs(ctx, dgd, rollingUpdateCtx.NewWorkerHash)
	}

	oldDCDs := make([]nvidiacomv1beta1.DynamoComponentDeployment, 0, len(rollingUpdateCtx.ObservedOldWorkerDCDs))
	for _, dcd := range rollingUpdateCtx.ObservedOldWorkerDCDs {
		if dcd != nil {
			oldDCDs = append(oldDCDs, *dcd.DeepCopy())
		}
	}
	return oldDCDs, nil
}

// listOldWorkerDCDs returns all worker DCDs for this DGD whose worker hash label
// does NOT match the given newWorkerHash. This captures all old generations (including legacy).
func (r *dgdWorkerRolloutReconciler) listOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, err
	}

	var workers []nvidiacomv1beta1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			continue
		}
		if dcd.Labels[consts.KubeLabelDynamoWorkerHash] != newWorkerHash {
			workers = append(workers, dcd)
		}
	}
	return workers, nil
}

// aggregateOldWorkerComponentStatuses fetches all non-current worker DCDs and returns their
// aggregated component statuses keyed by component name. Accumulates across multiple old generations.
func (r *dgdWorkerRolloutReconciler) aggregateOldWorkerComponentStatuses(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) (map[string]nvidiacomv1beta1.ComponentReplicaStatus, error) {
	oldStatuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus)
	oldDCDsByComponent := make(map[string][]nvidiacomv1beta1.DynamoComponentDeployment)

	oldDCDs, err := r.oldWorkerDCDsForRollingUpdate(ctx, dgd, rollingUpdateCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to list old worker DCDs for status aggregation: %w", err)
	}

	for _, dcd := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&dcd)
		if _, inRollout := rollingUpdateCtx.OldWorkerReplicaTargetsByComponent[componentName]; !inRollout {
			continue
		}
		if dcd.Status.Component == nil {
			continue
		}
		oldDCDsByComponent[componentName] = append(oldDCDsByComponent[componentName], dcd)
	}

	for componentName, dcds := range oldDCDsByComponent {
		sortOldWorkerDCDsNewestFirst(dcds)
		status := aggregateOldWorkerDCDStatuses(dcds)
		status.RuntimeNamespace = selectOldWorkerRuntimeNamespace(
			dcds,
			rollingUpdateCtx.OldWorkerReplicaTargetsByDCD,
		)
		oldStatuses[componentName] = status
	}

	return oldStatuses, nil
}

func sortOldWorkerDCDsNewestFirst(dcds []nvidiacomv1beta1.DynamoComponentDeployment) {
	sort.Slice(dcds, func(i, j int) bool {
		if dcds[i].CreationTimestamp.Time.Equal(dcds[j].CreationTimestamp.Time) {
			return dcds[i].Name > dcds[j].Name
		}
		return dcds[i].CreationTimestamp.Time.After(dcds[j].CreationTimestamp.Time)
	})
}

func aggregateOldWorkerDCDStatuses(
	dcds []nvidiacomv1beta1.DynamoComponentDeployment,
) nvidiacomv1beta1.ComponentReplicaStatus {
	var status nvidiacomv1beta1.ComponentReplicaStatus
	for i, dcd := range dcds {
		componentStatus := dcd.Status.Component
		if i == 0 {
			status = *componentStatus
			status.ComponentNames = componentReplicaResourceNames(componentStatus, dcd.Name)
			status.RuntimeNamespace = ""
			continue
		}

		status.Replicas += componentStatus.Replicas
		status.ReadyReplicas = addOptionalInt32(status.ReadyReplicas, componentStatus.ReadyReplicas)
		status.AvailableReplicas = addOptionalInt32(status.AvailableReplicas, componentStatus.AvailableReplicas)
		status.ComponentNames = append(status.ComponentNames, componentReplicaResourceNames(componentStatus, dcd.Name)...)
	}
	return status
}

func selectOldWorkerRuntimeNamespace(
	dcds []nvidiacomv1beta1.DynamoComponentDeployment,
	replicaTargets map[string]int32,
) string {
	for _, dcd := range dcds {
		if replicaTargets[dcd.Name] > 0 {
			return oldWorkerRuntimeNamespace(dcd)
		}
	}

	if len(dcds) == 0 {
		return ""
	}
	return oldWorkerRuntimeNamespace(dcds[0])
}

func oldWorkerRuntimeNamespace(dcd nvidiacomv1beta1.DynamoComponentDeployment) string {
	if dcd.Status.Component != nil && dcd.Status.Component.RuntimeNamespace != "" {
		return dcd.Status.Component.RuntimeNamespace
	}
	return dynamo.GetDCDRuntimeNamespace(&dcd)
}

// resolveRollingUpdateParams resolves maxSurge and maxUnavailable to concrete
// replica counts for a RollingUpdate component. Recreate components bypass
// this calculation.
// Defaults: maxSurge=25%, maxUnavailable=25% (matches Kubernetes Deployment defaults).
func resolveRollingUpdateParams(annotations map[string]string, desiredReplicas int32) (maxSurge int32, maxUnavailable int32) {
	surgeValue := intstr.FromString("25%")
	unavailValue := intstr.FromString("25%")

	if v := annotations[KubeAnnotationDeploymentRollingUpdateMaxSurge]; v != "" {
		surgeValue = intstr.Parse(v)
	}
	if v := annotations[KubeAnnotationDeploymentRollingUpdateMaxUnavailable]; v != "" {
		unavailValue = intstr.Parse(v)
	}

	// Resolve percentages against desiredReplicas. Round up for surge (more aggressive scale-up),
	// round down for unavailable (more conservative, matches Kubernetes deployment controller behavior).
	// https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#max-unavailable
	surge, _ := intstr.GetScaledValueFromIntOrPercent(&surgeValue, int(desiredReplicas), true)
	unavail, _ := intstr.GetScaledValueFromIntOrPercent(&unavailValue, int(desiredReplicas), false)

	// Ensure at least one of surge/unavailable is > 0 to guarantee progress
	if surge == 0 && unavail == 0 {
		surge = 1
	}

	return int32(surge), int32(unavail)
}

// buildRollingUpdateContext creates a RollingUpdateContext.
func (r *dgdWorkerRolloutReconciler) buildRollingUpdateContext(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dynamo.RollingUpdateContext, error) {
	logger := log.FromContext(ctx)

	desiredHashes, err := desiredWorkerHashes(dgd)
	if err != nil {
		return dynamo.RollingUpdateContext{}, err
	}
	newWorkerHash := r.activeWorkerHashForDCDGeneration(dgd, desiredHashes)
	currentHashes := r.currentWorkerHashes(dgd)

	if currentHashes.contains(newWorkerHash) {
		return dynamo.RollingUpdateContext{
			NewWorkerHash:                      newWorkerHash,
			OldWorkerReplicaTargetsByComponent: make(map[string]int32),
			OldWorkerReplicaTargetsByDCD:       make(map[string]int32),
			NewWorkerReplicaTargetsByComponent: make(map[string]int32),
		}, nil
	}

	oldDCDsByComponent, oldStates, err := r.getOldWorkerDCDsByComponent(ctx, dgd, newWorkerHash)
	if err != nil {
		return dynamo.RollingUpdateContext{}, fmt.Errorf("failed to get old worker component states: %w", err)
	}

	oldWorkerComponentReplicas := make(map[string]int32)
	oldWorkerDCDReplicas := make(map[string]int32)
	newWorkerReplicas := make(map[string]int32)

	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		componentName := spec.ComponentName
		if !dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			continue
		}

		desired := int32(1)
		if spec.Replicas != nil {
			desired = *spec.Replicas
		}

		var newState dcdComponentState
		newDCDName := dynamo.GetDCDResourceName(dgd, componentName, newWorkerHash)
		newDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := r.Get(ctx, types.NamespacedName{Name: newDCDName, Namespace: dgd.Namespace}, newDCD); err == nil {
			newState = dcdComponentStateFromDCD(newDCD)
		} else if !apierrors.IsNotFound(err) {
			return dynamo.RollingUpdateContext{}, fmt.Errorf("failed to get new worker DCD %s: %w", newDCDName, err)
		}

		oldState := oldStates[componentName]
		annotations := dynamo.GetDGDComponentResourceAnnotations(dgd, componentName, spec)
		strategy := deploymentStrategyFromAnnotations(annotations)

		var oldTarget, newTarget, maxSurge, maxUnavailable, minAvailable int32
		switch strategy {
		case common.DeploymentStrategyRecreate:
			// Recreate deliberately permits the full component to be unavailable.
			// Keep declaring every old generation at zero until their controllers
			// have observed the drain and every old workload pod is terminal; only
			// then start the replacement generation.
			maxUnavailable = desired
			oldDCDs := oldDCDsByComponent[componentName]
			drained, err := r.oldWorkerComponentDrained(ctx, dgd, componentName, oldDCDs)
			if err != nil {
				return dynamo.RollingUpdateContext{}, err
			}
			if drained {
				newTarget = desired
			}
		default:
			maxSurge, maxUnavailable = resolveRollingUpdateParams(annotations, desired)
			minAvailable = desired - maxUnavailable

			newUnavailable := max(int32(0), newState.Spec-newState.Available)
			// maxScaledDown is the maximum number of old replicas that can be scaled down
			maxScaledDown := max(int32(0), (oldState.Spec+newState.Spec)-minAvailable-newUnavailable)
			oldUnhealthy := max(int32(0), oldState.Spec-oldState.Available)
			// availableSurplus is how many extra available replicas we have above minAvailable (min 0)
			availableSurplus := max(int32(0), (oldState.Available+newState.Available)-minAvailable)
			oldTarget = max(int32(0), oldState.Spec-min(maxScaledDown, oldUnhealthy+availableSurplus))

			// Surge budget uses Spec (declared intent) like K8s Deployment controller; scheduler enforces actual resource constraints.
			scaleUpBudget := max(int32(0), desired+maxSurge-oldState.Spec-newState.Spec)
			newTarget = min(desired, newState.Spec+scaleUpBudget)
		}

		oldWorkerComponentReplicas[componentName] = oldTarget
		newWorkerReplicas[componentName] = newTarget

		for dcdName, target := range allocateOldWorkerDCDReplicas(oldDCDsByComponent[componentName], oldTarget) {
			oldWorkerDCDReplicas[dcdName] = target
		}

		logger.V(1).Info("Worker update replica calculation",
			"component", componentName,
			"strategy", strategy,
			"desired", desired,
			"maxSurge", maxSurge,
			"maxUnavailable", maxUnavailable,
			"minAvailable", minAvailable,
			"old", oldState,
			"new", newState,
			"oldTarget", oldTarget,
			"newTarget", newTarget)
	}

	return dynamo.RollingUpdateContext{
		NewWorkerHash:                      newWorkerHash,
		OldWorkerReplicaTargetsByComponent: oldWorkerComponentReplicas,
		OldWorkerReplicaTargetsByDCD:       oldWorkerDCDReplicas,
		NewWorkerReplicaTargetsByComponent: newWorkerReplicas,
	}, nil
}

// mergeWorkerComponentStatuses merges old worker component statuses into the existing component statuses.
// For each worker component present in both maps, it aggregates replica counts so that the status
// reflects the total across old and new worker DCDs during a rolling update.
func mergeWorkerComponentStatuses(
	componentStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus,
	oldWorkerStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus,
) {
	for componentName, oldStatus := range oldWorkerStatuses {
		newStatus, exists := componentStatuses[componentName]
		if !exists {
			oldStatus.UpdatedReplicas = 0
			componentStatuses[componentName] = oldStatus
			continue
		}

		if oldStatus.RuntimeNamespace != "" {
			// Keep routing consumers on the old active worker namespace until rollout cutover.
			newStatus.RuntimeNamespace = oldStatus.RuntimeNamespace
		}

		// Build sorted ComponentNames from old and new DCD names.
		componentNames := append(slices.Clone(oldStatus.ComponentNames), newStatus.ComponentNames...)
		slices.Sort(componentNames)
		newStatus.ComponentNames = componentNames

		// Aggregate replica counts
		newStatus.Replicas += oldStatus.Replicas
		// UpdatedReplicas stays as-is (only new are "updated")
		newStatus.ReadyReplicas = addOptionalInt32(newStatus.ReadyReplicas, oldStatus.ReadyReplicas)
		newStatus.AvailableReplicas = addOptionalInt32(newStatus.AvailableReplicas, oldStatus.AvailableReplicas)

		componentStatuses[componentName] = newStatus
	}
}

func componentReplicaResourceNames(status *nvidiacomv1beta1.ComponentReplicaStatus, fallback string) []string {
	if status == nil {
		return nil
	}
	if len(status.ComponentNames) > 0 {
		return slices.Clone(status.ComponentNames)
	}
	if fallback == "" {
		return nil
	}
	return []string{fallback}
}

// addOptionalInt32 adds two optional int32 pointers. Returns nil only if both are nil.
func addOptionalInt32(a, b *int32) *int32 {
	if a == nil && b == nil {
		return nil
	}
	var sum int32
	if a != nil {
		sum += *a
	}
	if b != nil {
		sum += *b
	}
	return &sum
}
