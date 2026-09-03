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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

// managedWorkerRollout is the controller's observed worker-DCD state and the
// exact actions permitted from that observation. It is the single source of
// truth for a managed rolling update.
type managedWorkerRollout struct {
	desiredV2Hash      string
	targetDCDSuffix    string
	targetsByComponent map[string]*nvidiacomv1beta1.DynamoComponentDeployment
	oldDCDs            []managedOldWorkerDCD

	newReplicaTargetsByComponent map[string]int32
}

// managedOldWorkerDCD couples an observed DCD to the replica target computed
// from that same observation. A nil target means no destructive action is
// permitted yet.
type managedOldWorkerDCD struct {
	dcd            nvidiacomv1beta1.DynamoComponentDeployment
	targetReplicas *int32
}

func (r managedWorkerRollout) targetPending() bool {
	for _, target := range r.targetsByComponent {
		if target == nil {
			return true
		}
	}
	return false
}

func (r managedWorkerRollout) targetsReady() bool {
	if r.targetPending() {
		return false
	}
	for _, target := range r.targetsByComponent {
		if !workerDCDReady(target) {
			return false
		}
	}
	return true
}

func (r managedWorkerRollout) inProgress() bool {
	return r.targetPending() || !r.targetsReady() || len(r.oldDCDs) > 0
}

// buildManagedWorkerRollout resolves the owned DCD cohort by its canonical
// name. A persisted v1 suffix is eligible only as an explicit v1-to-v2 bridge;
// worker specs are not used to identify a generation.
func (r *dgdWorkerRolloutReconciler) buildManagedWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (managedWorkerRollout, error) {
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return managedWorkerRollout{}, fmt.Errorf("compute desired worker hash: %w", err)
	}

	rollout := managedWorkerRollout{
		desiredV2Hash:                hash,
		targetDCDSuffix:              hash,
		targetsByComponent:           make(map[string]*nvidiacomv1beta1.DynamoComponentDeployment),
		newReplicaTargetsByComponent: make(map[string]int32),
	}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			rollout.targetsByComponent[component.ComponentName] = nil
		}
	}

	observed, err := r.listOwnedWorkerDCDs(ctx, dgd)
	if err != nil {
		return managedWorkerRollout{}, err
	}
	nonTerminatingByName := make(map[string]*nvidiacomv1beta1.DynamoComponentDeployment, len(observed))
	for i := range observed {
		if observed[i].DeletionTimestamp.IsZero() {
			nonTerminatingByName[observed[i].Name] = &observed[i]
		}
	}

	rollout.targetDCDSuffix = selectManagedWorkerTargetDCDSuffix(
		dgd,
		hash,
		rollout.targetsByComponent,
		nonTerminatingByName,
	)

	matched := make(map[string]struct{}, len(observed))
	for componentName := range rollout.targetsByComponent {
		targetName := dynamo.GetDCDResourceName(dgd, componentName, rollout.targetDCDSuffix)
		if target, ok := nonTerminatingByName[targetName]; ok {
			rollout.targetsByComponent[componentName] = target
			matched[target.Name] = struct{}{}
			continue
		}
	}
	for i := range observed {
		if _, ok := matched[observed[i].Name]; !ok {
			rollout.oldDCDs = append(rollout.oldDCDs, managedOldWorkerDCD{dcd: observed[i]})
		}
	}

	if len(rollout.targetsByComponent) == 0 {
		for i := range rollout.oldDCDs {
			target := int32(0)
			rollout.oldDCDs[i].targetReplicas = &target
		}
		return rollout, nil
	}

	if rollout.targetPending() {
		// A Recreate target must exist at zero before old DCDs can be drained.
		// Its desired replicas are set only after a later observation confirms
		// that the old worker component is completely gone.
		for i := range dgd.Spec.Components {
			component := &dgd.Spec.Components[i]
			if dynamo.IsWorkerComponent(string(component.ComponentType)) &&
				deploymentStrategyFromAnnotations(dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component)) == common.DeploymentStrategyRecreate {
				rollout.newReplicaTargetsByComponent[component.ComponentName] = 0
			}
		}
		return rollout, nil
	}
	if len(rollout.oldDCDs) > 0 {
		if err := r.populateManagedOldWorkerTargets(ctx, dgd, &rollout); err != nil {
			return managedWorkerRollout{}, err
		}
	}
	return rollout, nil
}

func managedWorkerBridgeHash(dgd *nvidiacomv1beta1.DynamoGraphDeployment, desiredHash string) string {
	current := currentWorkerHashes(dgd)
	if current.v1 == "" || current.v1 == consts.LegacyWorkerHash || current.v2 != desiredHash {
		return ""
	}
	return current.v1
}

// selectManagedWorkerTargetDCDSuffix prefers the canonical v2 cohort. It
// retains a v1 suffix only for the persisted {v1:H, v2:A} bridge when no v2
// target has been observed.
func selectManagedWorkerTargetDCDSuffix(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desiredV2Hash string,
	targetsByComponent map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	nonTerminatingByName map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
) string {
	if hasManagedWorkerTarget(targetsByComponent, nonTerminatingByName, dgd, desiredV2Hash) {
		return desiredV2Hash
	}
	bridgeHash := managedWorkerBridgeHash(dgd, desiredV2Hash)
	if bridgeHash != "" && hasManagedWorkerTarget(targetsByComponent, nonTerminatingByName, dgd, bridgeHash) {
		return bridgeHash
	}
	return desiredV2Hash
}

func hasManagedWorkerTarget(
	targetsByComponent map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	nonTerminatingByName map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	hash string,
) bool {
	for componentName := range targetsByComponent {
		if _, ok := nonTerminatingByName[dynamo.GetDCDResourceName(dgd, componentName, hash)]; ok {
			return true
		}
	}
	return false
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
	rollout *managedWorkerRollout,
) error {
	oldByComponent := make(map[string][]*managedOldWorkerDCD)
	oldStates := make(map[string]dcdComponentState)
	for i := range rollout.oldDCDs {
		old := &rollout.oldDCDs[i]
		componentName := dynamo.GetDCDComponentName(&old.dcd)
		oldByComponent[componentName] = append(oldByComponent[componentName], old)
		state := dcdComponentStateFromDCD(&old.dcd)
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
		newState := dcdComponentStateFromDCD(rollout.targetsByComponent[componentName])
		oldState := oldStates[componentName]
		annotations := dynamo.GetDGDComponentResourceAnnotations(dgd, componentName, spec)
		strategy := deploymentStrategyFromAnnotations(annotations)

		var oldTarget, newTarget int32
		if strategy == common.DeploymentStrategyRecreate {
			oldDCDs := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, len(oldByComponent[componentName]))
			for _, old := range oldByComponent[componentName] {
				oldDCDs = append(oldDCDs, &old.dcd)
			}
			drained, err := r.oldWorkerComponentDrained(ctx, dgd, componentName, oldDCDs)
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
		rollout.newReplicaTargetsByComponent[componentName] = newTarget
		oldDCDs := make([]*nvidiacomv1beta1.DynamoComponentDeployment, 0, len(oldByComponent[componentName]))
		for _, old := range oldByComponent[componentName] {
			oldDCDs = append(oldDCDs, &old.dcd)
		}
		oldTargets := allocateOldWorkerDCDReplicas(oldDCDs, oldTarget)
		for _, old := range oldByComponent[componentName] {
			target := oldTargets[old.dcd.Name]
			old.targetReplicas = &target
		}
	}
	return nil
}

func (r *dgdWorkerRolloutReconciler) advanceManagedWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	rollout managedWorkerRollout,
) error {
	if len(rollout.targetsByComponent) == 0 {
		if len(rollout.oldDCDs) == 0 {
			if err := r.clearProjectedWorkerHashes(ctx, dgd); err != nil {
				return err
			}
			completeManagedRollingUpdate(status, nil)
			return nil
		}
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		drained, err := r.managedOldWorkersDrained(ctx, dgd, rollout.oldDCDs)
		if err != nil {
			return err
		}
		if drained {
			return r.deleteObservedOldWorkerDCDs(ctx, rollout.oldDCDs)
		}
		return nil
	}

	if rollout.targetPending() {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhasePending)
		return nil
	}

	targetsReady := rollout.targetsReady()
	if len(rollout.oldDCDs) > 0 {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		status.RollingUpdate.UpdatedComponents = managedUpdatedWorkerComponents(dgd, rollout)
		if !targetsReady {
			return nil
		}
		drained, err := r.managedOldWorkersDrained(ctx, dgd, rollout.oldDCDs)
		if err != nil {
			return err
		}
		if drained {
			return r.deleteObservedOldWorkerDCDs(ctx, rollout.oldDCDs)
		}
		return nil
	}

	if !targetsReady {
		setManagedRollingUpdatePhase(status, nvidiacomv1beta1.RollingUpdatePhaseInProgress)
		return nil
	}
	if err := r.projectCompletedWorkerHash(ctx, dgd, rollout.desiredV2Hash, rollout.targetDCDSuffix); err != nil {
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
	rollout managedWorkerRollout,
) []string {
	oldReady := make(map[string]int32)
	for i := range rollout.oldDCDs {
		dcd := &rollout.oldDCDs[i].dcd
		if dcd.Status.Component != nil && dcd.Status.Component.ReadyReplicas != nil {
			oldReady[dynamo.GetDCDComponentName(dcd)] += *dcd.Status.Component.ReadyReplicas
		}
	}
	updated := make([]string, 0)
	for componentName := range rollout.targetsByComponent {
		if workerDCDReady(rollout.targetsByComponent[componentName]) && oldReady[componentName] == 0 {
			updated = append(updated, componentName)
		}
	}
	sort.Strings(updated)
	return updated
}

func (r *dgdWorkerRolloutReconciler) managedOldWorkersDrained(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	oldDCDs []managedOldWorkerDCD,
) (bool, error) {
	byComponent := make(map[string][]*nvidiacomv1beta1.DynamoComponentDeployment)
	for i := range oldDCDs {
		dcd := &oldDCDs[i].dcd
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
	oldDCDs []managedOldWorkerDCD,
) error {
	for i := range oldDCDs {
		dcd := &oldDCDs[i].dcd
		uid := dcd.UID
		if err := r.Delete(ctx, dcd, client.Preconditions{UID: &uid}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("delete old worker DCD %s: %w", dcd.Name, err)
		}
	}
	return nil
}

func (r *dgdWorkerRolloutReconciler) projectCompletedWorkerHash(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desiredHash string,
	targetHash string,
) error {
	next := workerGenerationHashes{v2: desiredHash}
	if targetHash != desiredHash {
		next.v1 = targetHash
	}
	if currentWorkerHashes(dgd) == next {
		return nil
	}
	r.setCurrentWorkerHashes(dgd, next)
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

func currentWorkerHash(dgd *nvidiacomv1beta1.DynamoGraphDeployment) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHash]
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
	rollout managedWorkerRollout,
) error {
	logger := log.FromContext(ctx)

	if !rollout.inProgress() || rollout.targetPending() {
		return nil
	}

	for i := range rollout.oldDCDs {
		old := &rollout.oldDCDs[i]
		if old.targetReplicas == nil {
			return fmt.Errorf("missing old worker DCD replica target for %s", old.dcd.Name)
		}
		dcd := old.dcd.DeepCopy()
		componentName := dynamo.GetDCDComponentName(dcd)
		desiredReplicas := *old.targetReplicas

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

// aggregateOldWorkerComponentStatuses returns status from the observed old DCDs
// selected by the managed rollout.
func (r *dgdWorkerRolloutReconciler) aggregateOldWorkerComponentStatuses(
	rollout managedWorkerRollout,
) map[string]nvidiacomv1beta1.ComponentReplicaStatus {
	oldStatuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus)
	oldDCDsByComponent := make(map[string][]nvidiacomv1beta1.DynamoComponentDeployment)
	replicaTargets := make(map[string]int32)

	for _, old := range rollout.oldDCDs {
		if old.targetReplicas == nil || old.dcd.Status.Component == nil {
			continue
		}
		componentName := dynamo.GetDCDComponentName(&old.dcd)
		oldDCDsByComponent[componentName] = append(oldDCDsByComponent[componentName], old.dcd)
		replicaTargets[old.dcd.Name] = *old.targetReplicas
	}

	for componentName, dcds := range oldDCDsByComponent {
		sortOldWorkerDCDsNewestFirst(dcds)
		status := aggregateOldWorkerDCDStatuses(dcds)
		status.RuntimeNamespace = selectOldWorkerRuntimeNamespace(dcds, replicaTargets)
		oldStatuses[componentName] = status
	}

	return oldStatuses
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
