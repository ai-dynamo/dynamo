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
	"errors"
	"fmt"
	"maps"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type dgdScalingAdaptersReconciler struct {
	dgdResourceSyncer
}

func newDGDScalingAdaptersReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
) *dgdScalingAdaptersReconciler {
	return &dgdScalingAdaptersReconciler{
		dgdResourceSyncer: newDGDResourceSyncer(kubeClient, recorder),
	}
}

// Reconcile ensures a DynamoGraphDeploymentScalingAdapter exists for each DGD
// component that has scaling explicitly enabled.
func (r *dgdScalingAdaptersReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	_, err := r.reconcile(ctx, dgd)
	return err
}

// ReconcileCreateTimeSeeds returns true when at least one DGDSA was created.
// Transactional callers use that signal to observe the complete create-only
// seed vector on a later pass before entering workload reconciliation.
func (r *dgdScalingAdaptersReconciler) ReconcileCreateTimeSeeds(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	return r.reconcile(ctx, dgd)
}

// ReconcileTransactionalReplicas admits one complete requested vector, persists
// its reservation, and mirrors only a previously persisted vector into the DGD.
// The returned wait flag keeps workload reconciliation behind each durability
// boundary while allowing a rejected request to leave the current vector live.
func (r *dgdScalingAdaptersReconciler) ReconcileTransactionalReplicas(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	created, err := r.reconcile(ctx, dgd)
	if err != nil || created {
		return created, err
	}

	// Observe the durable budget and the complete request vector after seed creation.
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{}
	key := types.NamespacedName{Namespace: dgd.Namespace, Name: dgd.Name}
	if err := r.Get(ctx, key, dgpb); err != nil {
		return false, fmt.Errorf("read DynamoGraphPowerBudget %s for replica admission: %w", key, err)
	}
	requested, err := r.transactionalRequestedReplicaVector(ctx, dgd)
	if err != nil {
		return false, err
	}

	// Persist the zero bootstrap vector before evaluating or mirroring any seed.
	committed := maps.Clone(dgpb.Status.CommittedReplicaTargets)
	if len(committed) == 0 {
		committed = zeroReplicaVector(requested)
		if err := r.persistCommittedReplicaVector(ctx, dgpb, committed, dgpb.Status.Ledger, false); err != nil {
			return false, err
		}
		if err := r.publishTransactionalAdapterStatuses(
			ctx,
			dgd,
			requested,
			committed,
			nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline,
		); err != nil {
			return false, err
		}
		return true, nil
	}
	if !sameReplicaVectorComponents(committed, requested) {
		return false, fmt.Errorf("committed replica vector does not match the transactional request components")
	}
	bootstrap := dgpb.Status.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing &&
		replicaVectorAllZero(committed)
	if !bootstrap {
		// Finish a previously committed mirror before evaluating a newer request.
		changed, err := mirrorCommittedReplicaVector(dgd, committed)
		if err != nil {
			return false, err
		}
		if changed {
			pendingReason := nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason("")
			if !maps.Equal(requested, committed) {
				pendingReason = nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonUnenforcedBaseline
			}
			if err := r.publishTransactionalAdapterStatuses(ctx, dgd, requested, committed, pendingReason); err != nil {
				return false, err
			}
			if err := r.Update(ctx, dgd); err != nil {
				return false, fmt.Errorf("mirror committed replica vector to DGD: %w", err)
			}
			log.FromContext(ctx).Info("Mirrored committed replica vector to DGD", "dgd", dgd.Name, "committed", committed)
			return true, nil
		}
	}

	// Commit an accepted request and stop before the DGD write boundary.
	if bootstrap || !maps.Equal(committed, requested) {
		decision, err := admitTransactionalReplicaVector(dgd, dgpb, committed, requested)
		if err != nil {
			return false, err
		}
		if decision.Accepted {
			if err := r.persistCommittedReplicaVector(
				ctx,
				dgpb,
				decision.Committed,
				powerbudget.NewLedgerStatus(decision.Ledger),
				true,
			); err != nil {
				return false, err
			}
			if err := r.publishTransactionalAdapterStatuses(ctx, dgd, requested, decision.Committed, ""); err != nil {
				return false, err
			}
			return true, nil
		}
		if err := r.publishTransactionalAdapterStatuses(ctx, dgd, requested, committed, decision.PendingReason); err != nil {
			return false, err
		}
		log.FromContext(ctx).Info(
			"Transactional replica request remains pending",
			"dgd", dgd.Name,
			"reason", decision.PendingReason,
			"requested", requested,
			"committed", committed,
		)
	}
	if maps.Equal(committed, requested) && !bootstrap {
		if err := r.publishTransactionalAdapterStatuses(ctx, dgd, requested, committed, ""); err != nil {
			return false, err
		}
	}

	// Mirror only the vector already durable in DGPB status.
	changed, err := mirrorCommittedReplicaVector(dgd, committed)
	if err != nil || !changed {
		return false, err
	}
	if err := r.Update(ctx, dgd); err != nil {
		return false, fmt.Errorf("mirror committed replica vector to DGD: %w", err)
	}
	log.FromContext(ctx).Info("Mirrored committed replica vector to DGD", "dgd", dgd.Name, "committed", committed)
	return true, nil
}

// publishTransactionalAdapterStatuses exposes the request, durable commitment,
// and observed capacity without making DGDSA status an admission authority. An
// empty pendingReason preserves an existing reason while request and commit
// differ, and clears it once they agree.
func (r *dgdScalingAdaptersReconciler) publishTransactionalAdapterStatuses(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	requested powerbudget.ReplicaVector,
	committed powerbudget.ReplicaVector,
	pendingReason nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReason,
) error {
	recordPowerRequestVector(pendingReason == "" && maps.Equal(requested, committed), pendingReason)
	for componentName, requestedReplicas := range requested {
		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
		key := types.NamespacedName{
			Namespace: dgd.Namespace,
			Name:      generateAdapterName(dgd.Name, componentName),
		}
		if err := r.Get(ctx, key, adapter); err != nil {
			return fmt.Errorf("read transactional scaling adapter %s status: %w", key, err)
		}

		committedReplicas, found := committed[componentName]
		if !found {
			return fmt.Errorf("committed replica vector is missing worker component %q", componentName)
		}
		actualReplicas := dgd.Status.Components[componentName].Replicas
		before := adapter.DeepCopy()
		adapter.Status.Replicas = actualReplicas
		adapter.Status.RequestedReplicas = requestedReplicas
		adapter.Status.CommittedReplicas = committedReplicas
		adapter.Status.ActualReplicas = actualReplicas
		if pendingReason != "" {
			adapter.Status.PendingReason = nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterPendingReason(pendingReason)
		} else if requestedReplicas == committedReplicas {
			adapter.Status.PendingReason = ""
		}

		if adapter.Status == before.Status {
			continue
		}
		if err := r.Status().Patch(
			ctx,
			adapter,
			client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{}),
		); err != nil {
			return fmt.Errorf("publish transactional scaling adapter %s status: %w", key, err)
		}
	}
	return nil
}

func (r *dgdScalingAdaptersReconciler) transactionalRequestedReplicaVector(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (powerbudget.ReplicaVector, error) {
	requested := make(powerbudget.ReplicaVector)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}

		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{}
		key := types.NamespacedName{
			Namespace: dgd.Namespace,
			Name:      generateAdapterName(dgd.Name, component.ComponentName),
		}
		if err := r.Get(ctx, key, adapter); err != nil {
			return nil, fmt.Errorf("read transactional scaling adapter %s: %w", key, err)
		}
		owner := metav1.GetControllerOf(adapter)
		if owner == nil || owner.UID != dgd.UID || adapter.Spec.DGDRef.Name != dgd.Name ||
			adapter.Spec.DGDRef.ServiceName != component.ComponentName {
			return nil, fmt.Errorf("transactional scaling adapter %s is not bound to DGD component %q", key, component.ComponentName)
		}
		requested[component.ComponentName] = adapter.Spec.Replicas
	}
	if len(requested) == 0 {
		return nil, fmt.Errorf("transactional DGD has no power-managed worker components")
	}
	return requested, nil
}

func (r *dgdScalingAdaptersReconciler) persistCommittedReplicaVector(
	ctx context.Context,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	committed powerbudget.ReplicaVector,
	ledger nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus,
	applying bool,
) error {
	before := dgpb.DeepCopy()
	dgpb.Status.CommittedReplicaTargets = maps.Clone(committed)
	dgpb.Status.Ledger = ledger
	if applying && (dgpb.Status.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseInitializing ||
		dgpb.Status.Phase == nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle) {
		dgpb.Status.Phase = nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseApplying
	}
	if _, err := powerbudget.EncodeStatusSnapshot(dgpb.Status); err != nil {
		return fmt.Errorf("validate committed replica vector: %w", err)
	}
	if err := r.Status().Patch(
		ctx,
		dgpb,
		client.MergeFromWithOptions(before, client.MergeFromWithOptimisticLock{}),
	); err != nil {
		return fmt.Errorf("persist committed replica vector: %w", err)
	}
	return nil
}

func admitTransactionalReplicaVector(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	committed powerbudget.ReplicaVector,
	requested powerbudget.ReplicaVector,
) (powerbudget.AdmissionDecision, error) {
	spec, err := powerbudget.NewSpec(dgpb.Spec)
	if err != nil {
		return powerbudget.AdmissionDecision{}, err
	}
	current, err := powerbudget.BuildObservedAdmissionProjection(committed, dgpb.Status.Components, dgpb.Status.Ledger)
	if err != nil {
		return powerbudget.AdmissionDecision{}, fmt.Errorf("build current replica projection: %w", err)
	}
	projected, err := powerbudget.BuildIncrementalAdmissionProjection(
		committed,
		requested,
		dgpb.Status.Components,
		current,
	)
	if err != nil {
		if errors.Is(err, powerbudget.ErrInvalidTarget) {
			return powerbudget.AdmissionDecision{
				Committed:     maps.Clone(committed),
				Ledger:        current.Ledger(),
				PendingReason: nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonInvalidTarget,
			}, nil
		}
		return powerbudget.AdmissionDecision{}, fmt.Errorf("build requested replica projection: %w", err)
	}
	return powerbudget.AdmitVector(
		spec,
		committed,
		requested,
		current,
		projected,
		powerbudget.AdmissionState{
			Phase:             dgpb.Status.Phase,
			TopologySupported: !dgd.HasAnyMultinodeComponent() && !dgdHasCheckpointConfiguration(dgd),
			HardwareQualified: dgpb.Status.Phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseUnqualified,
		},
	), nil
}

func zeroReplicaVector(requested powerbudget.ReplicaVector) powerbudget.ReplicaVector {
	zero := make(powerbudget.ReplicaVector, len(requested))
	for component := range requested {
		zero[component] = 0
	}
	return zero
}

func replicaVectorAllZero(vector powerbudget.ReplicaVector) bool {
	for _, replicas := range vector {
		if replicas != 0 {
			return false
		}
	}
	return true
}

func sameReplicaVectorComponents(left powerbudget.ReplicaVector, right powerbudget.ReplicaVector) bool {
	if len(left) == 0 || len(left) != len(right) {
		return false
	}
	for component := range left {
		if _, exists := right[component]; !exists {
			return false
		}
	}
	return true
}

func mirrorCommittedReplicaVector(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	committed powerbudget.ReplicaVector,
) (bool, error) {
	changed := false
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		target, found := committed[component.ComponentName]
		if !found {
			return false, fmt.Errorf("committed replica vector is missing worker component %q", component.ComponentName)
		}
		if ptr.Deref(component.Replicas, int32(1)) == target {
			continue
		}
		component.Replicas = ptr.To(target)
		changed = true
	}
	return changed, nil
}

func (r *dgdScalingAdaptersReconciler) reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	logger := log.FromContext(ctx)
	created := false

	// Reconcile adapters for current components while preserving adapter-owned replicas.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentName := component.ComponentName
		adapterName := generateAdapterName(dgd.Name, componentName)
		adapter := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapter{
			ObjectMeta: metav1.ObjectMeta{
				Name:      adapterName,
				Namespace: dgd.Namespace,
			},
		}

		transactionalWorker := dgd.Annotations[nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation] ==
			nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence &&
			dynamo.IsWorkerComponent(string(component.ComponentType))

		// Remove static adapters after opt-out; transactional workers always retain request ingress.
		if component.ScalingAdapter == nil && !transactionalWorker {
			if err := r.Delete(ctx, adapter); err != nil {
				if apierrors.IsNotFound(err) {
					continue
				}
				logger.Error(err, "Failed to delete DynamoGraphDeploymentScalingAdapter", "component", componentName)
				return false, err
			}

			logger.Info("Deleted DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterDeleted",
					"Delete",
					"Deleted scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
			continue
		}

		initialReplicas := ptr.Deref(component.Replicas, int32(1))
		operation, err := controllerutil.CreateOrPatch(ctx, r.Client, adapter, func() error {
			if adapter.Labels == nil {
				adapter.Labels = map[string]string{}
			}
			adapter.Labels[consts.KubeLabelDynamoGraphDeploymentName] = dgd.Name
			adapter.Labels[consts.KubeLabelDynamoComponent] = componentName
			adapter.Spec.DGDRef = nvidiacomv1alpha1.DynamoGraphDeploymentServiceRef{
				Name:        dgd.Name,
				ServiceName: componentName,
			}

			// Seed replicas only when creating the adapter; it owns subsequent changes.
			if adapter.GetResourceVersion() == "" {
				adapter.Spec.Replicas = initialReplicas
			}

			return controllerutil.SetControllerReference(dgd, adapter, r.Scheme())
		})
		if err != nil {
			logger.Error(err, "Failed to reconcile DynamoGraphDeploymentScalingAdapter", "component", componentName)
			return false, err
		}

		// Emit resource events only after the corresponding mutation succeeds.
		switch operation {
		case controllerutil.OperationResultCreated:
			created = true
			logger.Info("Created DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterCreated",
					"Create",
					"Created scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
		case controllerutil.OperationResultUpdated:
			logger.Info("Updated DynamoGraphDeploymentScalingAdapter", "adapter", adapterName, "component", componentName)
			if r.recorder != nil {
				r.recorder.Eventf(
					dgd,
					adapter,
					corev1.EventTypeNormal,
					"AdapterUpdated",
					"Update",
					"Updated scaling adapter %s for component %s",
					adapterName,
					componentName,
				)
			}
		}
	}

	// Delete adapters whose components have been removed from the DGD.
	adapterList := &nvidiacomv1alpha1.DynamoGraphDeploymentScalingAdapterList{}
	if err := r.List(
		ctx,
		adapterList,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		logger.Error(err, "Failed to list DynamoGraphDeploymentScalingAdapters")
		return false, err
	}

	for i := range adapterList.Items {
		adapter := &adapterList.Items[i]
		componentName := adapter.Spec.DGDRef.ServiceName
		if dgd.GetComponentByName(componentName) != nil {
			continue
		}

		logger.Info("Deleting orphaned DynamoGraphDeploymentScalingAdapter", "adapter", adapter.Name, "component", componentName)
		if err := r.Delete(ctx, adapter); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			logger.Error(err, "Failed to delete orphaned adapter", "adapter", adapter.Name)
			return false, err
		}
		if r.recorder != nil {
			r.recorder.Eventf(
				dgd,
				adapter,
				corev1.EventTypeNormal,
				"AdapterDeleted",
				"Delete",
				"Deleted orphaned scaling adapter %s for removed component %s",
				adapter.Name,
				componentName,
			)
		}
	}

	return created, nil
}

func generateAdapterName(dgdName, componentName string) string {
	return fmt.Sprintf("%s-%s", dgdName, strings.ToLower(componentName))
}
