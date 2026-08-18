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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
)

type workloadProgramRequest struct {
	// DGD is the mutable primary object. Programs may mutate and directly
	// persist non-status fields; status is returned through workloadProgramResult.
	DGD *nvidiacomv1beta1.DynamoGraphDeployment
}

type workloadProgramEvent struct {
	Type    string
	Reason  string
	Message string
}

type workloadProgramResult struct {
	ctrl.Result
	// Status is the complete desired DGD status after this attempt. It remains
	// meaningful when Reconcile returns an error so the outer reconciler can
	// persist any partial progress before returning that error.
	Status nvidiacomv1beta1.DynamoGraphDeploymentStatus
	// Events contains status-transition events that become real only after
	// Status has been persisted successfully.
	Events []workloadProgramEvent
}

type programRestart struct {
	State  *dynamo.RestartState
	Status *nvidiacomv1beta1.RestartStatus
}

type restartProgressResolver func(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeployment,
	[]string,
) []string

// workloadProgram owns the complete graph-workload state machine for one
// pathway. The common DGD controller selects one program and invokes it once;
// it does not drive provider rendering, rollout, readiness, or cleanup through
// lifecycle callbacks.
type workloadProgram interface {
	Reconcile(context.Context, workloadProgramRequest) (workloadProgramResult, error)
}

func (r *DynamoGraphDeploymentReconciler) selectWorkloadProgram(
	provider workloadProvider,
) (workloadProgram, error) {
	// Dispatch each durable provider value to its complete workload program.
	switch provider {
	case workloadProviderGrove:
		return r.newGroveProgram(), nil
	case workloadProviderComponent:
		return r.newComponentProgram(), nil
	default:
		return nil, fmt.Errorf("unsupported workload provider %q", provider)
	}
}

func newWorkloadProgramResult(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workloadProgramResult {
	status := dgd.DeepCopy().Status
	return workloadProgramResult{Status: status}
}

func (r *workloadProgramResult) Eventf(
	eventType string,
	reason string,
	format string,
	args ...any,
) {
	r.Events = append(r.Events, workloadProgramEvent{
		Type:    eventType,
		Reason:  reason,
		Message: fmt.Sprintf(format, args...),
	})
}

func recordRestartTransition(
	previous *nvidiacomv1beta1.RestartStatus,
	next *nvidiacomv1beta1.RestartStatus,
	result *workloadProgramResult,
) {
	if next == nil || next.Phase != nvidiacomv1beta1.RestartPhaseSuperseded {
		return
	}
	if previous != nil && previous.ObservedID == next.ObservedID && previous.Phase == next.Phase {
		return
	}
	result.Eventf(
		corev1.EventTypeWarning,
		"RestartSuperseded",
		"Restart %s superseded by rolling update",
		next.ObservedID,
	)
}

func (r *workloadProgramResult) Fail(generation int64, reason Reason, err error) {
	r.Status.State = nvidiacomv1beta1.DGDStateFailed
	meta.SetStatusCondition(&r.Status.Conditions, metav1.Condition{
		Type:               "Ready",
		Status:             metav1.ConditionFalse,
		ObservedGeneration: generation,
		Reason:             string(reason),
		Message:            err.Error(),
	})
}

func (r *workloadProgramResult) applyReconcileResult(
	generation int64,
	result ReconcileResult,
) {
	r.Status.State = result.State
	r.Status.Components = result.ComponentStatus
	if rollingUpdateInProgress(r.Status.RollingUpdate) {
		r.Status.State = nvidiacomv1beta1.DGDStatePending
	}
	meta.SetStatusCondition(&r.Status.Conditions, readyCondition(generation, r.Status, result))
	r.Status.ObservedGeneration = generation
}

func readyCondition(
	generation int64,
	status nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	workloads ReconcileResult,
) metav1.Condition {
	if rollingUpdateInProgress(status.RollingUpdate) {
		return metav1.Condition{
			Type:               "Ready",
			Status:             metav1.ConditionFalse,
			ObservedGeneration: generation,
			Reason:             "rolling_update_in_progress",
			Message:            "Rolling update in progress",
		}
	}

	conditionStatus := metav1.ConditionFalse
	if workloads.State == nvidiacomv1beta1.DGDStateSuccessful {
		conditionStatus = metav1.ConditionTrue
	}
	return metav1.Condition{
		Type:               "Ready",
		Status:             conditionStatus,
		ObservedGeneration: generation,
		Reason:             string(workloads.Reason),
		Message:            string(workloads.Message),
	}
}

func rollingUpdateInProgress(status *nvidiacomv1beta1.RollingUpdateStatus) bool {
	if status == nil {
		return false
	}
	return status.Phase == nvidiacomv1beta1.RollingUpdatePhasePending ||
		status.Phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress
}

func rollingUpdatePhase(status *nvidiacomv1beta1.RollingUpdateStatus) nvidiacomv1beta1.RollingUpdatePhase {
	if status == nil {
		return nvidiacomv1beta1.RollingUpdatePhaseNone
	}
	return status.Phase
}

type workloadProgramFailure struct {
	reason Reason
	err    error
}

func (e *workloadProgramFailure) Error() string {
	return e.err.Error()
}

func (e *workloadProgramFailure) Unwrap() error {
	return e.err
}

func failWorkloadProgram(reason Reason, err error) error {
	return &workloadProgramFailure{reason: reason, err: err}
}

func workloadProgramFailureReason(err error) (Reason, bool) {
	var failure *workloadProgramFailure
	if !errors.As(err, &failure) {
		return "", false
	}
	return failure.reason, true
}
