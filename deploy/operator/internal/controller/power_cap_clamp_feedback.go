/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"sync"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const powerCapClampedReason = "PowerCapClamped"

type powerCapClampObservation struct {
	Component      string
	Product        string
	Direction      string
	RequestedWatts int64
	EffectiveWatts int64
	MinWatts       int64
	MaxWatts       int64
}

type trackedPowerCapClamp struct {
	Scope       string
	Observation powerCapClampObservation
}

type powerCapClampEventTracker struct {
	mu       sync.Mutex
	observed map[string]trackedPowerCapClamp
}

func (tracker *powerCapClampEventTracker) unseen(
	scope string,
	observations []powerCapClampObservation,
) []powerCapClampObservation {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	if tracker.observed == nil {
		tracker.observed = make(map[string]trackedPowerCapClamp)
	}

	live := make(map[string]struct{}, len(observations))
	unseen := make([]powerCapClampObservation, 0, len(observations))
	for _, observation := range observations {
		key := scope + "\x00" + observation.Component
		live[key] = struct{}{}
		previous, exists := tracker.observed[key]
		if exists && previous.Scope == scope && previous.Observation == observation {
			continue
		}
		tracker.observed[key] = trackedPowerCapClamp{
			Scope:       scope,
			Observation: observation,
		}
		unseen = append(unseen, observation)
	}
	for key, previous := range tracker.observed {
		if previous.Scope != scope {
			continue
		}
		if _, exists := live[key]; !exists {
			delete(tracker.observed, key)
		}
	}
	return unseen
}

func (tracker *powerCapClampEventTracker) forgetScope(scope string) {
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	for key, previous := range tracker.observed {
		if previous.Scope == scope {
			delete(tracker.observed, key)
		}
	}
}

func (r *DynamoGraphDeploymentReconciler) emitPowerCapClampFeedback(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	observations := make([]powerCapClampObservation, 0)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		config, err := componentPowerConfig(component, r.PowerQualification)
		if err != nil || config.requestedCapWatts == config.bounds.InGateWatts {
			continue
		}
		direction := "above_max"
		if config.requestedCapWatts < config.bounds.QualifiedMinWatts {
			direction = "below_min"
		}
		observations = append(observations, powerCapClampObservation{
			Component:      component.ComponentName,
			Product:        config.qualifiedProduct,
			Direction:      direction,
			RequestedWatts: config.requestedCapWatts,
			EffectiveWatts: config.bounds.InGateWatts,
			MinWatts:       config.bounds.QualifiedMinWatts,
			MaxWatts:       config.bounds.QualifiedMaxWatts,
		})
	}

	for _, observation := range r.powerClampEvents.unseen(string(dgd.UID), observations) {
		recordPowerCapClamp(observation.Direction, observation.Product)
		message := fmt.Sprintf(
			"component %q requested %dW per GPU; using qualified %dW for product %q range [%d,%d]W",
			observation.Component,
			observation.RequestedWatts,
			observation.EffectiveWatts,
			observation.Product,
			observation.MinWatts,
			observation.MaxWatts,
		)
		log.FromContext(ctx).Info(
			"transactional power intent clamped to qualified product range",
			"component", observation.Component,
			"product", observation.Product,
			"direction", observation.Direction,
			"requestedWatts", observation.RequestedWatts,
			"effectiveWatts", observation.EffectiveWatts,
			"qualifiedMinWatts", observation.MinWatts,
			"qualifiedMaxWatts", observation.MaxWatts,
		)
		if r.Recorder != nil {
			r.Recorder.Eventf(
				dgd,
				nil,
				corev1.EventTypeWarning,
				powerCapClampedReason,
				"ValidatePowerCap",
				"%s",
				message,
			)
		}
	}
}
