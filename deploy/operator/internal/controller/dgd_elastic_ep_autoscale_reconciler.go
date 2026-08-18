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
	"net/http"
	"sync"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// elasticEPFollowerSuffix mirrors commonconsts.GroveRoleSuffixFollower, which lands with the
// Phase-4 follower (a sibling branch). It is duplicated here so Phase 6 stays independent of
// Phase 4; when both merge this reverts to the shared const.
const elasticEPFollowerSuffix = "flw"

// dgdElasticEPAutoscaleReconciler is Phase 6: it reads the get_ep_capacity endpoint over the
// Phase-3 leader Service and converges the engine's data-parallel size on the owned capacity
// that has joined. It does nothing unless the ElasticEPAutoscale gate is on (default off,
// until the scale_elastic_ep rollback #12991 lands) -- an operator that can wedge a serving
// engine unattended is worse than a human choosing to take the risk.
type dgdElasticEPAutoscaleReconciler struct {
	client client.Client
	// podReader is an UNCACHED reader. The manager's cached client has no Pod informer (this
	// controller does not watch Pods), so listing pods through it returns empty -- which a
	// cluster run caught as observedOwned=0 for a leader whose IP plainly matched Ray.
	podReader    client.Reader
	gate         features.Gate
	httpClient   *http.Client
	settleWindow time.Duration
	now          func() time.Time

	mu    sync.Mutex
	state map[string]*epAutoscaleState
}

// epAutoscaleState is the per-leader settle-window and fire-once memory. In-memory is enough
// for a draft (it resets on operator restart, which only re-opens a settle window); the design
// calls for recording the last-applied topology in status, which is the durable home.
type epAutoscaleState struct {
	lastObserved int
	observedAt   time.Time
	lastApplied  int
}

func newDGDElasticEPAutoscaleReconciler(c client.Client, podReader client.Reader, gate features.Gate) *dgdElasticEPAutoscaleReconciler {
	return &dgdElasticEPAutoscaleReconciler{
		client:       c,
		podReader:    podReader,
		gate:         gate,
		httpClient:   &http.Client{Timeout: 30 * time.Second},
		settleWindow: 60 * time.Second, // starting point; tune from join times on the target cluster
		now:          time.Now,
		state:        map[string]*epAutoscaleState{},
	}
}

// Reconcile is a no-op unless the gate is on. Per-leader failures are logged and swallowed: an
// HTTP hiccup or a busy engine must not wedge the whole DGD reconcile.
func (r *dgdElasticEPAutoscaleReconciler) Reconcile(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	if r.gate == nil || !r.gate.Enabled(features.ElasticEPAutoscale) {
		return nil
	}
	logger := log.FromContext(ctx)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if c := dynamo.GetMainContainer(component); c == nil || !dynamo.IsElasticEPRayLaunch(c) {
			continue
		}
		if err := r.reconcileComponent(ctx, dgd, component); err != nil {
			logger.Info("elastic-EP autoscale: deferring", "component", component.ComponentName, "reason", err.Error())
		}
	}
	return nil
}

func (r *dgdElasticEPAutoscaleReconciler) reconcileComponent(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	logger := log.FromContext(ctx)
	componentName := component.ComponentName
	// The leader's headless Service is named the same way the render names it.
	leaderService := dynamo.GetDCDResourceName(dgd, componentName, "")
	baseURL := elasticEPLeaderBaseURL(leaderService, dgd.Namespace)
	key := dgd.Namespace + "/" + componentName

	cap, err := readEPCapacity(ctx, r.httpClient, baseURL)
	if err != nil {
		return err
	}

	ownedIPs, err := r.ownedPodIPs(ctx, dgd, componentName)
	if err != nil {
		return err
	}
	observed := dynamo.OwnedRayNodeCount(cap, ownedIPs)
	desired := 1 + r.desiredFollowers(ctx, dgd, componentName)
	decision := dynamo.DecideEPScale(cap.DataParallelSize, observed, desired)

	// Phase 7b visibility: the three numbers that make the deployment's state legible from
	// outside -- desired followers, joined capacity, and the engine's current size -- plus the
	// Ray total so a superseded-generation node shows up as the difference.
	logger.Info("elastic-EP autoscale",
		"component", componentName,
		"desired", desired, "observedOwned", observed, "current", cap.DataParallelSize,
		"rayNodesTotal", len(cap.Nodes), "action", decision.Action.String(), "target", decision.Target)

	switch decision.Action {
	case dynamo.EPScaleFault:
		logger.Info("elastic-EP autoscale: FAULT -- owned capacity fell below the running size; not scaling (recovery, not scale-down)",
			"component", componentName, "current", cap.DataParallelSize, "observedOwned", observed)
		return nil
	case dynamo.EPScaleNone:
		r.markObserved(key, observed)
		return nil
	}

	// Grow waits for the settle window so a spike of several followers costs one rebuild, not
	// several; shrink is desire-driven and need not wait.
	if decision.Action == dynamo.EPScaleGrow && !r.settled(key, observed) {
		logger.Info("elastic-EP autoscale: waiting for arrivals to settle before growing",
			"component", componentName, "observedOwned", observed, "settleWindow", r.settleWindow)
		return nil
	}

	// Fire once per change: reconcilers run continuously, and the endpoint takes a target, so a
	// repeated call to the same size is at best wasteful and at worst a needless rebuild.
	if r.alreadyApplied(key, decision.Target) {
		return nil
	}
	if err := scaleElasticEP(ctx, r.httpClient, baseURL, decision.Target); err != nil {
		if errors.Is(err, errEngineBusy) {
			logger.Info("elastic-EP autoscale: engine busy, will retry on the next reconcile", "component", componentName)
			return nil
		}
		return fmt.Errorf("scale to %d: %w", decision.Target, err)
	}
	r.recordApplied(key, decision.Target)
	logger.Info("elastic-EP autoscale: applied", "component", componentName,
		"action", decision.Action.String(), "newDataParallelSize", decision.Target)
	return nil
}

// ownedPodIPs returns the pod IPs of this deployment's Running elastic-EP worker pods -- the
// leader and its followers. OwnedRayNodeCount intersects these with what Ray reports, so a
// superseded-generation pod still alive in Ray is filtered out.
func (r *dgdElasticEPAutoscaleReconciler) ownedPodIPs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) (map[string]struct{}, error) {
	pods := &corev1.PodList{}
	if err := r.podReader.List(ctx, pods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
		},
	); err != nil {
		return nil, err
	}
	ips := make(map[string]struct{}, len(pods.Items))
	for i := range pods.Items {
		p := &pods.Items[i]
		if p.Status.Phase == corev1.PodRunning && p.Status.PodIP != "" && p.DeletionTimestamp == nil {
			ips[p.Status.PodIP] = struct{}{}
		}
	}
	return ips, nil
}

// desiredFollowers is the follower count the scaling adapter is driving -- the resting Deployment
// replica count of the synthesized "<leader>-flw" follower. Zero when the follower Deployment is
// absent (e.g. before the follower has been created), which yields a desired data-parallel size
// of one: the leader alone.
func (r *dgdElasticEPAutoscaleReconciler) desiredFollowers(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) int {
	followerDCD := dynamo.GetDCDResourceName(dgd, componentName, "") + "-" + elasticEPFollowerSuffix
	dep := &appsv1.Deployment{}
	if err := r.client.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: followerDCD}, dep); err != nil {
		return 0
	}
	if dep.Spec.Replicas == nil {
		return 0
	}
	return int(*dep.Spec.Replicas)
}

// --- settle window + fire-once state ---

func (r *dgdElasticEPAutoscaleReconciler) settled(key string, observed int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.state[key]
	if s == nil || s.lastObserved != observed {
		r.state[key] = &epAutoscaleState{lastObserved: observed, observedAt: r.now()}
		return false // capacity just changed -> (re)open the window
	}
	return r.now().Sub(s.observedAt) >= r.settleWindow
}

func (r *dgdElasticEPAutoscaleReconciler) markObserved(key string, observed int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.state[key]
	if s == nil || s.lastObserved != observed {
		r.state[key] = &epAutoscaleState{lastObserved: observed, observedAt: r.now()}
	}
}

func (r *dgdElasticEPAutoscaleReconciler) alreadyApplied(key string, target int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.state[key]
	return s != nil && s.lastApplied == target
}

func (r *dgdElasticEPAutoscaleReconciler) recordApplied(key string, target int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s := r.state[key]
	if s == nil {
		s = &epAutoscaleState{}
		r.state[key] = s
	}
	s.lastApplied = target
}
