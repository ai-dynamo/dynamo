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
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/yaml"
)

const (
	// LabelRunName ties a candidate to its run.
	LabelRunName = "nvidia.com/dgdr-run-name"
	// LabelCandidateID records the stable candidate id (the identity read back on list).
	LabelCandidateID = "nvidia.com/dgdr-candidate-id"
)

// KubeCluster is the Cluster implementation backed by a controller-runtime client. It
// should be a direct (uncached) client: the publisher is a short-lived sidecar, so it
// neither needs nor wants an informer cache.
//
// Required RBAC (namespaced, Role bound to the run's ServiceAccount): candidates
// get/list/create/delete and candidates/status update; runs get and runs/status patch;
// pods get (own pod, for Sweeper crash detection; resourceNames cannot be used because
// Job pod names are generated).
type KubeCluster struct {
	Client           client.Client
	Namespace        string
	RunName          string
	PodName          string
	SweeperContainer string
}

var _ Cluster = (*KubeCluster)(nil)

func (k *KubeCluster) ListCandidates(ctx context.Context) ([]dgdcreconcile.CurrentDGDC, error) {
	var list v1beta2.DynamoGraphDeploymentCandidateList
	if err := k.Client.List(ctx, &list, client.InNamespace(k.Namespace), client.MatchingLabels{LabelRunName: k.RunName}); err != nil {
		return nil, err
	}
	out := make([]dgdcreconcile.CurrentDGDC, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, dgdcreconcile.CurrentDGDC{
			Name: list.Items[i].Name,
			ID:   list.Items[i].Labels[LabelCandidateID],
		})
	}
	return out, nil
}

// CreateCandidate creates the candidate and populates its status exactly once. Both
// steps are idempotent so a retry after a partial failure converges.
func (k *KubeCluster) CreateCandidate(ctx context.Context, name string, candidate dgdcreconcile.DesiredCandidate) error {
	var doc struct {
		Spec v1beta1.DynamoGraphDeploymentSpec `json:"spec"`
	}
	if err := yaml.Unmarshal([]byte(candidate.Spec), &doc); err != nil {
		return fmt.Errorf("decoding manifest of candidate %s: %w", candidate.ID, err)
	}
	var run v1beta2.DynamoGraphDeploymentRun
	if err := k.Client.Get(ctx, types.NamespacedName{Namespace: k.Namespace, Name: k.RunName}, &run); err != nil {
		return fmt.Errorf("getting run: %w", err)
	}
	parameters, err := rawExtension(candidate.Parameters)
	if err != nil {
		return err
	}
	object := &v1beta2.DynamoGraphDeploymentCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: k.Namespace,
			Name:      name,
			Labels:    map[string]string{LabelRunName: k.RunName, LabelCandidateID: candidate.ID},
		},
		Spec: v1beta2.DynamoGraphDeploymentCandidateSpec{
			DynamoGraphDeploymentSpec: doc.Spec,
			Parameters:                parameters,
		},
	}
	if err := controllerutil.SetControllerReference(&run, object, k.Client.Scheme()); err != nil {
		return err
	}
	if err := k.Client.Create(ctx, object); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	var stored v1beta2.DynamoGraphDeploymentCandidate
	if err := k.Client.Get(ctx, types.NamespacedName{Namespace: k.Namespace, Name: name}, &stored); err != nil {
		return err
	}
	if len(stored.Status.Conditions) > 0 {
		return nil // status already populated; candidates are immutable afterwards
	}
	metrics, err := rawExtension(candidate.Metrics)
	if err != nil {
		return err
	}
	stored.Status.Metrics = metrics
	meta.SetStatusCondition(&stored.Status.Conditions, metav1.Condition{
		Type:    "Evaluated",
		Status:  metav1.ConditionTrue,
		Reason:  "Evaluated",
		Message: "evaluated by the Sweeper",
	})
	return k.Client.Status().Update(ctx, &stored)
}

func rawExtension(value map[string]any) (*runtime.RawExtension, error) {
	if value == nil {
		value = map[string]any{}
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return &runtime.RawExtension{Raw: raw}, nil
}

func (k *KubeCluster) DeleteCandidate(ctx context.Context, name string) error {
	object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{Namespace: k.Namespace, Name: name}}
	return client.IgnoreNotFound(k.Client.Delete(ctx, object))
}

func (k *KubeCluster) PatchRunStatus(ctx context.Context, status RunStatus) error {
	var run v1beta2.DynamoGraphDeploymentRun
	if err := k.Client.Get(ctx, types.NamespacedName{Namespace: k.Namespace, Name: k.RunName}, &run); err != nil {
		return err
	}
	base := run.DeepCopy()
	run.Status.Message = status.Message
	run.Status.Progress = &v1beta2.RunProgress{Round: status.Round, Evaluated: status.Evaluated}
	if !status.LastProgress.IsZero() {
		progressed := metav1.NewTime(status.LastProgress)
		run.Status.LastProgressTime = &progressed
	}
	refs := make([]v1beta2.CandidateRef, 0, len(status.CandidateNames))
	for _, name := range status.CandidateNames {
		refs = append(refs, v1beta2.CandidateRef{Name: name})
	}
	run.Status.CandidateRefs = refs
	return k.Client.Status().Patch(ctx, &run, client.MergeFrom(base))
}

// SweeperState reads the Sweeper container's status from this pod's own status.
func (k *KubeCluster) SweeperState(ctx context.Context) (SweeperState, error) {
	var pod corev1.Pod
	if err := k.Client.Get(ctx, types.NamespacedName{Namespace: k.Namespace, Name: k.PodName}, &pod); err != nil {
		return SweeperState{}, err
	}
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name != k.SweeperContainer {
			continue
		}
		if terminated := status.State.Terminated; terminated != nil {
			return SweeperState{Exited: true, ExitCode: terminated.ExitCode}, nil
		}
		return SweeperState{}, nil
	}
	return SweeperState{}, nil
}
