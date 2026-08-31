/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	dgdPodSnapshotRefIndex = "checkpoint.podsnapshot.dgd.nvidia.com"
	dcdPodSnapshotRefIndex = "checkpoint.podsnapshot.dcd.nvidia.com"
)

func dgdPodSnapshotRefIndexValues(raw client.Object) []string {
	dgd, ok := raw.(*nvidiacomv1beta1.DynamoGraphDeployment)
	if !ok {
		return nil
	}

	// De-duplicate references shared by multiple components so one Snapshot
	// event produces one request for the owning graph.
	seen := make(map[string]struct{})
	for i := range dgd.Spec.Components {
		config := dynamo.GetCheckpoint(&dgd.Spec.Components[i])
		if config == nil || config.CheckpointRef == nil {
			continue
		}
		name := strings.TrimSpace(*config.CheckpointRef)
		if name != "" {
			seen[name] = struct{}{}
		}
	}
	refs := make([]string, 0, len(seen))
	for name := range seen {
		refs = append(refs, name)
	}
	return refs
}

func dcdPodSnapshotRefIndexValues(raw client.Object) []string {
	dcd, ok := raw.(*nvidiacomv1beta1.DynamoComponentDeployment)
	if !ok {
		return nil
	}
	if dcd.Annotations[consts.CheckpointSourceKindAnnotation] == consts.CheckpointSourceKindLegacy {
		return nil
	}
	config := dynamo.GetCheckpoint(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
	if config == nil || config.CheckpointRef == nil {
		return nil
	}
	name := strings.TrimSpace(*config.CheckpointRef)
	if name == "" {
		return nil
	}
	return []string{name}
}

func (r *DynamoGraphDeploymentReconciler) mapPodSnapshotToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	graphs := &nvidiacomv1beta1.DynamoGraphDeploymentList{}
	if err := r.List(
		ctx,
		graphs,
		client.InNamespace(obj.GetNamespace()),
		client.MatchingFields{dgdPodSnapshotRefIndex: obj.GetName()},
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoGraphDeployments for PodSnapshot event")
		return nil
	}

	requests := make([]ctrl.Request, 0, len(graphs.Items))
	for i := range graphs.Items {
		graph := &graphs.Items[i]
		requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
			Namespace: graph.Namespace,
			Name:      graph.Name,
		}})
	}
	return requests
}

func (r *DynamoComponentDeploymentReconciler) mapPodSnapshotToDCDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	components := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.List(
		ctx,
		components,
		client.InNamespace(obj.GetNamespace()),
		client.MatchingFields{dcdPodSnapshotRefIndex: obj.GetName()},
	); err != nil {
		log.FromContext(ctx).Error(err, "Failed to list DynamoComponentDeployments for PodSnapshot event")
		return nil
	}

	requests := make([]ctrl.Request, 0, len(components.Items))
	for i := range components.Items {
		component := &components.Items[i]
		requests = append(requests, ctrl.Request{NamespacedName: types.NamespacedName{
			Namespace: component.Namespace,
			Name:      component.Name,
		}})
	}
	return requests
}
