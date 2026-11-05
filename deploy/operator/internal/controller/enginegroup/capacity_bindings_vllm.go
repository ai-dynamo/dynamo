/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/workload"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type vllmCapacityBindings struct {
	Version int                                     `json:"version"`
	WorldID types.UID                               `json:"cliqueUID"`
	Slots   map[domain.CapacitySlotID]domain.PodUID `json:"slots"`
}

func (a *VLLMCapacity) capacityBindings() (vllmCapacityBindings, error) {
	// Missing authority cannot be reconstructed from replacement Pods that happen to occupy the same slots.
	literal := a.Group.Annotations[consts.KubeAnnotationDynamoVLLMCapacityBindings]
	var bindings vllmCapacityBindings
	decoder := json.NewDecoder(strings.NewReader(literal))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&bindings); err != nil || !json.Valid([]byte(literal)) || bindings.Version != 1 || bindings.Slots == nil ||
		(bindings.WorldID != "" && bindings.WorldID != a.WorldID) || (bindings.WorldID == "" && len(bindings.Slots) != 0) {
		return vllmCapacityBindings{}, fmt.Errorf("%w: vLLM capacity bindings are missing, invalid, or belong to another physical world", domain.ErrRecoveryRequired)
	}
	for slot, uid := range bindings.Slots {
		if slot == "" || uid == "" {
			return vllmCapacityBindings{}, fmt.Errorf("%w: incomplete vLLM slot binding", domain.ErrRecoveryRequired)
		}
	}
	return bindings, nil
}

func (a *VLLMCapacity) bindAllocations(ctx context.Context, allocations []workload.Allocation) error {
	// Binding records physical identity only; native bootstrap and membership commit do not wait for this write.
	bindings, err := a.capacityBindings()
	if err != nil {
		return err
	}
	before, err := json.Marshal(bindings)
	if err != nil {
		return err
	}
	bindings.WorldID = a.WorldID
	for _, allocation := range allocations {
		if previous, found := bindings.Slots[allocation.SlotID]; found && previous != allocation.Pod.UID {
			return fmt.Errorf("%w: vLLM slot %s belongs to another Pod UID", domain.ErrRecoveryRequired, allocation.SlotID)
		}
		bindings.Slots[allocation.SlotID] = allocation.Pod.UID
	}
	after, err := json.Marshal(bindings)
	if err != nil || bytes.Equal(before, after) {
		return err
	}

	// Optimistic UID/RV preconditions prevent stale observations from overwriting a later physical binding.
	patch, err := json.Marshal(map[string]any{"metadata": map[string]any{
		"uid": a.Group.UID, "resourceVersion": a.Group.ResourceVersion,
		"annotations": map[string]string{consts.KubeAnnotationDynamoVLLMCapacityBindings: string(after)},
	}})
	if err != nil {
		return err
	}
	if err := a.Client.Patch(ctx, a.Group, client.RawPatch(types.MergePatchType, patch)); err != nil {
		return fmt.Errorf("persist exact vLLM capacity bindings: %w", err)
	}
	return nil
}
