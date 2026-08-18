/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	"k8s.io/apimachinery/pkg/api/equality"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	powerInventoryStateAnnotation = "dynamo.nvidia.com/power-inventory-state"
	powerInventoryStateVersion    = 1
	maxPowerInventoryStateBytes   = 131072
	maxReportedPodUIDs            = 1024
	maxPowerInventoryPodUIDBytes  = 64
)

// powerInventoryState is bounded operator-private convergence metadata. It is
// deliberately not part of the public DGPB status contract. ReportedPodUIDs
// remembers the conservative class transition for an assigned Pod: once that
// Pod reports, later missing evidence is U_c. Saturation fails conservative.
type powerInventoryState struct {
	Version                  int      `json:"v"`
	TargetEpoch              int64    `json:"e"`
	Fingerprint              string   `json:"f"`
	RecoveryFitSinceUnixNano int64    `json:"r,omitempty"`
	AllPodsReported          bool     `json:"a,omitempty"`
	ReportedPodUIDs          []string `json:"p,omitempty"`
}

type powerReportHistory struct {
	All     bool
	PodUIDs []string
}

func (h powerReportHistory) Contains(podUID string) bool {
	if h.All {
		return true
	}
	index := sort.SearchStrings(h.PodUIDs, podUID)
	return index < len(h.PodUIDs) && h.PodUIDs[index] == podUID
}

func loadPowerInventoryState(dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget) (powerInventoryState, bool) {
	encoded := dgpb.Annotations[powerInventoryStateAnnotation]
	if encoded == "" || len(encoded) > maxPowerInventoryStateBytes {
		return powerInventoryState{}, false
	}
	state := powerInventoryState{}
	if err := json.Unmarshal([]byte(encoded), &state); err != nil ||
		state.Version != powerInventoryStateVersion ||
		state.TargetEpoch < 0 ||
		state.RecoveryFitSinceUnixNano < 0 ||
		len(state.ReportedPodUIDs) > maxReportedPodUIDs ||
		!validPowerInventoryFingerprint(state.Fingerprint) {
		return powerInventoryState{}, false
	}
	if state.AllPodsReported && len(state.ReportedPodUIDs) != 0 {
		return powerInventoryState{}, false
	}
	for i, uid := range state.ReportedPodUIDs {
		if uid == "" || len(uid) > maxPowerInventoryPodUIDBytes ||
			(i > 0 && state.ReportedPodUIDs[i-1] >= uid) {
			return powerInventoryState{}, false
		}
	}
	return state, true
}

func validPowerInventoryFingerprint(fingerprint string) bool {
	if len(fingerprint) != 64 || strings.ToLower(fingerprint) != fingerprint {
		return false
	}
	_, err := hex.DecodeString(fingerprint)
	return err == nil
}

func mergeReportedPowerPods(
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	inventory dgdPowerBudgetInventory,
	state powerInventoryState,
	stateValid bool,
) powerReportHistory {
	reported := make(map[string]struct{}, len(state.ReportedPodUIDs))
	all := false
	if stateValid {
		all = state.AllPodsReported
		for _, uid := range state.ReportedPodUIDs {
			reported[uid] = struct{}{}
		}
	} else if dgpb.Status.InventoryEpoch > 0 {
		// Missing or corrupt convergence metadata on an established DGPB must
		// never make assigned capacity look new again.
		all = true
	}
	if all {
		return powerReportHistory{All: true}
	}
	for i := range inventory.Pods {
		pod := &inventory.Pods[i]
		if pod.Annotations[powerbudget.AgentReportAnnotation] == "" {
			continue
		}
		if pod.UID == "" || len(pod.UID) > maxPowerInventoryPodUIDBytes {
			return powerReportHistory{All: true}
		}
		uid := string(pod.UID)
		if _, exists := reported[uid]; exists {
			continue
		}
		if len(reported) == maxReportedPodUIDs {
			return powerReportHistory{All: true}
		}
		reported[uid] = struct{}{}
	}

	uids := make([]string, 0, len(reported))
	for uid := range reported {
		uids = append(uids, uid)
	}
	sort.Strings(uids)
	return powerReportHistory{PodUIDs: uids}
}

func powerReportHistoryEqual(state powerInventoryState, history powerReportHistory) bool {
	return state.AllPodsReported == history.All &&
		slices.Equal(state.ReportedPodUIDs, history.PodUIDs)
}

func powerInventoryStatusSemanticallyEqual(
	current nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
	desired nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
) bool {
	current.InventoryEpoch = 0
	desired.InventoryEpoch = 0
	return equality.Semantic.DeepEqual(current, desired)
}

func persistPowerInventoryState(
	ctx context.Context,
	kubeClient client.Client,
	dgpb *nvidiacomv1beta1.DynamoGraphPowerBudget,
	state powerInventoryState,
) error {
	encoded, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encode power inventory state: %w", err)
	}
	if len(encoded) > maxPowerInventoryStateBytes {
		return fmt.Errorf("power inventory state exceeds %d bytes", maxPowerInventoryStateBytes)
	}
	before := dgpb.DeepCopy()
	if dgpb.Annotations == nil {
		dgpb.Annotations = map[string]string{}
	}
	dgpb.Annotations[powerInventoryStateAnnotation] = string(encoded)
	if err := kubeClient.Patch(ctx, dgpb, client.MergeFrom(before)); err != nil {
		return fmt.Errorf("patch DynamoGraphPowerBudget inventory state: %w", err)
	}
	return nil
}
