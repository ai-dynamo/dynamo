/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"slices"
)

// recordAcceptance retains immutable client-scoped evidence, not a second execution phase.
// state.Target is the durable pending target; its original base tokens remain frozen until commit.
func (a *MembershipAdapter) recordAcceptance(state *membershipBinding, receipt SubmissionReceipt) (bool, error) {
	if err := a.validateAcceptance(*state, receipt); err != nil {
		return false, err
	}
	if slices.ContainsFunc(state.Accepted, func(accepted SubmissionReceipt) bool { return accepted.FirstRank == receipt.FirstRank }) {
		return false, nil
	}

	// Progress flags are live observations. Acceptance history pins only the request and ownership.
	receipt.Started, receipt.Finished, receipt.Error = false, false, nil
	state.Accepted = append(slices.Clone(state.Accepted), receipt)
	slices.SortFunc(state.Accepted, func(left, right SubmissionReceipt) int { return int(left.FirstRank - right.FirstRank) })
	return true, nil
}

func (a *MembershipAdapter) validateAcceptance(state membershipBinding, receipt SubmissionReceipt) error {
	if receipt.FirstRank < 0 || int(receipt.FirstRank) >= len(state.Instances) {
		return fmt.Errorf("vLLM acceptance receipt has no original client")
	}
	process, err := a.process(receipt.FirstRank)
	if err != nil {
		return err
	}
	size, failure := a.resolvePlan(state.Target.BaseTopology, state.Target.Plan)
	if failure != nil {
		return fmt.Errorf("vLLM pending plan is unsupported: %s", failure.Message)
	}
	expected := ScaleRequest{OperationID: state.Target.TransitionID, ExpectedInstanceID: state.Instances[receipt.FirstRank],
		BaseTopologyToken: state.TopologyTokens[receipt.FirstRank], NewDataParallelSize: size, DrainTimeout: state.DrainTimeout}
	if receipt.FirstRank != process.Rank || receipt.LocalSize != int32(len(process.Membership.Members)) ||
		receipt.Request != expected || receipt.Error != nil {
		return fmt.Errorf("vLLM local acceptance does not match the original client, range and immutable request")
	}
	return nil
}
