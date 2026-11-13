//go:build clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
)

type clusterParetoObservation struct {
	Snapshots     []dgdrrunpublisher.Snapshot
	Records       []clusterParetoRecord
	TerminalFront []string
}

type clusterParetoRecord struct {
	ID         string             `json:"id"`
	Status     string             `json:"status"`
	Objectives map[string]float64 `json:"objectives"`
}

type clusterParetoObservationResult struct {
	Observation clusterParetoObservation
	Err         error
}

// observeClusterPareto reads existing output files through the real sweeper container.
func observeClusterPareto(ctx context.Context, config *rest.Config, namespace, podName, script string) <-chan clusterParetoObservationResult {
	results := make(chan clusterParetoObservationResult, 1)
	go func() {
		observation, err := readClusterParetoObservation(ctx, config, namespace, podName, script)
		results <- clusterParetoObservationResult{Observation: observation, Err: err}
	}()
	return results
}

func readClusterParetoObservation(ctx context.Context, config *rest.Config, namespace, podName, script string) (clusterParetoObservation, error) {
	// Exec only a read-only observer; the production command and output callbacks remain intact.
	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return clusterParetoObservation{}, err
	}
	request := kubeClient.CoreV1().RESTClient().Post().Namespace(namespace).Resource("pods").Name(podName).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: dgdrRunSweeperContainer,
			Command:   []string{"python", "-u", "-c", script, dgdrRunSnapshotPath},
			Stdout:    true,
			Stderr:    true,
		}, clientgoscheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(config, "POST", request.URL())
	if err != nil {
		return clusterParetoObservation{}, err
	}

	// Preserve diagnostics if the observer cannot read the real result before the container exits.
	var stdout, stderr bytes.Buffer
	streamErr := executor.StreamWithContext(ctx, remotecommand.StreamOptions{Stdout: &stdout, Stderr: &stderr})
	var observation clusterParetoObservation
	decoder := json.NewDecoder(&stdout)
	for {
		var event struct {
			Snapshot      *dgdrrunpublisher.Snapshot `json:"snapshot"`
			Records       []clusterParetoRecord      `json:"records"`
			TerminalFront []string                   `json:"terminalFront"`
		}
		if err := decoder.Decode(&event); err == io.EOF {
			break
		} else if err != nil {
			return observation, fmt.Errorf("decode AISimulate observation: %w\nstderr: %s", err, stderr.String())
		}
		if event.Snapshot != nil {
			observation.Snapshots = append(observation.Snapshots, *event.Snapshot)
		}
		if event.Records != nil {
			observation.Records = event.Records
			observation.TerminalFront = event.TerminalFront
		}
	}
	if observation.Records == nil {
		return observation, fmt.Errorf("AISimulate ledger was not captured (exec: %v)\nstderr: %s", streamErr, stderr.String())
	}

	// Keep complete evidence even if the sweeper exits immediately after emitting it.
	return observation, nil
}

func assertClusterParetoPublication(t *testing.T, observation clusterParetoObservation, liveRuns []v1beta2.DynamoGraphDeploymentRun, completed *v1beta2.DynamoGraphDeploymentRun) {
	t.Helper()
	t.Log("Require a nontrivial real Pareto search and live publication, not just terminal success")
	t.Logf("Captured %d live snapshots and %d evaluated records; terminal front: %v", len(observation.Snapshots), len(observation.Records), observation.TerminalFront)
	limit := int(completed.Spec.Recommendation.MaxCandidates)
	require.Greater(t, len(observation.TerminalFront), limit, "fixture must find more Pareto points than the publication cap")
	require.NotEmpty(t, liveRuns, "publisher never exposed a Running Run with Candidates")
	require.NotEmpty(t, observation.Snapshots, "no live snapshots were observed in the real Job")
	fronts := make([][]string, len(observation.Records))
	feasibleCounts := make([]int, len(observation.Records))
	for i := range observation.Records {
		fronts[i] = clusterParetoFront(t, observation.Records[:i+1])
		if i > 0 {
			feasibleCounts[i] = feasibleCounts[i-1]
		}
		if observation.Records[i].Status == "feasible" {
			feasibleCounts[i]++
		}
	}
	require.Greater(t, feasibleCounts[len(fronts)-1], len(fronts[len(fronts)-1]), "fixture must evaluate at least one dominated point")
	t.Logf("Final evaluated pool: %d feasible points, %d nondominated points", feasibleCounts[len(fronts)-1], len(fronts[len(fronts)-1]))

	// Check each actual live snapshot against its evaluated prefix's two maximizing objectives.
	sawTradeoffWithDominatedPoint := false
	for _, snapshot := range observation.Snapshots {
		evaluated := int(snapshot.Progress.Evaluated)
		require.Positive(t, evaluated)
		require.LessOrEqual(t, evaluated, len(fronts), "snapshot evaluated count exceeds the SDK ledger")
		front := fronts[evaluated-1]
		require.Equal(t, len(front), int(snapshot.Progress.ParetoFront),
			"live front cardinality differs from AISimulate after %d evaluations", evaluated)
		require.LessOrEqual(t, len(snapshot.Candidates), limit)
		for _, candidate := range snapshot.Candidates {
			require.Contains(t, front, candidate.ID, "live snapshot published a dominated point after %d evaluations", evaluated)
		}
		if len(front) > limit && feasibleCounts[evaluated-1] > len(front) {
			sawTradeoffWithDominatedPoint = true
		}
	}
	require.True(t, sawTradeoffWithDominatedPoint, "fixture must observe a live front larger than the cap and an evaluated dominated point")

	// Correlate a live publication without assuming the two readers sample every same file version.
	sawBoundedLiveFront := false
	for _, run := range liveRuns {
		progress := run.Status.Progress.Candidates
		matchedSnapshot := false
		for _, snapshot := range observation.Snapshots {
			if int64(snapshot.Progress.ParetoFront) != progress.ParetoFront {
				continue
			}
			allowed := make(map[string]bool, len(snapshot.Candidates))
			for _, candidate := range snapshot.Candidates {
				if candidate.Outcome == dgdrrunpublisher.OutcomeMaterialized {
					allowed[dgdrrunpublisher.CandidateName(run.Name, string(run.UID), candidate.ID)] = true
				}
			}
			allOnFront := len(allowed) == len(run.Status.CandidateRefs)
			for _, ref := range run.Status.CandidateRefs {
				allOnFront = allOnFront && allowed[ref.Name]
			}
			matchedSnapshot = matchedSnapshot || allOnFront
		}
		require.Equal(t, len(run.Status.CandidateRefs), int(progress.Published))
		require.LessOrEqual(t, len(run.Status.CandidateRefs), limit)
		sawBoundedLiveFront = sawBoundedLiveFront || (matchedSnapshot && int(progress.ParetoFront) > limit)
	}
	require.True(t, sawBoundedLiveFront, "publisher must expose a larger-than-cap front while the Run is still Running")

	t.Log("Verify terminal Run references come from the SDK's final selected front")
	require.Equal(t, len(observation.TerminalFront), int(completed.Status.Progress.Candidates.ParetoFront))
	allowed := make(map[string]bool, len(observation.TerminalFront))
	for _, id := range observation.TerminalFront {
		allowed[dgdrrunpublisher.CandidateName(completed.Name, string(completed.UID), id)] = true
	}
	for _, ref := range completed.Status.CandidateRefs {
		require.True(t, allowed[ref.Name], "terminal Candidate %s is outside AISimulate's selected front", ref.Name)
	}
}

// clusterParetoFront independently checks the two supported maximizing objectives in test data.
func clusterParetoFront(t *testing.T, records []clusterParetoRecord) []string {
	t.Helper()
	var front []string
	for _, candidate := range records {
		if candidate.Status != "feasible" {
			continue
		}
		require.Contains(t, candidate.Objectives, "throughput_per_gpu")
		require.Contains(t, candidate.Objectives, "throughput_per_user")
		dominated := false
		for _, other := range records {
			if other.Status != "feasible" {
				continue
			}
			gpu, user := candidate.Objectives["throughput_per_gpu"], candidate.Objectives["throughput_per_user"]
			otherGPU, otherUser := other.Objectives["throughput_per_gpu"], other.Objectives["throughput_per_user"]
			dominated = dominated || (otherGPU >= gpu && otherUser >= user && (otherGPU > gpu || otherUser > user))
		}
		if !dominated {
			front = append(front, candidate.ID)
		}
	}
	return front
}
