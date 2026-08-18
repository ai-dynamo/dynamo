/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestMetricLabelsBoundedAndRetireGPUIdentity(t *testing.T) {
	t.Log("Require every power metric to use only bounded non-identity label dimensions")
	if len(powerMetricCollectors) != len(powerMetricVariableLabels) {
		t.Fatalf("metric collectors = %d, label declarations = %d", len(powerMetricCollectors), len(powerMetricVariableLabels))
	}
	for _, labels := range powerMetricVariableLabels {
		for _, label := range labels {
			if powerMetricLabelIsIdentity(label) {
				t.Fatalf("power metric exposes identity label %q", label)
			}
		}
	}

	t.Log("Normalize arbitrary report content instead of retaining a stale GPU identity as a label value")
	recordPowerRequestVector(false, nvidiacomv1beta1.DynamoGraphPowerBudgetPendingReasonBudgetExceeded)
	recordPowerBudgetStatus(nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
		Phase: nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseRecovering,
		Ledger: nvidiacomv1beta1.DynamoGraphPowerBudgetLedgerStatus{
			EnforcedWatts: 100, UnknownWatts: 200, InGateReservedWatts: 300, RolloutExtraWatts: 400,
		},
	})
	recordPowerReportEvidence(podPowerEvidence{Reason: "GPU-deadbeef on node-secret failed"})
	recordPowerGateWait("untrusted GPU-deadbeef", 3*time.Second, true)
	recordPowerGateWait("success", 2*time.Second, true)
	recordPowerRecoveryAction("GPU-deadbeef")

	registry := prometheus.NewPedanticRegistry()
	for _, collector := range powerMetricCollectors {
		if err := registry.Register(collector); err != nil {
			t.Fatalf("register power metric: %v", err)
		}
	}
	families, err := registry.Gather()
	if err != nil {
		t.Fatalf("gather power metrics: %v", err)
	}
	for _, family := range families {
		for _, metric := range family.Metric {
			for _, label := range metric.Label {
				if powerMetricLabelIsIdentity(label.GetName()) {
					t.Fatalf("metric %q gathered identity label %q", family.GetName(), label.GetName())
				}
				value := strings.ToLower(label.GetValue())
				if strings.Contains(value, "deadbeef") || strings.Contains(value, "node-secret") {
					t.Fatalf("metric %q retained unbounded identity value %q", family.GetName(), label.GetValue())
				}
			}
		}
	}

	t.Log("Observe one successful wait per live Pod and retire its internal deduplication identity")
	startedAt := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "worker", UID: types.UID("pod-uid")},
		Status:     corev1.PodStatus{StartTime: &metav1.Time{Time: startedAt}},
	}
	tracker := powerGateEventTracker{}
	elapsed, known, unseen := tracker.observeSuccess("dgd-uid", &pod, startedAt.Add(7*time.Second))
	if !unseen || !known || elapsed != 7*time.Second {
		t.Fatalf("first successful wait = (%s, %v, %v)", elapsed, known, unseen)
	}
	if _, _, unseen = tracker.observeSuccess("dgd-uid", &pod, startedAt.Add(8*time.Second)); unseen {
		t.Fatal("repeated healthy report recorded a second gate wait")
	}
	tracker.unseen("dgd-uid", nil)
	if _, _, unseen = tracker.observeSuccess("dgd-uid", &pod, startedAt.Add(9*time.Second)); !unseen {
		t.Fatal("deleted Pod identity was not retired from gate-wait telemetry")
	}
}
