/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
)

func TestPowerCapClampFeedbackIsBoundedPerIntentObservation(t *testing.T) {
	dgd, _, _ := statusTestObjects(1)
	dgd.UID = types.UID("clamp-feedback-dgd")
	component := &dgd.Spec.Components[0]
	component.PodTemplate.Spec.NodeSelector[qualifiedGPUProductLabel] = "feedback-sku"
	component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "100"
	recorder := events.NewFakeRecorder(4)
	reconciler := &DynamoGraphDeploymentReconciler{
		Recorder: recorder,
		PowerQualification: powerbudget.QualificationIndex{
			"feedback-sku": {MinWatts: 200, DefaultWatts: 400, MaxWatts: 700},
		},
	}

	belowMetric := powerCapClamped.WithLabelValues("below_min", "feedback-sku")
	beforeBelow := powerCounterValue(t, belowMetric)
	reconciler.emitPowerCapClampFeedback(context.Background(), dgd)
	if got := powerCounterValue(t, belowMetric); got != beforeBelow+1 {
		t.Fatalf("below-min metric = %v, want %v", got, beforeBelow+1)
	}
	select {
	case event := <-recorder.Events:
		for _, want := range []string{
			"Warning", powerCapClampedReason, "requested 100W", "qualified 200W", "feedback-sku", "[200,700]W",
		} {
			if !strings.Contains(event, want) {
				t.Fatalf("event = %q, missing %q", event, want)
			}
		}
	default:
		t.Fatal("below-min clamp Event was not emitted")
	}

	t.Log("Unchanged reconciliation does not repeat Event, metric, or log observation")
	reconciler.emitPowerCapClampFeedback(context.Background(), dgd)
	if got := powerCounterValue(t, belowMetric); got != beforeBelow+1 {
		t.Fatalf("deduplicated below-min metric = %v, want %v", got, beforeBelow+1)
	}
	select {
	case duplicate := <-recorder.Events:
		t.Fatalf("duplicate clamp event = %q", duplicate)
	default:
	}

	t.Log("A distinct above-max intent observation uses the bounded direction label")
	component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "900"
	aboveMetric := powerCapClamped.WithLabelValues("above_max", "feedback-sku")
	beforeAbove := powerCounterValue(t, aboveMetric)
	reconciler.emitPowerCapClampFeedback(context.Background(), dgd)
	if got := powerCounterValue(t, aboveMetric); got != beforeAbove+1 {
		t.Fatalf("above-max metric = %v, want %v", got, beforeAbove+1)
	}
	select {
	case event := <-recorder.Events:
		if !strings.Contains(event, "requested 900W") || !strings.Contains(event, "qualified 700W") {
			t.Fatalf("above-max event = %q", event)
		}
	default:
		t.Fatal("above-max clamp Event was not emitted")
	}
}

func powerCounterValue(t *testing.T, metric interface{ Write(*dto.Metric) error }) float64 {
	t.Helper()
	encoded := &dto.Metric{}
	if err := metric.Write(encoded); err != nil {
		t.Fatalf("encode Prometheus counter: %v", err)
	}
	return encoded.GetCounter().GetValue()
}

func TestPowerCapClampTrackerForgetsDeletedDGDScope(t *testing.T) {
	observation := powerCapClampObservation{Component: "worker", Direction: "below_min"}
	tracker := powerCapClampEventTracker{}
	if got := tracker.unseen("dgd-uid", []powerCapClampObservation{observation}); len(got) != 1 {
		t.Fatalf("initial unseen = %v, want one", got)
	}
	if got := tracker.unseen("dgd-uid", []powerCapClampObservation{observation}); len(got) != 0 {
		t.Fatalf("duplicate unseen = %v, want none", got)
	}
	tracker.forgetScope("dgd-uid")
	if got := tracker.unseen("dgd-uid", []powerCapClampObservation{observation}); len(got) != 1 {
		t.Fatalf("post-delete unseen = %v, want one", got)
	}
}
