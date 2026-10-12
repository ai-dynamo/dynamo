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
	"reflect"
	"testing"
	"time"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func podWith(statuses ...corev1.ContainerStatus) *corev1.Pod {
	return &corev1.Pod{Status: corev1.PodStatus{ContainerStatuses: statuses}}
}

func TestSweeperStateOfReadsOnlyTheSweeperContainer(t *testing.T) {
	terminated := func(name string, code int32) corev1.ContainerStatus {
		return corev1.ContainerStatus{Name: name, State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: code}}}
	}
	running := corev1.ContainerStatus{Name: "sweeper", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{}}}

	cases := map[string]struct {
		pod  *corev1.Pod
		want SweeperState
	}{
		"no statuses yet":      {podWith(), SweeperState{}},
		"running":              {podWith(running), SweeperState{}},
		"exited non-zero":      {podWith(terminated("sweeper", 3)), SweeperState{Exited: true, ExitCode: 3}},
		"exited zero":          {podWith(terminated("sweeper", 0)), SweeperState{Exited: true}},
		"other container only": {podWith(terminated("publisher", 1)), SweeperState{}},
	}
	for name, tc := range cases {
		if got := sweeperStateOf(tc.pod, "sweeper"); got != tc.want {
			t.Errorf("%s: got %+v, want %+v", name, got, tc.want)
		}
	}
}

func TestObservedStatusMirrorsTheRunStatus(t *testing.T) {
	when := metav1.NewTime(time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC))
	run := &v1beta2.DynamoGraphDeploymentRun{Status: v1beta2.DynamoGraphDeploymentRunStatus{
		Message:          "m",
		Progress:         &v1beta2.RunProgress{Round: 2, Evaluated: 5},
		LastProgressTime: &when,
		CandidateRefs:    []v1beta2.CandidateRef{{Name: "x"}, {Name: "y"}},
	}}
	want := RunStatus{Message: "m", HasProgress: true, Round: 2, Evaluated: 5, LastProgress: when.Time, CandidateNames: []string{"x", "y"}}
	if got := observedStatus(run); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if got := observedStatus(&v1beta2.DynamoGraphDeploymentRun{}); got.HasProgress || !got.LastProgress.IsZero() {
		t.Fatalf("an empty status must not look like reported progress: %+v", got)
	}
}

func TestCacheOptionsCoverOnlyThisRunItsCandidatesAndItsPod(t *testing.T) {
	cfg := Config{Namespace: "ns", RunName: "run", RunUID: "uid-1", PodName: "pod-1"}
	opts := cacheOptions(cfg)
	if _, ok := opts.DefaultNamespaces["ns"]; !ok || len(opts.DefaultNamespaces) != 1 {
		t.Fatalf("cache must be limited to the run's namespace: %v", opts.DefaultNamespaces)
	}
	if len(opts.ByObject) != 3 {
		t.Fatalf("want selectors for the run, its candidates and the pod, got %d", len(opts.ByObject))
	}
	for object, selector := range opts.ByObject {
		switch object.(type) {
		case *v1beta2.DynamoGraphDeploymentRun:
			if selector.Field.String() != "metadata.name=run" {
				t.Errorf("run selector = %q", selector.Field)
			}
		case *v1beta2.DynamoGraphDeploymentCandidate:
			if selector.Label.String() != LabelRunUID+"=uid-1" {
				t.Errorf("candidate selector = %q", selector.Label)
			}
		case *corev1.Pod:
			if selector.Field.String() != "metadata.name=pod-1" {
				t.Errorf("pod selector = %q", selector.Field)
			}
		default:
			t.Errorf("unexpected cached type %T", object)
		}
	}
}
