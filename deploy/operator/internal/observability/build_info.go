/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package observability

import (
	"runtime"
	"runtime/debug"

	"github.com/prometheus/client_golang/prometheus"
)

// unknownBuildValue is the fallback for build metadata that cannot be
// determined (e.g. container builds whose context has no .git directory).
const unknownBuildValue = "unknown"

type vcsInfo struct {
	commit     string
	commitDate string
	goVersion  string
}

// readVCSInfo reads the binary's embedded build info. Note that vcs.time is
// the source revision's commit timestamp, not the time the binary was built.
// Missing values fall back to unknownBuildValue; this never fails.
func readVCSInfo() vcsInfo {
	info := vcsInfo{
		commit:     unknownBuildValue,
		commitDate: unknownBuildValue,
		goVersion:  unknownBuildValue,
	}
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return info
	}
	if bi.GoVersion != "" {
		info.goVersion = bi.GoVersion
	}
	for _, s := range bi.Settings {
		switch s.Key {
		case "vcs.revision":
			if s.Value != "" {
				info.commit = s.Value
			}
		case "vcs.time":
			if s.Value != "" {
				info.commitDate = s.Value
			}
		}
	}
	return info
}

// buildInfoCollector implements the standard Prometheus *_build_info
// convention: a constant-1 gauge carrying build metadata as labels.
func buildInfoCollector(operatorVersion string) prometheus.Collector {
	vcs := readVCSInfo()
	return prometheus.NewGaugeFunc(
		prometheus.GaugeOpts{
			Namespace: metricsNamespace,
			Name:      "build_info",
			Help:      "Dynamo operator build metadata exposed as labels with a constant value of 1.",
			ConstLabels: prometheus.Labels{
				"version":         operatorVersion,
				"git_commit":      vcs.commit,
				"git_commit_date": vcs.commitDate,
				"go_version":      vcs.goVersion,
				"platform":        runtime.GOOS + "/" + runtime.GOARCH,
			},
		},
		func() float64 { return 1 },
	)
}
