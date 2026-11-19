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

package aisimulate

import (
	"fmt"
	"strings"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

// TraceMountPath is the directory of the Run Job's sweeper container that holds the trace file.
const TraceMountPath = "/var/run/aisimulate/trace"

// TracePath is where the Run Job mounts the trace file at subPath of the claim: one fixed file
// under TraceMountPath. The Job mounts only that file, through a subPath of the claim, so the
// kubelet resolves subPath and refuses a symlink that resolves outside the claim. The name keeps
// the ".gz" suffix, by which AISimulate recognizes gzip-compressed JSON Lines.
func TracePath(subPath string) string {
	if strings.HasSuffix(subPath, ".gz") {
		return TraceMountPath + "/trace.jsonl.gz"
	}
	return TraceMountPath + "/trace.jsonl"
}

// traceSource returns the traffic.source object a typed trace lowers to: a trace in AISimulate's
// native format read from its one claim file under TraceMountPath. trace must not be nil. A trace
// without a persistentVolumeClaim source has no paths; the CRD schema rejects that shape.
func traceSource(trace *v1beta2.TraceWorkloadSpec) (map[string]any, error) {
	format, err := nativeTraceFormat(trace.Format)
	if err != nil {
		return nil, err
	}
	source := map[string]any{"type": "trace", "format": format}
	if claim := trace.Source.PersistentVolumeClaim; claim != nil {
		source["paths"] = []any{TracePath(claim.SubPath)}
	}
	return source, nil
}

// nativeTraceFormat maps each API trace format to AISimulate's traffic.source.format spelling.
// The CRD schema defaults an omitted format, so an empty format is an invalid input here.
func nativeTraceFormat(format v1beta2.TraceFormat) (string, error) {
	switch format {
	case v1beta2.TraceFormatDynamo:
		return "dynamo", nil
	default:
		return "", fmt.Errorf("spec.workload.trace.format: unsupported trace format %q", format)
	}
}
