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

package v1alpha1

import (
	"reflect"
	"slices"
	"strings"
	"testing"
	"unicode"

	"github.com/google/go-cmp/cmp"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const knownV1Beta1ConversionFieldSet = `
DynamoComponentDeploymentSpec.backendFramework
DynamoComponentDeploymentSpec.compilationCache.mountPath
DynamoComponentDeploymentSpec.compilationCache.pvcName
DynamoComponentDeploymentSpec.eppConfig.config
DynamoComponentDeploymentSpec.eppConfig.configMapRef
DynamoComponentDeploymentSpec.experimental.checkpoint.checkpointRef
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.backendFramework
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.dtype
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.dynamoVersion
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.extraParameters
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.maxModelLen
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.model
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.pipelineParallelSize
DynamoComponentDeploymentSpec.experimental.checkpoint.identity.tensorParallelSize
DynamoComponentDeploymentSpec.experimental.checkpoint.mode
DynamoComponentDeploymentSpec.experimental.failover.mode
DynamoComponentDeploymentSpec.experimental.failover.numShadows
DynamoComponentDeploymentSpec.experimental.gpuMemoryService.deviceClassName
DynamoComponentDeploymentSpec.experimental.gpuMemoryService.mode
DynamoComponentDeploymentSpec.frontendSidecar
DynamoComponentDeploymentSpec.globalDynamoNamespace
DynamoComponentDeploymentSpec.modelRef.name
DynamoComponentDeploymentSpec.modelRef.revision
DynamoComponentDeploymentSpec.multinode.nodeCount
DynamoComponentDeploymentSpec.name
DynamoComponentDeploymentSpec.podTemplate
DynamoComponentDeploymentSpec.replicas
DynamoComponentDeploymentSpec.scalingAdapter
DynamoComponentDeploymentSpec.sharedMemorySize
DynamoComponentDeploymentSpec.topologyConstraint.packDomain
DynamoComponentDeploymentSpec.type
DynamoComponentDeploymentStatus.component.availableReplicas
DynamoComponentDeploymentStatus.component.componentKind
DynamoComponentDeploymentStatus.component.componentNames
DynamoComponentDeploymentStatus.component.readyReplicas
DynamoComponentDeploymentStatus.component.replicas
DynamoComponentDeploymentStatus.component.updatedReplicas
DynamoComponentDeploymentStatus.conditions
DynamoComponentDeploymentStatus.observedGeneration
DynamoGraphDeploymentScalingAdapterSpec.dgdRef.componentName
DynamoGraphDeploymentScalingAdapterSpec.dgdRef.name
DynamoGraphDeploymentScalingAdapterSpec.replicas
DynamoGraphDeploymentScalingAdapterStatus.lastScaleTime
DynamoGraphDeploymentScalingAdapterStatus.replicas
DynamoGraphDeploymentScalingAdapterStatus.selector
DynamoGraphDeploymentSpec.annotations
DynamoGraphDeploymentSpec.backendFramework
DynamoGraphDeploymentSpec.components.compilationCache.mountPath
DynamoGraphDeploymentSpec.components.compilationCache.pvcName
DynamoGraphDeploymentSpec.components.eppConfig.config
DynamoGraphDeploymentSpec.components.eppConfig.configMapRef
DynamoGraphDeploymentSpec.components.experimental.checkpoint.checkpointRef
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.backendFramework
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.dtype
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.dynamoVersion
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.extraParameters
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.maxModelLen
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.model
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.pipelineParallelSize
DynamoGraphDeploymentSpec.components.experimental.checkpoint.identity.tensorParallelSize
DynamoGraphDeploymentSpec.components.experimental.checkpoint.mode
DynamoGraphDeploymentSpec.components.experimental.failover.mode
DynamoGraphDeploymentSpec.components.experimental.failover.numShadows
DynamoGraphDeploymentSpec.components.experimental.gpuMemoryService.deviceClassName
DynamoGraphDeploymentSpec.components.experimental.gpuMemoryService.mode
DynamoGraphDeploymentSpec.components.frontendSidecar
DynamoGraphDeploymentSpec.components.globalDynamoNamespace
DynamoGraphDeploymentSpec.components.modelRef.name
DynamoGraphDeploymentSpec.components.modelRef.revision
DynamoGraphDeploymentSpec.components.multinode.nodeCount
DynamoGraphDeploymentSpec.components.name
DynamoGraphDeploymentSpec.components.podTemplate
DynamoGraphDeploymentSpec.components.replicas
DynamoGraphDeploymentSpec.components.scalingAdapter
DynamoGraphDeploymentSpec.components.sharedMemorySize
DynamoGraphDeploymentSpec.components.topologyConstraint.packDomain
DynamoGraphDeploymentSpec.components.type
DynamoGraphDeploymentSpec.env
DynamoGraphDeploymentSpec.labels
DynamoGraphDeploymentSpec.restart.id
DynamoGraphDeploymentSpec.restart.strategy.order
DynamoGraphDeploymentSpec.restart.strategy.type
DynamoGraphDeploymentSpec.topologyConstraint.clusterTopologyName
DynamoGraphDeploymentSpec.topologyConstraint.packDomain
DynamoGraphDeploymentStatus.checkpoints.checkpointName
DynamoGraphDeploymentStatus.checkpoints.identityHash
DynamoGraphDeploymentStatus.checkpoints.ready
DynamoGraphDeploymentStatus.components.availableReplicas
DynamoGraphDeploymentStatus.components.componentKind
DynamoGraphDeploymentStatus.components.componentNames
DynamoGraphDeploymentStatus.components.readyReplicas
DynamoGraphDeploymentStatus.components.replicas
DynamoGraphDeploymentStatus.components.updatedReplicas
DynamoGraphDeploymentStatus.conditions
DynamoGraphDeploymentStatus.observedGeneration
DynamoGraphDeploymentStatus.restart.inProgress
DynamoGraphDeploymentStatus.restart.observedID
DynamoGraphDeploymentStatus.restart.phase
DynamoGraphDeploymentStatus.rollingUpdate.endTime
DynamoGraphDeploymentStatus.rollingUpdate.phase
DynamoGraphDeploymentStatus.rollingUpdate.startTime
DynamoGraphDeploymentStatus.rollingUpdate.updatedComponents
DynamoGraphDeploymentStatus.state
`

func TestV1Beta1ConversionFieldSetIsAcknowledged(t *testing.T) {
	want := strings.Fields(strings.TrimSpace(knownV1Beta1ConversionFieldSet))
	got := v1beta1ConversionFieldSet()
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("v1beta1 conversion field set changed (-want +got).\n\nEvery v1beta1 field added here needs an explicit conversion decision: map it natively, preserve it sparsely, or document why it is intentionally ignored. See deploy/operator/api/CONVERSION.md.\n\n%s", diff)
	}
}

func v1beta1ConversionFieldSet() []string {
	roots := []struct {
		name string
		typ  reflect.Type
	}{
		{"DynamoGraphDeploymentSpec", reflect.TypeFor[v1beta1.DynamoGraphDeploymentSpec]()},
		{"DynamoGraphDeploymentStatus", reflect.TypeFor[v1beta1.DynamoGraphDeploymentStatus]()},
		{"DynamoComponentDeploymentSpec", reflect.TypeFor[v1beta1.DynamoComponentDeploymentSpec]()},
		{"DynamoComponentDeploymentStatus", reflect.TypeFor[v1beta1.DynamoComponentDeploymentStatus]()},
		{"DynamoGraphDeploymentScalingAdapterSpec", reflect.TypeFor[v1beta1.DynamoGraphDeploymentScalingAdapterSpec]()},
		{"DynamoGraphDeploymentScalingAdapterStatus", reflect.TypeFor[v1beta1.DynamoGraphDeploymentScalingAdapterStatus]()},
	}

	var paths []string
	for _, root := range roots {
		paths = appendV1Beta1ConversionFieldPaths(paths, root.name, root.typ, nil)
	}
	slices.Sort(paths)
	return slices.Compact(paths)
}

func appendV1Beta1ConversionFieldPaths(paths []string, prefix string, typ reflect.Type, stack []reflect.Type) []string {
	typ = conversionFieldBaseType(typ)
	if typ.Kind() != reflect.Struct || typ.PkgPath() != reflect.TypeFor[v1beta1.DynamoGraphDeploymentSpec]().PkgPath() {
		return append(paths, prefix)
	}
	if slices.Contains(stack, typ) {
		return append(paths, prefix)
	}

	stack = append(stack, typ)
	before := len(paths)
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if field.PkgPath != "" {
			continue
		}

		name, inline, ok := conversionFieldJSONName(field)
		if !ok {
			continue
		}
		fieldPrefix := prefix
		if !inline {
			fieldPrefix += "." + name
		}
		paths = appendV1Beta1ConversionFieldPaths(paths, fieldPrefix, field.Type, stack)
	}
	if len(paths) == before {
		paths = append(paths, prefix)
	}
	return paths
}

func conversionFieldBaseType(typ reflect.Type) reflect.Type {
	for {
		switch typ.Kind() {
		case reflect.Pointer:
			typ = typ.Elem()
		case reflect.Slice, reflect.Array:
			typ = typ.Elem()
		case reflect.Map:
			typ = typ.Elem()
		default:
			return typ
		}
	}
}

func conversionFieldJSONName(field reflect.StructField) (name string, inline bool, ok bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, false
	}
	parts := strings.Split(tag, ",")
	name = parts[0]
	inline = slices.Contains(parts[1:], "inline")
	if name == "" {
		name = lowerFirst(field.Name)
	}
	return name, inline, true
}

func lowerFirst(s string) string {
	if s == "" {
		return ""
	}
	runes := []rune(s)
	runes[0] = unicode.ToLower(runes[0])
	return string(runes)
}
