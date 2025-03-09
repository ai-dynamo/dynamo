/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	dynamoCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
)

type Duration time.Duration

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Duration(d).String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var v interface{}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch value := v.(type) {
	case float64:
		*d = Duration(time.Duration(value))
		return nil
	case string:
		tmp, err := time.ParseDuration(value)
		if err != nil {
			return err
		}
		*d = Duration(tmp)
		return nil
	default:
		return errors.New("invalid duration")
	}
}

type DeploymentTargetType string

const (
	DeploymentTargetTypeStable DeploymentTargetType = "stable"
	DeploymentTargetTypeCanary DeploymentTargetType = "canary"
)

var DeploymentTargetTypeAddrs = map[DeploymentTargetType]string{
	DeploymentTargetTypeStable: "stb",
	DeploymentTargetTypeCanary: "cnr",
}

type DeploymentTargetResourceItem struct {
	CPU    string            `json:"cpu,omitempty"`
	Memory string            `json:"memory,omitempty"`
	GPU    string            `json:"gpu,omitempty"`
	Custom map[string]string `json:"custom,omitempty"`
}

func (in *DeploymentTargetResourceItem) DeepCopyInto(out *DeploymentTargetResourceItem) {
	*out = *in
	if in.Custom != nil {
		in, out := &in.Custom, &out.Custom
		*out = make(map[string]string, len(*in))
		for key, val := range *in {
			(*out)[key] = val
		}
	}
}

type DeploymentTargetResources struct {
	Requests *DeploymentTargetResourceItem `json:"requests,omitempty"`
	Limits   *DeploymentTargetResourceItem `json:"limits,omitempty"`
}

func (in *DeploymentTargetResources) DeepCopy() (out *DeploymentTargetResources) {
	if in == nil {
		return nil
	}
	out = new(DeploymentTargetResources)
	in.DeepCopyInto(out)
	return
}

func (in *DeploymentTargetResources) DeepCopyInto(out *DeploymentTargetResources) {
	*out = *in
	if in.Requests != nil {
		in, out := &in.Requests, &out.Requests
		*out = new(DeploymentTargetResourceItem)
		(*in).DeepCopyInto(*out)
	}
	if in.Limits != nil {
		in, out := &in.Limits, &out.Limits
		*out = new(DeploymentTargetResourceItem)
		(*in).DeepCopyInto(*out)
	}
}

type HPAMetricType string

const (
	HPAMetricTypeMemory HPAMetricType = "memory"
	HPAMetricTypeCPU    HPAMetricType = "cpu"
	HPAMetricTypeGPU    HPAMetricType = "gpu"
	HPAMetricTypeQPS    HPAMetricType = "qps"
)

type HPAMetric struct {
	Type  HPAMetricType      `json:"type"`
	Value *resource.Quantity `json:"value"`
}

func (in *HPAMetric) DeepCopy() (out *HPAMetric) {
	if in == nil {
		return nil
	}
	out = new(HPAMetric)
	in.DeepCopyInto(out)
	return
}

func (in *HPAMetric) DeepCopyInto(out *HPAMetric) {
	*out = *in
	if in.Value != nil {
		in, out := &in.Value, &out.Value
		*out = new(resource.Quantity)
		**out = (*in).DeepCopy()
	}
}

type HPAScaleBehavior string

const (
	HPAScaleBehaviorDisabled HPAScaleBehavior = "disabled"
	HPAScaleBehaviorStable   HPAScaleBehavior = "stable"
	HPAScaleBehaviorFast     HPAScaleBehavior = "fast"
)

type HPAPolicy struct {
	Metrics           []HPAMetric       `json:"metrics,omitempty"`
	ScaleDownBehavior *HPAScaleBehavior `json:"scale_down_behavior,omitempty"`
	ScaleUpBehavior   *HPAScaleBehavior `json:"scale_up_behavior,omitempty"`
}

func (in *HPAPolicy) DeepCopy() (out *HPAPolicy) {
	if in == nil {
		return nil
	}
	out = new(HPAPolicy)
	in.DeepCopyInto(out)
	return
}

func (in *HPAPolicy) DeepCopyInto(out *HPAPolicy) {
	*out = *in
	if in.Metrics != nil {
		in, out := &in.Metrics, &out.Metrics
		*out = make([]HPAMetric, len(*in))
		copy(*out, *in)
	}
	if in.ScaleDownBehavior != nil {
		out.ScaleDownBehavior = new(HPAScaleBehavior)
		*out.ScaleDownBehavior = *in.ScaleDownBehavior
	}
	if in.ScaleUpBehavior != nil {
		out.ScaleUpBehavior = new(HPAScaleBehavior)
		*out.ScaleUpBehavior = *in.ScaleUpBehavior
	}
}

type DeploymentTargetHPAConf struct {
	CPU         *int32     `json:"cpu,omitempty"`
	GPU         *int32     `json:"gpu,omitempty"`
	Memory      *string    `json:"memory,omitempty"`
	QPS         *int64     `json:"qps,omitempty"`
	MinReplicas *int32     `json:"min_replicas,omitempty"`
	MaxReplicas *int32     `json:"max_replicas,omitempty"`
	Policy      *HPAPolicy `json:"policy,omitempty"`
}

func (in *DeploymentTargetHPAConf) DeepCopy() (out *DeploymentTargetHPAConf) {
	if in == nil {
		return nil
	}
	out = new(DeploymentTargetHPAConf)
	in.DeepCopyInto(out)
	return
}

func (in *DeploymentTargetHPAConf) DeepCopyInto(out *DeploymentTargetHPAConf) {
	*out = *in
	if in.CPU != nil {
		out.CPU = new(int32)
		*out.CPU = *in.CPU
	}
	if in.GPU != nil {
		out.GPU = new(int32)
		*out.GPU = *in.GPU
	}
	if in.Memory != nil {
		out.Memory = new(string)
		*out.Memory = *in.Memory
	}
	if in.QPS != nil {
		out.QPS = new(int64)
		*out.QPS = *in.QPS
	}
	if in.MinReplicas != nil {
		out.MinReplicas = new(int32)
		*out.MinReplicas = *in.MinReplicas
	}
	if in.MaxReplicas != nil {
		out.MaxReplicas = new(int32)
		*out.MaxReplicas = *in.MaxReplicas
	}
}

type BentoRequestOverrides struct {
	ImageBuildTimeout *Duration `json:"imageBuildTimeout,omitempty"`

	ImageBuilderExtraPodMetadata   *dynamoCommon.ExtraPodMetadata `json:"imageBuilderExtraPodMetadata,omitempty"`
	ImageBuilderExtraPodSpec       *dynamoCommon.ExtraPodSpec     `json:"imageBuilderExtraPodSpec,omitempty"`
	ImageBuilderExtraContainerEnv  []corev1.EnvVar                    `json:"imageBuilderExtraContainerEnv,omitempty"`
	ImageBuilderContainerResources *corev1.ResourceRequirements       `json:"imageBuilderContainerResources,omitempty"`

	DockerConfigJSONSecretName string `json:"dockerConfigJsonSecretName,omitempty"`

	DownloaderContainerEnvFrom []corev1.EnvFromSource `json:"downloaderContainerEnvFrom,omitempty"`
}

type ApiServerBentoDeploymentOverrides struct {
	MonitorExporter  *dynamoCommon.MonitorExporterSpec `json:"monitorExporter,omitempty"`
	ExtraPodMetadata *dynamoCommon.ExtraPodMetadata    `json:"extraPodMetadata,omitempty"`
	ExtraPodSpec     *dynamoCommon.ExtraPodSpec        `json:"extraPodSpec,omitempty"`
}

type RunnerBentoDeploymentOverrides struct {
	ExtraPodMetadata *dynamoCommon.ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	ExtraPodSpec     *dynamoCommon.ExtraPodSpec     `json:"extraPodSpec,omitempty"`
}

type DeploymentTargetRunnerConfig struct {
	ResourceInstance                       *string                         `json:"resource_instance,omitempty"`
	Resources                              *DeploymentTargetResources      `json:"resources,omitempty"`
	HPAConf                                *DeploymentTargetHPAConf        `json:"hpa_conf,omitempty"`
	Envs                                   *[]*LabelItemSchema             `json:"envs,omitempty"`
	EnableStealingTrafficDebugMode         *bool                           `json:"enable_stealing_traffic_debug_mode,omitempty"`
	EnableDebugMode                        *bool                           `json:"enable_debug_mode,omitempty"`
	EnableDebugPodReceiveProductionTraffic *bool                           `json:"enable_debug_pod_receive_production_traffic,omitempty"`
	DeploymentStrategy                     *DeploymentStrategy             `json:"deployment_strategy,omitempty"`
	BentoDeploymentOverrides               *RunnerBentoDeploymentOverrides `json:"bento_deployment_overrides,omitempty"`
	TrafficControl                         *TrafficControlConfig           `json:"traffic_control,omitempty"`
	DeploymentColdStartWaitTimeout         *Duration                       `json:"deployment_cold_start_wait_timeout,omitempty"`
}

func (in *DeploymentTargetRunnerConfig) DeepCopy() (out *DeploymentTargetRunnerConfig) {
	if in == nil {
		return nil
	}
	out = new(DeploymentTargetRunnerConfig)
	in.DeepCopyInto(out)
	return
}

func (in *DeploymentTargetRunnerConfig) DeepCopyInto(out *DeploymentTargetRunnerConfig) {
	*out = *in
	if in.Resources != nil {
		in, out := &in.Resources, &out.Resources
		*out = new(DeploymentTargetResources)
		(*in).DeepCopyInto(*out)
	}
	if in.HPAConf != nil {
		in, out := &in.HPAConf, &out.HPAConf
		*out = new(DeploymentTargetHPAConf)
		(*in).DeepCopyInto(*out)
	}
	if in.Envs != nil {
		out.Envs = new([]*LabelItemSchema)
		for _, item := range *in.Envs {
			newItem := new(LabelItemSchema)
			item.DeepCopyInto(newItem)
			*out.Envs = append(*out.Envs, newItem)
		}
	}
}

type DeploymentStrategy string

const (
	DeploymentStrategyRollingUpdate               DeploymentStrategy = "RollingUpdate"
	DeploymentStrategyRecreate                    DeploymentStrategy = "Recreate"
	DeploymentStrategyRampedSlowRollout           DeploymentStrategy = "RampedSlowRollout"
	DeploymentStrategyBestEffortControlledRollout DeploymentStrategy = "BestEffortControlledRollout"
)

type TrafficControlConfig struct {
	Timeout      *Duration           `json:"timeout,omitempty"`
	RequestQueue *RequestQueueConfig `json:"request_queue,omitempty"`
}

type RequestQueueConfig struct {
	Enabled               *bool  `json:"enabled,omitempty"`
	MaxConsumeConcurrency *int32 `json:"max_consume_concurrency,omitempty"`
}

type DeploymentTargetConfig struct {
	KubeResourceUid                        string                                  `json:"kubeResourceUid"`
	KubeResourceVersion                    string                                  `json:"kubeResourceVersion"`
	ResourceInstance                       *string                                 `json:"resource_instance,omitempty"`
	Resources                              *DeploymentTargetResources              `json:"resources"`
	HPAConf                                *DeploymentTargetHPAConf                `json:"hpa_conf,omitempty"`
	Envs                                   *[]*LabelItemSchema                     `json:"envs,omitempty"`
	Runners                                map[string]DeploymentTargetRunnerConfig `json:"runners,omitempty"`
	EnableIngress                          *bool                                   `json:"enable_ingress,omitempty"`
	EnableStealingTrafficDebugMode         *bool                                   `json:"enable_stealing_traffic_debug_mode,omitempty"`
	EnableDebugMode                        *bool                                   `json:"enable_debug_mode,omitempty"`
	EnableDebugPodReceiveProductionTraffic *bool                                   `json:"enable_debug_pod_receive_production_traffic,omitempty"`
	DeploymentStrategy                     *DeploymentStrategy                     `json:"deployment_strategy,omitempty"`
	BentoDeploymentOverrides               *ApiServerBentoDeploymentOverrides      `json:"bento_deployment_overrides,omitempty"`
	BentoRequestOverrides                  *BentoRequestOverrides                  `json:"bento_request_overrides,omitempty"`
	TrafficControl                         *TrafficControlConfig                   `json:"traffic_control,omitempty"`
	DeploymentColdStartWaitTimeout         *Duration                               `json:"deployment_cold_start_wait_timeout,omitempty"`
}

func (in *DeploymentTargetConfig) DeepCopy() (out *DeploymentTargetConfig) {
	if in == nil {
		return nil
	}
	out = new(DeploymentTargetConfig)
	in.DeepCopyInto(out)
	return
}

func (in *DeploymentTargetConfig) DeepCopyInto(out *DeploymentTargetConfig) {
	*out = *in
	if in.Resources != nil {
		in, out := &in.Resources, &out.Resources
		*out = new(DeploymentTargetResources)
		(*in).DeepCopyInto(*out)
	}
	if in.HPAConf != nil {
		in, out := &in.HPAConf, &out.HPAConf
		*out = new(DeploymentTargetHPAConf)
		(*in).DeepCopyInto(*out)
	}
	if in.Envs != nil {
		out.Envs = new([]*LabelItemSchema)
		for _, item := range *in.Envs {
			newItem := new(LabelItemSchema)
			item.DeepCopyInto(newItem)
			*out.Envs = append(*out.Envs, newItem)
		}
	}
}

func (c *DeploymentTargetConfig) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *DeploymentTargetConfig) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
