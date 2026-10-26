# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render a regional bastion-only ALB backed by a stable EKS NodePort."""

import argparse
import ipaddress
import json
from pathlib import Path


def render(config):
    source = ipaddress.ip_network(config["bastionCidr"])
    checks = (
        source.version == 4,
        source.prefixlen == 32,
        source.network_address.is_global,
    )
    if not all(checks):
        raise ValueError(
            "bastionCidr must be the public IPv4 /32 of the approved bastion"
        )
    port = config.get("nodePort", 30800)
    if not 30000 <= port <= 32767:
        raise ValueError("nodePort must be in the default Kubernetes NodePort range")
    if len(config["publicSubnetIds"]) < 2 or not config["gpuInstanceIds"]:
        raise ValueError(
            "at least two public subnets and one existing node are required"
        )
    if not config["gpuSecurityGroups"]:
        raise ValueError("existing node security groups are required")
    namespace = config.get("namespace", "global-router-qwen")
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "global-router-alb", "namespace": namespace},
        "spec": {
            "type": "NodePort",
            "externalTrafficPolicy": "Cluster",
            "selector": {"app": "global-router"},
            "ports": [
                {"name": "http", "port": 8000, "targetPort": "http", "nodePort": port}
            ],
        },
    }
    resources = {
        "NodePortService": {
            "Type": "Custom::AWSCDK-EKS-KubernetesResource",
            "Properties": {
                "ServiceToken": config["kubectlProviderServiceToken"],
                "ClusterName": config["clusterName"],
                "Manifest": json.dumps([service]),
                "Overwrite": True,
            },
        },
        "AlbSecurityGroup": {
            "Type": "AWS::EC2::SecurityGroup",
            "Properties": {
                "GroupDescription": "Regional Qwen ingress restricted to the approved bastion",
                "VpcId": config["vpcId"],
                "SecurityGroupIngress": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 80,
                        "ToPort": 80,
                        "CidrIp": str(source),
                        "Description": "Approved bastion only",
                    }
                ],
                "SecurityGroupEgress": [
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "CidrIp": config["vpcCidr"],
                    }
                ],
            },
        },
        "Alb": {
            "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
            "Properties": {
                "Type": "application",
                "Scheme": "internet-facing",
                "IpAddressType": "ipv4",
                "Subnets": config["publicSubnetIds"],
                "SecurityGroups": [{"Ref": "AlbSecurityGroup"}],
                "LoadBalancerAttributes": [
                    {"Key": "idle_timeout.timeout_seconds", "Value": "120"},
                    {
                        "Key": "routing.http.drop_invalid_header_fields.enabled",
                        "Value": "true",
                    },
                ],
                "Tags": [{"Key": "Project", "Value": "global-router-qwen"}],
            },
        },
        "Targets": {
            "Type": "AWS::ElasticLoadBalancingV2::TargetGroup",
            "DependsOn": "NodePortService",
            "Properties": {
                "VpcId": config["vpcId"],
                "Protocol": "HTTP",
                "Port": port,
                "TargetType": "instance",
                "HealthCheckPath": "/health",
                "HealthCheckIntervalSeconds": 15,
                "HealthyThresholdCount": 2,
                "UnhealthyThresholdCount": 2,
                "Matcher": {"HttpCode": "200"},
                "Targets": [
                    {"Id": instance, "Port": port}
                    for instance in config["gpuInstanceIds"]
                ],
                "TargetGroupAttributes": [
                    {"Key": "deregistration_delay.timeout_seconds", "Value": "10"}
                ],
            },
        },
        "Listener": {
            "Type": "AWS::ElasticLoadBalancingV2::Listener",
            "Properties": {
                "LoadBalancerArn": {"Ref": "Alb"},
                "Port": 80,
                "Protocol": "HTTP",
                "DefaultActions": [
                    {"Type": "forward", "TargetGroupArn": {"Ref": "Targets"}}
                ],
            },
        },
    }
    for index, group in enumerate(config["gpuSecurityGroups"]):
        resources[f"NodePortIngress{index}"] = {
            "Type": "AWS::EC2::SecurityGroupIngress",
            "Properties": {
                "GroupId": group,
                "SourceSecurityGroupId": {"Ref": "AlbSecurityGroup"},
                "IpProtocol": "tcp",
                "FromPort": port,
                "ToPort": port,
                "Description": "Regional ALB to global ingress NodePort",
            },
        }
    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": resources,
        "Outputs": {
            "Url": {
                "Value": {
                    "Fn::Join": ["", ["http://", {"Fn::GetAtt": ["Alb", "DNSName"]}]]
                }
            },
            "TargetGroupArn": {"Value": {"Ref": "Targets"}},
            "AlbArn": {"Value": {"Ref": "Alb"}},
            "AlbSecurityGroupId": {"Value": {"Ref": "AlbSecurityGroup"}},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(json.loads(args.config.read_text())), indent=2))
