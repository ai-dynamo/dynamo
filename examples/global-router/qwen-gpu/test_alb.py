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

"""Check ALB exposure boundaries and compute preservation."""

import unittest

import alb


class AlbTests(unittest.TestCase):
    def config(self):
        return {
            "bastionCidr": "8.8.8.8/32",
            "vpcId": "vpc-example",
            "vpcCidr": "10.0.0.0/16",
            "publicSubnetIds": ["subnet-a", "subnet-b"],
            "gpuInstanceIds": ["i-example-a", "i-example-b"],
            "gpuSecurityGroups": ["sg-example"],
            "kubectlProviderServiceToken": "arn:example:provider",
            "clusterName": "example",
        }

    def test_alb_exposes_only_bastion_and_does_not_provision_compute(self):
        config = self.config()
        template = alb.render(config)
        rules = template["Resources"]["AlbSecurityGroup"]["Properties"][
            "SecurityGroupIngress"
        ]
        self.assertEqual(
            rules,
            [
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "CidrIp": "8.8.8.8/32",
                    "Description": "Approved bastion only",
                }
            ],
        )
        types = {v["Type"] for v in template["Resources"].values()}
        self.assertFalse(
            types.intersection(
                {
                    "AWS::EC2::Instance",
                    "AWS::EKS::Nodegroup",
                    "AWS::AutoScaling::AutoScalingGroup",
                }
            )
        )
        targets = template["Resources"]["Targets"]["Properties"]
        self.assertEqual(
            {v["Id"] for v in targets["Targets"]}, set(config["gpuInstanceIds"])
        )
        self.assertEqual(targets["Port"], 30800)

    def test_open_or_private_source_ranges_are_rejected(self):
        for cidr in ["0.0.0.0/0", "8.8.8.0/24", "10.0.0.1/32", "::/0"]:
            config = self.config()
            config["bastionCidr"] = cidr
            with self.assertRaises(ValueError):
                alb.render(config)


if __name__ == "__main__":
    unittest.main()
