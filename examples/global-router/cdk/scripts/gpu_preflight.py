# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only AWS checks before provisioning reserved GPU compute (AWS CLI required)."""
import argparse
import json
from pathlib import Path
import subprocess


def aws(region, service, *args):
    return json.loads(subprocess.check_output([
        "aws", service, *args, "--region", region, "--output", "json", "--no-cli-pager"
    ], text=True))


def check(config, region_filter=None):
    regions = config["regions"]
    if not 1 <= len(regions) <= 4 or len({r["region"] for r in regions}) != len(regions):
        raise ValueError("Specify one to four unique regions")
    if sum(r["count"] for r in regions) > 16:
        raise ValueError("Fleet exceeds 16 nodes")
    owner = aws("us-east-1", "sts", "get-caller-identity")["Account"]
    selected = [r for r in regions if region_filter is None or r["region"] == region_filter]
    if not selected:
        raise ValueError("Requested region is not configured")
    total = 0.0
    for r in selected:
        region = r["region"]
        count = r["count"]
        if not isinstance(count, int) or not 2 <= count <= 4:
            raise ValueError("Specify two to four nodes per region")
        cr = aws(region, "ec2", "describe-capacity-reservations", "--capacity-reservation-ids", r["reservationId"])["CapacityReservations"][0]
        expected = {"State": "active", "OwnerId": owner, "InstanceType": r["instanceType"],
                    "AvailabilityZone": r["availabilityZone"], "InstancePlatform": "Linux/UNIX", "Tenancy": "default"}
        for key, value in expected.items():
            if cr.get(key) != value:
                raise ValueError(f"{region}: reservation {key} does not match")
        if cr["TotalInstanceCount"] < count:
            raise ValueError(f"{region}: reservation too small")
        existing = aws(region, "ec2", "describe-instances", "--filters",
            f"Name=capacity-reservation-id,Values={r['reservationId']}",
            "Name=instance-state-name,Values=pending,running")["Reservations"]
        owned = [i for group in existing for i in group["Instances"]
                 if {t["Key"]: t["Value"] for t in i.get("Tags", [])}.get("Project") == "global-router-gpu"]
        if cr["AvailableInstanceCount"] + len(owned) < count:
            raise ValueError(f"{region}: slots occupied by other compute")
        filters = [{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in {
            "regionCode": region, "instanceType": r["instanceType"], "operatingSystem": "Linux",
            "tenancy": "Shared", "preInstalledSw": "NA", "capacitystatus": "Used"}.items()]
        prices = aws("us-east-1", "pricing", "get-products", "--service-code", "AmazonEC2", "--filters", json.dumps(filters))["PriceList"]
        rates = [float(d["pricePerUnit"]["USD"]) for p in prices
                 for term in json.loads(p)["terms"].get("OnDemand", {}).values()
                 for d in term["priceDimensions"].values() if "On Demand Linux" in d.get("description", "")]
        if len(set(rates)) != 1 or not 0 < rates[0] <= 7:
            raise ValueError(f"{region}: price missing, ambiguous, or above $7/hour")
        if abs(rates[0] - r["hourlyRate"]) > 0.00001:
            raise ValueError(f"{region}: configured price differs from AWS; refresh hourlyRate")
        if r.get("subnetId"):
            subnet = aws(region, "ec2", "describe-subnets", "--subnet-ids", r["subnetId"])["Subnets"][0]
            cluster = aws(region, "eks", "describe-cluster", "--name", r["clusterName"])["cluster"]
            if (subnet["AvailabilityZone"] != r["availabilityZone"] or subnet["MapPublicIpOnLaunch"] or
                subnet["VpcId"] != cluster["resourcesVpcConfig"]["vpcId"] or subnet["AvailableIpAddressCount"] < count * 10 or
                cluster["status"] != "ACTIVE" or cluster["version"] != "1.35"):
                raise ValueError(f"{region}: subnet or cluster is incompatible")
        total += count * rates[0]
        print(f"{region}: {count} x {r['instanceType']}, reservation verified, ${count * rates[0]:.5f}/hour; expires {cr.get('EndDate', 'no fixed end')}")
    print(f"Selected compute total: ${total:.5f}/hour (excludes EBS/network/EKS/NAT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--region")
    args = parser.parse_args()
    check(json.loads(args.config.read_text()), args.region)
