#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preview or seed the contributor compiler cache without overwriting its keys."""

import argparse
import json
from datetime import datetime, timedelta, timezone

SOURCE = "ai-dynamo-sccache"
DESTINATION = "dynamo-ci-210086341041-us-west-2-pr-sccache"
OWNER = "210086341041"


def seed(
    s3,
    *,
    execute=False,
    max_objects=100000,
    max_bytes=50 * 1024**3,
    max_age_days=30,
    prefix="",
    after_key="",
    now=None,
):
    """Preserve cache keys; conditional server-side copies never replace PR data."""
    if max_objects <= 0 or max_bytes <= 0 or max_age_days < 0:
        raise ValueError("Cache copy bounds must be positive (age may be zero)")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days) if max_age_days else None
    result = {
        "dry_run": not execute,
        "source": SOURCE,
        "destination": DESTINATION,
        "considered": 0,
        "bytes": 0,
        "copied": 0,
        "raced_or_existing": 0,
        "bounded": False,
        "resume_after": after_key,
    }
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=SOURCE, Prefix=prefix, StartAfter=after_key, ExpectedBucketOwner=OWNER
    )
    for page in pages:
        for item in page.get("Contents", []):
            key, size = item["Key"], item["Size"]
            if cutoff and item["LastModified"] < cutoff:
                result["resume_after"] = key
                continue
            if size > 5 * 1024**3:
                raise ValueError("Compiler cache object exceeds the atomic-copy limit")
            if (
                result["considered"] >= max_objects
                or result["bytes"] + size > max_bytes
            ):
                result["bounded"] = True
                return result
            result["considered"] += 1
            result["bytes"] += size
            if execute:
                try:
                    s3.copy_object(
                        Bucket=DESTINATION,
                        Key=key,
                        CopySource={"Bucket": SOURCE, "Key": key},
                        CopySourceIfMatch=item["ETag"],
                        IfNoneMatch="*",
                        ExpectedBucketOwner=OWNER,
                        ExpectedSourceBucketOwner=OWNER,
                        MetadataDirective="REPLACE",
                        TaggingDirective="REPLACE",
                        ServerSideEncryption="AES256",
                    )
                    result["copied"] += 1
                except Exception as error:
                    status = (
                        getattr(error, "response", {})
                        .get("ResponseMetadata", {})
                        .get("HTTPStatusCode")
                    )
                    if status != 412:
                        raise
                    # Either destination already exists or the source changed
                    # since enumeration. Neither condition permits replacement.
                    result["raced_or_existing"] += 1
            result["resume_after"] = key
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Copy; otherwise metadata-only preview"
    )
    parser.add_argument("--max-objects", type=int, default=100000)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--after-key", default="")
    args = parser.parse_args()
    import boto3

    s3 = boto3.client("s3", region_name="us-west-2")
    shape = s3.meta.service_model.operation_model("CopyObject").input_shape
    if "IfNoneMatch" not in shape.members:
        raise RuntimeError("This SDK cannot enforce conditional destination copies")
    print(
        json.dumps(
            seed(
                s3,
                execute=args.execute,
                max_objects=args.max_objects,
                max_bytes=args.max_bytes,
                max_age_days=args.max_age_days,
                prefix=args.prefix,
                after_key=args.after_key,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
