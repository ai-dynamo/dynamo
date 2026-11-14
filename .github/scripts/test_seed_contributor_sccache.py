# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location(
    "seed_cache", Path(__file__).with_name("seed-contributor-sccache.py")
)
seed_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_cache)
NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


class S3:
    def __init__(self):
        self.source = [
            dict(Key=f"a/b/{i}", Size=10, ETag='"etag"', LastModified=NOW)
            for i in range(3)
        ]
        self.copies = []
        self.error = None

    def get_paginator(self, operation):
        return self

    def paginate(self, **kwargs):
        self.listing = kwargs
        return [{"Contents": self.source[:1]}, {"Contents": self.source[1:]}]

    def copy_object(self, **kwargs):
        if self.error:
            raise self.error
        self.copies.append(kwargs)


class CacheSeedTests(unittest.TestCase):
    def test_preview_makes_no_write_and_keeps_source_prefix(self):
        s3 = S3()
        result = seed_cache.seed(s3, prefix="a/b/", now=NOW)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["considered"], 3)
        self.assertFalse(s3.copies)
        self.assertEqual(s3.listing["Bucket"], "ai-dynamo-sccache")

    def test_copy_is_one_way_key_preserving_and_conditional(self):
        s3 = S3()
        result = seed_cache.seed(s3, execute=True, now=NOW)
        self.assertEqual(result["copied"], 3)
        for copy in s3.copies:
            self.assertEqual(copy["CopySource"]["Bucket"], seed_cache.SOURCE)
            self.assertEqual(copy["Bucket"], seed_cache.DESTINATION)
            self.assertEqual(copy["Key"], copy["CopySource"]["Key"])
            self.assertEqual(copy["IfNoneMatch"], "*")
            self.assertEqual(copy["CopySourceIfMatch"], '"etag"')

    def test_bounds_stop_before_exceeding_budget(self):
        for kwargs in [{"max_objects": 1}, {"max_bytes": 15}]:
            s3 = S3()
            result = seed_cache.seed(s3, execute=True, now=NOW, **kwargs)
            self.assertTrue(result["bounded"])
            self.assertEqual(len(s3.copies), 1)
            self.assertEqual(result["resume_after"], "a/b/0")

    def test_existing_destination_or_changed_source_cannot_be_overwritten(self):
        s3 = S3()
        s3.error = ClientError(
            {
                "Error": {"Code": "PreconditionFailed"},
                "ResponseMetadata": {"HTTPStatusCode": 412},
            },
            "CopyObject",
        )
        result = seed_cache.seed(s3, execute=True, now=NOW)
        self.assertEqual(result["copied"], 0)
        self.assertEqual(result["raced_or_existing"], 3)

    def test_access_denial_does_not_become_a_cache_miss(self):
        s3 = S3()
        s3.error = ClientError(
            {
                "Error": {"Code": "AccessDenied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "CopyObject",
        )
        with self.assertRaises(ClientError):
            seed_cache.seed(s3, execute=True, now=NOW)

    def test_old_entries_are_excluded_without_copying(self):
        s3 = S3()
        s3.source[0]["LastModified"] = NOW - timedelta(days=90)
        self.assertEqual(seed_cache.seed(s3, execute=True, now=NOW)["copied"], 2)


if __name__ == "__main__":
    unittest.main()
