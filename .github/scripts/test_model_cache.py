# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache integrity, publication boundary, and real offline HF lookup regressions."""

import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "model_cache", Path(__file__).with_name("model-cache.py")
)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
MODEL = {"id": "example/tiny", "revision": "a" * 40}
DIGEST = "b" * 64


class S3:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def upload_file(self, path, bucket, key, **kwargs):
        self.objects[bucket, key] = Path(path).read_bytes()
        self.writes.append((bucket, key))

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Bucket, Key] = Body
        self.writes.append((Bucket, Key))

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Bucket, Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[bucket, key])


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.s3 = S3()

    def snapshot(self, repo, revision, cache_dir, **kwargs):
        snapshot = Path(cache_dir) / "hub" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text('{"model_type":"example"}')
        return str(snapshot)

    def publish(self):
        cache.publish(
            self.s3,
            [MODEL],
            DIGEST,
            [cache.PR_BUCKET, cache.APPROVED_BUCKET],
            self.snapshot,
            self.root,
        )

    def test_both_stores_receive_files_before_their_completion_index(self):
        self.publish()
        for bucket in cache.BUCKETS:
            writes = [key for b, key in self.s3.writes if b == bucket]
            self.assertTrue(writes[-1].endswith("/index.json"))
            self.assertTrue(writes[0].endswith("/files/config.json"))
        self.assertEqual(len(self.s3.objects), 4)

    def test_single_store_publication_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both"):
            cache.publish(
                self.s3, [MODEL], DIGEST, [cache.PR_BUCKET], self.snapshot, self.root
            )
        self.assertFalse(self.s3.writes)

    def test_offline_hf_lookup_resolves_the_pinned_revision_without_aws(self):
        from huggingface_hub import hf_hub_download

        self.publish()
        destination = self.root / "models"
        cache.hydrate(self.s3, [MODEL], DIGEST, cache.PR_BUCKET, destination)
        path = hf_hub_download(
            MODEL["id"],
            "config.json",
            cache_dir=destination / "hub",
            local_files_only=True,
            token=False,
        )
        self.assertIn(MODEL["revision"], path)
        self.assertEqual(json.loads(Path(path).read_text()), {"model_type": "example"})

    def test_corrupted_download_does_not_publish_a_ref(self):
        self.publish()
        key = cache.model_prefix(DIGEST, MODEL) + "/files/config.json"
        self.s3.objects[cache.PR_BUCKET, key] = b"corrupted"
        destination = self.root / "models"
        with self.assertRaisesRegex(ValueError, "integrity"):
            cache.hydrate(self.s3, [MODEL], DIGEST, cache.PR_BUCKET, destination)
        self.assertFalse(list(destination.rglob("refs/main")))

    def test_missing_completion_index_fails(self):
        with self.assertRaises(KeyError):
            cache.hydrate(
                self.s3, [MODEL], DIGEST, cache.PR_BUCKET, self.root / "models"
            )

    def test_inventory_mismatch_and_path_traversal_fail(self):
        for field, value in [
            ("inventory_sha256", "c" * 64),
            ("files", [{"path": "../escape", "size": 1, "sha256": "a" * 64}]),
        ]:
            self.publish()
            key = cache.model_prefix(DIGEST, MODEL) + "/index.json"
            index = json.loads(self.s3.objects[cache.PR_BUCKET, key])
            index[field] = value
            self.s3.objects[cache.PR_BUCKET, key] = json.dumps(index).encode()
            with self.assertRaises(ValueError):
                cache.hydrate(
                    self.s3, [MODEL], DIGEST, cache.PR_BUCKET, self.root / "models"
                )

    def test_existing_symlink_cannot_redirect_model_writes(self):
        self.publish()
        destination = self.root / "models"
        destination.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (destination / "hub").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.hydrate(self.s3, [MODEL], DIGEST, cache.PR_BUCKET, destination)
        self.assertFalse(list(outside.iterdir()))

    def test_mutable_and_duplicate_model_revisions_are_rejected(self):
        for models in [[dict(MODEL, revision="main")], [MODEL, MODEL]]:
            path = self.root / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "models": models}))
            with self.assertRaises(ValueError):
                cache.inventory(path)

    def test_cache_namespace_changes_when_approved_inventory_changes(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "models": [MODEL]}))
        _, first = cache.inventory(path)
        path.write_text(
            json.dumps(
                {"schema_version": 1, "models": [dict(MODEL, revision="c" * 40)]}
            )
        )
        _, second = cache.inventory(path)
        self.assertNotEqual(first, second)

    def test_pytest_receives_models_dir_only_when_explicitly_configured(self):
        script = Path(__file__).with_name("pytest-local.sh").resolve()
        for enabled in [False, True]:
            with (
                self.subTest(enabled=enabled),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                binary = root / "bin"
                binary.mkdir()
                python = binary / "python"
                python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARG_LOG"\n')
                python.chmod(0o755)
                models = root / "models"
                models.mkdir()
                env = dict(
                    os.environ,
                    PATH=f"{binary}:{os.environ['PATH']}",
                    ARG_LOG=str(root / "args"),
                    PYTEST_MARKS="unit",
                    CONTAINER_WORKSPACE=str(root),
                    GITHUB_WORKSPACE=str(root),
                    GITHUB_WORKFLOW="fixture",
                    GITHUB_RUN_ID="1",
                    GITHUB_ENV=str(root / "env"),
                    TEST_SUITE_NAME="fixture",
                    TEST_TYPE="cpu",
                    PLATFORM_ARCH="amd64",
                    PARALLEL_MODE="none",
                    JOB_CHECK_RUN_ID="1",
                    GPU_TESTS="false",
                )
                env.pop("MODELS_DIR", None)
                if enabled:
                    env["MODELS_DIR"] = str(models)
                subprocess.run(
                    ["bash", str(script)],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                )
                args = (root / "args").read_text().splitlines()
                self.assertEqual(
                    any(arg.startswith("--models-dir=") for arg in args), enabled
                )


if __name__ == "__main__":
    unittest.main()
