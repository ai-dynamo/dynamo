# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test external publication references without publishing any crates."""

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import tomllib
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "stage_crates", Path(__file__).with_name("stage_crates.py")
)
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)
INDEX = "sparse+https://artifactory.example/artifactory/api/cargo/dynamo/index/"
DEPENDENCY_INDEX = (
    "sparse+https://artifactory.example/artifactory/api/cargo/aisim/index/"
)
REVISION = "a" * 40
REFERENCES = {
    "artifactory": {
        "dependencies": {
            "aisim-core": {
                "version": "=0.14.0-rc.0",
                "registry": "aisim",
                "git-rev": REVISION,
            }
        }
    }
}


class ExternalRegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "Cargo.toml"
        self.manifest.write_text(
            '[workspace]\nmembers = ["lib/consumer"]\nresolver = "2"\n'
            "[workspace.dependencies]\n"
            'aisimulate-core = { package = "aisim-core", version = "=0.14.0", default-features = false }\n'
            "[patch.crates-io]\n"
            f'aisim-core = {{ git = "https://example.invalid/source", rev = "{REVISION}" }}\n'
            'other-core = { git = "https://example.invalid/other", rev = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }\n'
        )
        member = self.root / "lib/consumer"
        (member / "src").mkdir(parents=True)
        self.member = member / "Cargo.toml"
        self.member.write_text(
            '[package]\nname = "consumer"\nversion = "1.0.0"\nedition = "2021"\n'
            'description = "Registry test fixture"\nlicense = "Apache-2.0"\n'
            "[dependencies]\naisimulate-core = { workspace = true }\n"
        )
        (member / "src/lib.rs").write_text(
            "pub fn value() -> u8 { aisimulate_core::value() }\n"
        )
        self.manifests = [self.manifest, self.member]
        self.env = {
            "ARTIFACTORY_TOKEN": "test-token",
            "ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES": json.dumps(
                {"aisim": DEPENDENCY_INDEX}
            ),
        }

    def configure(self, metadata=None):
        return stage.configure_dependency_registries(
            self.root,
            self.manifests,
            metadata if metadata is not None else REFERENCES,
            "artifactory",
            INDEX,
            self.env,
        )

    def test_no_references_preserves_existing_behavior(self):
        before = [p.read_bytes() for p in self.manifests]
        self.env["ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES"] = ""
        self.assertEqual(self.configure({}), {})
        self.assertEqual([p.read_bytes() for p in self.manifests], before)
        self.assertNotIn("CARGO_REGISTRIES_AISIM_TOKEN", self.env)

    def test_alias_and_workspace_inheritance_survive(self):
        self.member.write_text(
            self.member.read_text()
            + 'direct = { package = "aisim-core", version = "=0.14.0", optional = true, features = ["python"] }\n'
        )
        self.member.write_text(
            self.member.read_text() + "[target.'cfg(unix)'.build-dependencies]\n"
            'target_alias = { package = "aisim-core", version = "=0.14.0" }\n'
        )
        registries = self.configure()
        target = tomllib.loads(self.member.read_text())["target"]["cfg(unix)"][
            "build-dependencies"
        ]["target_alias"]
        self.assertEqual(target["version"], "=0.14.0-rc.0")
        self.assertEqual(target["registry"], "aisim")
        root = tomllib.loads(self.manifest.read_text())
        dependency = root["workspace"]["dependencies"]["aisimulate-core"]
        self.assertEqual(
            dependency,
            {
                "package": "aisim-core",
                "version": "=0.14.0-rc.0",
                "registry": "aisim",
                "default-features": False,
            },
        )
        member = tomllib.loads(self.member.read_text())["dependencies"]
        self.assertEqual(member["aisimulate-core"], {"workspace": True})
        self.assertEqual(
            member["direct"],
            {
                "package": "aisim-core",
                "version": "=0.14.0-rc.0",
                "registry": "aisim",
                "optional": True,
                "features": ["python"],
            },
        )
        self.assertEqual(set(root["patch"]["crates-io"]), {"other-core"})
        stage.strip_git_deps(self.root, [self.member])
        self.assertEqual(tomllib.loads(self.member.read_text())["dependencies"], member)
        (self.root / ".cargo").mkdir()
        (self.root / ".cargo/config.toml").write_text(
            '[build]\nrustflags = ["--cfg", "tokio_unstable"]\n'
        )
        stage.write_cargo_config(self.root, "artifactory", INDEX, registries)
        config = tomllib.loads((self.root / ".cargo/config.toml").read_text())
        self.assertEqual(config["build"]["rustflags"], ["--cfg", "tokio_unstable"])
        self.assertEqual(config["registries"]["aisim"]["index"], DEPENDENCY_INDEX)
        self.assertEqual(config["registry"]["default"], "artifactory")
        self.assertEqual(self.env["CARGO_REGISTRIES_AISIM_TOKEN"], "Bearer test-token")
        self.assertNotIn("test-token", (self.root / ".cargo/config.toml").read_text())

    def test_invalid_reference_cannot_mutate_manifests(self):
        for field, value in [
            ("version", "^0.14"),
            ("git-rev", "b" * 40),
            ("registry", "artifactory"),
            ("registry", "bad-name"),
        ]:
            with self.subTest(field=field, value=value):
                metadata = copy.deepcopy(REFERENCES)
                metadata["artifactory"]["dependencies"]["aisim-core"][field] = value
                before = [p.read_bytes() for p in self.manifests]
                with self.assertRaises(RuntimeError):
                    self.configure(metadata)
                self.assertEqual([p.read_bytes() for p in self.manifests], before)

    def test_invalid_registry_configuration_cannot_receive_token(self):
        cases = ["", "[]", "{}"] + [
            json.dumps({"aisim": url})
            for url in (
                "sparse+https://different.example/artifactory/api/cargo/aisim/index/",
                "sparse+http://artifactory.example/artifactory/api/cargo/aisim/index/",
                DEPENDENCY_INDEX + "?token=bad",
                "sparse+https://user:password@artifactory.example/artifactory/api/cargo/aisim/index/",
            )
        ]
        for value in cases:
            with self.subTest(value=value):
                before = [p.read_bytes() for p in self.manifests]
                self.env["ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES"] = value
                with self.assertRaises(RuntimeError):
                    self.configure()
                self.assertEqual([p.read_bytes() for p in self.manifests], before)
                self.assertNotIn("CARGO_REGISTRIES_AISIM_TOKEN", self.env)

    def test_unsupported_multiline_declaration_fails_before_mutation(self):
        self.manifest.write_text(
            self.manifest.read_text().replace(
                'aisimulate-core = { package = "aisim-core", version = "=0.14.0", default-features = false }',
                '[workspace.dependencies.aisimulate-core]\npackage = "aisim-core"\nversion = "=0.14.0"',
            )
        )
        before = [p.read_bytes() for p in self.manifests]
        with self.assertRaisesRegex(RuntimeError, "inline dependencies"):
            self.configure()
        self.assertEqual([p.read_bytes() for p in self.manifests], before)

    def test_mixed_declaration_layouts_fail_before_any_write(self):
        original = self.member.read_text()
        for section in (
            "dependencies.direct",
            "build-dependencies.direct",
            "target.'cfg(unix)'.dependencies.direct",
        ):
            with self.subTest(section=section):
                self.member.write_text(
                    original
                    + f'[{section}]\npackage = "aisim-core"\nversion = "=0.14.0"\n'
                )
                before = [p.read_bytes() for p in self.manifests]
                with self.assertRaisesRegex(RuntimeError, "unsupported declaration"):
                    self.configure()
                self.assertEqual([p.read_bytes() for p in self.manifests], before)

    def test_package_metadata_is_not_a_dependency(self):
        self.member.write_text(
            self.member.read_text()
            + '[package.metadata.tool.dependencies]\nexample = { package = "aisim-core", version = "=0.14.0" }\n'
        )
        before = tomllib.loads(self.member.read_text())["package"]["metadata"]
        self.configure()
        self.assertEqual(
            tomllib.loads(self.member.read_text())["package"]["metadata"], before
        )

    def test_misleading_metadata_text_cannot_be_mutated(self):
        original = self.member.read_text()
        quote = "'" * 3
        variants = [
            "[package.metadata.tool] # trailing comment\n"
            'aisim-core = { package = "aisim-core", version = "=0.14.0" }\n',
            f"[package.metadata.tool]\ntext = {quote}\n[dependencies]\n"
            'aisim-core = { version = "=0.14.0" }\n' + f"{quote}\n",
        ]
        for metadata in variants:
            with self.subTest(metadata=metadata):
                self.member.write_text(original + metadata)
                before = [p.read_bytes() for p in self.manifests]
                with self.assertRaisesRegex(RuntimeError, "unsupported declaration"):
                    self.configure()
                self.assertEqual([p.read_bytes() for p in self.manifests], before)

    def test_expanded_reference_cannot_retain_another_source(self):
        self.member.write_text(
            self.member.read_text() + '[dependencies.direct]\npackage = "aisim-core"\n'
            'version = "=0.14.0-rc.0"\nregistry = "aisim"\npath = "../other-source"\n'
        )
        before = [p.read_bytes() for p in self.manifests]
        with self.assertRaisesRegex(RuntimeError, "reviewed crates.io Git patch"):
            self.configure()
        self.assertEqual([p.read_bytes() for p in self.manifests], before)

    def test_patch_package_identity_must_match(self):
        self.manifest.write_text(
            self.manifest.read_text().replace(
                "aisim-core = { git =",
                'aisim-core = { package = "unrelated-core", git =',
            )
        )
        before = [p.read_bytes() for p in self.manifests]
        with self.assertRaisesRegex(RuntimeError, "current Git source pin"):
            self.configure()
        self.assertEqual([p.read_bytes() for p in self.manifests], before)

    @unittest.skipUnless(
        shutil.which("cargo"), "Cargo is needed for the package contract"
    )
    def test_cargo_package_records_the_external_registry(self):
        # Serve a tiny real Cargo dependency so package normalization is tested,
        # including authenticated sparse-index/download requests, without upload.
        self.manifest.write_text(
            "\n".join(
                line
                for line in self.manifest.read_text().splitlines()
                if not line.startswith("other-core =")
            )
            + "\n"
        )
        self.manifest.write_text(
            self.manifest.read_text()
            + "[workspace.metadata.artifactory.dependencies.aisim-core]\n"
            'version = "=0.14.0-rc.0"\nregistry = "aisim"\n'
            f'git-rev = "{REVISION}"\n'
        )
        metadata = stage.cargo_metadata(self.root)["metadata"]
        self.assertEqual(metadata, REFERENCES)
        self.configure(metadata)
        crate = io.BytesIO()
        prefix = "aisim-core-0.14.0-rc.0"
        files = {
            "Cargo.toml": '[package]\nname = "aisim-core"\nversion = "0.14.0-rc.0"\nedition = "2021"\n[lib]\nname = "aisimulate_core"\n',
            "src/lib.rs": "pub fn value() -> u8 { 1 }\n",
        }
        with tarfile.open(fileobj=crate, mode="w:gz") as archive:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        payload = crate.getvalue()
        row = (
            json.dumps(
                {
                    "name": "aisim-core",
                    "vers": "0.14.0-rc.0",
                    "deps": [],
                    "cksum": hashlib.sha256(payload).hexdigest(),
                    "features": {},
                    "yanked": False,
                }
            ).encode()
            + b"\n"
        )
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                if self.path.endswith("/config.json"):
                    repo = self.path.split("/api/cargo/", 1)[1].split("/", 1)[0]
                    body = json.dumps(
                        {
                            "dl": f"{base}/api/cargo/{repo}/v1/crates",
                            "api": f"{base}/api/cargo/{repo}",
                            "auth-required": True,
                        }
                    ).encode()
                elif self.headers.get("Authorization") != "Bearer test-token":
                    self.send_error(401)
                    return
                elif self.path == "/api/cargo/aisim/index/ai/si/aisim-core":
                    body = row
                elif (
                    self.path
                    == "/api/cargo/aisim/v1/crates/aisim-core/0.14.0-rc.0/download"
                ):
                    body = payload
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        base = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            dependency_index = f"sparse+{base}/api/cargo/aisim/index/"
            stage.write_cargo_config(
                self.root,
                "artifactory",
                f"sparse+{base}/api/cargo/dynamo/index/",
                {"aisim": dependency_index},
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("CARGO_", "GIT_"))
                and key not in {"RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}
            }
            env.update(
                {
                    "CARGO_HOME": str(self.root / "cargo-home"),
                    "CARGO_REGISTRIES_AISIM_TOKEN": "Bearer test-token",
                    "CARGO_REGISTRIES_ARTIFACTORY_TOKEN": "Bearer test-token",
                    "CARGO_REGISTRIES_CRATES_IO_INDEX": f"sparse+{base}/unused-cratesio/",
                    "NO_PROXY": "127.0.0.1,localhost",
                }
            )
            result = subprocess.run(
                [
                    "cargo",
                    "package",
                    "--manifest-path",
                    str(self.member),
                    "--allow-dirty",
                    "--registry",
                    "artifactory",
                ],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with tarfile.open(
                self.root / "target/package/consumer-1.0.0.crate"
            ) as package:
                normalized = tomllib.loads(
                    package.extractfile("consumer-1.0.0/Cargo.toml").read().decode()
                )
            dependency = normalized["dependencies"]["aisimulate-core"]
            self.assertEqual(dependency["package"], "aisim-core")
            self.assertEqual(dependency["version"], "=0.14.0-rc.0")
            self.assertEqual(dependency["registry-index"], dependency_index)
            self.assertNotIn("git", dependency)
            self.assertNotIn("path", dependency)
            self.assertTrue(
                any(
                    path.endswith("/download") and auth == "Bearer test-token"
                    for path, auth in requests
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
