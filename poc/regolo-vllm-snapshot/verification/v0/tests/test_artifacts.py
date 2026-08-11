import json
import pathlib
import stat
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]


class ImplementationArtifactContract(unittest.TestCase):
    def test_entrypoint_contract(self):
        path = ROOT / "implementation" / "image" / "snapshot-entrypoint"
        body = path.read_text()
        self.assertIn("SNAPSHOT_READY_URL", body)
        self.assertIn("DYN_SNAPSHOT_CONTROL_DIR", body)
        self.assertIn("DYN_SNAPSHOT_RESTORE_STANDBY", body)
        self.assertIn("ready-for-snapshot", body)
        self.assertTrue(path.stat().st_mode & stat.S_IXUSR)

    def test_placeholder_is_layered_on_pinned_base_and_has_entrypoint(self):
        body = (ROOT / "implementation" / "image" / "Dockerfile").read_text()
        self.assertIn("ARG SNAPSHOT_TOOLS_IMAGE", body)
        self.assertIn('ENTRYPOINT ["/usr/local/bin/snapshot-entrypoint"]', body)

    def test_workload_is_single_gpu_server_container(self):
        manifest = json.loads(
            (ROOT / "implementation" / "k8s" / "pod.template.json").read_text()
        )
        self.assertEqual(manifest["kind"], "Pod")
        containers = manifest["spec"]["containers"]
        self.assertEqual([c["name"] for c in containers], ["server"])
        self.assertEqual(containers[0]["resources"]["limits"]["nvidia.com/gpu"], 1)
        self.assertEqual(containers[0]["resources"]["requests"]["nvidia.com/gpu"], 1)
        self.assertEqual(
            manifest["metadata"]["annotations"]["nvidia.com/snapshot-target-containers"],
            "server",
        )
        self.assertEqual(manifest["spec"]["restartPolicy"], "Never")

    def test_rbac_is_strictly_namespace_scoped(self):
        docs = json.loads((ROOT / "implementation" / "k8s" / "rbac.json").read_text())
        self.assertEqual(docs["kind"], "List")
        kinds = {item["kind"] for item in docs["items"]}
        self.assertEqual(kinds, {"Role", "RoleBinding"})

    def test_pinned_revision_and_chart_only_install(self):
        config = json.loads((ROOT / "implementation" / "config.json").read_text())
        self.assertEqual(config["dynamo_version"], "v1.3.0")
        self.assertEqual(
            config["dynamo_commit"], "8ce9e22f11576402102ea9d8b8e46233f5430a0d"
        )
        install = (ROOT / "implementation" / "bin" / "install-snapshot-chart").read_text()
        self.assertIn("deploy/helm/charts/snapshot", install)
        self.assertNotIn("charts/platform", install)


if __name__ == "__main__":
    unittest.main()
