from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import os

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name == "wheel":
            bin_path = os.getenv("DYNAMO_BIN_PATH", "target/release")
            build_data["force_include"] = {
                f"{bin_path}/dynamo-run": "dynamo/sdk/cli/bin/dynamo-run",
                f"{bin_path}/llmctl": "dynamo/sdk/cli/bin/llmctl",
                f"{bin_path}/http": "dynamo/sdk/cli/bin/http",
                f"{bin_path}/metrics": "dynamo/sdk/cli/bin/metrics",
                f"{bin_path}/mock_worker": "dynamo/sdk/cli/bin/mock_worker"
            }