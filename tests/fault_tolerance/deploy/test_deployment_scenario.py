# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import pytest

from tests.fault_tolerance.deploy.scenarios import DeletePodFailure
from tests.utils.managed_aiperf_deployment import LoadConfig, ManagedAIPerfDeployment
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_rolling_restart(
    request,
    image: str,
    namespace: str,
    skip_service_restart: bool,
):
    logger = logging.getLogger(request.node.name)

    deployment_spec = DeploymentSpec(
        "/workspace/examples/backends/trtllm/deploy/disagg.yaml"
    )

    if image:
        deployment_spec.set_image(image)

    _model = deployment_spec.get_model()  # noqa: F841

    deployment_spec.set_logging(True, "info")

    endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        #       time.sleep(10)

        #        <k8s-ns>-<lowercase(dgd-name)>.<k8s-ns>.svc.cluster.local

        endpoint_url = f"http://{deployment_spec.name.lower()}-{deployment_spec.frontend_service.name.lower()}.{namespace.lower()}.svc.cluster.local:{deployment_spec.port}"
        print(endpoint_url)

        async with ManagedAIPerfDeployment(
            namespace=namespace,
            load_config=LoadConfig(endpoint_url=endpoint_url, duration_minutes=100),
            log_dir=request.node.name,
        ) as load:
            _results = await load.run(wait_for_completion=False)  # noqa: F841

            await load._wait_for_started()

            time.sleep(30)

            await DeletePodFailure(0, ["frontend"]).execute(deployment, logger)

            await deployment.wait_for_unready(timeout=60, log_interval=10)
            await deployment._wait_for_ready(timeout=1800)

            time.sleep(30)

            await load.terminate()

            # print(results)


#        print(model)


# Populate shared context for validation
#        validation_context["deployment"] = deployment
#       validation_context["namespace"] = namespace

#        with _clients(
#           logger,
#          request.node.name,
#         deployment_spec,
#        namespace,
#       model,
#      scenario.load,  # Pass entire Load config object
#  ) as client_procs:
#     time.sleep(100)
#    # Inject failures and capture which pods were affected
#   affected_pods = await _inject_failures(
#      scenario.failures, logger, deployment
#  )
#  logger.info(f"Affected pods during test: {affected_pods}")

#  if scenario.load.continuous_load:
#     _terminate_client_processes(client_procs, logger)
