# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.e2e]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("use_tls", [False, True], ids=["http", "https"])
@pytest.mark.filterwarnings(
    "ignore:Unverified HTTPS request:urllib3.exceptions.InsecureRequestWarning"
)
def test_custom_routes_and_built_ins_share_frontend(
    request, dynamo_dynamic_ports, tmp_path, use_tls: bool
) -> None:
    plugin = tmp_path / "custom_routes.py"
    plugin.write_text(
        "from dynamo.frontend.routes import Request, Response, Router\n"
        "from dynamo.llm import HttpError\n"
        "from dynamo.runtime import DistributedRuntime\n"
        "\n"
        "async def register_routes(\n"
        "    router: Router, runtime: DistributedRuntime\n"
        ") -> None:\n"
        "    @router.route('/custom/{tenant}', methods=['GET', 'POST'])\n"
        "    async def custom(request: Request) -> Response:\n"
        "        return Response.json({\n"
        "            'method': request.method,\n"
        "            'tenant': request.path_params['tenant'],\n"
        "            'query': request.query_params,\n"
        "            'headers': request.headers.getall('X-Test'),\n"
        "            'body': request.body.decode(),\n"
        "            'context_id': request.context.id(),\n"
        "            'metadata': request.context.metadata.copy(),\n"
        "        })\n"
        "\n"
        "    @router.get('/custom-http-error')\n"
        "    async def http_error(request: Request) -> Response:\n"
        "        raise HttpError(418, 'teapot')\n"
        "\n"
        "    @router.get('/custom-internal-error')\n"
        "    async def internal_error(request: Request) -> Response:\n"
        "        raise RuntimeError('private traceback detail')\n"
    )
    frontend_port = dynamo_dynamic_ports.frontend_port
    extra_args = ["--custom-routes", str(plugin)]
    if use_tls:
        certificate = tmp_path / "certificate.pem"
        private_key = tmp_path / "private-key.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(private_key),
                "-out",
                str(certificate),
                "-days",
                "1",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        extra_args.extend(
            ["--tls-cert-path", str(certificate), "--tls-key-path", str(private_key)]
        )
    frontend = DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=extra_args,
        extra_env={"DYN_DISCOVERY_BACKEND": "mem", "DYN_REQUEST_PLANE": "tcp"},
        terminate_all_matching_process_names=False,
    )
    frontend.health_check_ports = [frontend_port]
    base_url = f"{'https' if use_tls else 'http'}://localhost:{frontend_port}"
    verify = not use_tls

    with frontend:
        response = requests.post(
            f"{base_url}/custom/acme?tag=a&tag=b",
            data=b"payload",
            headers={"X-Test": "value", "X-Dynamo-Meta-Scope": "gold"},
            timeout=10,
            verify=verify,
        )
        assert response.status_code == 200
        payload = response.json()
        context_id = payload.pop("context_id")
        assert payload == {
            "method": "POST",
            "tenant": "acme",
            "query": {"tag": ["a", "b"]},
            "headers": ["value"],
            "body": "payload",
            "metadata": {"scope": "gold"},
        }
        assert context_id

        models = requests.get(f"{base_url}/v1/models", timeout=10, verify=verify)
        assert models.status_code == 200

        openapi = requests.get(
            f"{base_url}/openapi.json", timeout=10, verify=verify
        ).json()
        assert "get" in openapi["paths"]["/custom/{tenant}"]
        assert "post" in openapi["paths"]["/custom/{tenant}"]

        http_error = requests.get(
            f"{base_url}/custom-http-error", timeout=10, verify=verify
        )
        assert http_error.status_code == 418
        assert http_error.text == "teapot"

        internal_error = requests.get(
            f"{base_url}/custom-internal-error", timeout=10, verify=verify
        )
        assert internal_error.status_code == 500
        assert internal_error.text == "Internal server error"
        assert "private traceback detail" not in internal_error.text
