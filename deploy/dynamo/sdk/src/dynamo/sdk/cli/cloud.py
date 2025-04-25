#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import rich
import typer
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import CloudClientConfig, CloudClientContext
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import CLIException, CloudRESTApiClientError

app = typer.Typer(
    help="Interact with your Dynamo Cloud Server",
    add_completion=True,
    no_args_is_help=True,
)
console = rich.console.Console()


@app.command()
def login(
    endpoint: str = typer.Argument(
        ..., help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD_API_ENDPOINT"
    )
) -> None:
    """Connect to your Dynamo Cloud. You can find deployment instructions for this in our docs"""
    try:
        api_token = ""  # Using empty string for now as it's not used
        cloud_rest_client = RestApiClient(endpoint, api_token)
        user = cloud_rest_client.v1.get_current_user()

        if user is None:
            raise CLIException("current user is not found")

        org = cloud_rest_client.v1.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

        current_context_name = CloudClientConfig.get_config().current_context_name
        cloud_context = BentoMLContainer.cloud_context.get()

        ctx = CloudClientContext(
            name=cloud_context if cloud_context is not None else current_context_name,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        ctx.save()
        console.print(
            f":white_check_mark: Configured Dynamo Cloud credentials (current-context: {ctx.name})"
        )
        console.print(
            f":white_check_mark: Logged in as [blue]{user.email}[/] at [blue]{org.name}[/] organization"
        )
    except CloudRESTApiClientError as e:
        if e.error_code == 401:
            console.print(
                f":police_car_light: Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)"
            )
        else:
            console.print(
                f":police_car_light: Error validating token: HTTP {e.error_code}"
            )
