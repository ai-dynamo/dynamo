from __future__ import annotations

import contextlib
import sys
import urllib.parse
import webbrowser

import click
import rich

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import DEFAULT_ENDPOINT
from bentoml._internal.cloud.config import CloudClientConfig
from bentoml._internal.cloud.config import CloudClientContext
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils.cattr import bentoml_cattr
from bentoml.exceptions import CLIException
from bentoml.exceptions import CloudRESTApiClientError
from bentoml_cli.auth_server import AuthCallbackHttpServer
from bentoml_cli.utils import BentoMLCommandGroup


@click.group(name="cloud", cls=BentoMLCommandGroup)
def cloud_command():
    """BentoCloud Subcommands Groups."""


@cloud_command.command()
@click.option(
    "--endpoint",
    type=click.STRING,
    help="Dynamo Cloud endpoint",
    default=DEFAULT_ENDPOINT,
    envvar="DYNAMO_CLOUD_API_ENDPOINT",
    show_default=True,
    show_envvar=True,
    required=True,
)
@click.option(
    "--api-token",
    type=click.STRING,
    help="Dynamo Cloud user API token",
    envvar="DYNAMO_CLOUD_API_KEY",
    show_envvar=True,
    required=True,
)
def login(endpoint: str, api_token: str) -> None:  # type: ignore (not accessed)
    """Authenticate to Dynamo Cloud. You can find deployment instructions for this in our docs"""
    try:
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
        rich.print(
            f":white_check_mark: Configured BentoCloud credentials (current-context: {ctx.name})"
        )
        rich.print(
            f":white_check_mark: Logged in as [blue]{user.email}[/] at [blue]{org.name}[/] organization"
        )
    except CloudRESTApiClientError as e:
        if e.error_code == 401:
            rich.print(
                f":police_car_light: Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)",
                file=sys.stderr,
            )
        else:
            rich.print(
                f":police_car_light: Error validating token: HTTP {e.error_code}",
                file=sys.stderr,
            )


@cloud_command.command()
def current_context() -> None:  # type: ignore (not accessed)
    """Get current cloud context."""
    rich.print_json(
        data=bentoml_cattr.unstructure(CloudClientConfig.get_config().get_context())
    )


@cloud_command.command()
def list_context() -> None:  # type: ignore (not accessed)
    """List all available context."""
    config = CloudClientConfig.get_config()
    rich.print_json(data=bentoml_cattr.unstructure([i.name for i in config.contexts]))


@cloud_command.command()
@click.argument("context_name", type=click.STRING)
def update_current_context(context_name: str) -> None:  # type: ignore (not accessed)
    """Update current context"""
    ctx = CloudClientConfig.get_config().set_current_context(context_name)
    rich.print(f"Successfully switched to context: {ctx.name}")