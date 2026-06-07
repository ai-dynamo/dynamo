import os
from dataclasses import dataclass
from typing import List


@dataclass
class FlexPriceConfig:
    """Configuration for the auth + FlexPrice billing layer.

    The proxy activates only when DYN_AUTH_ENABLED=true.  FlexPrice usage
    emission is optional on top — it requires DYN_FLEXPRICE_ENABLED=true,
    which in turn requires DYN_AUTH_ENABLED=true because the org UUID must
    come from the authenticated JWT.

    Auth env vars (required to activate the proxy):
        DYN_AUTH_ENABLED    - Enable JWT authentication (default: false)
        DYN_AUTH_SECRET_KEY - HMAC secret(s) for JWT validation, comma-separated for key rotation
        DYN_AUTH_VALID_ORGS - Comma-separated org UUID allowlist; empty = allow all authenticated orgs

    FlexPrice env vars (optional; requires auth):
        DYN_FLEXPRICE_ENABLED              - Enable async usage event emission (default: false)
        DYN_FLEXPRICE_API_KEY              - FlexPrice API key (required when enabled)
        DYN_FLEXPRICE_API_HOST             - FlexPrice API host, e.g. "api.flexprice.io"
        DYN_FLEXPRICE_EVENT_NAME           - Override billing event name (default: "{model}-llm-usage")
        DYN_FLEXPRICE_SOURCE_NAME          - Override billing source name (default: "{model}")
        DYN_FLEXPRICE_INTERNAL_PORT_OFFSET - Port offset for the internal Dynamo HTTP service (default: 1)
    """

    # Auth (master switch for the proxy)
    auth_enabled: bool
    auth_secret_keys: List[str]
    auth_valid_orgs: List[str]

    # FlexPrice billing (optional; requires auth)
    enabled: bool
    api_key: str
    api_host: str
    event_name: str
    source_name: str
    internal_port_offset: int

    @classmethod
    def from_env(cls) -> "FlexPriceConfig":
        auth_enabled = os.environ.get("DYN_AUTH_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )
        enabled = os.environ.get("DYN_FLEXPRICE_ENABLED", "false").lower() in (
            "true", "1", "yes",
        )

        raw_keys = os.environ.get("DYN_AUTH_SECRET_KEY", "")
        auth_secret_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

        raw_orgs = os.environ.get("DYN_AUTH_VALID_ORGS", "")
        auth_valid_orgs = [o.strip() for o in raw_orgs.split(",") if o.strip()]

        return cls(
            auth_enabled=auth_enabled,
            auth_secret_keys=auth_secret_keys,
            auth_valid_orgs=auth_valid_orgs,
            enabled=enabled,
            api_key=os.environ.get("DYN_FLEXPRICE_API_KEY", ""),
            api_host=os.environ.get("DYN_FLEXPRICE_API_HOST", "").rstrip("/"),
            event_name=os.environ.get("DYN_FLEXPRICE_EVENT_NAME", ""),
            source_name=os.environ.get("DYN_FLEXPRICE_SOURCE_NAME", ""),
            internal_port_offset=int(
                os.environ.get("DYN_FLEXPRICE_INTERNAL_PORT_OFFSET", "1")
            ),
        )

    def validate(self) -> None:
        if self.enabled and not self.auth_enabled:
            raise ValueError(
                "DYN_FLEXPRICE_ENABLED=true requires DYN_AUTH_ENABLED=true "
                "(org ID is sourced from the authenticated JWT)"
            )
        if self.auth_enabled and not self.auth_secret_keys:
            raise ValueError(
                "DYN_AUTH_SECRET_KEY is required when DYN_AUTH_ENABLED=true"
            )
        if self.enabled:
            if not self.api_key:
                raise ValueError(
                    "DYN_FLEXPRICE_API_KEY is required when DYN_FLEXPRICE_ENABLED=true"
                )
            if not self.api_host:
                raise ValueError(
                    "DYN_FLEXPRICE_API_HOST is required when DYN_FLEXPRICE_ENABLED=true"
                )
            if self.internal_port_offset < 1:
                raise ValueError("DYN_FLEXPRICE_INTERNAL_PORT_OFFSET must be >= 1")

    @property
    def proxy_required(self) -> bool:
        """True when the Python proxy layer must be inserted in front of Dynamo."""
        return self.auth_enabled

    def resolve_event_name(self, model_name: str = "") -> str:
        if self.event_name:
            return self.event_name
        return f"{model_name}-llm-usage" if model_name else "dynamo-llm-usage"

    def resolve_source_name(self, model_name: str = "") -> str:
        if self.source_name:
            return self.source_name
        return model_name if model_name else "dynamo"
