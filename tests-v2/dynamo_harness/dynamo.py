# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The facade a test receives.

Two construction paths, and the difference is the whole point:

    Dynamo.attach(url)          -> query only; `deployment` is None
    Dynamo.deploy(Docker(...))  -> also start/stop/kill/restart

Components that are not part of a deployment stay ``None``, so a test touching
one fails immediately and by name instead of timing out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .capabilities import Capability, Report, Verdict, all_unknown
from .components import Frontend, Worker
from .deployment import Attached, Deployment, NotControllable
from .transport import Http


@dataclass
class Dynamo:
    frontend: Optional[Frontend] = None
    worker: Optional[Worker] = None
    deployment: Optional[Deployment] = None
    headers: Dict[str, str] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def attach(
        cls, base_url: str, *, model: Optional[str] = None, **http_kw: Any
    ) -> "Dynamo":
        """Point at a Dynamo somebody else deployed. No lifecycle control."""
        http = Http(base_url, **http_kw)
        # Attached: no deployment handle, so components cannot be controlled.
        return cls(frontend=Frontend(http, model=model), worker=None, deployment=None)

    @classmethod
    def deploy(
        cls, deployment: Deployment, *, model: Optional[str] = None, **http_kw: Any
    ) -> "Dynamo":
        """Bring a deployment up and bind components to it."""
        base_url = deployment.start()
        http = Http(base_url, **http_kw)
        resolved = model or getattr(deployment, "model", None)
        return cls(
            frontend=Frontend(http, deployment=deployment, model=resolved),
            worker=Worker(
                http, deployment=deployment, backend=getattr(deployment, "backend", "")
            ),
            deployment=deployment,
        )

    # -- convenience -------------------------------------------------------
    @property
    def base_url(self) -> str:
        if not self.frontend:
            raise RuntimeError("this Dynamo has no frontend")
        return self.frontend.http.base_url

    def wait_until_serving(self, timeout: float = 900.0) -> str:
        if not self.frontend:
            raise RuntimeError("this Dynamo has no frontend")
        return self.frontend.wait_until_serving(timeout=timeout)

    def capabilities(self) -> Dict[Capability, Report]:
        if self.deployment is None:
            return all_unknown("no deployment handle")
        return self.deployment.capabilities()

    def check(self, capability: Capability) -> Report:
        """Three-valued: SATISFIED / UNSATISFIED / UNKNOWN, with a reason."""
        return self.capabilities().get(
            capability, Report(capability, Verdict.UNKNOWN, "not modelled", "n/a")
        )

    def require(self, capability: Capability) -> None:
        """Skip, with attribution, unless the deployment supports this.

        UNKNOWN is reported distinctly from UNSATISFIED so that "we could not
        tell" is never silently recorded as "not supported" -- and so CI can
        gate on it (see --on-unknown-requirement).
        """
        import pytest  # local: the harness itself does not depend on pytest

        report = self.check(capability)
        if report.verdict is Verdict.SATISFIED:
            return
        if report.verdict is Verdict.UNKNOWN:
            pytest.skip(f"UNKNOWN requirement: {report}")
        pytest.skip(f"unsupported: {report}")

    def require_deployment(self) -> Deployment:
        """For tests that need to restart or inspect infrastructure."""
        if self.deployment is None or isinstance(self.deployment, Attached):
            raise NotControllable(
                "this test needs lifecycle control but Dynamo was attached to an "
                "existing deployment; run with --dynamo-image instead of "
                "--dynamo-url"
            )
        return self.deployment

    def close(self) -> None:
        if self.deployment is not None:
            try:
                self.deployment.stop()
            except NotControllable:
                pass
