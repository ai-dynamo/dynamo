# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal in-cluster Kubernetes client shared by Helm preflight Jobs."""

import json
import ssl
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener


def kubernetes_client():
    """Create an authenticated JSON GET client from the mounted service account."""
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    context = ssl.create_default_context(cafile=str(account / "ca.crt"))
    token = (account / "token").read_text().strip()
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context))

    def get(path, missing_ok=False):
        request = Request(
            "https://kubernetes.default.svc" + path,
            headers={"Authorization": "Bearer " + token},
        )
        try:
            with opener.open(request, timeout=10) as response:
                return json.load(response)
        except HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            raise

    return get
