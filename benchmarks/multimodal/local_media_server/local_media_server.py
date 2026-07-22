# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class LocalMediaServer(BaseHTTPRequestHandler):
    image_store: dict[str, bytes] = {}
    processing_time_mean_ms: float = 0.0
    processing_time_variance_ms: float = 0.0

    @classmethod
    def set_images(cls, images: dict[str, bytes]) -> None:
        cls.image_store = dict(images)

    def _sample_processing_time_s(self) -> float:
        mean = self.processing_time_mean_ms
        variance = self.processing_time_variance_ms
        if variance <= 0.0:
            return max(mean, 0.0) / 1000.0
        sample_ms = random.normalvariate(mean, math.sqrt(variance))
        return max(sample_ms, 0.0) / 1000.0

    def do_GET(self) -> None:
        start = time.monotonic()
        target_s = self._sample_processing_time_s()

        parsed_path = urlparse(self.path)
        resource = parsed_path.path.lstrip("/")

        if resource and resource in self.image_store:
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            self.wfile.write(self.image_store[resource])
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Image not found")

        # wfile.write above starts the byte transfer, but with no
        # Content-Length and HTTP/1.0 the response body runs until
        # connection close — the client only sees end-of-body when do_GET
        # returns and the stdlib closes the socket (sending FIN) for us.
        # Sleeping here therefore extends the client's observed response
        # time even though the bytes are already in flight.
        remaining = target_s - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)


def run_server(
    port: int,
    images: dict[str, bytes],
    processing_time_mean_ms: float,
    processing_time_variance_ms: float,
) -> None:
    LocalMediaServer.set_images(images)
    LocalMediaServer.processing_time_mean_ms = processing_time_mean_ms
    LocalMediaServer.processing_time_variance_ms = processing_time_variance_ms
    httpd = ThreadingHTTPServer(("", port), LocalMediaServer)
    print(
        f"Server running on port {port} "
        f"(processing_time_mean_ms={processing_time_mean_ms}, "
        f"processing_time_variance_ms={processing_time_variance_ms})"
    )
    httpd.serve_forever()
