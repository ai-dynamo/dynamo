# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse

import requests

# [NOTE] this is keep as a reference in case we need to run a local media server to eliminate image server influence.
# However, this implementation is not used as it is actually slower than directly using public image URLs in our benchmark experiments.
#
# Example usage:
# python local_media_server.py \
#     --image test.jpg:https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg &
# IMG_SERVER_PID=$!
# trap "kill $IMG_SERVER_PID" EXIT

# # Wait for the server to start
# for i in {1..10}; do
#     HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" localhost:8233/test.jpg)
#     if [[ "$HTTP_CODE" -eq 200 ]]; then
#         echo "Server is responding with HTTP 200."
#         break
#     else
#         echo "Server did not respond with HTTP 200. Response code: $HTTP_CODE. Retrying in 1 second..."
#         sleep 1
#     fi
#     if [[ $i -eq 10 ]]; then
#         echo "Server did not respond with HTTP 200 after 10 attempts. Exiting."
#         exit 1
#     fi
# done


class LocalMediaServer(BaseHTTPRequestHandler):
    image_store = {}

    @classmethod
    def initialize_images(cls, images):
        for name, url in images.items():
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    cls.image_store[name] = BytesIO(response.content)
                else:
                    print(f"Failed to load image from {url}")
            except Exception as e:
                print(f"Error loading image from {url}: {e}")

    def do_GET(self):
        parsed_path = urlparse(self.path)
        resource = parsed_path.path.lstrip("/")

        if resource and resource in self.image_store:
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            self.wfile.write(self.image_store[resource].getvalue())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Image not found")


def run_server(port, images):
    LocalMediaServer.initialize_images(images)
    server_address = ("", port)
    httpd = HTTPServer(server_address, LocalMediaServer)
    print(f"Server running on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Start a local media server.")
    parser.add_argument(
        "--image",
        action="append",
        help='Specify images in the format "file_name:url". Can be used multiple times.',
        required=True,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8233,
        help="Specify the port number for the server. Default is 8233.",
    )
    args = parser.parse_args()

    images = {}
    for image_arg in args.image:
        try:
            file_name, url = image_arg.split(":", 1)
            images[file_name] = url
        except ValueError:
            print(
                f"Invalid format for image argument: {image_arg}. Expected format is 'file_name:url'."
            )
            exit(1)
    run_server(args.port, images)
