import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import urlparse

import requests


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
