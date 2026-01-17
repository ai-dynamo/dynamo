#!/usr/bin/env python3
# type: ignore  # Ignore all mypy errors in this file

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
import os
import re
import subprocess
from contextlib import contextmanager
from functools import partial

# Get the directory of the current file
dynamo_docs_abspath = os.path.dirname(os.path.abspath(__file__))
dynamo_abspath = os.path.dirname(dynamo_docs_abspath)
repo_url = "https://github.com/ai-dynamo/dynamo/blob/main/"

# Regex patterns
http_patn = r"^https?://"
http_reg = re.compile(http_patn)
tag_patn = "/(?:blob|tree)/main"
dynamo_repo_patn = rf"{http_patn}github.com/ai-dynamo/dynamo"
dynamo_github_url_reg = re.compile(
    rf"{dynamo_repo_patn}/([^/#]+)(?:{tag_patn})?/*([^#]*)\s*(?=#|$)"
)
# relpath_patn = r"]\s*\(\s*([^)]+)\)"
# Hyperlink in a .md file, excluding embedded images.
hyperlink_reg = re.compile(r"((?<!\!)\[[^\]]+\]\s*\(\s*)([^)]+?)(\s*\))")

exclusions = None
with open(f"{dynamo_docs_abspath}/exclusions.txt", "r") as f:
    exclusions = f.read()
    f.close()
exclude_patterns = exclusions.strip().split("\n")


def setup_logger():
    """
    This function is to setup logging
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    # Set the log level
    logger.setLevel(logging.INFO)
    # Create handlers
    stream_handler = logging.StreamHandler()
    # Create formatters and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_message(message):
    """
    This function is for logging to /tmp
    - message: Message to log
    """
    # Setup the logger
    logger = setup_logger()
    # Log the message
    logger.info(message)


def run_command(command):
    """
    This function runs any command using subprocess and logs failures
    - command: Command to execute
    """
    log_message(f"Running command: {command}")
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        raise (e)


def is_excluded(file_path):
    for exclude_pattern in exclude_patterns:
        file_abspath = os.path.abspath(file_path)
        exclude_pattern = os.path.abspath(exclude_pattern)
        if os.path.commonpath([file_abspath, exclude_pattern]) == exclude_pattern:
            return True
    return False


def replace_url_with_relpath(url, src_doc_path):
    """
    This function replaces Triton Inference Server GitHub URLs with relative paths in following cases.
    1. URL is a doc file, e.g. ".md" file.
    2. URL is a directory which contains README.md and URL ends with "#<section>".

    Examples:
        https://github.com/triton-inference-server/server/blob/main/docs/protocol#restricted-protocols
        https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher

    Keep URL in the following cases:
        https://github.com/triton-inference-server/server/tree/r24.02
        https://github.com/triton-inference-server/server/blob/main/build.py
        https://github.com/triton-inference-server/server/blob/main/qa
        https://github.com/triton-inference-server/server/blob/main/CONTRIBUTING.md
    """
    m = dynamo_github_url_reg.match(url)
    # Do not replace URL if it is not a Triton GitHub file.
    if not m:
        return url

    target_repo_name = m.group(1)
    target_relpath_from_target_repo = os.path.normpath(m.groups("")[1])
    section = url[len(m.group(0)) :]
    valid_hashtag = section not in ["", "#"] and section.startswith("#")

    if target_repo_name == "dynamo":
        target_path = os.path.join(dynamo_abspath, target_relpath_from_target_repo)
    else:
        target_path = os.path.join(
            dynamo_docs_abspath, target_repo_name, target_relpath_from_target_repo
        )

    # Return URL if it points to a path outside server/docs.
    if os.path.commonpath([dynamo_docs_abspath, target_path]) != dynamo_docs_abspath:
        return url

    if (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and not is_excluded(target_path)
    ):
        pass
    elif (
        os.path.isdir(target_path)
        and os.path.isfile(os.path.join(target_path, "README.md"))
        and valid_hashtag
        and not is_excluded(os.path.join(target_path, "README.md"))
    ):
        target_path = os.path.join(target_path, "README.md")
    else:
        return url

    # The "target_path" must be a file at this line.
    relpath = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))
    return re.sub(dynamo_github_url_reg, relpath, url, count=1)


def replace_relpath_with_url(relpath, src_doc_path):
    """
    This function replaces relative paths with Triton Inference Server GitHub URLs in following cases.
    1. Relative path is a file that is not ".md" type inside the current repo.
    2. Relative path is a directory but not (has "README.md" and ends with "#<section>").
    3. Relative path does not exist (shows 404 page).

    Examples:
        ../examples/model_repository
        ../examples/model_repository/inception_graphdef/config.pbtxt

    Keep relpath in the following cases:
        build.md
        build.md#building-with-docker
        #building-with-docker
        ../getting_started/quickstart.md
        ../protocol#restricted-protocols
    """
    target_path = relpath.rsplit("#")[0]
    section = relpath[len(target_path) :]
    valid_hashtag = section not in ["", "#"]
    if relpath.startswith("#"):
        target_path = os.path.basename(src_doc_path)
    target_path = os.path.join(os.path.dirname(src_doc_path), target_path)
    target_path = os.path.normpath(target_path)

    # Assert target path is under the current repo directory.
    assert os.path.commonpath([dynamo_abspath, target_path]) == dynamo_abspath

    target_path_from_src_repo = os.path.relpath(target_path, start=dynamo_abspath)

    # For example, target_path of "../protocol#restricted-protocols" should be "<path-to-server>/server/docs/protocol/README.md"
    if (
        os.path.isdir(target_path)
        and valid_hashtag
        and os.path.isfile(os.path.join(target_path, "README.md"))
    ):
        relpath = os.path.join(relpath.rsplit("#")[0], "README.md") + section
        target_path = os.path.join(target_path, "README.md")

    if (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and os.path.commonpath([dynamo_docs_abspath, target_path])
        == dynamo_docs_abspath
        and not is_excluded(target_path)
    ):
        return relpath
    else:
        return repo_url + target_path_from_src_repo + section


def replace_hyperlink(m, src_doc_path):
    """
    TODO: Support of HTML tags for future docs.
    Markdown allows <link>, e.g. <a href=[^>]+>. Whether we want to
    find and replace the link depends on if they link to internal .md files
    or allows relative paths. I haven't seen one such case in our doc so
    should be safe for now.
    """

    hyperlink_str = m.group(2)
    match = http_reg.match(hyperlink_str)

    if match:
        # Hyperlink is a URL.
        res = replace_url_with_relpath(hyperlink_str, src_doc_path)
    else:
        # Hyperlink is a relative path.
        res = replace_relpath_with_url(hyperlink_str, src_doc_path)

    return m.group(1) + res + m.group(3)


def preprocess_docs(exclude_paths=[]):
    # Find all ".md" files inside the current repo.
    if exclude_paths:
        cmd = (
            ["find", dynamo_docs_abspath, "-type", "d", "\\("]
            + " -o ".join([f"-path './{dir}'" for dir in exclude_paths]).split(" ")
            + ["\\)", "-prune", "-o", "-type", "f", "-name", "'*.md'", "-print"]
        )
    else:
        cmd = ["find", dynamo_docs_abspath, "-name", "'*.md'"]
    cmd = " ".join(cmd)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    docs_list = list(filter(None, result.stdout.split("\n")))

    # Read, preprocess and write back to each document file.
    for doc_abspath in docs_list:
        if is_excluded(doc_abspath):
            continue

        content = None
        with open(doc_abspath, "r") as f:
            content = f.read()

        content = hyperlink_reg.sub(
            partial(replace_hyperlink, src_doc_path=doc_abspath),
            content,
        )

        with open(doc_abspath, "w") as f:
            f.write(content)


@contextmanager
def change_directory(path):
    """
    Context manager for changing the current working directory
    """
    original_directory = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_directory)


def update_project_json():
    """Update project.json with the current version from DYNAMO_DOCS_VERSION env var."""
    version = os.environ.get("DYNAMO_DOCS_VERSION", "dev")
    project_json_path = os.path.join(dynamo_docs_abspath, "project.json")

    project_data = {"name": "NVIDIA Dynamo", "version": version}

    with open(project_json_path, "w") as f:
        json.dump(project_data, f)

    log_message(f"Updated project.json with version: {version}")


def main():
    with change_directory(dynamo_docs_abspath):
        run_command("make clean")
        update_project_json()
        preprocess_docs()
        run_command("make html")


if __name__ == "__main__":
    main()
