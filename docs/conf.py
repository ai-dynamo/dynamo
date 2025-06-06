#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import json
import os
from datetime import date

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import httplib2
from packaging.version import Version

# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- conf.py setup -----------------------------------------------------------

# conf.py needs to be run in the top level 'docs'
# directory but the calling build script needs to
# be called from the current working directory. We
# change to the 'docs' dir here and then revert back
# at the end of the file.
# current_dir = os.getcwd()
# os.chdir("docs")

# -- Project information -----------------------------------------------------

project = "Dynamo"
copyright = "2025-{}, NVIDIA Corporation".format(date.today().year)
author = "NVIDIA"

# Get the version of dynamo this is building.
version_long = "0.1.0"

version_short = version_long
version_short_split = version_short.split(".")
one_before = f"{version_short_split[0]}.{int(version_short_split[1]) - 1}.{version_short_split[2]}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "ablog",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx-prompt",
    # "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
    "sphinxcontrib.mermaid",
]

suppress_warnings = ["myst.domains", "ref.ref", "myst.header"]

source_suffix = [".rst", ".md"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}

autosummary_generate = True
autosummary_mock_imports = [
    "tritonclient.grpc.model_config_pb2",
    "tritonclient.grpc.service_pb2",
    "tritonclient.grpc.service_pb2_grpc",
]

napoleon_include_special_with_doc = True

numfig = True

# final location of docs for seo/sitemap
html_baseurl = "https://docs.nvidia.com/dynamo/latest/"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    "html_image",
    "colon_fence",
    # "smartquotes",
    "replacements",
    # "linkify",
    "substitution",
]
myst_heading_anchors = 5
myst_fence_as_directive = ["mermaid"]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"] # disable it for nvidia-sphinx-theme to show footer


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "nvidia_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_js_files = ["custom.js"]
# html_css_files = ["custom.css"] # Not needed with new theme

html_theme_options = {
    "collapse_navigation": False,
    "github_url": "https://github.com/ai-dynamo/dynamo",
    # "switcher": {
    # use for local testing
    # "json_url": "http://localhost:8000/_static/switcher.json",
    # "json_url": "https://docs.nvidia.com/dynamo/latest/_static/switcher.json",
    # "version_match": one_before if "dev" in version_long else version_short,
    # },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "primary_sidebar_end": [],
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options.update(
    {
        "collapse_navigation": False,
    }
)

deploy_ngc_org = "nvidia"
deploy_ngc_team = "dynamo"
myst_substitutions = {
    "VersionNum": version_short,
    "deploy_ngc_org_team": f"{deploy_ngc_org}/{deploy_ngc_team}"
    if deploy_ngc_team
    else deploy_ngc_org,
}


def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


# this is a necessary hack to allow us to fill in variables that exist in code blocks
ultimate_replacements = {
    "{VersionNum}": version_short,
    "{SamplesVersionNum}": version_short,
    "{NgcOrgTeam}": f"{deploy_ngc_org}/{deploy_ngc_team}"
    if deploy_ngc_team
    else deploy_ngc_org,
}

# bibtex_bibfiles = ["references.bib"]
# To test that style looks good with common bibtex config
# bibtex_reference_style = "author_year"
# bibtex_default_style = "plain"

### We currently use Myst: https://myst-nb.readthedocs.io/en/latest/use/execute.html
nb_execution_mode = "off"  # Global execution disable
# execution_excludepatterns = ['tutorials/tts-python-basics.ipynb']  # Individual notebook disable

###############################
# SETUP SWITCHER
###############################
switcher_path = os.path.join(html_static_path[0], "switcher.json")
versions = []
# Triton 2 releases
correction = -1 if "dev" in version_long else 0
upper_bound = version_short.split(".")[1]
for i in range(2, int(version_short.split(".")[1]) + correction):
    versions.append((f"2.{i}.0", f"dynamo{i}0"))

# Patch releases
# Add here.

versions = sorted(versions, key=lambda v: Version(v[0]), reverse=True)

# Build switcher data
json_data = []
for v in versions:
    json_data.append(
        {
            "name": v[0],
            "version": v[0],
            "url": f"https://docs.nvidia.com/dynamo/archives/{v[1]}/user-guide/docs",
        }
    )
if "dev" in version_long:
    json_data.insert(
        0,
        {
            "name": f"{one_before} (current_release)",
            "version": f"{one_before}",
            "url": "https://docs.nvidia.com/dynamo/latest/index.html",
        },
    )
else:
    json_data.insert(
        0,
        {
            "name": f"{version_short} (current release)",
            "version": f"{version_short}",
            "url": "https://docs.nvidia.com/dynamo/latest/index.html",
        },
    )

# Trim to last N releases.
json_data = json_data[0:12]

json_data.append(
    {
        "name": "older releases",
        "version": "archives",
        "url": "https://docs.nvidia.com/dynamo/archives/",
    }
)

# validate the links
for i, d in enumerate(json_data):
    h = httplib2.Http()
    resp = h.request(d["url"], "HEAD")
    if int(resp[0]["status"]) >= 400:
        print(d["url"], "NOK", resp[0]["status"])
        # exit(1)

# Write switcher data to file
with open(switcher_path, "w") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
