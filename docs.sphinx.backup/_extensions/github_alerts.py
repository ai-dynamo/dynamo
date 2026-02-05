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

"""
AST-based Sphinx extension to convert GitHub-flavored markdown alerts to MyST admonitions.

This extension works on the parsed document AST, making it more robust than text preprocessing.
It finds blockquote nodes that match GitHub alert patterns and replaces them with admonition nodes.
"""

import re
from typing import Any, Dict

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging

__version__ = "0.2.0"

# Set up logger for the extension
logger = logging.getLogger(__name__)

# Log when the extension module is imported
logger.info(f"GitHub alerts extension v{__version__} imported successfully")


class GitHubAlertsTransformer:
    """AST transformer for GitHub alerts to MyST admonitions."""

    # Mapping of GitHub alert types to MyST admonition types
    ALERT_MAPPING = {
        "note": nodes.note,
        "tip": nodes.tip,
        "important": nodes.important,
        "warning": nodes.warning,
        "caution": nodes.caution,
        "danger": nodes.danger,
        "info": nodes.note,  # Map info to note
        "hint": nodes.tip,  # Map hint to tip
    }

    def __init__(self):
        # Regex to match GitHub alert syntax in text
        self.alert_pattern = re.compile(r"^\[!(.*?)\](?:\s+(.*))?$")

    def is_github_alert_blockquote(self, node: nodes.block_quote) -> bool:
        """
        Check if a blockquote node represents a GitHub alert.

        Returns:
            bool: True if this is a GitHub alert blockquote, False otherwise
        """
        if not isinstance(node, nodes.block_quote):
            return False

        # GitHub alerts start with a paragraph containing [!TYPE]
        if not node.children or not isinstance(node.children[0], nodes.paragraph):
            return False

        first_para = node.children[0]
        if not first_para.children or not isinstance(
            first_para.children[0], nodes.Text
        ):
            return False

        first_text = first_para.children[0].astext()
        match = self.alert_pattern.match(first_text.strip())

        return match is not None

    def create_admonition_node(self, blockquote: nodes.block_quote) -> nodes.admonition:
        """
        Create a docutils admonition node from a GitHub alert blockquote.

        Args:
            blockquote: The blockquote node containing the GitHub alert

        Returns:
            The created admonition node
        """
        # Extract alert information from the blockquote
        first_para = blockquote.children[0]
        first_text = first_para.children[0].astext()
        match = self.alert_pattern.match(first_text.strip())

        if not match:
            raise ValueError("Not a valid GitHub alert blockquote")

        alert_type = match.group(1).lower().strip()
        title = match.group(2).strip() if match.group(2) else None

        # Extract content nodes (everything after the first paragraph)
        content_nodes = []

        # If there's a title, check if there's more content in the first paragraph
        if title and len(first_para.children) > 1:
            # Create new paragraph with remaining content
            remaining_para = nodes.paragraph()
            # Properly detach and add child nodes
            for child in first_para.children[1:]:
                child.parent = None  # Detach from current parent
                remaining_para.append(child)
            content_nodes.append(remaining_para)
        elif not title and len(first_para.children) > 1:
            # No title, but there's content after [!TYPE] - treat as content
            content_para = nodes.paragraph()
            # Properly detach and add child nodes
            for child in first_para.children[1:]:
                child.parent = None  # Detach from current parent
                content_para.append(child)
            content_nodes.append(content_para)

        # Add any additional paragraphs/content
        for child in blockquote.children[1:]:
            child.parent = None  # Detach from current parent
            content_nodes.append(child)

        # Map to MyST admonition type
        admonition_class = self.ALERT_MAPPING.get(alert_type, nodes.note)
        admonition = admonition_class()

        # Add title if present
        if title:
            title_node = nodes.title(title, title)
            admonition.append(title_node)

        # Add content nodes
        for content_node in content_nodes:
            content_node.parent = None  # Ensure node is properly detached
            admonition.append(content_node)

        return admonition

    def transform_document(self, document: nodes.document) -> None:
        """Transform all GitHub alert blockquotes in the document."""

        # Find all blockquote nodes
        blockquotes = document.traverse(nodes.block_quote)

        for blockquote in blockquotes:
            if self.is_github_alert_blockquote(blockquote):
                # Create admonition node from blockquote
                admonition = self.create_admonition_node(blockquote)

                # Replace blockquote with admonition
                blockquote.parent.replace(blockquote, admonition)


def transform_github_alerts(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """
    Transform GitHub alerts in the document tree.

    This function is connected to Sphinx's 'doctree-resolved' event.

    Args:
        app: The Sphinx application instance
        doctree: The document tree to transform
        docname: The document name being processed
    """
    # Check if this is a markdown file by looking at the source file
    # Sphinx strips extensions from docnames, so we need to check the source
    env = app.env
    source_file = env.doc2path(docname, base=None)
    is_markdown = source_file and source_file.suffix in (".md", ".markdown")

    if not is_markdown:
        return

    # Check if the extension is enabled
    if not app.config.github_alerts_enabled:
        return

    logger.debug(f"Processing GitHub alerts in {docname}")

    try:
        # Get the transformer instance
        transformer = getattr(app, "_github_alerts_transformer", None)
        if transformer is None:
            transformer = GitHubAlertsTransformer()
            app._github_alerts_transformer = transformer

        # Count blockquotes before transformation
        initial_blockquotes = list(doctree.traverse(nodes.block_quote))
        initial_admonitions = list(doctree.traverse(nodes.Admonition))
        alert_blockquotes = [
            bq
            for bq in initial_blockquotes
            if transformer.is_github_alert_blockquote(bq)
        ]

        if alert_blockquotes:
            logger.info(
                f"GitHub alerts: Converting {len(alert_blockquotes)} alert(s) in {docname}"
            )

            # Transform the document
            transformer.transform_document(doctree)

            # Count remaining blockquotes and new admonitions for verification
            remaining_blockquotes = list(doctree.traverse(nodes.block_quote))
            remaining_admonitions = list(doctree.traverse(nodes.Admonition))

            logger.debug(
                f"GitHub alerts: {docname} - {len(initial_blockquotes)} → {len(remaining_blockquotes)} blockquotes, {len(remaining_admonitions) - len(initial_admonitions)} admonitions created"
            )
        else:
            logger.debug(f"GitHub alerts: No alerts found in {docname}")
    except Exception as e:
        logger.error(f"GitHub alerts: Error processing {docname}: {e}")
        raise


def setup(app: Sphinx) -> Dict[str, Any]:
    """
    Setup function for the Sphinx extension.

    Args:
        app: The Sphinx application instance

    Returns:
        Extension metadata
    """
    logger.info("GitHub alerts extension setup() called")

    try:
        # Connect our transformer to the doctree-resolved event
        # This happens after parsing but before writing
        app.connect("doctree-resolved", transform_github_alerts)
        logger.info("GitHub alerts extension connected to 'doctree-resolved' event")

        # Add configuration values
        app.add_config_value("github_alerts_enabled", True, "env")

        logger.info("GitHub alerts extension setup completed")

        return {
            "version": __version__,
            "parallel_read_safe": True,
            "parallel_write_safe": True,
        }
    except Exception as e:
        logger.error(f"GitHub alerts extension setup failed: {e}")
        raise
