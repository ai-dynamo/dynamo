#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to detect broken links in markdown files and problematic symbolic links within a git repository.

This script:
1. Finds all .md files in the specified directory (recursively)
2. Parses each file to extract links to other .md files
3. Validates if the linked files exist
4. Detects problematic symbolic links (broken, circular, outside repo)
5. Generates a JSON or HTML report of broken links and problematic symlinks with line numbers
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def setup_logging(
    verbose: bool = False, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("detect_broken_links")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")

    return logger


def path_relative_to_git_root(
    file_path: Path, git_root_dir: str, logger: logging.Logger
) -> str:
    """
    Convert a file path to be relative to the git repository root.

    Args:
        file_path: The file path to convert
        git_root_dir: The git repository root directory
        logger: Logger instance for logging

    Returns:
        Path relative to git repository root as string
    """
    try:
        git_root_abs = Path(git_root_dir).resolve()
        file_abs = Path(file_path).resolve()

        # Calculate relative path from git repo root to file
        relative_path = file_abs.relative_to(git_root_abs)
        result = str(relative_path)
        logger.debug(f"Converted {file_path} to git-relative path: {result}")
        return result

    except ValueError as e:
        # If relative_to fails, it means the file is not under the git repo
        logger.warning(
            f"File {file_path} is not under Git repository {git_root_dir}: {e}"
        )
        return str(file_path)
    except Exception as e:
        logger.warning(f"Error converting path {file_path} to git-relative: {e}")
        return str(file_path)


def get_git_info(
    logger: logging.Logger, start_path: str = "."
) -> Optional[Dict[str, str]]:
    """
    Get Git repository information including remote URL and current branch.

    Args:
        logger: Logger instance for logging
        start_path: Starting path to search for git repository

    Returns:
        Dictionary with 'remote_url', 'branch', and 'git_root_dir' keys, or None if not in a Git repo
    """
    try:
        logger.debug(f"Attempting to get Git repository information from: {start_path}")

        # Find the git repository root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=start_path,
        )
        if result.returncode != 0:
            logger.warning("Not in a Git repository or Git not available")
            return None

        git_root_dir = result.stdout.strip()
        logger.debug(f"Found Git repository root: {git_root_dir}")

        # Check if this repository has a remote origin URL
        remote_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            cwd=git_root_dir,
        )
        if remote_result.returncode != 0 or not remote_result.stdout.strip():
            logger.warning("Git repository has no remote origin URL")
            return None

        remote_url = remote_result.stdout.strip()
        logger.debug(f"Remote URL: {remote_url}")

        # Get current branch
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=git_root_dir,
        )
        if branch_result.returncode != 0:
            logger.warning("Could not get current branch")
            return None

        branch = branch_result.stdout.strip()
        logger.debug(f"Current branch: {branch}")

        # Get current commit hash
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=git_root_dir,
        )
        if commit_result.returncode != 0:
            logger.warning("Could not get current commit hash")
            commit_hash = "unknown"
        else:
            commit_hash = commit_result.stdout.strip()
            logger.debug(f"Current commit: {commit_hash}")

        # Get short commit hash (first 7 characters)
        short_commit = commit_hash[:7] if commit_hash != "unknown" else "unknown"

        return {
            "remote_url": remote_url,
            "branch": branch,
            "commit_hash": commit_hash,
            "short_commit": short_commit,
            "git_root_dir": git_root_dir,
        }

    except Exception as e:
        logger.warning(f"Error getting Git information: {e}")
        return None


def construct_github_url(
    file_path: str,
    git_info: Dict[str, str],
    logger: logging.Logger,
    line_number: Optional[int] = None,
) -> str:
    """
    Construct a GitHub URL for a file, optionally linking to a specific line.

    Args:
        file_path: Path to the file relative to the repository root
        git_info: Git repository information
        logger: Logger instance for logging
        line_number: Optional line number to link to (adds #L{line_number} anchor)

    Returns:
        GitHub URL for the file, optionally with line anchor
    """
    try:
        remote_url = git_info["remote_url"]
        branch = git_info["branch"]

        # Handle SSH and HTTPS URLs
        if remote_url.startswith("git@github.com:"):
            # Convert SSH to HTTPS
            repo_path = remote_url.replace("git@github.com:", "")
            if repo_path.endswith(".git"):
                repo_path = repo_path[:-4]
            github_base = f"https://github.com/{repo_path}"
        elif remote_url.startswith("https://github.com/"):
            # Already HTTPS
            github_base = remote_url
            if github_base.endswith(".git"):
                github_base = github_base[:-4]
        else:
            logger.warning(f"Unsupported remote URL format: {remote_url}")
            return ""

        # Handle file path - if it's already relative to git root, use as-is
        # If it's absolute, convert it to be relative to the repository root
        if os.path.isabs(file_path):
            git_root_dir = git_info.get("git_root_dir", ".")

            try:
                # Get absolute paths
                git_repo_abs_path = Path(git_root_dir).resolve()
                file_abs_path = Path(file_path).resolve()

                # Calculate relative path from git repo root to file
                relative_path = file_abs_path.relative_to(git_repo_abs_path)
                file_path = str(relative_path)
                logger.debug(
                    f"Calculated relative path from Git repo root: {file_path}"
                )

            except ValueError as e:
                # If relative_to fails, it means the file is not under the git repo
                logger.warning(
                    f"File {file_path} is not under Git repository {git_root_dir}: {e}"
                )
                return ""
        else:
            # Path is already relative to git root, use as-is
            logger.debug(f"Using git-relative path as-is: {file_path}")

        # Construct the GitHub URL
        github_url = f"{github_base}/blob/{branch}/{file_path}"

        # Add line anchor if specified
        if line_number is not None:
            # For markdown files, we need to add ?plain=1 to make line anchors work
            if file_path.lower().endswith(".md"):
                github_url += f"?plain=1#L{line_number}"
            else:
                github_url += f"#L{line_number}"
            logger.debug(f"Constructed GitHub URL with line anchor: {github_url}")
        else:
            logger.debug(f"Constructed GitHub URL: {github_url}")

        return github_url

    except Exception as e:
        logger.warning(f"Error constructing GitHub URL: {e}")
        return ""


def find_markdown_files(root_dir: str, logger: logging.Logger) -> List[Path]:
    """Find all .md files recursively in the given directory or specific file."""
    root_path = Path(root_dir)
    logger.debug(f"Searching for markdown files in: {root_dir}")

    if not root_path.exists():
        error_msg = f"Path does not exist: {root_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    md_files = []

    if root_path.is_file():
        # If it's a specific file, check if it's a markdown file
        if root_path.suffix.lower() == ".md":
            md_files.append(root_path)
            logger.debug(f"Added single markdown file: {root_path}")
        else:
            logger.warning(f"Specified file {root_path} is not a markdown file")
    else:
        # If it's a directory, find all .md files recursively
        logger.debug(f"Scanning directory recursively: {root_path}")
        for file_path in root_path.rglob("*.md"):
            md_files.append(file_path)

        logger.info(f"Found {len(md_files)} markdown files in {root_dir}")
        logger.debug(f"Markdown files: {[str(f) for f in md_files]}")

    return sorted(md_files)


def extract_markdown_links(
    content: str, file_path: Path, logger: logging.Logger
) -> List[Tuple[str, int, str]]:
    """
    Extract markdown links from file content.

    Args:
        content: File content as string
        file_path: Path to the file being processed
        logger: Logger instance for logging

    Returns:
        List of tuples: (link_text, line_number, link_url)
    """
    links = []
    logger.debug(f"Extracting markdown links from: {file_path}")

    # Regex to match markdown links: [text](url)
    # This captures the link text and URL, excluding image links (![text](url))
    link_pattern = r"(?<!\!)\[([^\]]+)\]\(([^)]+)\)"

    for line_num, line in enumerate(content.split("\n"), 1):
        matches = re.finditer(link_pattern, line)
        for match in matches:
            link_text = match.group(1)
            link_url = match.group(2)

            # Only process links that could be to .md files
            # Skip external URLs, anchors, and non-markdown files
            if (
                not link_url.startswith("http")
                and not link_url.startswith("mailto:")
                and not link_url.startswith("#")
                and not link_url.startswith("ftp://")
                and not link_url.startswith("//")
            ):
                links.append((link_text, line_num, link_url))
                logger.debug(
                    f"Found link at line {line_num}: [{link_text}]({link_url})"
                )

    logger.debug(f"Extracted {len(links)} potential markdown links from {file_path}")
    return links


def resolve_link_path(
    link_url: str,
    source_file: Path,
    logger: logging.Logger,
    git_root_dir: Optional[str] = None,
) -> Path:
    """
    Resolve a relative or absolute link to an absolute file path.

    Args:
        link_url: The link URL from the markdown file
        source_file: The source markdown file path
        logger: Logger instance for logging
        git_root_dir: Git repository root directory for resolving GitHub-style absolute paths

    Returns:
        Resolved absolute path to the target file
    """
    logger.debug(f"Resolving link: {link_url} from source: {source_file}")

    # Remove any anchor/fragment from the URL
    original_link_url = link_url
    link_url = link_url.split("#")[0]

    if not link_url:
        logger.debug(f"Empty link URL, returning source file: {source_file}")
        return source_file

    # Handle GitHub-style absolute paths (starting with /) that should be relative to git repo root
    if link_url.startswith("/") and git_root_dir:
        # Strip the leading '/' and resolve relative to git repository root
        repo_relative_path = link_url[1:]  # Remove leading '/'
        git_root_path = Path(git_root_dir)
        resolved_path = git_root_path / repo_relative_path
        final_path = resolved_path.resolve()
        logger.debug(
            f"GitHub-style absolute path resolved: {original_link_url} -> {final_path}"
        )
        return final_path

    # If it's already an absolute path (and not a GitHub-style path), return as is
    if os.path.isabs(link_url):
        logger.debug(f"Absolute filesystem path detected: {link_url}")
        return Path(link_url)

    # Handle symlinks: resolve relative paths from the target's directory, not the symlink's directory
    if source_file.is_symlink():
        # Get the real path (target) of the symlink
        real_source_file = source_file.resolve()
        source_dir = real_source_file.parent
        logger.debug(f"Source file is a symlink, using target directory: {source_dir}")
    else:
        # Resolve relative path from the source file's directory
        source_dir = source_file.parent

    resolved_path = source_dir / link_url

    # Normalize the path (resolve any .. or . components)
    final_path = resolved_path.resolve()
    logger.debug(f"Resolved {original_link_url} -> {final_path}")

    return final_path


def is_markdown_file(file_path: Path) -> bool:
    """Check if a file is a markdown file."""
    return file_path.suffix.lower() == ".md"


def validate_links(
    md_files: List[Path],
    logger: logging.Logger,
    git_root_dir: Optional[str] = None,
    git_info: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Validate all links in markdown files.

    Args:
        md_files: List of markdown files to validate
        logger: Logger instance for logging
        git_root_dir: Git repository root directory for path normalization
        git_info: Git repository information for generating GitHub URLs

    Returns:
        Dictionary with file paths as keys and lists of broken links as values
    """
    broken_links_report = {}
    total_links_checked = 0
    total_broken_links = 0

    logger.info(f"Starting validation of {len(md_files)} markdown files")

    for md_file in md_files:
        logger.debug(f"Processing file: {md_file}")

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read {len(content)} characters from {md_file}")
        except Exception as e:
            logger.error(f"Could not read {md_file}: {e}")
            continue

        links = extract_markdown_links(content, md_file, logger)
        broken_links = []
        file_links_checked = 0

        for link_text, line_num, link_url in links:
            file_links_checked += 1
            total_links_checked += 1

            try:
                resolved_path = resolve_link_path(
                    link_url, md_file, logger, git_root_dir
                )

                # Only check if the target should be a markdown file
                if is_markdown_file(resolved_path):
                    if not resolved_path.exists():
                        # Generate GitHub URL for the broken link line
                        file_for_github = (
                            path_relative_to_git_root(md_file, git_root_dir, logger)
                            if git_root_dir
                            else str(md_file)
                        )
                        github_url = (
                            construct_github_url(
                                file_for_github, git_info, logger, line_num
                            )
                            if git_info
                            else ""
                        )

                        broken_link_info = {
                            "line": line_num,
                            "link_text": link_text,
                            "link_url": link_url,
                            "resolved_path": str(resolved_path),
                            "github_url": github_url,
                        }
                        broken_links.append(broken_link_info)
                        total_broken_links += 1
                        logger.warning(
                            f"Broken link found in {md_file}:{line_num} - {link_url} -> {resolved_path}"
                        )
                    else:
                        logger.debug(
                            f"Valid link in {md_file}:{line_num} - {link_url} -> {resolved_path}"
                        )
                else:
                    logger.debug(
                        f"Skipping non-markdown link in {md_file}:{line_num} - {link_url}"
                    )

            except Exception as e:
                # If we can't resolve the path, consider it broken
                # Generate GitHub URL for the broken link line
                file_for_github = (
                    path_relative_to_git_root(md_file, git_root_dir, logger)
                    if git_root_dir
                    else str(md_file)
                )
                github_url = (
                    construct_github_url(file_for_github, git_info, logger, line_num)
                    if git_info
                    else ""
                )

                broken_link_info = {
                    "line": line_num,
                    "link_text": link_text,
                    "link_url": link_url,
                    "resolved_path": f"ERROR: {e}",
                    "github_url": github_url,
                }
                broken_links.append(broken_link_info)
                total_broken_links += 1
                logger.error(
                    f"Error resolving link in {md_file}:{line_num} - {link_url}: {e}"
                )

        if broken_links:
            # Use git-relative path for the report key if git_root_dir is available
            report_key = (
                path_relative_to_git_root(md_file, git_root_dir, logger)
                if git_root_dir
                else str(md_file)
            )
            broken_links_report[report_key] = broken_links
            logger.info(
                f"File {md_file}: {len(broken_links)} broken links out of {file_links_checked} total links"
            )
        else:
            logger.debug(f"File {md_file}: All {file_links_checked} links are valid")

    logger.info(
        f"Validation complete: {total_broken_links} broken links found out of {total_links_checked} total links checked"
    )
    return broken_links_report


def find_symbolic_links(root_dir: str, logger: logging.Logger) -> List[Path]:
    """
    Find all symbolic links in the given directory recursively.

    Args:
        root_dir: Root directory to search for symbolic links
        logger: Logger instance for logging

    Returns:
        List of Path objects representing symbolic links
    """
    symlinks = []
    root_path = Path(root_dir).resolve()

    logger.debug(f"Searching for symbolic links in: {root_path}")

    try:
        for item in root_path.rglob("*"):
            if item.is_symlink():
                symlinks.append(item)
                logger.debug(f"Found symbolic link: {item}")
    except (OSError, PermissionError) as e:
        logger.warning(f"Error accessing path during symlink search: {e}")

    logger.info(f"Found {len(symlinks)} symbolic links in {root_dir}")
    return symlinks


def detect_problematic_symlinks(
    symlinks: List[Path], git_root_dir: Optional[str], logger: logging.Logger
) -> Dict[str, List[Dict[str, str]]]:
    """
    Detect problematic symbolic links including broken, circular, and external links.

    Args:
        symlinks: List of symbolic link paths to check
        git_root_dir: Git repository root directory for relative path calculation
        logger: Logger instance for logging

    Returns:
        Dictionary with categories of problematic symlinks
    """
    problematic_symlinks = {
        "broken": [],
        "circular": [],
        "external": [],
        "suspicious": [],
    }

    git_root_path = Path(git_root_dir).resolve() if git_root_dir else None

    for symlink in symlinks:
        try:
            symlink_path = symlink.resolve()
            target_path = symlink.readlink()

            # Get relative path from git root for reporting
            if git_root_path:
                try:
                    relative_symlink = symlink.relative_to(git_root_path)
                except ValueError:
                    relative_symlink = symlink
            else:
                relative_symlink = symlink

            symlink_info = {
                "symlink_path": str(relative_symlink),
                "target_path": str(target_path),
                "absolute_symlink_path": str(symlink),
                "issue": "",
            }

            # Check if symlink is broken (target doesn't exist)
            if not symlink_path.exists():
                symlink_info[
                    "issue"
                ] = f"Broken symlink: target '{target_path}' does not exist"
                problematic_symlinks["broken"].append(symlink_info)
                logger.warning(f"Broken symlink found: {symlink} -> {target_path}")
                continue

            # Check for circular symlinks
            try:
                # Try to resolve the symlink completely
                resolved_path = symlink.resolve(strict=True)
                if resolved_path == symlink:
                    symlink_info["issue"] = "Circular symlink: points to itself"
                    problematic_symlinks["circular"].append(symlink_info)
                    logger.warning(f"Circular symlink found: {symlink}")
                    continue
            except (OSError, RuntimeError) as e:
                if "Too many levels of symbolic links" in str(e):
                    symlink_info[
                        "issue"
                    ] = "Circular symlink: too many levels of symbolic links"
                    problematic_symlinks["circular"].append(symlink_info)
                    logger.warning(f"Circular symlink found: {symlink}")
                    continue

            # Check if symlink points outside the repository
            if git_root_path:
                try:
                    symlink_path.relative_to(git_root_path)
                except ValueError:
                    symlink_info[
                        "issue"
                    ] = f"External symlink: points outside repository to '{symlink_path}'"
                    problematic_symlinks["external"].append(symlink_info)
                    logger.warning(
                        f"External symlink found: {symlink} -> {symlink_path}"
                    )
                    continue

            # Check for suspicious patterns (e.g., very long paths, unusual targets)
            if len(str(target_path)) > 200:
                symlink_info[
                    "issue"
                ] = f"Suspicious symlink: unusually long target path ({len(str(target_path))} characters)"
                problematic_symlinks["suspicious"].append(symlink_info)
                logger.info(f"Suspicious symlink found: {symlink} (long path)")

            # Check if target is in a different directory tree (potential maintenance issue)
            if "../" in str(target_path) and str(target_path).count("../") > 3:
                symlink_info[
                    "issue"
                ] = f"Suspicious symlink: target requires many directory traversals ('{target_path}')"
                problematic_symlinks["suspicious"].append(symlink_info)
                logger.info(f"Suspicious symlink found: {symlink} (many traversals)")

        except (OSError, PermissionError) as e:
            symlink_info = {
                "symlink_path": str(symlink),
                "target_path": "unknown",
                "absolute_symlink_path": str(symlink),
                "issue": f"Error accessing symlink: {e}",
            }
            problematic_symlinks["broken"].append(symlink_info)
            logger.error(f"Error processing symlink {symlink}: {e}")

    # Log summary
    total_issues = sum(len(issues) for issues in problematic_symlinks.values())
    if total_issues > 0:
        logger.warning(f"Found {total_issues} problematic symbolic links:")
        for category, issues in problematic_symlinks.items():
            if issues:
                logger.warning(f"  {category}: {len(issues)}")
    else:
        logger.info("No problematic symbolic links found")

    return problematic_symlinks


def main():
    parser = argparse.ArgumentParser(
        description="Detect broken links in markdown files and problematic symbolic links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all markdown files in the current directory
  python detect_broken_links.py

  # Check markdown files in a specific directory
  python detect_broken_links.py /path/to/docs

  # Check markdown files in multiple directories
  python detect_broken_links.py /path/to/docs1 /path/to/docs2

  # Output to JSON file
  python detect_broken_links.py --format json --output report.json

  # Verbose logging
  python detect_broken_links.py --verbose

  # Log to file
  python detect_broken_links.py --log-file debug.log
        """,
    )

    parser.add_argument(
        "directories",
        nargs="*",
        default=["."],
        help="Directories to scan for markdown files (default: current directory)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for report (default: print to stdout)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including file processing information",
    )

    parser.add_argument(
        "--log-file", type=str, help="Log file path for detailed logging"
    )

    parser.add_argument(
        "--check-symlinks",
        action="store_true",
        help="Also check for problematic symbolic links (broken, circular, external)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger.info("Starting broken links detection")
    logger.info(
        f"Arguments: directories={args.directories}, format={args.format}, output={args.output}, verbose={args.verbose}"
    )

    # Get git repository information for path normalization
    git_info = get_git_info(logger, args.directories[0] if args.directories else ".")
    git_root_dir = git_info.get("git_root_dir") if git_info else None

    all_broken_links = {}
    total_files_processed = 0
    all_processed_files = []

    for directory in args.directories:
        logger.info(f"Processing directory: {directory}")

        try:
            md_files = find_markdown_files(directory, logger)
            total_files_processed += len(md_files)

            # Convert file paths to git-relative paths for the report
            if git_root_dir:
                git_relative_files = [
                    path_relative_to_git_root(f, git_root_dir, logger) for f in md_files
                ]
                all_processed_files.extend(git_relative_files)
            else:
                all_processed_files.extend([str(f) for f in md_files])

            logger.info(f"Found {len(md_files)} markdown files in {directory}")

            broken_links = validate_links(md_files, logger, git_root_dir, git_info)
            all_broken_links.update(broken_links)

        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            continue

    # Check for problematic symbolic links if requested
    all_problematic_symlinks = {}
    if args.check_symlinks:
        logger.info("Checking for problematic symbolic links...")
        for directory in args.directories:
            try:
                symlinks = find_symbolic_links(directory, logger)
                if symlinks:
                    problematic_symlinks = detect_problematic_symlinks(
                        symlinks, git_root_dir, logger
                    )
                    # Only include categories that have issues
                    for category, issues in problematic_symlinks.items():
                        if issues:
                            if category not in all_problematic_symlinks:
                                all_problematic_symlinks[category] = []
                            all_problematic_symlinks[category].extend(issues)
            except Exception as e:
                logger.error(f"Error checking symlinks in directory {directory}: {e}")
                continue

    # Prepare the final report
    total_problematic_symlinks = sum(
        len(issues) for issues in all_problematic_symlinks.values()
    )
    report = {
        "summary": {
            "total_files_processed": total_files_processed,
            "files_with_broken_links": len(all_broken_links),
            "total_broken_links": sum(
                len(links) for links in all_broken_links.values()
            ),
            "total_problematic_symlinks": total_problematic_symlinks,
            "symlink_check_enabled": args.check_symlinks,
        },
        "broken_links": all_broken_links,
        "problematic_symlinks": all_problematic_symlinks,
        "all_processed_files": sorted(all_processed_files),
    }

    logger.info(f"Report summary: {report['summary']}")

    # Generate JSON output
    logger.info("Generating JSON report")
    # Create a cleaned version of the report without resolved_path for JSON output
    cleaned_broken_links = {}
    for file_path, links in report["broken_links"].items():
        cleaned_links = []
        for link in links:
            # Create a copy of the link without resolved_path
            cleaned_link = {k: v for k, v in link.items() if k != "resolved_path"}
            cleaned_links.append(cleaned_link)
        cleaned_broken_links[file_path] = cleaned_links

    cleaned_report = {
        "summary": report["summary"],
        "broken_links": cleaned_broken_links,
        "problematic_symlinks": report["problematic_symlinks"],
        "all_processed_files": report["all_processed_files"],
    }
    output_content = json.dumps(cleaned_report, indent=2, ensure_ascii=False)

    # Output the report
    if args.output:
        try:
            logger.info(f"Writing report to: {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)
            logger.info(f"Report successfully written to: {args.output}")
            print(f"Report written to: {args.output}", file=sys.stderr)
        except Exception as e:
            logger.error(f"Error writing to {args.output}: {e}")
            print(f"Error writing to {args.output}: {e}", file=sys.stderr)
            print(output_content)
    else:
        logger.info("Writing report to stdout")
        print(output_content)

    # Exit with error code if broken links or problematic symlinks were found
    # Note: "suspicious" symlinks are warnings only and don't cause failure
    has_broken_links = bool(all_broken_links)

    # Only count critical symlink issues (broken, circular, external) as errors
    critical_symlink_categories = ["broken", "circular", "external"]
    critical_symlinks = {
        category: issues
        for category, issues in all_problematic_symlinks.items()
        if category in critical_symlink_categories
    }
    has_critical_symlinks = bool(critical_symlinks)
    total_critical_symlinks = sum(len(issues) for issues in critical_symlinks.values())

    # Log suspicious symlinks separately as warnings
    suspicious_symlinks = all_problematic_symlinks.get("suspicious", [])
    if suspicious_symlinks:
        logger.warning(
            f"Found {len(suspicious_symlinks)} suspicious symlinks (warnings only, not causing failure)"
        )

    if has_broken_links or has_critical_symlinks:
        error_msg = []
        if has_broken_links:
            error_msg.append(f"{len(all_broken_links)} files with broken links")
        if has_critical_symlinks:
            error_msg.append(f"{total_critical_symlinks} critical problematic symlinks")

        logger.warning(f"Exiting with error code 1 due to: {', '.join(error_msg)}")
        sys.exit(1)
    else:
        success_msg = "No broken links found"
        if args.check_symlinks:
            success_msg += " and no critical problematic symlinks found"
        logger.info(f"{success_msg} - exiting successfully")


if __name__ == "__main__":
    main()
