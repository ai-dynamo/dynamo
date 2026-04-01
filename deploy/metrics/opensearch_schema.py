#!/usr/bin/env python3
"""
OpenSearch Schema Definitions for GitHub Metrics

This file contains all field name constants used across different OpenSearch indices.
Modify this file to update field names across all metrics uploaders.

Field naming conventions:
- s_* : String fields
- l_* : Long (integer) fields
- f_* : Float fields
- b_* : Boolean fields
- ts_*: Timestamp fields
"""

# ============================================================================
# Test Result Fields
# ============================================================================

TEST_FIELDS = {
    # Test identification
    "test_name": "s_test_name",
    "test_classname": "s_test_classname",
    "test_duration_ms": "l_test_duration_ms",
    "test_status": "s_test_status",
    "error_message": "s_error_message",
}

# ============================================================================
# Build Metrics Fields
# ============================================================================

# Container-level fields
CONTAINER_FIELDS = {
    "build_duration_sec": "l_build_duration_sec",
    "build_start_time": "ts_build_start_time",
    "build_end_time": "ts_build_end_time",
    "build_framework": "s_build_framework",
    "build_target": "s_build_target",
    "build_platform": "s_build_platform",
    "build_size_bytes": "l_build_size_bytes",
    "total_size_transferred": "l_total_size_transferred_bytes",
    "total_steps": "l_total_steps",
    "cached_steps": "l_cached_steps",
    "built_steps": "l_built_steps",
    "cache_hit_rate": "f_cache_hit_rate",
}

# Stage-level fields
STAGE_FIELDS = {
    "stage_name": "s_stage_name",
    "stage_total_steps": "l_stage_total_steps",
    "stage_cached_steps": "l_stage_cached_steps",
    "stage_built_steps": "l_stage_built_steps",
    "stage_duration_sec": "f_stage_duration_sec",
    "stage_cache_hit_rate": "f_stage_cache_hit_rate",
}

# Layer-level fields
LAYER_FIELDS = {
    "layer_step_number": "l_step_number",
    "layer_step_name": "s_step_name",
    "layer_command": "s_command",
    "layer_status": "s_layer_status",
    "layer_cached": "b_cached",
    "layer_duration_sec": "f_duration_sec",
    "layer_size_transferred": "l_size_transferred",
    "layer_stage": "s_stage",
}

# ============================================================================
# Workflow Metrics Fields
# ============================================================================

WORKFLOW_FIELDS = {
    # Workflow identification
    "workflow_id": "s_workflow_id",
    "workflow_name": "s_workflow_name",

    # Status fields
    "status": "s_status",
    "status_number": "l_status_number",

    # Timing fields
    "queue_time_sec": "l_queue_time_sec",
    "duration_sec": "l_duration_sec",

    # Retry fields
    "run_attempt": "l_run_attempt",
    "retry_count": "l_retry_count",

    # Annotation fields
    "annotation_count": "l_annotation_count",
    "annotation_failure_count": "l_annotation_failure_count",
    "annotation_warning_count": "l_annotation_warning_count",
    "annotation_notice_count": "l_annotation_notice_count",
    "annotation_messages": "s_annotation_messages",
}

JOB_FIELDS = {
    # Job identification
    "job_id": "s_job_id",
    "job_name": "s_job_name",

    # Runner fields
    "runner_id": "s_runner_id",
    "runner_name": "s_runner_name",
    "runner_prefix": "s_runner_prefix",

    # Status fields
    "status": "s_status",
    "status_number": "l_status_number",

    # Timing fields
    "queue_time_sec": "l_queue_time_sec",
    "duration_sec": "l_duration_sec",

    # Retry fields
    "run_attempt": "l_run_attempt",
    "retry_count": "l_retry_count",

    # Annotation fields
    "annotation_count": "l_annotation_count",
    "annotation_failure_count": "l_annotation_failure_count",
    "annotation_warning_count": "l_annotation_warning_count",
    "annotation_notice_count": "l_annotation_notice_count",
    "annotation_messages": "s_annotation_messages",
}

STEP_FIELDS = {
    # Step identification
    "step_id": "s_step_id",
    "step_run_id": "s_step_run_id",
    "step_name": "s_step_name",
    "step_number": "l_step_number",
    "command": "s_command",

    # Status fields
    "status": "s_status",
    "status_number": "l_status_number",

    # Timing fields
    "duration_sec": "l_duration_sec",

    # Retry fields
    "run_attempt": "l_run_attempt",
    "retry_count": "l_retry_count",
}

# ============================================================================
# GitHub Stats Fields
# ============================================================================

PR_FIELDS = {
    "pr_number": "l_pr_number",
    "author": "s_author",
    "author_association": "s_author_association",
    "is_external": "b_is_external",
    "is_first_time_contributor": "b_is_first_time_contributor",
    "state": "s_state",
    "merged": "b_merged",
    "draft": "b_draft",
    "merged_by": "s_merged_by",
    "created_at": "ts_created_at",
    "merged_at": "ts_merged_at",
    "time_to_merge_hours": "l_time_to_merge_hours",
    "base_branch": "s_base_branch",
}

REVIEW_FIELDS = {
    "reviewer": "s_reviewer",
    "state": "s_state",  # APPROVED, CHANGES_REQUESTED, COMMENTED
}

COMMIT_FIELDS = {
    "sha": "s_sha",
    "author": "s_author",
    "committer": "s_committer",
    "branch": "s_branch",
    "committed_at": "ts_committed_at",
}

ISSUE_FIELDS = {
    "issue_number": "l_issue_number",
    "author": "s_author",
    "author_association": "s_author_association",
    "is_external": "b_is_external",
    "state": "s_state",
    "comments_count": "l_comments_count",
    "labels": "s_labels",
    "assignee": "s_assignee",
    "closed_by": "s_closed_by",
    "created_at": "ts_created_at",
    "closed_at": "ts_closed_at",
    "time_to_close_hours": "l_time_to_close_hours",
}

# ============================================================================
# Common Context Fields (shared across all metrics)
# ============================================================================

COMMON_FIELDS = {
    # Document ID
    "id": "_id",

    # Repository context
    "repo": "s_repo",
    "user_alias": "s_user_alias",
    "commit_sha": "s_commit_sha",
    "branch": "s_branch",
    "pr_id": "s_pr_id",

    # Workflow context
    "workflow_id": "s_workflow_id",
    "workflow_name": "s_workflow_name",
    "github_event": "s_github_event",
    "run_id": "s_run_id",

    # Job context
    "job_id": "s_job_id",
    "job_name": "s_job_name",
    "runner_name": "s_runner_name",

    # Step context
    "step_id": "s_step_id",

    # Test/Build context
    "status": "s_status",
    "framework": "s_framework",
    "test_type": "s_test_type",
    "arch": "s_arch",

    # Timestamp
    "timestamp": "@timestamp",
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_field(category: str, field_name: str) -> str:
    """
    Get a field name from a category

    Args:
        category: One of 'test', 'container', 'stage', 'layer', 'workflow',
                 'job', 'step', 'pr', 'review', 'commit', 'issue', 'common'
        field_name: The field name (without prefix)

    Returns:
        The full OpenSearch field name (with prefix)

    Example:
        >>> get_field('common', 'job_id')
        's_job_id'
    """
    categories = {
        'test': TEST_FIELDS,
        'container': CONTAINER_FIELDS,
        'stage': STAGE_FIELDS,
        'layer': LAYER_FIELDS,
        'workflow': WORKFLOW_FIELDS,
        'job': JOB_FIELDS,
        'step': STEP_FIELDS,
        'pr': PR_FIELDS,
        'review': REVIEW_FIELDS,
        'commit': COMMIT_FIELDS,
        'issue': ISSUE_FIELDS,
        'common': COMMON_FIELDS,
    }

    if category not in categories:
        raise ValueError(f"Unknown category: {category}")

    if field_name not in categories[category]:
        raise ValueError(f"Unknown field '{field_name}' in category '{category}'")

    return categories[category][field_name]


def get_all_fields(category: str) -> dict:
    """
    Get all fields for a category

    Args:
        category: One of 'test', 'container', 'stage', 'layer', etc.

    Returns:
        Dictionary mapping field names to OpenSearch field names
    """
    categories = {
        'test': TEST_FIELDS,
        'container': CONTAINER_FIELDS,
        'stage': STAGE_FIELDS,
        'layer': LAYER_FIELDS,
        'workflow': WORKFLOW_FIELDS,
        'job': JOB_FIELDS,
        'step': STEP_FIELDS,
        'pr': PR_FIELDS,
        'review': REVIEW_FIELDS,
        'commit': COMMIT_FIELDS,
        'issue': ISSUE_FIELDS,
        'common': COMMON_FIELDS,
    }

    if category not in categories:
        raise ValueError(f"Unknown category: {category}")

    return categories[category].copy()


# ============================================================================
# Constants for backward compatibility
# ============================================================================

# Export all field constants for direct import
# This maintains backward compatibility with existing code

# Test fields
FIELD_TEST_NAME = TEST_FIELDS["test_name"]
FIELD_TEST_CLASSNAME = TEST_FIELDS["test_classname"]
FIELD_TEST_DURATION = TEST_FIELDS["test_duration_ms"]
FIELD_TEST_STATUS = TEST_FIELDS["test_status"]
FIELD_ERROR_MESSAGE = TEST_FIELDS["error_message"]

# Container fields
FIELD_BUILD_DURATION_SEC = CONTAINER_FIELDS["build_duration_sec"]
FIELD_BUILD_START_TIME = CONTAINER_FIELDS["build_start_time"]
FIELD_BUILD_END_TIME = CONTAINER_FIELDS["build_end_time"]
FIELD_BUILD_FRAMEWORK = CONTAINER_FIELDS["build_framework"]
FIELD_BUILD_TARGET = CONTAINER_FIELDS["build_target"]
FIELD_BUILD_PLATFORM = CONTAINER_FIELDS["build_platform"]
FIELD_BUILD_SIZE_BYTES = CONTAINER_FIELDS["build_size_bytes"]
FIELD_TOTAL_SIZE_TRANSFERRED = CONTAINER_FIELDS["total_size_transferred"]
FIELD_TOTAL_STEPS = CONTAINER_FIELDS["total_steps"]
FIELD_CACHED_STEPS = CONTAINER_FIELDS["cached_steps"]
FIELD_BUILT_STEPS = CONTAINER_FIELDS["built_steps"]
FIELD_CACHE_HIT_RATE = CONTAINER_FIELDS["cache_hit_rate"]

# Stage fields
FIELD_STAGE_NAME = STAGE_FIELDS["stage_name"]
FIELD_STAGE_TOTAL_STEPS = STAGE_FIELDS["stage_total_steps"]
FIELD_STAGE_CACHED_STEPS = STAGE_FIELDS["stage_cached_steps"]
FIELD_STAGE_BUILT_STEPS = STAGE_FIELDS["stage_built_steps"]
FIELD_STAGE_DURATION_SEC = STAGE_FIELDS["stage_duration_sec"]
FIELD_STAGE_CACHE_HIT_RATE = STAGE_FIELDS["stage_cache_hit_rate"]

# Layer fields
FIELD_LAYER_STEP_NUMBER = LAYER_FIELDS["layer_step_number"]
FIELD_LAYER_STEP_NAME = LAYER_FIELDS["layer_step_name"]
FIELD_LAYER_COMMAND = LAYER_FIELDS["layer_command"]
FIELD_LAYER_STATUS = LAYER_FIELDS["layer_status"]
FIELD_LAYER_CACHED = LAYER_FIELDS["layer_cached"]
FIELD_LAYER_DURATION_SEC = LAYER_FIELDS["layer_duration_sec"]
FIELD_LAYER_SIZE_TRANSFERRED = LAYER_FIELDS["layer_size_transferred"]
FIELD_LAYER_STAGE = LAYER_FIELDS["layer_stage"]

# Workflow/Job/Step annotation fields
FIELD_ANNOTATION_COUNT = WORKFLOW_FIELDS["annotation_count"]
FIELD_ANNOTATION_FAILURE_COUNT = WORKFLOW_FIELDS["annotation_failure_count"]
FIELD_ANNOTATION_WARNING_COUNT = WORKFLOW_FIELDS["annotation_warning_count"]
FIELD_ANNOTATION_NOTICE_COUNT = WORKFLOW_FIELDS["annotation_notice_count"]
FIELD_ANNOTATION_MESSAGES = WORKFLOW_FIELDS["annotation_messages"]
FIELD_RUNNER_PREFIX = JOB_FIELDS["runner_prefix"]
FIELD_RUN_ATTEMPT = WORKFLOW_FIELDS["run_attempt"]
FIELD_RETRY_COUNT = WORKFLOW_FIELDS["retry_count"]

# Common context fields
FIELD_ID = COMMON_FIELDS["id"]
FIELD_JOB_NAME = COMMON_FIELDS["job_name"]
FIELD_JOB_ID = COMMON_FIELDS["job_id"]
FIELD_STEP_ID = COMMON_FIELDS["step_id"]
FIELD_STATUS = COMMON_FIELDS["status"]
FIELD_FRAMEWORK = COMMON_FIELDS["framework"]
FIELD_TEST_TYPE = COMMON_FIELDS["test_type"]
FIELD_ARCH = COMMON_FIELDS["arch"]
FIELD_USER_ALIAS = COMMON_FIELDS["user_alias"]
FIELD_REPO = COMMON_FIELDS["repo"]
FIELD_WORKFLOW_NAME = COMMON_FIELDS["workflow_name"]
FIELD_GITHUB_EVENT = COMMON_FIELDS["github_event"]
FIELD_BRANCH = COMMON_FIELDS["branch"]
FIELD_WORKFLOW_ID = COMMON_FIELDS["workflow_id"]
FIELD_COMMIT_SHA = COMMON_FIELDS["commit_sha"]
FIELD_PR_ID = COMMON_FIELDS["pr_id"]
FIELD_RUN_ID = COMMON_FIELDS["run_id"]
FIELD_RUNNER_NAME = COMMON_FIELDS["runner_name"]
