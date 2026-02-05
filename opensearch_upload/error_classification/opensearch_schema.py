"""
OpenSearch index schema for error classifications.
"""
from typing import Dict, Any


ERROR_CLASSIFICATIONS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            # Identity & References
            "_id": {"type": "keyword"},
            "s_error_id": {"type": "keyword"},
            "s_error_hash": {"type": "keyword"},  # For deduplication

            # Source References (link to existing indexes)
            "s_workflow_id": {"type": "keyword"},
            "s_job_id": {"type": "keyword"},
            "s_step_id": {"type": "keyword"},
            "s_test_name": {"type": "keyword"},

            # Error Source
            "s_error_source": {"type": "keyword"},  # pytest|buildkit|rust_test|github_annotation
            "s_framework": {"type": "keyword"},  # vllm|sglang|trtllm|rust

            # Common Context (inherited pattern)
            "s_user_alias": {"type": "keyword"},
            "s_repo": {"type": "keyword"},
            "s_workflow_name": {"type": "keyword"},
            "s_branch": {"type": "keyword"},
            "s_pr_id": {"type": "keyword"},
            "s_commit_sha": {"type": "keyword"},

            # Error Content
            "s_error_snippet": {"type": "text"},  # First 500 chars
            "s_error_full_text": {"type": "text"},  # Up to 10KB

            # AI Classification Results
            "s_primary_category": {"type": "keyword"},
            "s_subcategory": {"type": "keyword"},  # For future Phase 2
            "f_confidence_score": {"type": "float"},
            "s_root_cause_summary": {"type": "text"},

            # Metadata
            "s_classification_method": {"type": "keyword"},  # realtime|batch
            "s_model_version": {"type": "keyword"},
            "ts_classified_at": {"type": "date"},
            "b_is_duplicate": {"type": "boolean"},

            # Tracking
            "l_occurrence_count": {"type": "long"},
            "ts_first_seen": {"type": "date"},
            "ts_last_seen": {"type": "date"},

            # API Usage
            "l_prompt_tokens": {"type": "long"},
            "l_completion_tokens": {"type": "long"},
            "l_cached_tokens": {"type": "long"},

            "@timestamp": {"type": "date"},
        }
    },
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "refresh_interval": "5s"
        }
    }
}


def create_index_if_not_exists(opensearch_client: Any, index_name: str) -> bool:
    """
    Create the error_classifications index if it doesn't exist.

    Args:
        opensearch_client: OpenSearch client instance
        index_name: Name of the index to create

    Returns:
        True if index was created, False if it already existed
    """
    try:
        if opensearch_client.indices.exists(index=index_name):
            print(f"✓ Index {index_name} already exists")
            return False

        opensearch_client.indices.create(
            index=index_name,
            body=ERROR_CLASSIFICATIONS_INDEX_MAPPING
        )
        print(f"✓ Created index {index_name}")
        return True

    except Exception as e:
        print(f"✗ Error creating index {index_name}: {e}")
        raise


def get_index_mapping() -> Dict[str, Any]:
    """Get the error classifications index mapping."""
    return ERROR_CLASSIFICATIONS_INDEX_MAPPING
