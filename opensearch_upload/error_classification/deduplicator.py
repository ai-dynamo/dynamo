"""
Error deduplication via content-based hashing.
"""
import hashlib
import re
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta


class ErrorDeduplicator:
    """Deduplicate errors using content-based hashing."""

    def __init__(self, opensearch_client: Any = None, config: Any = None):
        """
        Initialize deduplicator.

        Args:
            opensearch_client: OpenSearch client for cache lookups
            config: Configuration object
        """
        self.opensearch_client = opensearch_client
        self.config = config

    def normalize_error_text(self, error_text: str) -> str:
        """
        Normalize error text by removing variable content.

        This removes timestamps, UUIDs, file paths, line numbers, and other
        variable content to enable deduplication of similar errors.

        Args:
            error_text: Raw error message

        Returns:
            Normalized error text
        """
        text = error_text

        # Remove timestamps (various formats)
        # 2025-01-15 10:30:45
        text = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(\.\d+)?', 'TIMESTAMP', text)
        # Jan 15 10:30:45
        text = re.sub(r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', 'TIMESTAMP', text)
        # Unix timestamps
        text = re.sub(r'\b\d{10,13}\b', 'TIMESTAMP', text)

        # Remove UUIDs and hex IDs
        # UUID format: 550e8400-e29b-41d4-a716-446655440000
        text = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', 'UUID', text, flags=re.IGNORECASE)
        # Short hex IDs: abc123def
        text = re.sub(r'\b[0-9a-f]{6,40}\b', 'HEXID', text, flags=re.IGNORECASE)

        # Remove absolute file paths, keep relative paths and filenames
        # /home/user/path/to/file.py -> file.py
        text = re.sub(r'/[\w\-./]+/([^/\s]+\.\w+)', r'\1', text)
        # Windows paths: C:\Users\path\to\file.py -> file.py
        text = re.sub(r'[A-Z]:\\[\w\-\\]+\\([^\\]+\.\w+)', r'\1', text)

        # Remove specific line and column numbers
        # "line 123" -> "line NUM"
        text = re.sub(r'\bline\s+\d+\b', 'line NUM', text, flags=re.IGNORECASE)
        # "column 45" -> "column NUM"
        text = re.sub(r'\bcolumn\s+\d+\b', 'column NUM', text, flags=re.IGNORECASE)
        # file.py:123 -> file.py:NUM
        text = re.sub(r'(\.\w+):(\d+)', r'\1:NUM', text)

        # Remove memory addresses
        # 0x7f8b9c0a1234 -> MEMADDR
        text = re.sub(r'\b0x[0-9a-f]+\b', 'MEMADDR', text, flags=re.IGNORECASE)

        # Remove PID/thread IDs
        # [PID 12345] -> [PID NUM]
        text = re.sub(r'\b(pid|tid|thread)\s+\d+\b', r'\1 NUM', text, flags=re.IGNORECASE)

        # Remove port numbers (but keep common ones like 80, 443)
        text = re.sub(r':(\d{4,5})\b', ':PORT', text)

        # Remove durations/timings
        # "took 123.45s" -> "took NUMs"
        text = re.sub(r'\b\d+\.?\d*\s*(ms|milliseconds?|s|seconds?|m|minutes?|h|hours?)\b', 'NUM TIME_UNIT', text, flags=re.IGNORECASE)

        # Remove sizes/counts (but be careful not to remove version numbers)
        # "12345 bytes" -> "NUM bytes"
        text = re.sub(r'\b\d+\s*(bytes?|kb|mb|gb|tb)\b', 'NUM SIZE_UNIT', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def compute_error_hash(self, error_text: str) -> str:
        """
        Compute content hash of normalized error text.

        Args:
            error_text: Raw error message

        Returns:
            16-character hex hash for deduplication
        """
        normalized = self.normalize_error_text(error_text)
        full_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        # Return first 16 chars for shorter IDs
        return full_hash[:16]

    def find_similar_classification(
        self,
        error_hash: str,
        index_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find existing classification for similar error.

        Args:
            error_hash: Hash of the error text
            index_name: OpenSearch index to search (from config if not provided)

        Returns:
            Existing classification dict or None
        """
        if not self.opensearch_client or not self.config:
            return None

        if not index_name:
            index_name = self.config.error_classification_index

        if not index_name:
            return None

        try:
            # Calculate time threshold for cache
            cache_ttl_hours = self.config.classification_cache_ttl_hours
            min_timestamp = datetime.now(timezone.utc) - timedelta(hours=cache_ttl_hours)

            # Query for existing classification with same hash
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"s_error_hash": error_hash}},
                            {"range": {"ts_classified_at": {"gte": min_timestamp.isoformat()}}}
                        ]
                    }
                },
                "sort": [{"ts_classified_at": {"order": "desc"}}],
                "size": 1
            }

            response = self.opensearch_client.search(
                index=index_name,
                body=query
            )

            hits = response.get('hits', {}).get('hits', [])

            if hits:
                classification = hits[0]['_source']
                confidence = classification.get('f_confidence_score', 0)

                # Only return if confidence is high enough
                if confidence >= self.config.min_confidence_for_reuse:
                    return classification

            return None

        except Exception as e:
            print(f"  ⚠️  Error querying for similar classification: {e}")
            return None

    def increment_occurrence_count(
        self,
        error_hash: str,
        index_name: str = None
    ) -> bool:
        """
        Increment occurrence count for an existing classification.

        Args:
            error_hash: Hash of the error
            index_name: OpenSearch index

        Returns:
            True if successfully incremented, False otherwise
        """
        if not self.opensearch_client or not self.config:
            return False

        if not index_name:
            index_name = self.config.error_classification_index

        if not index_name:
            return False

        try:
            # Find the document
            query = {
                "query": {
                    "term": {"s_error_hash": error_hash}
                },
                "size": 1
            }

            response = self.opensearch_client.search(
                index=index_name,
                body=query
            )

            hits = response.get('hits', {}).get('hits', [])

            if not hits:
                return False

            doc_id = hits[0]['_id']
            current_count = hits[0]['_source'].get('l_occurrence_count', 1)

            # Update the document
            update_body = {
                "doc": {
                    "l_occurrence_count": current_count + 1,
                    "ts_last_seen": datetime.now(timezone.utc).isoformat()
                }
            }

            self.opensearch_client.update(
                index=index_name,
                id=doc_id,
                body=update_body
            )

            return True

        except Exception as e:
            print(f"  ⚠️  Error incrementing occurrence count: {e}")
            return False
