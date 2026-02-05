"""
Unit tests for error deduplicator.
"""
import pytest
from opensearch_upload.error_classification.deduplicator import ErrorDeduplicator


class TestErrorDeduplicator:
    """Test error deduplication functionality."""

    def test_normalize_error_text_timestamps(self):
        """Test that timestamps are normalized."""
        deduplicator = ErrorDeduplicator()

        error_with_timestamp = "Error occurred at 2025-01-15 10:30:45"
        normalized = deduplicator.normalize_error_text(error_with_timestamp)

        assert "2025-01-15" not in normalized
        assert "10:30:45" not in normalized
        assert "TIMESTAMP" in normalized

    def test_normalize_error_text_uuids(self):
        """Test that UUIDs are normalized."""
        deduplicator = ErrorDeduplicator()

        error_with_uuid = "Request ID: 550e8400-e29b-41d4-a716-446655440000 failed"
        normalized = deduplicator.normalize_error_text(error_with_uuid)

        assert "550e8400" not in normalized
        assert "UUID" in normalized

    def test_normalize_error_text_paths(self):
        """Test that file paths are normalized."""
        deduplicator = ErrorDeduplicator()

        error_with_path = "Error in /home/user/project/src/module.py"
        normalized = deduplicator.normalize_error_text(error_with_path)

        assert "/home/user" not in normalized
        assert "module.py" in normalized

    def test_normalize_error_text_line_numbers(self):
        """Test that line numbers are normalized."""
        deduplicator = ErrorDeduplicator()

        error_with_line = "Error at line 123 in file.py"
        normalized = deduplicator.normalize_error_text(error_with_line)

        assert "line 123" not in normalized
        assert "line NUM" in normalized

    def test_compute_error_hash_consistency(self):
        """Test that same error produces same hash."""
        deduplicator = ErrorDeduplicator()

        error1 = "Error at 2025-01-15 10:30:45: Connection refused"
        error2 = "Error at 2026-02-03 14:20:10: Connection refused"

        hash1 = deduplicator.compute_error_hash(error1)
        hash2 = deduplicator.compute_error_hash(error2)

        # Should be same because timestamps are normalized
        assert hash1 == hash2

    def test_compute_error_hash_different_errors(self):
        """Test that different errors produce different hashes."""
        deduplicator = ErrorDeduplicator()

        error1 = "Connection refused"
        error2 = "Connection timeout"

        hash1 = deduplicator.compute_error_hash(error1)
        hash2 = deduplicator.compute_error_hash(error2)

        assert hash1 != hash2

    def test_normalize_memory_addresses(self):
        """Test that memory addresses are normalized."""
        deduplicator = ErrorDeduplicator()

        error = "Segfault at address 0x7f8b9c0a1234"
        normalized = deduplicator.normalize_error_text(error)

        assert "0x7f8b9c0a1234" not in normalized
        assert "MEMADDR" in normalized

    def test_normalize_pids(self):
        """Test that process IDs are normalized."""
        deduplicator = ErrorDeduplicator()

        error = "Process PID 12345 crashed"
        normalized = deduplicator.normalize_error_text(error)

        assert "12345" not in normalized
        assert "PID NUM" in normalized.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
