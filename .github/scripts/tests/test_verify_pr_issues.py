# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic verifier tests; run with PYTHONPATH=.github/scripts unittest discovery."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import verify_pr_issues as verifier


class VerifyPRIssuesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.summary = Path(self.temp.name) / "summary.md"

    def run_verifier(self, body=None, title="", key="test-key"):
        logs = io.StringIO()
        with redirect_stdout(logs):
            code = verifier.verify(
                {"title": title, "body": body},
                "ai-dynamo/dynamo",
                "github-key",
                key,
                self.summary,
            )
        return code, logs.getvalue(), self.summary.read_text(encoding="utf-8")

    def test_reference_formats_and_deduplication(self):
        refs = verifier.references(
            """Closes #123, GH-123,
            https://github.com/ai-dynamo/dynamo/issues/123#issuecomment-789
            other/repo#456 https://github.com/other/repo/issues/456
            DYN-321 https://linear.app/team/issue/DYN-321/a-title""",
            "ai-dynamo/dynamo",
        )
        self.assertEqual(
            [ref.ref for ref in refs],
            ["ai-dynamo/dynamo#123", "other/repo#456", "DYN-321"],
        )

    def test_ignores_template_comments_and_unrelated_urls(self):
        self.assertEqual(
            verifier.references(
                """<!-- Closes #8480 --> Closes #XXXX
                https://github.com/ai-dynamo/dynamo/pull/123#456
                https://example.com/path#789 <!-- DYN-321""",
                "ai-dynamo/dynamo",
            ),
            [],
        )

    @patch.object(verifier, "request_json")
    def test_missing_reference_fails_without_requests(self, request_json):
        code, logs, summary = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertIn("::error::Failed: no issue listed", logs)
        self.assertIn("Failed: no issue listed", summary)
        request_json.assert_not_called()

    @patch.object(
        verifier,
        "request_json",
        return_value={"title": "GitHub issue", "state": "closed"},
    )
    def test_closed_issue_in_title_succeeds(self, request_json):
        code, logs, summary = self.run_verifier(title="fix: bug #123")
        self.assertEqual(code, 0)
        self.assertIn("Found: ai-dynamo/dynamo#123 — GitHub issue", logs)
        self.assertIn("GitHub issue", summary)
        request = request_json.call_args.args[0]
        self.assertEqual(
            request.full_url, "https://api.github.com/repos/ai-dynamo/dynamo/issues/123"
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer github-key")

    @patch.object(
        verifier,
        "request_json",
        return_value={
            "data": {"issue": {"identifier": "DYN-321", "title": "Linear ticket"}}
        },
    )
    def test_linear_query_uses_variable_and_reports_title(self, request_json):
        code, logs, summary = self.run_verifier("DYN-321")
        self.assertEqual(code, 0)
        self.assertIn("Found: DYN-321 — Linear ticket", logs)
        self.assertIn("Linear ticket", summary)
        request = request_json.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.linear.app/graphql")
        self.assertEqual(request.get_header("Authorization"), "test-key")
        self.assertEqual(json.loads(request.data)["variables"], {"id": "DYN-321"})
        self.assertEqual(request.get_method(), "POST")

    @patch.object(verifier, "request_json")
    def test_both_trackers_and_escaped_summary(self, request_json):
        request_json.side_effect = [
            {"title": '<script>alert("x")</script> & | title\n::error::injection'},
            {"data": {"issue": {"identifier": "DYN-321", "title": "Linear ticket"}}},
        ]
        code, logs, summary = self.run_verifier("#123 DYN-321")
        self.assertEqual(code, 0)
        self.assertIn("2 of 2 referenced issues verified", summary)
        self.assertIn("&lt;script&gt;", summary)
        self.assertNotIn("<script>", summary)
        self.assertNotIn("\n::error::injection", logs)

    @patch.object(verifier, "request_json")
    def test_github_http_errors_and_pr_rejection(self, request_json):
        for status in (404, 403, 500):
            with self.subTest(status=status):
                request_json.side_effect = HTTPError(
                    "https://api.github.com", status, "secret error details", {}, None
                )
                code, logs, summary = self.run_verifier("#123")
                expected = (
                    "does not exist or is not accessible"
                    if status == 404
                    else f"GitHub API HTTP {status}"
                )
                self.assertEqual(code, 1)
                self.assertIn(expected, logs)
                self.assertIn(expected, summary)
                self.assertNotIn("secret error details", logs)
        request_json.side_effect = None
        request_json.return_value = {"title": "PR", "pull_request": {}}
        code, logs, _ = self.run_verifier("#123")
        self.assertEqual(code, 1)
        self.assertIn("is a pull request, not an issue", logs)

    @patch.object(verifier, "request_json")
    def test_linear_errors_are_distinct(self, request_json):
        cases = [
            ({"data": {"issue": None}}, "does not exist or is not accessible"),
            (
                {"errors": [{"message": "Entity not found"}]},
                "does not exist or is not accessible",
            ),
            (
                {
                    "errors": [{"message": "secret response detail"}],
                    "data": {"issue": {"title": "partial data"}},
                },
                "Linear GraphQL error",
            ),
            ({"data": {}}, "unexpected Linear API response"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                request_json.return_value = payload
                code, logs, summary = self.run_verifier("DYN-321")
                self.assertEqual(code, 1)
                self.assertIn(expected, logs)
                self.assertIn(expected, summary)
                self.assertNotIn("secret response detail", logs)

    @patch.object(verifier, "request_json")
    def test_linear_missing_key_and_http_errors(self, request_json):
        code, logs, _ = self.run_verifier("DYN-321", key="")
        self.assertEqual(code, 1)
        self.assertIn("LINEAR_ACCESS_KEY secret is missing", logs)
        request_json.assert_not_called()
        for status in (401, 429):
            with self.subTest(status=status):
                request_json.side_effect = HTTPError(
                    "https://api.linear.app/graphql", status, "private", {}, None
                )
                code, logs, _ = self.run_verifier("DYN-321")
                self.assertEqual(code, 1)
                self.assertIn(f"Linear API HTTP {status}", logs)

    @patch.object(verifier, "request_json")
    def test_network_and_json_failures(self, request_json):
        for error in (
            URLError("secret request headers"),
            TimeoutError(),
            json.JSONDecodeError("private", "", 0),
        ):
            for body in ("#123", "DYN-321"):
                with self.subTest(error=type(error).__name__, body=body):
                    request_json.side_effect = error
                    code, logs, summary = self.run_verifier(body)
                    self.assertEqual(code, 1)
                    self.assertIn("request failed or timed out", logs)
                    self.assertIn("request failed or timed out", summary)
                    self.assertNotIn("secret request headers", logs)

    @patch.object(verifier, "request_json")
    def test_valid_reference_does_not_hide_invalid_one(self, request_json):
        request_json.side_effect = [
            {"title": "GitHub issue"},
            {"data": {"issue": None}},
        ]
        code, logs, summary = self.run_verifier("#123 DYN-321")
        self.assertEqual(code, 1)
        self.assertIn("Failed: issue listed DYN-321", logs)
        self.assertIn("GitHub issue", summary)
        self.assertIn("1 of 2 referenced issues verified", summary)

    @patch.object(verifier, "request_json")
    def test_request_limit(self, request_json):
        code, logs, _ = self.run_verifier(" ".join(f"#{i + 1}" for i in range(51)))
        self.assertEqual(code, 1)
        self.assertIn("more than 50", logs)
        request_json.assert_not_called()

    def test_annotation_escapes_workflow_commands(self):
        logs = io.StringIO()
        with redirect_stdout(logs):
            verifier.annotation("bad % input\r\n::warning::injection")
        self.assertEqual(
            logs.getvalue(), "::error::bad %25 input%0D%0A::warning::injection\n"
        )

    @patch.object(verifier, "build_opener")
    def test_request_timeout_and_no_redirect(self, build_opener):
        response = build_opener.return_value.open.return_value.__enter__.return_value
        response.read.return_value = b'{"title": "Issue"}'
        request = Request("https://api.github.com")
        self.assertEqual(verifier.request_json(request), {"title": "Issue"})
        build_opener.return_value.open.assert_called_once_with(request, timeout=15)
        redirect = build_opener.call_args.args[0]
        self.assertIsNone(
            redirect.redirect_request(
                request, None, 302, "Found", {}, "https://example.com"
            )
        )

    @patch.object(verifier, "request_json", return_value={"title": "Issue"})
    def test_main_reads_event_and_runner_environment(self, request_json):
        event = Path(self.temp.name) / "event.json"
        event.write_text(
            json.dumps({"pull_request": {"title": "fix #123", "body": None}}),
            encoding="utf-8",
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "GITHUB_EVENT_PATH": str(event),
                    "GITHUB_REPOSITORY": "ai-dynamo/dynamo",
                    "GITHUB_STEP_SUMMARY": str(self.summary),
                    "GITHUB_TOKEN": "github-key",
                },
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(verifier.main(), 0)
        self.assertIn("Issue", self.summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
