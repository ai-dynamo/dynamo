# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "run_media_test", Path(__file__).resolve().parents[1] / "run_image_test.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class MediaResponseTests(unittest.TestCase):
    def test_image_requires_left_to_right_color_order(self):
        response = {"prompt_tokens": 82, "completion_tokens": 20}
        runner.validate_response("image", "- Red\n- Blue", response, 19)
        for caption in ("Blue, red.", "Red.", "A green square."):
            with self.subTest(caption=caption), self.assertRaises(AssertionError):
                runner.validate_response("image", caption, response, 19)

    def test_video_requires_the_fixture_subject(self):
        response = {"prompt_tokens": 1024, "completion_tokens": 30}
        for caption in ("A rotating triangle.", "A triangular shape moves."):
            runner.validate_response("video", caption, response, 21)
        with self.assertRaisesRegex(AssertionError, "triangle"):
            runner.validate_response("video", "A red square moves.", response, 21)

    def test_video_requires_expanded_prompt_and_completion_usage(self):
        for response in (
            {"prompt_tokens": 21, "completion_tokens": 30},
            {"prompt_tokens": 1024, "completion_tokens": 0},
        ):
            with (
                self.subTest(response=response),
                self.assertRaisesRegex(AssertionError, "token usage"),
            ):
                runner.validate_response("video", "A rotating triangle.", response, 21)

    def test_audio_requires_expected_transcription_and_completion(self):
        response = {"prompt_tokens": 0, "completion_tokens": 17}
        runner.validate_response("audio", "Thoughts affected Hester.", response, 0)
        with self.assertRaisesRegex(AssertionError, "transcription"):
            runner.validate_response("audio", "An unrelated sentence.", response, 0)
        with self.assertRaisesRegex(AssertionError, "token usage"):
            runner.validate_response(
                "audio",
                "Thoughts affected Hester.",
                {"prompt_tokens": 0, "completion_tokens": 0},
                0,
            )

    def test_audio_selects_whisper_without_changing_image_identity(self):
        config = {
            "model": "vision",
            "model_revision": "vision-revision",
            "image": "official@sha256:digest",
            "upstream_commit": "commit",
            "audio": {"model": "whisper", "model_revision": "audio-revision"},
        }
        selected = runner.model_config(config, "audio")
        self.assertEqual(selected["model"], "whisper")
        self.assertEqual(selected["model_revision"], "audio-revision")
        self.assertEqual(selected["image"], config["image"])
        self.assertEqual(config["model"], "vision")

    def test_changed_audio_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(runner.urllib.request, "urlopen") as fetch,
                self.assertRaisesRegex(ValueError, "Audio fixture"),
            ):
                fetch.return_value.__enter__.return_value.read.return_value = (
                    b"wrong audio"
                )
                runner.prepare_audio(Path(tmp))
            self.assertFalse((Path(tmp) / "audio-data-uri.txt").exists())

    def test_video_data_uri_contains_the_checked_in_clip(self):
        uri = runner.video_data_uri()
        header, encoded = uri.split(",", 1)
        self.assertEqual(header, "data:video/mp4;base64")
        self.assertEqual(base64.b64decode(encoded), runner.VIDEO_PATH.read_bytes())

    def test_changed_video_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"not the validated video")
            with self.assertRaisesRegex(ValueError, "Video fixture changed"):
                runner.video_data_uri(path)

    def test_zero_or_ignored_rust_tests_cannot_pass(self):
        for stdout in (
            "test result: ok. 0 passed; 0 failed; 0 ignored;",
            "test result: ok. 0 passed; 0 failed; 1 ignored;",
        ):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.CompletedProcess([], 0, stdout=stdout)
                with (
                    patch.object(runner.subprocess, "run", return_value=result),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(
                        RuntimeError, "did not execute exactly once"
                    ),
                ):
                    runner.run_rust_test(Path("binary"), {}, "video-test", Path(tmp))
                self.assertEqual((Path(tmp) / "rust-test.log").read_text(), stdout)

    def test_rust_video_test_is_explicitly_selected_and_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.CompletedProcess(
                [], 0, stdout="test result: ok. 1 passed; 0 failed; 0 ignored;"
            )
            with (
                patch.object(runner.subprocess, "run", return_value=result) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.run_rust_test(Path("binary"), {}, "video-test", Path(tmp))
            self.assertEqual(
                run.call_args.args[0][:4],
                ["binary", "--exact", "video-test", "--ignored"],
            )

    def test_video_failures_are_recorded_as_failures_in_junit(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run_image_test.py",
                "--media",
                "video",
                "--binary",
                "unused",
                "--config",
                "unused",
                "--results",
                tmp,
            ]
            with (
                patch("sys.argv", argv),
                patch.object(runner, "run", side_effect=RuntimeError("engine failed")),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(), 1)
            suite = ET.parse(Path(tmp) / "junit.xml").getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertEqual(suite.attrib["skipped"], "0")
            case = suite.find("testcase")
            self.assertEqual(
                case.attrib["name"], "tests::e2e::e2e_real_openengine_video"
            )
            self.assertIn("engine failed", case.find("failure").text)


if __name__ == "__main__":
    unittest.main()
