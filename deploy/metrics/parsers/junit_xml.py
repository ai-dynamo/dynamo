"""JUnit XML parser — extracted from TestResultsProcessor._parse_junit_xml."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List


def parse_junit_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a JUnit XML file and extract test results.

    Returns:
        List of dicts with keys: classname, name, time (float seconds),
        status (passed|failed|error|skipped), error_message
    """
    tests: List[Dict[str, Any]] = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle both <testsuites> and <testsuite> as root
        testsuites = root.findall(".//testsuite")
        if not testsuites and root.tag == "testsuite":
            testsuites = [root]

        for testsuite in testsuites:
            for testcase in testsuite.findall("testcase"):
                test_classname = testcase.get("classname", "")
                test_name = testcase.get("name", "")
                test_time = float(testcase.get("time", 0))
                test_status = "passed"
                error_msg = ""

                failure_elem = testcase.find("failure")
                if failure_elem is not None:
                    test_status = "failed"
                    error_msg = failure_elem.get("message", "")
                    if not error_msg and failure_elem.text:
                        error_msg = failure_elem.text

                error_elem = testcase.find("error")
                if error_elem is not None:
                    test_status = "error"
                    error_msg = error_elem.get("message", "")
                    if not error_msg and error_elem.text:
                        error_msg = error_elem.text

                skipped_elem = testcase.find("skipped")
                if skipped_elem is not None:
                    test_status = "skipped"
                    error_msg = skipped_elem.get("message", "")

                tests.append(
                    {
                        "classname": test_classname,
                        "name": test_name,
                        "time": test_time,
                        "status": test_status,
                        "error_message": error_msg,
                    }
                )

    except Exception as e:
        print(f"Error parsing JUnit XML {xml_path}: {e}")

    return tests


def parse_junit_xml_directory(dir_path: Path) -> List[Dict[str, Any]]:
    """Find all .xml files recursively under *dir_path* and parse each as JUnit XML."""
    all_tests: List[Dict[str, Any]] = []
    xml_files = sorted(dir_path.rglob("*.xml"))
    for xml_file in xml_files:
        all_tests.extend(parse_junit_xml(xml_file))
    return all_tests
