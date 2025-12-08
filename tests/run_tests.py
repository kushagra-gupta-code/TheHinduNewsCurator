"""
Test runner for The Hindu News Curator
Provides a unified interface to run all test suites
"""

import os
import sys
import subprocess
from pathlib import Path


def run_test_suite(test_name, test_file):
    """Run a specific test suite"""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name} Tests")
    print(f"{'=' * 60}")

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the test file
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / test_file)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        return result.returncode == 0

    except Exception as e:
        print(f"ERROR running {test_name}: {e}")
        return False


def main():
    """Main test runner"""
    print("The Hindu News Curator - Test Suite")
    print("=" * 60)

    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "config.py").exists():
        print("ERROR: config.py not found. Make sure you're in the project root.")
        return 1

    # Test suites to run
    test_suites = [
        ("Integration", "tests/test_integration.py"),
        ("OpenRouter Fallback", "tests/test_openrouter_consolidated.py"),
        ("Flask Scores", "tests/test_flask_scores.py"),
    ]

    results = {}

    for test_name, test_file in test_suites:
        if not (project_root / test_file).exists():
            print(f"WARNING: {test_file} not found, skipping {test_name} tests")
            continue

        success = run_test_suite(test_name, test_file)
        results[test_name] = success

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Suite Summary")
    print("=" * 60)

    passed = sum(1 for success in results.values() if success)
    total = len(results)

    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {test_name:20s} : {status}")

    print(f"\nOverall: {passed}/{total} test suites passed")

    if passed == total:
        print("ALL TEST SUITES PASSED!")
        return 0
    else:
        print("SOME TEST SUITES FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
