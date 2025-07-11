#!/usr/bin/env python3
"""
Test runner script for the RAG Chat application.
Provides different test execution modes and reporting options.
"""

import os
import sys
import subprocess
import argparse

# from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0


def run_unit_tests():
    """Run unit tests."""
    cmd = "python -m pytest tests/ -m unit -v"
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = "python -m pytest tests/ -m integration -v"
    return run_command(cmd, "Integration Tests")


def run_all_tests():
    """Run all tests."""
    cmd = "python -m pytest tests/ -v"
    return run_command(cmd, "All Tests")


def run_tests_with_coverage():
    """Run tests with coverage report."""
    cmd = "python -m pytest tests/ --cov=. --cov-report=html \
        --cov-report=term-missing -v"
    return run_command(cmd, "Tests with Coverage")


def run_performance_tests():
    """Run performance tests."""
    cmd = "python -m pytest tests/ -m performance -v"
    return run_command(cmd, "Performance Tests")


def run_streamlit_tests():
    """Run Streamlit-specific tests."""
    cmd = "python -m pytest tests/test_streamlit_app.py -v"
    return run_command(cmd, "Streamlit Tests")


def run_agent_tests():
    """Run agent-specific tests."""
    cmd = "python -m pytest tests/test_agent_core.py -v"
    return run_command(cmd, "Agent Tests")


def run_linting():
    """Run code linting."""
    commands = [
        (
            "python -m black --check .",
            "Black formatting check",
        ),
        ("python -m flake8 .", "Flake8 linting"),
        (
            "python -m isort --check-only .",
            "Import sorting check",
        ),
    ]

    success = True
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            success = False

    return success


def run_type_checking():
    """Run type checking."""
    cmd = "python -m mypy ."
    return run_command(cmd, "Type Checking")


def run_security_check():
    """Run security checks."""
    cmd = "python -m bandit -r . -f json"
    return run_command(cmd, "Security Check")


def setup_test_environment():
    """Set up test environment."""
    print("Setting up test environment...")

    # Create necessary directories
    os.makedirs("tests", exist_ok=True)
    os.makedirs("htmlcov", exist_ok=True)

    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["GOOGLE_API_KEY"] = "test_key"

    print("Test environment setup complete.")


def generate_test_report():
    """Generate comprehensive test report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 60)

    results = {}

    # Run different test suites
    test_suites = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("Streamlit Tests", run_streamlit_tests),
        ("Agent Tests", run_agent_tests),
        ("Code Quality", run_linting),
        ("Type Checking", run_type_checking),
    ]

    for name, test_func in test_suites:
        try:
            results[name] = test_func()
        except Exception as e:
            results[name] = False
            print(f"Error running {name}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name: <20} {status}")

    # Overall result
    all_passed = all(results.values())
    overall_status = (
        "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    )
    print(f"\nOverall Status: {overall_status}")

    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for RAG Chat application"
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run Streamlit tests only",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run agent tests only",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests only",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage",
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting only",
    )
    parser.add_argument(
        "--type-check",
        action="store_true",
        help="Run type checking only",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security checks only",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive test report",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up test environment",
    )

    args = parser.parse_args()

    # Set up test environment
    setup_test_environment()

    success = True

    if args.setup:
        setup_test_environment()
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.streamlit:
        success = run_streamlit_tests()
    elif args.agent:
        success = run_agent_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.coverage:
        success = run_tests_with_coverage()
    elif args.lint:
        success = run_linting()
    elif args.type_check:
        success = run_type_checking()
    elif args.security:
        success = run_security_check()
    elif args.all:
        success = run_all_tests()
    elif args.report:
        success = generate_test_report()
    else:
        # Default: run all tests
        success = run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
