#!/bin/bash

# =================================================================
# OpenParse Test Environment Setup
# =================================================================
#
# Sets up Python testing environment for OpenParse:
# - Creates virtual environment
# - Installs test dependencies
# - Configures pytest
# - Runs test suite
#
# Usage:
#   ./setup_pytest.sh
#
# Requirements:
#   - Python 3.x
#   - pip
#   - git (for finding project root)
#
# Environment:
#   Creates:
#   - venv/          Virtual environment
#   - pytest.ini     Pytest configuration
#
# Tests:
#   Runs test suite from:
#   - src/tests/test_docparser_0.7.2.py
# =================================================================


# Find project root (where setup.py lives)
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
if [ ! -f "$PROJECT_ROOT/setup.py" ]; then
    echo "Error: Cannot find project root (setup.py)"
    exit 1
fi

# Create and activate virtual environment
# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Install only test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests
python -m pytest src/tests/test_docparser_0.7.2.py -v