import pytest
from pathlib import Path


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    expected_dir = Path(__file__).resolve().parent / "src"

    current_dir = Path.cwd()

    if not current_dir == expected_dir:
        pytest.exit(
            "Pytest must be run from the project root directory",
            returncode=1,
        )

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent