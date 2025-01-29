## Test Environment Setup

```bash
# Create and activate virtual environment.
# From the project root:
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows

# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests in venv
pytest ./ -v
```

## Running Tests
``` bash
# Run all tests
pytest ./ -v

# Run specific test file
pytest ./test_docparser_0_7_2.py -v

# Run with coverage
pytest ./ --cov=openparse
```