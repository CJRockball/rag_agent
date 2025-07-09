# Test Suite for RAG Chat Application

This directory contains a comprehensive test suite for the RAG Chat application, including unit tests, integration tests, and performance tests.

## Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── test_agent_core.py       # Unit tests for RAG agent
├── test_streamlit_app.py    # Tests for Streamlit application
├── test_integration.py      # Integration tests
└── README.md               # This file
```

## Test Categories

### Unit Tests (`-m unit`)
- **test_agent_core.py**: Tests for individual agent functions
  - `retrieve()` function testing
  - `generate()` function testing  
  - `ask_rag()` function testing
  - State management testing
  - Error handling testing

- **test_streamlit_app.py**: Tests for Streamlit UI components
  - App initialization testing
  - User input processing
  - Chat history management
  - Error handling in UI

### Integration Tests (`-m integration`)
- **test_integration.py**: End-to-end system testing
  - Complete RAG pipeline testing
  - System resilience testing
  - Data flow validation
  - Performance under load

### Performance Tests (`-m performance`)
- Response time testing
- Memory usage testing  
- Concurrent request handling
- Load testing scenarios

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m performance   # Performance tests only

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test files
pytest tests/test_agent_core.py -v
pytest tests/test_streamlit_app.py -v
```

### Using the test runner script

```bash
# Make script executable
chmod +x run_tests.py

# Run all tests
python run_tests.py --all

# Run specific test suites
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --streamlit
python run_tests.py --agent

# Run with coverage
python run_tests.py --coverage

# Generate comprehensive report
python run_tests.py --report

# Run code quality checks
python run_tests.py --lint
python run_tests.py --type-check
```

## Test Configuration

### pytest.ini
Configures pytest behavior, markers, and coverage settings.

### conftest.py
Contains shared fixtures and test utilities:
- `test_environment`: Sets up test environment variables
- `sample_documents`: Provides sample documents for testing
- `mock_llm_response`: Mocks LLM responses
- `mock_chroma_db`: Mocks ChromaDB interactions

## Mocking Strategy

The test suite uses extensive mocking to:
- **Isolate components**: Each test focuses on specific functionality
- **Avoid external dependencies**: No real API calls during testing
- **Control test data**: Predictable inputs and outputs
- **Speed up execution**: Fast test runs without network calls

### Key Mocked Components
- **ChromaDB**: Database interactions mocked with controlled responses
- **Google Generative AI**: LLM calls mocked with predetermined responses
- **Streamlit components**: UI elements mocked for testing
- **Environment variables**: Test-specific configuration

## Test Data

### Sample Documents
```python
sample_documents = [
    Document(page_content="AI is the simulation of human intelligence..."),
    Document(page_content="Machine learning is a subset of AI..."),
    Document(page_content="Deep learning uses neural networks...")
]
```

### Test Questions
- "What is artificial intelligence?"
- "How does machine learning work?"
- "What are the applications of deep learning?"

## CI/CD Integration

### GitHub Actions Workflow
The test suite integrates with GitHub Actions for:
- **Multi-Python version testing**: Tests on Python 3.9, 3.10, 3.11
- **Code quality checks**: Linting, formatting, type checking
- **Coverage reporting**: Automated coverage reports
- **Performance monitoring**: Performance regression detection

### Workflow Triggers
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

## Performance Testing

### Metrics Tracked
- **Response Time**: Time from question to answer
- **Memory Usage**: Memory consumption during execution
- **Concurrent Requests**: System behavior under load
- **Error Rates**: Failure rates under stress

### Performance Thresholds
- Response time: < 5 seconds
- Memory usage: < 100MB increase
- Error rate: < 1% under normal load

## Error Simulation

The test suite simulates various error conditions:
- **Network errors**: Connection failures, timeouts
- **API errors**: Service unavailable, rate limiting
- **Database errors**: Connection issues, empty results
- **Input validation**: Invalid or malformed inputs

## Best Practices

### Writing Tests
1. **One assertion per test**: Clear, focused tests
2. **Descriptive names**: Test names explain what they test
3. **Arrange-Act-Assert**: Clear test structure
4. **Mock external dependencies**: Isolated, fast tests
5. **Test edge cases**: Error conditions, boundary values

### Test Maintenance
1. **Regular updates**: Keep tests current with code changes
2. **Performance monitoring**: Track test execution time
3. **Coverage goals**: Maintain >90% code coverage
4. **Documentation**: Keep test documentation updated

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install test dependencies
pip install -r test-requirements.txt
```

#### Mock Issues
```python
# Patch at the right level
@patch('module.function')  # Patch where it's used
def test_function(mock_func):
    pass
```

#### Streamlit Testing
```python
# Use Streamlit's testing framework
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("streamlit_app.py")
```

### Performance Issues
- Run tests with `--tb=short` for faster output
- Use `pytest-xdist` for parallel execution
- Skip slow tests with `-m "not slow"`

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Use existing fixtures where possible
4. Update documentation
5. Ensure tests pass in CI/CD

## Test Coverage Goals

- **Overall coverage**: >90%
- **Critical paths**: >95%
- **Error handling**: >80%
- **UI components**: >85%

## Dependencies

See `test-requirements.txt` for all testing dependencies:
- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-mock: Mocking utilities
- streamlit: For Streamlit testing
- Various mocking and utility libraries
