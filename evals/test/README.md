# Evaluation Tests

Tests for validating memory agent functionality.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest evals/test/

# Run with coverage
pytest evals/test/ --cov=memory --cov=utils

# Run specific test file
pytest evals/test/test_topic_retrieval.py
pytest evals/test/test_fact_extraction.py

# Run with verbose output
pytest evals/ -v
```

## Test Categories

### Topic Retrieval Tests (`test_topic_retrieval.py`)

Tests for semantic topic retrieval:

- **Short-term memory**: Tests retrieval within a session
  - Current topic messages are always included
  - Relevant closed topics are retrieved based on similarity
  - All closed topic summaries are included
  
- **Long-term memory**: Tests retrieval across sessions
  - Relevant topics from all past sessions are retrieved
  - Similarity threshold is respected (0.5)
  - Top-k limiting works correctly (max 3 topics)

### Fact Extraction Tests (`test_fact_extraction.py`)

Tests for extracting facts from conversations:

- Name extraction
- Location extraction
- Preferences extraction
- Multiple facts in one exchange
- Work/job information
- Interests/hobbies
- No false facts from generic greetings
- Only concrete facts, not inferences

## Test Data

Tests use temporary directories and clean up after themselves. No real data is modified.

## Notes

- Tests use real embeddings (not mocked) for accuracy
- Some tests may have non-deterministic results due to LLM variability
- Fact extraction tests check for key presence and value content, allowing for some variation in field names

