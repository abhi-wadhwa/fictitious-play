.PHONY: install dev test lint type-check format run-app clean docker-build docker-run

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

# Lint with ruff
lint:
	ruff check src/ tests/

# Type check with mypy
type-check:
	mypy src/

# Auto-format with ruff
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run Streamlit app
run-app:
	streamlit run src/viz/app.py

# Run CLI demo
demo:
	python examples/demo.py

# Build Docker image
docker-build:
	docker build -t fictitious-play .

# Run Docker container
docker-run:
	docker run -p 8501:8501 fictitious-play

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
