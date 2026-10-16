.PHONY: install test lint run-api run-app clean

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev,causal]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ api/ app/

run-api:
	uvicorn api.main:app --reload --port 8000

run-app:
	streamlit run app/dashboard.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
