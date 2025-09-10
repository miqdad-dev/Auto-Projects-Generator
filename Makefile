.PHONY: setup generate test lint

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

generate:
	python scripts/generate_next.py

test:
	@echo "Tests are per-generated project; run within the created folder."

lint:
	@echo "No repo-level lints; generated projects may include their own."
