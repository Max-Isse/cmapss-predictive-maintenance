.PHONY: image fetch bronze all test test-local lint shell

DC = docker compose run --rm pipeline

image:
	docker compose build pipeline

fetch:
	uv run cmapss fetch

bronze:
	$(DC) uv run cmapss bronze

all:
	$(DC) uv run cmapss all

test:
	$(DC) uv run pytest

test-local:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

shell:
	$(DC) bash
