.PHONY: image fetch bronze silver gold replay stream train all test test-local lint shell

DC = docker compose run --rm pipeline

image:
	docker compose build pipeline

fetch:
	uv run cmapss fetch

bronze:
	$(DC) uv run cmapss bronze

silver:
	$(DC) uv run cmapss silver

gold:
	$(DC) uv run cmapss gold

replay:
	$(DC) uv run cmapss replay --batches $(or $(BATCHES),1)

stream:
	$(DC) uv run cmapss stream

train:
	$(DC) uv run cmapss train

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
