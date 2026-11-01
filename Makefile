.PHONY: image fetch bronze silver gold replay stream train policy score monitor api-image up down all test test-local lint shell

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

policy:
	$(DC) uv run cmapss policy

score:
	$(DC) uv run cmapss score

monitor:
	$(DC) uv run cmapss monitor --retrain

api-image:
	docker compose build mlflow  # builds cmapss-pm-api, used by both mlflow and api

up:
	docker compose up -d mlflow api

down:
	docker compose down

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
