.PHONY: help install lock run test eval eval-fast spider-setup spider-eval lint up down logs rebuild ps clean

help:
	@echo "DataAgent — common tasks"
	@echo ""
	@echo "  make install   Install locked dependencies with uv (creates .venv)"
	@echo "  make lock      Re-lock deps and regenerate requirements.txt"
	@echo "  make run       Run Streamlit locally"
	@echo "  make test      Run pytest suite"
	@echo "  make eval      Run benchmark eval (LLM judge on)"
	@echo "  make eval-fast Run benchmark eval (no LLM judge)"
	@echo "  make spider-setup Download Spider 1.0 dev set (~95 MB)"
	@echo "  make spider-eval  Run Spider text-to-SQL eval (first 50)"
	@echo "  make up        Start Docker stack (build + detach)"
	@echo "  make down      Stop Docker stack"
	@echo "  make logs      Tail container logs"
	@echo "  make rebuild   Rebuild image from scratch and restart"
	@echo "  make ps        Show container status"
	@echo "  make clean     Remove caches and build artifacts"

UV ?= uv
RUN := $(UV) run

install:
	$(UV) sync

# requirements.txt is exported from uv.lock for Docker / Streamlit Cloud.
lock:
	$(UV) lock
	$(UV) export --no-hashes --no-dev --format requirements-txt -o requirements.txt

run:
	$(RUN) streamlit run app.py

test:
	$(RUN) pytest tests/ -q

eval:
	$(RUN) python -m benchmarks.runner

eval-fast:
	$(RUN) python -m benchmarks.runner --no-judge

spider-setup:
	bash benchmarks/spider_setup.sh

spider-eval:
	$(RUN) python -m benchmarks.spider_eval --limit 50

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

ps:
	docker compose ps

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
