.PHONY: help install lock run test eval eval-fast spider-setup spider-eval bird-setup bench bench-spider bench-bird bench-ablate bench-plan bench-oracle lint up down logs rebuild ps clean

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
	@echo "  make bird-setup   Download BIRD mini-dev (~800 MB)"
	@echo "  make bench-oracle Harness self-check on Spider+BIRD with gold SQL (no LLM)"
	@echo "  make bench-plan   Show uncached questions for the ablation set (no spend)"
	@echo "  make bench-spider Spider dev, 200-question stratified subset"
	@echo "  make bench-bird   BIRD mini-dev, evidence on and off"
	@echo "  make bench-ablate Run benchmarks/ablations.yaml with CIs + McNemar"
	@echo "  make bench        bench-spider + bench-bird + bench-ablate"
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

bird-setup:
	bash benchmarks/bird_setup.sh

bench-oracle:
	$(RUN) python -m benchmarks.spider_eval --oracle
	$(RUN) python -m benchmarks.bird_eval --oracle

bench-plan:
	$(RUN) python -m benchmarks.ablate --plan

bench-spider:
	$(RUN) python -m benchmarks.spider_eval --subset 200 --concurrency 4

bench-bird:
	$(RUN) python -m benchmarks.bird_eval --evidence on --concurrency 4
	$(RUN) python -m benchmarks.bird_eval --evidence off --concurrency 4

bench-ablate:
	$(RUN) python -m benchmarks.ablate

bench: bench-spider bench-bird bench-ablate

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
