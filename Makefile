.PHONY: setup test lint samples static web dev-api dev-web docker
setup:            ## install backend + frontend dependencies
	cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm ci
samples:          ## generate the synthetic sample scenes
	cd backend && .venv/bin/geoai-generate-samples --out data/samples
test:             ## run all tests
	cd backend && .venv/bin/pytest
	cd frontend && npm run typecheck && npm test
lint:
	cd backend && .venv/bin/ruff check src tests
static: samples   ## pre-compute the static demo used by GitHub Pages
	cd backend && .venv/bin/geoai-export-static --out ../frontend/public/demo
web: static       ## build the static site
	cd frontend && npm run build
dev-api:
	cd backend && .venv/bin/uvicorn --factory geoai.api.app:app_factory --reload --port 8000
dev-web:
	cd frontend && VITE_API_URL=http://localhost:8000 npm run dev
docker:
	docker compose up --build
