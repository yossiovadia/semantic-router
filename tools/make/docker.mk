# ======== docker.mk ============
# = Docker build and management =
# ======== docker.mk ============

##@ Docker

# Docker image tags
DOCKER_REGISTRY ?= ghcr.io/vllm-project/semantic-router
DOCKER_TAG ?= latest

# Default docker compose environment
# Point Compose to the relocated main stack by default; override by exporting COMPOSE_FILE
export COMPOSE_FILE ?= deploy/docker-compose/docker-compose.yml
# Keep a stable project name so network/volume names are predictable across runs
export COMPOSE_PROJECT_NAME ?= semantic-router

# Build all Docker images
docker-build-all: ## Build all Docker images
docker-build-all: docker-build-extproc docker-build-llm-katan docker-build-dashboard docker-build-precommit

# Build extproc Docker image
docker-build-extproc: ## Build extproc Docker image
docker-build-extproc:
	@$(LOG_TARGET)
	@echo "Building extproc Docker image..."
	@$(CONTAINER_RUNTIME) build -f Dockerfile.extproc -t $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG) .

# Build llm-katan Docker image
docker-build-llm-katan: ## Build llm-katan Docker image
docker-build-llm-katan:
	@$(LOG_TARGET)
	@echo "Building llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) build -f e2e-tests/llm-katan/Dockerfile -t $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG) e2e-tests/llm-katan/

# Build dashboard Docker image
docker-build-dashboard: ## Build dashboard Docker image
docker-build-dashboard:
	@$(LOG_TARGET)
	@echo "Building dashboard Docker image..."
	@$(CONTAINER_RUNTIME) build -f dashboard/backend/Dockerfile -t $(DOCKER_REGISTRY)/dashboard:$(DOCKER_TAG) .

# Build precommit Docker image
docker-build-precommit: ## Build precommit Docker image
docker-build-precommit:
	@$(LOG_TARGET)
	@echo "Building precommit Docker image..."
	@$(CONTAINER_RUNTIME) build -f Dockerfile.precommit -t $(DOCKER_REGISTRY)/precommit:$(DOCKER_TAG) .

# Test llm-katan Docker image locally
docker-test-llm-katan: ## Test llm-katan Docker image locally
docker-test-llm-katan:
	@$(LOG_TARGET)
	@echo "Testing llm-katan Docker image..."
	@curl -f http://localhost:8000/v1/models || (echo "Models endpoint failed" && exit 1)
	@echo "\n✅ llm-katan Docker image test passed"

# Run llm-katan Docker image locally
docker-run-llm-katan: ## Run llm-katan Docker image locally
docker-run-llm-katan: docker-build-llm-katan
	@$(LOG_TARGET)
	@echo "Running llm-katan Docker image on port 8000..."
	@echo "Access the server at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@$(CONTAINER_RUNTIME) run --rm -p 8000:8000 $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Run llm-katan with custom served model name
docker-run-llm-katan-custom: ## Run with custom served model name, by append SERVED_NAME=name
docker-run-llm-katan-custom:
	@$(LOG_TARGET)
	@echo "Running llm-katan with custom served model name..."
	@echo "Usage: make docker-run-llm-katan-custom SERVED_NAME=your-served-model-name"
	@if [ -z "$(SERVED_NAME)" ]; then \
		echo "Error: SERVED_NAME variable is required"; \
		echo "Example: make docker-run-llm-katan-custom SERVED_NAME=claude-3-haiku"; \
		exit 1; \
	fi
	@$(CONTAINER_RUNTIME) run --rm -p 8000:8000 $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG) \
		llm-katan --model "Qwen/Qwen3-0.6B" --served-model-name "$(SERVED_NAME)" --host 0.0.0.0 --port 8000

# Clean up Docker images
docker-clean: ## Clean up Docker images
docker-clean:
	@$(LOG_TARGET)
	@echo "Cleaning up Docker images..."
	@$(CONTAINER_RUNTIME) image prune -f
	@echo "Docker cleanup completed"

# Push Docker images (for CI/CD)
docker-push-all: ## Build all Docker images
docker-push-all: docker-push-extproc docker-push-llm-katan
	@$(LOG_TARGET)
	@echo "All Docker images pushed successfully"

docker-push-extproc: ## Push extproc Docker image
docker-push-extproc:
	@$(LOG_TARGET)
	@echo "Pushing extproc Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/extproc:$(DOCKER_TAG)

docker-push-llm-katan: ## Push llm-katan Docker image
docker-push-llm-katan:
	@$(LOG_TARGET)
	@echo "Pushing llm-katan Docker image..."
	@$(CONTAINER_RUNTIME) push $(DOCKER_REGISTRY)/llm-katan:$(DOCKER_TAG)

# Docker compose build flag logic
# Usage: make docker-compose-up REBUILD=1  (forces image rebuild)
REBUILD ?=
BUILD_FLAG=$(if $(REBUILD),--build,)

# Docker compose shortcuts (no rebuild by default)
docker-compose-up: ## Start services (default includes llm-katan; REBUILD=1 to rebuild)
docker-compose-up:
	@$(LOG_TARGET)
	@echo "Starting services with docker-compose (default includes llm-katan) (REBUILD=$(REBUILD))..."
	@docker compose --profile llm-katan up -d $(BUILD_FLAG)

docker-compose-up-testing: ## Start with testing profile (REBUILD=1 optional)
docker-compose-up-testing:
	@$(LOG_TARGET)
	@echo "Starting services with testing profile (REBUILD=$(REBUILD))..."
	@docker compose --profile testing up -d $(BUILD_FLAG)

docker-compose-up-llm-katan: ## Start with llm-katan profile (REBUILD=1 optional)
docker-compose-up-llm-katan:
	@$(LOG_TARGET)
	@echo "Starting services with llm-katan profile (REBUILD=$(REBUILD))..."
	@docker compose --profile llm-katan up -d $(BUILD_FLAG)

# Start core services only (closer to production; excludes llm-katan)
docker-compose-up-core: ## Start core services only (no llm-katan)
docker-compose-up-core:
	@$(LOG_TARGET)
	@echo "Starting core services (no llm-katan) (REBUILD=$(REBUILD))..."
	@docker compose up -d $(BUILD_FLAG)

# Explicit rebuild targets for convenience
docker-compose-rebuild: ## Force rebuild then start
docker-compose-rebuild: REBUILD=1
docker-compose-rebuild: docker-compose-up

docker-compose-rebuild-testing: ## Force rebuild (testing profile)
docker-compose-rebuild-testing: REBUILD=1
docker-compose-rebuild-testing: docker-compose-up-testing

docker-compose-rebuild-llm-katan: ## Force rebuild (llm-katan profile)
docker-compose-rebuild-llm-katan: REBUILD=1
docker-compose-rebuild-llm-katan: docker-compose-up-llm-katan

docker-compose-down:
docker-compose-down: ## Stop services (default includes llm-katan)
	@$(LOG_TARGET)
	@echo "Stopping docker-compose services (default includes llm-katan)..."
	@docker compose --profile llm-katan down

docker-compose-down-core: ## Stop core services only (no llm-katan)
docker-compose-down-core:
	@$(LOG_TARGET)
	@echo "Stopping core services only (no llm-katan)..."
	@docker compose down

docker-compose-down-testing: ## Stop services with testing profile
docker-compose-down-testing:
	@$(LOG_TARGET)
	@echo "Stopping services with testing profile..."
	@docker compose --profile testing down

docker-compose-down-llm-katan: ## Stop services with llm-katan profile
docker-compose-down-llm-katan:
	@$(LOG_TARGET)
	@echo "Stopping services with llm-katan profile..."
	@docker compose --profile llm-katan down

# Help target for Docker commands
docker-help:
docker-help: ## Show help for Docker-related make targets and environment variables
	@echo "Environment Variables:"
	@echo "  DOCKER_REGISTRY - Docker registry (default: ghcr.io/vllm-project/semantic-router)"
	@echo "  DOCKER_TAG      - Docker tag (default: latest)"
	@echo "  SERVED_NAME     - Served model name for custom runs"
