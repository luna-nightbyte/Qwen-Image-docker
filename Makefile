.PHONY: help init pull-models build pull-or-build up down restart logs clean env test shell

# Default target
.DEFAULT_GOAL := help

# Docker image settings
IMAGE_NAME := qwen-image
IMAGE_TAG := latest
FULL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

# Docker Compose settings
COMPOSE := docker-compose
COMPOSE_FILE := docker-compose.yml

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)Qwen-Image Docker - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. Run: make init          (Download models from HuggingFace)"
	@echo "  2. Run: make env           (Setup environment file)"
	@echo "  3. Run: make up            (Build and start services)"
	@echo "  4. Open: http://localhost:7860"

init: pull-models ## Initialize repository (download model submodules)
	@echo "$(GREEN)Repository initialized successfully!$(NC)"

pull-models: ## Download model submodules from HuggingFace
	@echo "$(CYAN)Downloading model submodules from HuggingFace...$(NC)"
	@if [ -d ".git" ]; then \
		git submodule init && \
		git submodule update --init --recursive --progress && \
		echo "$(GREEN)Model submodules downloaded successfully!$(NC)"; \
	else \
		echo "$(RED)Error: Not a git repository$(NC)" && exit 1; \
	fi

env: ## Setup environment file from example
	@if [ ! -f .env ]; then \
		echo "$(CYAN)Creating .env file from .env.example...$(NC)"; \
		cp .env.example .env && \
		echo "$(GREEN).env file created. Please edit it with your settings.$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists. Skipping...$(NC)"; \
	fi

pull-or-build: ## Attempt to pull Docker image, build if not available
	@echo "$(CYAN)Checking for pre-built Docker image...$(NC)"
	@if docker pull $(FULL_IMAGE) 2>/dev/null; then \
		echo "$(GREEN)Successfully pulled $(FULL_IMAGE)$(NC)"; \
	else \
		echo "$(YELLOW)No pre-built image found. Building locally...$(NC)"; \
		$(MAKE) build; \
	fi

build: ## Build Docker image
	@echo "$(CYAN)Building Docker image...$(NC)"
	$(COMPOSE) build
	@echo "$(GREEN)Docker image built successfully!$(NC)"

up: pull-or-build env ## Start services (pull/build image if needed, create env if needed)
	@echo "$(CYAN)Starting Qwen-Image services...$(NC)"
	$(COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "$(CYAN)Access the web interface at: http://localhost:7860$(NC)"

down: ## Stop and remove containers
	@echo "$(CYAN)Stopping services...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

restart: ## Restart services
	@echo "$(CYAN)Restarting services...$(NC)"
	$(COMPOSE) restart
	@echo "$(GREEN)Services restarted!$(NC)"

logs: ## Show container logs (follow mode)
	$(COMPOSE) logs -f

logs-tail: ## Show last 100 lines of logs
	$(COMPOSE) logs --tail=100

status: ## Show container status
	$(COMPOSE) ps

shell: ## Open bash shell in the running container
	$(COMPOSE) exec qwen-image bash

edit-up: env ## Start the edit demo service
	@echo "$(CYAN)Starting Qwen-Image Edit demo...$(NC)"
	$(COMPOSE) --profile edit up -d qwen-image-edit
	@echo "$(GREEN)Edit demo started!$(NC)"
	@echo "$(CYAN)Access the edit interface at: http://localhost:7861$(NC)"

edit-down: ## Stop the edit demo service
	$(COMPOSE) --profile edit down

rebuild: ## Rebuild and restart services (no cache)
	@echo "$(CYAN)Rebuilding Docker image (no cache)...$(NC)"
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d --force-recreate
	@echo "$(GREEN)Services rebuilt and restarted!$(NC)"

update: ## Update code and restart services
	@echo "$(CYAN)Updating repository...$(NC)"
	git pull
	$(MAKE) pull-models
	$(MAKE) rebuild

clean: ## Remove containers and volumes (WARNING: This deletes model cache!)
	@echo "$(RED)Removing all containers, networks, and volumes...$(NC)"
	@read -p "Are you sure? This will delete the HuggingFace cache! (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(COMPOSE) down -v && \
		echo "$(GREEN)Cleanup complete!$(NC)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled.$(NC)"; \
	fi

clean-containers: ## Remove containers only (keep volumes)
	@echo "$(CYAN)Removing containers...$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)Containers removed!$(NC)"

clean-images: ## Remove Docker images
	@echo "$(RED)Removing Docker images...$(NC)"
	@read -p "Are you sure? (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker rmi $(FULL_IMAGE) 2>/dev/null || true && \
		echo "$(GREEN)Images removed!$(NC)"; \
	else \
		echo "$(YELLOW)Image removal cancelled.$(NC)"; \
	fi

test-gpu: ## Test if GPU is accessible in Docker
	@echo "$(CYAN)Testing GPU accessibility...$(NC)"
	@docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi || \
		echo "$(RED)GPU test failed. Make sure nvidia-docker is installed.$(NC)"

check-env: ## Validate environment setup
	@echo "$(CYAN)Checking environment...$(NC)"
	@echo -n "Docker: "
	@docker --version || echo "$(RED)Not installed$(NC)"
	@echo -n "Docker Compose: "
	@docker-compose --version || echo "$(RED)Not installed$(NC)"
	@echo -n "NVIDIA Docker: "
	@docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 && \
		echo "$(GREEN)OK$(NC)" || echo "$(RED)Not available$(NC)"
	@echo -n "Git submodules: "
	@git submodule status | grep -q "^-" && echo "$(RED)Not initialized$(NC)" || echo "$(GREEN)OK$(NC)"
	@echo -n ".env file: "
	@[ -f .env ] && echo "$(GREEN)OK$(NC)" || echo "$(YELLOW)Missing (run 'make env')$(NC)"

install-deps: ## Install system dependencies (requires sudo)
	@echo "$(CYAN)Installing system dependencies...$(NC)"
	@echo "$(YELLOW)This requires sudo privileges.$(NC)"
	@command -v docker >/dev/null 2>&1 || { \
		echo "Installing Docker..."; \
		curl -fsSL https://get.docker.com -o get-docker.sh && \
		sudo sh get-docker.sh && \
		rm get-docker.sh; \
	}
	@echo "Installing Docker Compose plugin..."
	@sudo apt-get update && sudo apt-get install -y docker-compose-plugin
	@echo "$(GREEN)Dependencies installed!$(NC)"
	@echo "$(YELLOW)Note: You may need to install nvidia-docker separately for GPU support.$(NC)"

all: init env build up ## Run complete setup (init + env + build + start)
	@echo "$(GREEN)Complete setup finished!$(NC)"
	@echo "$(CYAN)Access the web interface at: http://localhost:7860$(NC)"
