.PHONY: demo build setup-env clean help start-web start-workers start-full list-videos

# Default variables
DOCKER_IMAGE ?= video-processor:latest
DOCKER_IMAGE_CPU ?= video-processor:cpu
DEMO_VIDEO ?= $(shell find sample-videos -name "*.mp4" 2>/dev/null | head -1)
ENV_FILE ?= .env.local

# Auto-detect GPU availability (can be overridden with FORCE_CPU=true)
HAS_GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU" && echo "true" || echo "false")
HAS_NVIDIA_DOCKER := $(shell docker info 2>/dev/null | grep -q "nvidia" && echo "true" || echo "false")

# Determine which setup to use
ifeq ($(FORCE_CPU),true)
    USE_GPU_SETUP := false
else ifeq ($(HAS_GPU),true)
    ifeq ($(HAS_NVIDIA_DOCKER),true)
        USE_GPU_SETUP := true
    else
        USE_GPU_SETUP := false
    endif
else
    USE_GPU_SETUP := false
endif

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "System detection:"
	@echo "  GPU detected: $(HAS_GPU)"
	@echo "  Nvidia Docker: $(HAS_NVIDIA_DOCKER)" 
	@echo "  Will use: $(if $(filter true,$(USE_GPU_SETUP)),GPU setup,CPU-only setup)"
	@echo ""
	@echo "Override with: make demo FORCE_CPU=true"

check-system: ## Show detected system capabilities
	@echo "System capabilities:"
	@echo "  GPU detected: $(HAS_GPU)"
	@echo "  Nvidia Docker: $(HAS_NVIDIA_DOCKER)"
	@echo "  Will use: $(if $(filter true,$(USE_GPU_SETUP)),GPU setup,CPU-only setup)"
	@echo ""
	@echo "Force CPU mode: make demo FORCE_CPU=true"

setup-env: ## Copy example env file if .env doesn't exist
	@if [ ! -f .env ]; then \
		echo "Creating .env from example..."; \
		cp .env.local.example .env; \
		echo "✓ Created .env with demo-ready defaults"; \
	else \
		echo "✓ .env already exists"; \
	fi

build: ## Build the appropriate Docker image (GPU or CPU based on detection)
ifeq ($(USE_GPU_SETUP),true)
	@echo "Building GPU Docker image: $(DOCKER_IMAGE) (this may take several minutes)..."
	docker build -t $(DOCKER_IMAGE) .
else
	@echo "Building CPU Docker image: $(DOCKER_IMAGE_CPU) (this may take several minutes)..."
	docker build -f Dockerfile.cpu -t $(DOCKER_IMAGE_CPU) .
endif

build-gpu: ## Build the GPU Docker image
	@echo "Building GPU Docker image: $(DOCKER_IMAGE) (this may take several minutes)..."
	docker build -t $(DOCKER_IMAGE) .

build-cpu: ## Build the CPU Docker image  
	@echo "Building CPU Docker image: $(DOCKER_IMAGE_CPU) (this may take several minutes)..."
	docker build -f Dockerfile.cpu -t $(DOCKER_IMAGE_CPU) .


demo: setup-env ## Process a demo video synchronously using Django management command
	@echo "🎬 Starting demo video processing..."
	@if [ -z "$(DEMO_VIDEO)" ]; then \
		if [ -d "sample-videos" ] && [ -n "$$(find sample-videos -name '*.mp4' 2>/dev/null | head -1)" ]; then \
			echo "Using video from sample-videos directory"; \
		else \
			echo "❌ No video specified and no sample-videos found"; \
			echo ""; \
			echo "Options:"; \
			echo "  1. Specify a video: make demo DEMO_VIDEO=path/to/your/video.mp4"; \
			echo "  2. Create sample-videos/ and add MP4 files"; \
			echo ""; \
			echo "Example: make demo DEMO_VIDEO=~/Downloads/my-video.mp4"; \
			exit 1; \
		fi; \
	elif [ ! -f "$(DEMO_VIDEO)" ]; then \
		echo "❌ Video file not found: $(DEMO_VIDEO)"; \
		echo "Please check the path and try again"; \
		exit 1; \
	fi
	@echo "Video: $(DEMO_VIDEO)"
ifeq ($(USE_GPU_SETUP),true)
	@echo "Mode: GPU-accelerated processing"
	@echo "Setting up .env for GPU mode..."
	@sed 's/USE_CPU_ONLY=true/USE_CPU_ONLY=false/' .env.local.example > .env
	@echo "DOCKER_IMAGE=$(DOCKER_IMAGE)" >> .env
	@echo "Building GPU image (this may take several minutes for MegaDetector installation)..."
	docker build -t $(DOCKER_IMAGE) .
else
	@echo "Mode: CPU-only processing (no GPU/Nvidia Container Toolkit required)"
	@echo "Setting up .env for CPU mode..."
	@cp .env.local.example .env
	@echo "DOCKER_IMAGE=$(DOCKER_IMAGE_CPU)" >> .env
	@echo "Building CPU image (this may take several minutes for MegaDetector installation)..."
	docker build -f Dockerfile.cpu -t $(DOCKER_IMAGE_CPU) .
endif
	@echo "Setting up environment..."
	docker compose -f docker-compose-local.yaml up -d db redis
	@echo "Waiting for database to be ready..."
	@sleep 5
	@echo "Creating any needed migrations..."
	docker compose -f docker-compose-local.yaml run --rm admin python manage.py makemigrations
	@echo "Running database migrations..."
	docker compose -f docker-compose-local.yaml run --rm admin python manage.py migrate
	@echo "Processing video synchronously..."
	docker compose -f docker-compose-local.yaml run --rm \
		-v "$(PWD)/$(DEMO_VIDEO):/input/video.mp4:ro" \
		admin python manage.py process_video /input/video.mp4
	@echo ""
	@echo "🌐 You can also start the web interface to view results:"
	@echo "   make start-web"
	@echo "   Then visit: http://localhost:8000/admin"

clean: ## Stop containers and clean up
	@echo "🧹 Cleaning up..."
	docker compose -f docker-compose-local.yaml down
	docker system prune -f

start-web: setup-env ## Start the web interface only
	@echo "🌐 Starting web interface..."
	docker compose -f docker-compose-local.yaml up admin

start-workers: setup-env ## Start all processing workers
	@echo "⚙️  Starting processing workers..."
	docker compose -f docker-compose-local.yaml up chunk_video extract detect create_output

start-full: setup-env ## Start complete system (web + workers + services)
	@echo "🚀 Starting complete system..."
	docker compose -f docker-compose-local.yaml up

migrate: setup-env ## Create and run database migrations
	@echo "🔄 Creating any needed migrations..."
	docker compose -f docker-compose-local.yaml up -d db
	@sleep 3
	docker compose -f docker-compose-local.yaml run --rm admin python manage.py makemigrations
	@echo "🔄 Running database migrations..."
	docker compose -f docker-compose-local.yaml run --rm admin python manage.py migrate