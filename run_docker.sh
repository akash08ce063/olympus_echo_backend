#!/bin/bash

# Olympus Echo Backend Docker Run Script
# This script provides easy commands to run the Docker container

set -e  # Exit on any error

# Configuration
IMAGE_NAME="olympus-echo-backend"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="olympus-echo-backend-container"
HOST_PORT=6068

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if container is running
is_container_running() {
    docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" | grep -q "${CONTAINER_NAME}"
}

# Function to stop container
stop_container() {
    if is_container_running; then
        print_info "Stopping running container..."
        docker stop "${CONTAINER_NAME}" > /dev/null 2>&1
        print_success "Container stopped"
    else
        print_info "No running container found"
    fi
}

# Function to remove container
remove_container() {
    if docker ps -a --filter "name=${CONTAINER_NAME}" | grep -q "${CONTAINER_NAME}"; then
        print_info "Removing existing container..."
        docker rm "${CONTAINER_NAME}" > /dev/null 2>&1
        print_success "Container removed"
    fi
}

# Main commands
case "${1:-run}" in
    "build")
        print_info "Building Docker image..."
        ./build_docker.sh
        ;;

    "run")
        # Check if image exists
        if ! docker images "${FULL_IMAGE_NAME}" | grep -q "${IMAGE_NAME}"; then
            print_error "Image ${FULL_IMAGE_NAME} not found. Building first..."
            ./build_docker.sh
        fi

        # Stop and remove existing container
        stop_container
        remove_container

        print_info "Starting container on port ${HOST_PORT}..."
        docker run -d \
            --name "${CONTAINER_NAME}" \
            -p "${HOST_PORT}:${HOST_PORT}" \
            -e PORT="${HOST_PORT}" \
            -e HOST=0.0.0.0 \
            -v "$(pwd)/logs:/usr/src/app/logs" \
            "${FULL_IMAGE_NAME}"

        print_success "Container started!"
        print_info "Container name: ${CONTAINER_NAME}"
        print_info "Access at: http://localhost:${HOST_PORT}"
        print_info "Check logs: docker logs ${CONTAINER_NAME}"
        ;;

    "stop")
        stop_container
        ;;

    "restart")
        stop_container
        print_info "Waiting 2 seconds..."
        sleep 2
        "$0" run
        ;;

    "logs")
        if is_container_running; then
            print_info "Showing container logs (press Ctrl+C to exit)..."
            docker logs -f "${CONTAINER_NAME}"
        else
            print_error "Container is not running"
            exit 1
        fi
        ;;

    "status")
        if is_container_running; then
            print_success "Container is running"
            docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        else
            print_warning "Container is not running"
        fi
        ;;

    "shell")
        if is_container_running; then
            print_info "Opening shell in container..."
            docker exec -it "${CONTAINER_NAME}" /bin/bash
        else
            print_error "Container is not running"
            exit 1
        fi
        ;;

    "clean")
        stop_container
        remove_container
        print_info "Cleaning up unused Docker resources..."
        docker image prune -f > /dev/null 2>&1
        docker container prune -f > /dev/null 2>&1
        print_success "Cleanup completed"
        ;;

    "help"|"-h"|"--help")
        echo "Olympus Echo Backend Docker Management Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run      - Build (if needed) and run the container (default)"
        echo "  build    - Build the Docker image only"
        echo "  stop     - Stop the running container"
        echo "  restart  - Restart the container"
        echo "  logs     - Show container logs"
        echo "  status   - Show container status"
        echo "  shell    - Open shell in running container"
        echo "  clean    - Stop, remove container and clean up resources"
        echo "  help     - Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  PORT     - Host port to bind (default: 6068)"
        echo "  HOST     - Container host binding (default: 0.0.0.0)"
        echo "  WORKERS  - Number of uvicorn workers (default: 1)"
        ;;

    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac