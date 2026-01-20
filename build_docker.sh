#!/bin/bash

# Olympus Echo Backend Docker Build Script
# This script builds the Docker image for the Olympus Echo backend service

set -e  # Exit on any error

# Configuration
IMAGE_NAME="olympus-echo-backend"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🏗️  Building Olympus Echo Backend Docker Image..."
echo "Image name: ${FULL_IMAGE_NAME}"

# Build the Docker image
docker build -t "${FULL_IMAGE_NAME}" .

# Get image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "table {{.Size}}" | tail -n 1)

echo "✅ Build completed successfully!"
echo "📦 Image: ${FULL_IMAGE_NAME}"
echo "📏 Size: ${IMAGE_SIZE}"

# Optional: Tag with registry if provided
if [ ! -z "$DOCKER_REGISTRY" ]; then
    REGISTRY_IMAGE="${DOCKER_REGISTRY}/${FULL_IMAGE_NAME}"
    docker tag "${FULL_IMAGE_NAME}" "${REGISTRY_IMAGE}"
    echo "🏷️  Tagged: ${REGISTRY_IMAGE}"
fi

echo ""
echo "🚀 To run the container:"
echo "   docker run -p 6068:6068 -e PORT=6068 ${FULL_IMAGE_NAME}"
echo ""
echo "📋 Available environment variables:"
echo "   - PORT: Server port (default: 6068)"
echo "   - HOST: Server host (default: 0.0.0.0)"
echo "   - WORKERS: Number of workers (default: 1)"
echo "   - RELOAD: Enable reload (default: false)"

echo ""
echo "🔍 To inspect the image:"
echo "   docker inspect ${FULL_IMAGE_NAME}"