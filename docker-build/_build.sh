#!/bin/bash
set -e

IMAGE_NAME="insect-detector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

echo "📦 Building Docker image..."
docker build -f docker-build/Dockerfile -t $IMAGE_NAME .
echo "✅ Build complete"