#!/bin/bash

set -e

IMAGE_NAME="insect-detector"
CONTAINER_NAME="insect-detector-AI"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Remove old container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🧹 Removing old container..."
    docker rm -f $CONTAINER_NAME
fi

# Create container with port mapping for frame receiver
echo "📦 Creating container..."
docker create \
    --name $CONTAINER_NAME \
    -p 8080:8080 \
    $IMAGE_NAME

# Start container
echo "🚀 Starting container..."
docker start -a $CONTAINER_NAME

echo "✅ Done"
