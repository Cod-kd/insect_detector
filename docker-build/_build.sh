#!/bin/bash
set -e

IMAGE_NAME="kamikou/insect-detector"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [ ! -d tflm_standalone ]; then
    echo "❌ tflm_standalone/ not found — see README: 'Telepítés és futtatás' step 1"
    exit 1
fi

# Generate models/model.h if missing or older than model.tflite
if [ ! -f models/model.h ] || [ models/model.tflite -nt models/model.h ]; then
    echo "🧠 Generating models/model.h..."
    if command -v xxd >/dev/null 2>&1; then
        xxd -i models/model.tflite > models/model.h
    else
        # xxd-compatible fallback (same symbol names: models_model_tflite[_len])
        python3 - <<'PY'
data = open("models/model.tflite", "rb").read()
lines = [", ".join(f"0x{b:02x}" for b in data[i:i + 12]) for i in range(0, len(data), 12)]
with open("models/model.h", "w") as f:
    f.write("unsigned char models_model_tflite[] = {\n  " + ",\n  ".join(lines) + "\n};\n")
    f.write(f"unsigned int models_model_tflite_len = {len(data)};\n")
PY
    fi
fi

echo "📦 Building Docker image..."
docker build -f docker-build/Dockerfile -t $IMAGE_NAME .
echo "✅ Build complete"
