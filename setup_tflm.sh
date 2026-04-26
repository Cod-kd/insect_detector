#!/bin/bash
# =============================================================================
# setup_tflm.sh  —  Csak a detector_micro.cpp fordításához szükséges fájlok
#
# Használat:
#   chmod +x setup_tflm.sh
#   ./setup_tflm.sh ./tflite-micro
# =============================================================================

set -e
TFLM_REPO="${1:-./tflite-micro}"

if [ ! -d "$TFLM_REPO" ]; then
    echo "[hiba] Nem található: $TFLM_REPO"
    echo "  git clone --depth=1 https://github.com/tensorflow/tflite-micro.git"
    exit 1
fi

TFLM_REPO="$(realpath "$TFLM_REPO")"
PROJECT="$(realpath "$(dirname "$0")")"

echo "Repo   : $TFLM_REPO"
echo "Projekt: $PROJECT"
echo ""

mkdir -p "$PROJECT/tflm_srcs"
mkdir -p "$PROJECT/tflm_includes"
mkdir -p "$PROJECT/third_party/flatbuffers/include"
mkdir -p "$PROJECT/third_party/gemmlowp"
mkdir -p "$PROJECT/models"

# ─── 1. Header fa ─────────────────────────────────────────────────────────────
echo "[1/4] Headerek másolása..."
cp -r "$TFLM_REPO/tensorflow" "$PROJECT/tflm_includes/"
echo "  ✓ tensorflow/ header fa"

# ─── 2. Core .cc forrásfájlok ─────────────────────────────────────────────────
echo "[2/4] Core TFLite Micro forrásfájlok..."
MICRO="$TFLM_REPO/tensorflow/lite/micro"

CORE_FILES=(
    "micro_interpreter.cc"
    "micro_allocator.cc"
    "micro_context.cc"
    "micro_graph.cc"
    "micro_log.cc"
    "micro_profiler.cc"
    "micro_resource_variable.cc"
    "micro_utils.cc"
    "recording_micro_allocator.cc"
    "system_setup.cc"
    "arena_allocator/non_persistent_arena_buffer_allocator.cc"
    "arena_allocator/persistent_arena_buffer_allocator.cc"
    "arena_allocator/single_arena_buffer_allocator.cc"
    "memory_planner/greedy_memory_planner.cc"
    "memory_planner/linear_memory_planner.cc"
    "memory_planner/non_persistent_buffer_planner_shim.cc"
)
for f in "${CORE_FILES[@]}"; do
    src="$MICRO/$f"
    [ -f "$src" ] \
        && cp "$src" "$PROJECT/tflm_srcs/$(basename "$f")" && echo "  ✓ $f" \
        || echo "  ! hiányzik: $f"
done

# ─── 3. Kernelek — csak a modell op-jai ───────────────────────────────────────
echo "[3/4] Kernel forrásfájlok (QUANTIZE, PAD, CONV_2D, DEPTHWISE_CONV_2D,"
echo "      ADD, MEAN, FULLY_CONNECTED, SOFTMAX, DEQUANTIZE)..."
KERNEL_FILES=(
    "kernels/quantize.cc"
    "kernels/dequantize.cc"
    "kernels/pad.cc"
    "kernels/conv.cc"
    "kernels/depthwise_conv.cc"
    "kernels/add.cc"
    "kernels/reduce.cc"
    "kernels/fully_connected.cc"
    "kernels/softmax.cc"
    "kernels/kernel_util.cc"
    "kernels/micro_utils.cc"
)
for f in "${KERNEL_FILES[@]}"; do
    src="$MICRO/$f"
    [ -f "$src" ] \
        && cp "$src" "$PROJECT/tflm_srcs/kernel_$(basename "$f")" && echo "  ✓ $f" \
        || echo "  ! hiányzik: $f"
done

# ─── 4. TFLite common + third_party ──────────────────────────────────────────
echo "[4/4] Common + third_party..."
LITE="$TFLM_REPO/tensorflow/lite"
LITE_FILES=(
    "core/c/common.cc"
    "core/api/error_reporter.cc"
    "core/api/flatbuffer_conversions.cc"
    "core/api/op_resolver.cc"
    "core/api/tensor_utils.cc"
    "kernels/internal/quantization_util.cc"
    "kernels/internal/portable_tensor_utils.cc"
    "kernels/internal/common.cc"
    "kernels/kernel_util.cc"
)
for f in "${LITE_FILES[@]}"; do
    src="$LITE/$f"
    [ -f "$src" ] \
        && cp "$src" "$PROJECT/tflm_srcs/lite_$(basename "$f")" && echo "  ✓ lite/$f" \
        || echo "  ! hiányzik: lite/$f"
done

FB="$TFLM_REPO/third_party/flatbuffers/include"
[ -d "$FB" ] \
    && cp -r "$FB/." "$PROJECT/third_party/flatbuffers/include/" && echo "  ✓ flatbuffers" \
    || echo "  ! flatbuffers nem találtam"

GEMM="$TFLM_REPO/third_party/gemmlowp"
[ -d "$GEMM" ] \
    && cp -r "$GEMM/." "$PROJECT/third_party/gemmlowp/" && echo "  ✓ gemmlowp" \
    || echo "  ! gemmlowp hiányzik"

# ─── Összefoglaló ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Kész! $(ls "$PROJECT/tflm_srcs/" | wc -l) .cc fájl → tflm_srcs/"
echo ""
echo "  Következő lépések:"
echo ""
echo "  1. Docker image build:"
echo "       docker build -t detector-build ."
echo ""
echo "  2. Fordítás (model.h generálás + g++):"
echo "       docker run --rm -v \$(pwd):/project detector-build bash /project/build.sh"
echo ""
echo "  3. Flask szerver + detektor indítása:"
echo "       python server.py &"
echo "       docker run --rm --device=/dev/video0 \\"
echo "         -e DISPLAY=\$DISPLAY \\"
echo "         -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "         -v \$(pwd):/project \\"
echo "         detector-build /project/detector"
echo "════════════════════════════════════════════════"
