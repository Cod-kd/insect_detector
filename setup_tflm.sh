#!/bin/bash
# =============================================================================
# setup_tflm.sh  —  Copies required files from tflite-micro repo
#
# Usage:
#   chmod +x setup_tflm.sh
#   ./setup_tflm.sh ./tflite-micro
# =============================================================================

set -e
TFLM_REPO="${1:-./tflite-micro}"

if [ ! -d "$TFLM_REPO" ]; then
    echo "[error] Not found: $TFLM_REPO"
    echo "  git clone --depth=1 https://github.com/tensorflow/tflite-micro.git"
    exit 1
fi

TFLM_REPO="$(realpath "$TFLM_REPO")"
PROJECT="$(realpath "$(dirname "$0")")"

echo "Repo   : $TFLM_REPO"
echo "Project: $PROJECT"
echo ""

mkdir -p "$PROJECT/tflm_srcs"
mkdir -p "$PROJECT/tflm_includes"
mkdir -p "$PROJECT/third_party"
mkdir -p "$PROJECT/models"

# ─── 1. Headers (full tree — keeps all internal includes intact) ──────────────
echo "[1/4] Copying headers..."
cp -r "$TFLM_REPO/tensorflow" "$PROJECT/tflm_includes/"

# signal/ library (required by micro_mutable_op_resolver.h)
if [ -d "$TFLM_REPO/signal" ]; then
    cp -r "$TFLM_REPO/signal" "$PROJECT/tflm_includes/"
    echo "  ✓ signal/ headers"
else
    echo "  ! signal/ not found in repo (may cause irfft.h error)"
fi
echo "  ✓ tensorflow/ header tree"

# ─── 2. Third-party (from repo — guaranteed version match) ───────────────────
echo "[2/4] Copying third_party from repo (version-matched)..."
cp -r "$TFLM_REPO/third_party/." "$PROJECT/third_party/"
echo "  ✓ third_party/"

# ─── 3. Core .cc sources ─────────────────────────────────────────────────────
echo "[3/4] Core TFLite Micro sources..."
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
        || echo "  ! missing: $f"
done

# ─── 4. Kernels (only what the model uses) ───────────────────────────────────
echo "[4/4] Kernel sources (QUANTIZE, PAD, CONV_2D, DEPTHWISE_CONV_2D,"
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
        || echo "  ! missing: $f"
done

# ─── TFLite common (lite/ level, not micro) ───────────────────────────────────
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
        || echo "  ! missing: lite/$f"
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Done! $(ls "$PROJECT/tflm_srcs/" | wc -l) .cc files → tflm_srcs/"
echo ""
echo "  Next step:"
echo "       ./docker-build/_build.sh"
echo "════════════════════════════════════════════════"
