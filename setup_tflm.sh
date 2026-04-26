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
mkdir -p "$PROJECT/third_party/ruy"
mkdir -p "$PROJECT/models"

# ─── 1. Header fa ─────────────────────────────────────────────────────────────
echo "[1/5] Headerek másolása..."
cp -r "$TFLM_REPO/tensorflow" "$PROJECT/tflm_includes/"
echo "  ✓ tensorflow/ header fa"

# ─── 2. Core .cc forrásfájlok ─────────────────────────────────────────────────
echo "[2/5] Core TFLite Micro forrásfájlok..."
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
echo "[3/5] Kernel forrásfájlok (QUANTIZE, PAD, CONV_2D, DEPTHWISE_CONV_2D,"
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

# ─── 4. TFLite common ─────────────────────────────────────────────────────────
echo "[4/5] TFLite common forrásfájlok..."
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

# ─── 5. Third-party headers ───────────────────────────────────────────────────
echo "[5/5] Third-party headerek..."

# flatbuffers → "flatbuffers/flatbuffers.h"
FB="$TFLM_REPO/third_party/flatbuffers/include"
if [ -d "$FB" ]; then
    cp -r "$FB/." "$PROJECT/third_party/flatbuffers/include/"
    echo "  ✓ flatbuffers (repóból)"
else
    echo "  ! flatbuffers nem a repóban — letöltés..."
    curl -sL https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz \
        | tar -xz -C /tmp
    cp -r /tmp/flatbuffers-23.5.26/include/flatbuffers \
          "$PROJECT/third_party/flatbuffers/include/"
    rm -rf /tmp/flatbuffers-23.5.26
    echo "  ✓ flatbuffers (letöltve)"
fi

# gemmlowp → "fixedpoint/fixedpoint.h"
GEMM="$TFLM_REPO/third_party/gemmlowp"
if [ -d "$GEMM" ]; then
    cp -r "$GEMM/." "$PROJECT/third_party/gemmlowp/"
    echo "  ✓ gemmlowp (repóból)"
else
    echo "  ! gemmlowp nem a repóban — letöltés..."
    curl -sL https://github.com/google/gemmlowp/archive/refs/heads/master.tar.gz \
        | tar -xz -C /tmp
    cp -r /tmp/gemmlowp-master/fixedpoint \
          /tmp/gemmlowp-master/internal \
          "$PROJECT/third_party/gemmlowp/"
    rm -rf /tmp/gemmlowp-master
    echo "  ✓ gemmlowp (letöltve)"
fi

# ruy → "ruy/profiler/instrumentation.h"
RUY="$TFLM_REPO/third_party/ruy"
if [ -d "$RUY" ]; then
    cp -r "$RUY/." "$PROJECT/third_party/ruy/"
    echo "  ✓ ruy (repóból)"
else
    echo "  ! ruy nem a repóban — letöltés..."
    curl -sL https://github.com/google/ruy/archive/refs/heads/master.tar.gz \
        | tar -xz -C /tmp
    cp -r /tmp/ruy-master/ruy "$PROJECT/third_party/ruy/"
    rm -rf /tmp/ruy-master
    echo "  ✓ ruy (letöltve)"
fi

# ─── Összefoglaló ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Kész! $(ls "$PROJECT/tflm_srcs/" | wc -l) .cc fájl → tflm_srcs/"
echo ""
echo "  Következő lépés:"
echo "       ./docker-build/_build.sh"
echo "════════════════════════════════════════════════"
