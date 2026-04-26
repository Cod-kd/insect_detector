# ===== BUILD STAGE =====
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libopencv-dev \
    libcurl4-openssl-dev \
    tzdata \
    xxd \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# ── Fetch exact third-party deps from tflite-micro repo at build time ─────────
# This guarantees version match regardless of what setup_tflm.sh copied locally.
# We only clone the specific third_party folders we need (sparse checkout).
RUN git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/tensorflow/tflite-micro.git /tmp/tflm && \
    cd /tmp/tflm && \
    git sparse-checkout set third_party && \
    cp -r /tmp/tflm/third_party /app/ && \
    rm -rf /tmp/tflm

# Generate model.h from .tflite if not already present
RUN if [ ! -f models/model.h ] && [ -f models/model.tflite ]; then \
        xxd -i models/model.tflite > models/model.h; \
    fi

# Compile
RUN g++ detector_micro.cpp tflm_srcs/*.cc \
    $(pkg-config --cflags --libs opencv4) \
    -Itflm_includes \
    -Ithird_party/flatbuffers/include \
    -Ithird_party/gemmlowp \
    -Ithird_party/ruy \
    -Ithird_party/kissfft \
    -Ithird_party \
    -lcurl -lm -lpthread \
    -DTF_LITE_STATIC_MEMORY \
    -DTF_LITE_DISABLE_X86_NEON \
    -std=c++17 -O2 \
    -o detector


# ===== RUNTIME STAGE =====
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/detector .

CMD ["./detector"]
