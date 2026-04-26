# ===== BUILD STAGE =====
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libopencv-dev \
    libcurl4-openssl-dev \
    xxd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Generate model.h if not present
RUN if [ ! -f models/model.h ] && [ -f models/model.tflite ]; then \
        xxd -i models/model.tflite > models/model.h; \
    fi

# Build the TFLite Micro static library from standalone
RUN cd tflm_standalone && \
    make -f tensorflow/lite/micro/tools/make/Makefile \
    microlite -j4 && \
    mkdir -p /app/tflm_lib && \
    cp tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a \
    /app/tflm_lib/

# Compile
RUN g++ detector_micro.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -Itflm_standalone \
    -Itflm_standalone/third_party/flatbuffers/include \
    -Itflm_standalone/third_party/gemmlowp \
    -Itflm_standalone/third_party/ruy \
    -Itflm_standalone/third_party \
    -Ltflm_lib -ltensorflow-microlite \
    -lcurl -lm -lpthread \
    -DTF_LITE_STATIC_MEMORY \
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