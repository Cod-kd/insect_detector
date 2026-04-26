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
    ccache \
    python3

ENV PATH="/usr/lib/ccache:$PATH"

WORKDIR /app

COPY tflm_standalone ./tflm_standalone

COPY docker-build/patch_compatibility.py .

# Fix compatibility.h
RUN python3 patch_compatibility.py

# Build TFLite as static lib — cached unless tflm_standalone changes
RUN find tflm_standalone -name "*.cc" \
        | grep -v "_test\." \
        | grep -v "test_" \
        | grep -v "mock_" \
        | grep -v "fake_" \
        | grep -v "decode_state" \
        | grep -v "tflm_standalone/signal/" \
        | grep -v "recording_micro_allocator" \
        | xargs -P4 -I{} g++ -c {} \
            -Itflm_standalone \
            -Itflm_standalone/third_party/flatbuffers/include \
            -Itflm_standalone/third_party/gemmlowp \
            -Itflm_standalone/third_party/ruy \
            -Itflm_standalone/third_party \
            -DTF_LITE_STATIC_MEMORY \
            -std=c++17 -O2 \
        && ar rcs tflm_lib.a *.o \
        && rm *.o

# Copy rest of project separately so above layer stays cached
COPY . .

# Generate model.h if not present
RUN if [ ! -f models/model.h ] && [ -f models/model.tflite ]; then \
        xxd -i models/model.tflite > models/model.h; \
    fi

# Only your code recompiles on changes
RUN g++ detector_micro.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -Itflm_standalone \
    -Itflm_standalone/third_party/flatbuffers/include \
    -Itflm_standalone/third_party/gemmlowp \
    -Itflm_standalone/third_party/ruy \
    -Itflm_standalone/third_party \
    tflm_lib.a \
    -lcurl -lm -lpthread \
    -DTF_LITE_STATIC_MEMORY \
    -std=c++17 -O2 \
    -o detector


# ===== RUNTIME STAGE =====
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libcurl4-openssl-dev

WORKDIR /app

COPY --from=builder /app/detector .

CMD ["./detector"]