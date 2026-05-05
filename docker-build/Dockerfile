# ===== BUILD STAGE =====
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libopencv-dev \
    libcurl4-openssl-dev \
    ccache \
    python3

ENV PATH="/usr/lib/ccache:$PATH"

WORKDIR /app

COPY tflm_standalone ./tflm_standalone

COPY docker-build/patch_compatibility.py .

# Fix compatibility.h
RUN python3 patch_compatibility.py

# Build TFLite as static lib — cached unless tflm_standalone changes
RUN mkdir -p tflm_objs && \
    find tflm_standalone -name "*.cc" \
        | grep -v "_test\." \
        | grep -v "test_" \
        | grep -v "mock_" \
        | grep -v "fake_" \
        | grep -v "decode_state" \
        | grep -v "tflm_standalone/signal/" \
        | grep -v "recording_micro_allocator" \
        | while read f; do \
            out="tflm_objs/$(echo $f | tr '/' '_').o"; \
            g++ -c "$f" -o "$out" \
                -Itflm_standalone \
                -Itflm_standalone/third_party/flatbuffers/include \
                -Itflm_standalone/third_party/gemmlowp \
                -Itflm_standalone/third_party/ruy \
                -Itflm_standalone/third_party \
                -DTF_LITE_STATIC_MEMORY \
                -std=c++17 -O2; \
          done && \
    ar rcs tflm_lib.a tflm_objs/*.o
    # && rm -rf tflm_objs   --- IN PROD ---

# Copy rest of project separately so above layer stays cached
COPY . .

# Compile — no xxd/model.h needed, model loaded at runtime from file
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
    # && rm -rf /var/lib/apt/lists/* --- IN PROD ---

WORKDIR /app

COPY --from=builder /app/detector .

# Model és config — futásidőben olvasódik
COPY models/model.tflite ./models/model.tflite
COPY detector.env .

EXPOSE 8080

CMD ["./detector"]
