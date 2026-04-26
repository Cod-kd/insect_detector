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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy EVERYTHING from project root (important!)
COPY . .

# model.h generálása a .tflite tömbből (ha még nincs meg)
RUN if [ ! -f models/model.h ] && [ -f models/model.tflite ]; then \
        xxd -i models/model.tflite > models/model.h; \
    fi

# Compile
RUN g++ detector_micro.cpp tflm_srcs/*.cc \
    $(pkg-config --cflags --libs opencv4) \
    -Itflm_includes \
    -Ithird_party/flatbuffers/include \
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
